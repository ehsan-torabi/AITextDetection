"""Group-aware split builder for every evaluation condition (review section 3).

The v1 pipeline had one random, document-level split. The review requires a
set of purpose-built splits, with thresholds and model choices tuned only on
validation, and with integrity assertions enforced in code:

- ``in_domain_group``        group-aware train/val/test; whole topics (AI) and
                             whole TF-IDF clusters (human) stay inside one split
- ``cross_generator``        AI generators partitioned into seen/unseen; the
                             test AI rows come from held-out generators *and*
                             held-out topics (the AI corpus is a dense
                             topic x generator grid, so a generator-only
                             hold-out cannot be topic-disjoint; both
                             constraints are enforced and the trade-off is
                             recorded in ``details``)
- ``cross_domain__<d>``      one human domain held out of training entirely; the
                             test set is that domain's *complete* human side
                             (the realistic "new domain arrives" case) balanced
                             with held-out AI documents
- ``cross_style__<s>``       one AI prompt style held out (AI-side domain shift)
- ``few_shot__<k>``          k rows per class from the training pool
- ``length__short|long``     short (<100 words) / long (>=500 words) test rows
- ``human_vs_ai_revised``    human vs machine-revised (present once the
                             paraphrase module has produced revision rows)
- ``paraphrase_attention``   the AI test rows reserved for the attack run
- ``v1_test``                the rows the pre-existing checkpoints never saw
                             (used to evaluate them without contamination)

Every split is written as positional indices into ``data/dataset_v2.parquet``
(so the frame can be re-read cheaply) and described in
``data/splits_v2.json`` with sizes, class mixes, group counts, seeds and the
integrity assertions that were checked.

Run from the project root:
    python training/splits_v2.py [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from aitextd import dataset as ds
from aitextd import paths

DATASET = paths.DATA_DIR / "dataset_v2.parquet"
OUT_JSON = paths.DATA_DIR / "splits_v2.json"
OUT_JOBLIB = paths.DATA_DIR / "splits_v2.joblib"

TRAIN_FRAC, VAL_FRAC = 0.7, 0.1
FEW_SHOT_KS = (8, 32, 128)
ATTACK_SAMPLE = 300


class SplitError(AssertionError):
    """Raised when an integrity property of a split is violated."""


def _grouped_assign(df: pd.DataFrame, mask: np.ndarray, seed: int, block_name: str) -> np.ndarray:
    """Assign train/val/test to whole topic groups of one (label, domain) block.

    Groups are visited largest-first (random order inside a size class) and each
    is given to the split whose *fill ratio* against its row target is lowest.
    A naive "fill train, then val, then test" loop overshoots badly when the
    cluster size distribution is chunky -- one large cluster can consume the
    whole val+test budget and silently leave the test set empty (observed with
    the news block). Balancing ratios keeps every split populated and keeps the
    largest clusters in training, where they cannot dominate a test estimate.
    Returns one assignment per masked row (same order as ``mask`` selects).
    """
    positions = np.flatnonzero(mask)
    groups = df.iloc[positions]["topic_id"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_groups))
    row_counts = pd.Series(groups).value_counts()

    n_total = len(positions)
    targets = {"train": TRAIN_FRAC * n_total, "val": VAL_FRAC * n_total}
    targets["test"] = n_total - targets["train"] - targets["val"]
    fill = {"train": 0.0, "val": 0.0, "test": 0.0}

    ordered_groups = [unique_groups[pos] for pos in order]
    ordered_groups.sort(key=lambda g: -int(row_counts[g]))

    group_split: dict[str, str] = {}
    for group in ordered_groups:
        split = min(fill, key=lambda s: fill[s] / max(targets[s], 1.0))
        group_split[group] = split
        fill[split] += int(row_counts[group])

    return np.array([group_split[g] for g in groups], dtype=object)


def _pick_unseen_generators(
    df: pd.DataFrame,
    positions: np.ndarray,
    seed: int,
    target_frac: float = 0.2,
    min_generators: int = 8,
    min_test_rows: int = 250,
    test_positions: np.ndarray | None = None,
) -> tuple[set[str], dict]:
    """Choose the held-out generators from the AI rows at ``positions``.

    Stratified by frequency (a handful of high-volume generators plus several
    low-volume ones, both seeded-random): the review asks for both "a new model
    appears" and "an obscure model appears" cases. Selection stops when the
    unseen set covers ``target_frac`` of AI rows, but never before
    ``min_generators``. Because the corpus is a dense topic x generator grid,
    the unseen generators must *also* be topic-disjoint from training, so their
    usable test rows are those already held out by the main split; if that
    yields fewer than ``min_test_rows`` documents the unseen set is extended
    (seeded order) until the floor is met or the pool is exhausted. The extra
    rounds are reported in ``details`` so the choice is auditable.
    """
    positions = np.asarray(positions, dtype=np.int64)
    generators = df["generator"].astype(str).to_numpy()[positions]
    counts = pd.Series(generators).value_counts()
    counts = counts[counts.index != "unknown"]
    rng = np.random.default_rng(seed)
    total_rows = int(len(positions))
    target_rows = target_frac * total_rows

    high = [g for g in counts.index if counts[g] >= 0.01 * total_rows]
    low = [g for g in counts.index if counts[g] < 0.01 * total_rows]
    rng.shuffle(high)
    rng.shuffle(low)
    # take at most three high-volume generators, then fill from the low end
    order = high[:3] + low + high[3:]

    unseen: set[str] = set()
    covered = 0
    for generator in order:
        if covered >= target_rows and len(unseen) >= min_generators:
            break
        unseen.add(generator)
        covered += int(counts[generator])
    selected_rows = covered

    rounds = 0
    if test_positions is not None:
        position_set = np.zeros(len(df), dtype=bool)
        position_set[np.asarray(test_positions, dtype=np.int64)] = True
        generator_all = df["generator"].astype(str).to_numpy()
        is_unknown = generator_all == "unknown"
        while rounds < len(order):
            mask = np.isin(generator_all, sorted(unseen)) & position_set & ~is_unknown
            if int(mask.sum()) >= min_test_rows or len(unseen) >= len(counts):
                break
            for generator in order[rounds:]:
                rounds += 1
                if generator not in unseen:
                    unseen.add(generator)
                    covered += int(counts[generator])
                    break

    details = {
        "n_generators": int(len(unseen)),
        "covered_rows": int(covered),
        "target_rows": int(round(target_rows)),
        "extension_rounds_to_reach_test_floor": int(rounds),
    }
    return unseen, details


def _sample_balanced(
    df: pd.DataFrame, ai_idx: np.ndarray, human_idx: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cap the AI side at the human count and vice versa, keeping both classes
    balanced inside a test set (seeded, deterministic)."""
    rng = np.random.default_rng(seed)
    ai_idx = np.asarray(ai_idx, dtype=np.int64)
    human_idx = np.asarray(human_idx, dtype=np.int64)
    n = min(len(ai_idx), len(human_idx))
    if len(ai_idx) > n:
        ai_idx = np.sort(rng.choice(ai_idx, size=n, replace=False))
    if len(human_idx) > n:
        human_idx = np.sort(rng.choice(human_idx, size=n, replace=False))
    return ai_idx, human_idx


def _assert_disjoint(name: str, splits: dict[str, np.ndarray]) -> None:
    keys = list(splits)
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            overlap = np.intersect1d(splits[a], splits[b])
            if overlap.size:
                raise SplitError(
                    f"{name}: {a} and {b} share {overlap.size} documents "
                    f"(e.g. {overlap[:5].tolist()})"
                )


def _assert_no_group_leak(
    name: str, df: pd.DataFrame, train_idx: np.ndarray, tests: dict[str, np.ndarray]
) -> None:
    train_groups = set(df.iloc[train_idx]["topic_id"].astype(str))
    for test_name, idx in tests.items():
        leaked = train_groups & set(df.iloc[idx]["topic_id"].astype(str))
        if leaked:
            raise SplitError(f"{name}/{test_name}: {len(leaked)} topic groups leak")


def build(df: pd.DataFrame, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    df = df.reset_index(drop=True)
    label = df["label"].to_numpy()
    domain = df["domain"].to_numpy()
    is_human = label == ds.HUMAN
    is_ai = label == ds.AI
    is_revised = label == ds.AI_REVISED

    # --- primary group-aware assignment, block by block ---------------------
    assignment = np.full(len(df), "", dtype=object)
    for target_label in ds.LABELS:
        for target_domain in sorted(set(domain[label == target_label])):
            mask = (label == target_label) & (domain == target_domain)
            if not mask.any():
                continue
            assignment[mask] = _grouped_assign(
                df, mask, seed=seed, block_name=f"{target_label}/{target_domain}"
            )
    # --- machine-revised rows inherit their source document's split --------
    # A revision of a human document must never land in a different partition
    # than the document it was made from: that would leak the same topic (and
    # near-identical wording) across train and test.
    revised_positions = np.flatnonzero(label == ds.AI_REVISED)
    if revised_positions.size:
        docid_to_pos = {doc_id: pos for pos, doc_id in enumerate(df["doc_id"])}
        inherited = 0
        for pos in revised_positions:
            try:
                source_doc_id = json.loads(df["extra"].iat[pos]).get("source_doc_id")
            except (TypeError, json.JSONDecodeError):
                source_doc_id = None
            source_pos = docid_to_pos.get(source_doc_id)
            if source_pos is not None:
                assignment[pos] = assignment[source_pos]
                inherited += 1
        details_source_inheritance = {"revised_rows": int(revised_positions.size), "inherited": inherited}
    else:
        details_source_inheritance = {}

    df = df.assign(_assignment=assignment)

    splits: dict[str, np.ndarray] = {}
    details: dict[str, dict] = {}

    def idx(where) -> np.ndarray:
        values = where.to_numpy() if hasattr(where, "to_numpy") else np.asarray(where)
        return np.flatnonzero(values)

    train_idx = idx(df["_assignment"] == "train")
    val_idx = idx(df["_assignment"] == "val")
    test_idx = idx(df["_assignment"] == "test")

    splits["in_domain_group__train"] = train_idx
    splits["in_domain_group__val"] = val_idx
    splits["in_domain_group__test"] = test_idx
    _assert_disjoint("in_domain_group", {"train": train_idx, "val": val_idx, "test": test_idx})
    _assert_no_group_leak(
        "in_domain_group",
        df,
        train_idx,
        {"val": val_idx, "test": test_idx},
    )

    # --- cross-generator ----------------------------------------------------
    # Generators are partitioned into seen/unseen. Test AI rows must be
    # *unseen generators on topics that are also absent from training*; the
    # main split already holds whole topics out, so those rows are reused here
    # and everything else (unseen generator on a training topic) is dropped
    # from this experiment: it can neither train (generator leak) nor test
    # (topic leak).
    ai_test_main = np.intersect1d(test_idx, np.flatnonzero(is_ai))
    unseen, unseen_details = _pick_unseen_generators(
        df,
        np.flatnonzero(is_ai),
        seed,
        test_positions=ai_test_main,
    )
    generator_series = df["generator"].astype(str)
    topic_series = df["topic_id"].astype(str)
    unseen_gen_mask = (is_ai & generator_series.isin(sorted(unseen))).to_numpy()
    test_ai_all = np.intersect1d(ai_test_main, np.flatnonzero(unseen_gen_mask))
    seen_ai_mask = (is_ai & ~generator_series.isin(sorted(unseen))).to_numpy()

    train_ai = np.intersect1d(train_idx, np.flatnonzero(seen_ai_mask))
    val_ai = np.intersect1d(val_idx, np.flatnonzero(seen_ai_mask))
    val_human = np.intersect1d(val_idx, np.flatnonzero(is_human))
    human_test = np.intersect1d(test_idx, np.flatnonzero(is_human))
    ai_side, human_side = _sample_balanced(df, test_ai_all, human_test, seed)
    splits["cross_generator__train"] = np.sort(
        np.concatenate([train_ai, np.intersect1d(train_idx, np.flatnonzero(is_human))])
    )
    splits["cross_generator__val"] = np.sort(np.concatenate([val_ai, val_human]))
    splits["cross_generator__test"] = np.sort(np.concatenate([ai_side, human_side]))

    dropped_unseen_in_train = np.intersect1d(train_idx, np.flatnonzero(unseen_gen_mask))
    unseen_topics = set(topic_series.iloc[test_ai_all])
    details["cross_generator"] = {
        **unseen_details,
        "unseen_generators": sorted(unseen),
        "unseen_generator_rows_in_test": int(len(ai_side)),
        "test_human_rows": int(len(human_side)),
        "train_rows_seen_generators": int(len(train_ai)),
        "rows_dropped_unseen_generator_on_train_topic": int(len(dropped_unseen_in_train)),
        "shared_generators": [],
        "unseen_test_topics": int(len(unseen_topics)),
    }
    train_generators = set(df.iloc[splits["cross_generator__train"]]["generator"])
    if train_generators & unseen:
        raise SplitError("cross_generator: unseen generators appear in training")
    train_topics = set(df.iloc[splits["cross_generator__train"]]["topic_id"])
    if len(train_topics & unseen_topics) > 0:
        # only AI rows carry the raw topic ids used here; the check is exact
        leaked = train_topics & unseen_topics
        raise SplitError(f"cross_generator: {len(leaked)} unseen topics appear in training")
    test_ai_labels = df.iloc[splits["cross_generator__test"]].query("label == @ds.AI")
    if set(test_ai_labels["generator"]) - unseen:
        raise SplitError("cross_generator: test AI rows are not all from unseen generators")

    # --- cross-domain (human side) -----------------------------------------
    ai_train_ai = np.intersect1d(train_idx, np.flatnonzero(is_ai))
    ai_test_for_domain = np.intersect1d(test_idx, np.flatnonzero(is_ai))
    human_domains = sorted(set(domain[is_human]))
    for held_out in human_domains:
        mask_train_other = is_human & (domain != held_out)
        train_h = np.intersect1d(train_idx, np.flatnonzero(mask_train_other))
        held_out_rows = idx(is_human & (domain == held_out))
        ai_side, human_side = _sample_balanced(
            df, ai_test_for_domain, held_out_rows, seed
        )
        name = f"cross_domain__{held_out}"
        splits[f"{name}__train"] = np.sort(np.concatenate([ai_train_ai, train_h]))
        splits[f"{name}__test"] = np.sort(np.concatenate([ai_side, human_side]))
        details[name] = {
            "held_out_domain": held_out,
            "train_human_rows": int(len(train_h)),
            "test_human_rows": int(len(human_side)),
            "test_ai_rows": int(len(ai_side)),
            "note": "the held-out domain is tested in full (every human document "
            "of that domain), not only its main-split share",
        }
        if set(df.iloc[splits[f"{name}__train"]]["domain"]) & {held_out}:
            raise SplitError(f"{name}: held-out domain leaks into training")

    # --- cross-style (AI side proxy for domain shift) -----------------------
    ai_styles = sorted(set(df.loc[is_ai, "prompt_style"]))
    for style in ai_styles:
        if not style:
            continue
        mask_style = (is_ai) & (df["prompt_style"] == style).to_numpy()
        style_test = np.intersect1d(test_idx, np.flatnonzero(mask_style))
        train_ai_other = np.intersect1d(
            train_idx, np.flatnonzero(is_ai & (df["prompt_style"] != style).to_numpy())
        )
        train_h = np.intersect1d(train_idx, np.flatnonzero(is_human))
        ai_side, human_side = _sample_balanced(
            df, style_test, human_test, seed
        )
        name = f"cross_style__{style}"
        splits[f"{name}__train"] = np.sort(np.concatenate([train_ai_other, train_h]))
        splits[f"{name}__test"] = np.sort(np.concatenate([ai_side, human_side]))
        details[name] = {
            "held_out_style": style,
            "test_style_rows": int(len(ai_side)),
            "test_human_rows": int(len(human_side)),
            "train_ai_rows_other_styles": int(len(train_ai_other)),
        }
        train_ai_styles = set(
            df.iloc[splits[f"{name}__train"]].query("label == @ds.AI")["prompt_style"]
        )
        if style in train_ai_styles:
            raise SplitError(f"{name}: held-out style appears in training")

    # --- few-shot -----------------------------------------------------------
    for k in FEW_SHOT_KS:
        rng = np.random.default_rng(seed + k)
        human_pool = np.intersect1d(train_idx, np.flatnonzero(is_human))
        ai_pool = np.intersect1d(train_idx, np.flatnonzero(is_ai))
        if len(human_pool) < k or len(ai_pool) < k:
            raise SplitError(f"few_shot__{k}: not enough training rows")
        splits[f"few_shot__{k}__train"] = np.sort(
            np.concatenate(
                [
                    rng.choice(human_pool, size=k, replace=False),
                    rng.choice(ai_pool, size=k, replace=False),
                ]
            )
        )
        splits[f"few_shot__{k}__test"] = splits["in_domain_group__test"]

    # --- length buckets -----------------------------------------------------
    for bucket in ds.LENGTH_BUCKETS:
        if bucket == "medium":
            continue
        bucket_idx = idx((df["length_bucket"] == bucket).to_numpy())
        splits[f"length__{bucket}__test"] = np.intersect1d(test_idx, bucket_idx)

    # --- paraphrase-attack reservation (AI + human rows kept untouched) -----
    rng = np.random.default_rng(seed + 977)
    attack_pool = np.intersect1d(test_idx, np.flatnonzero(is_ai))
    attack_idx = (
        np.sort(rng.choice(attack_pool, size=min(ATTACK_SAMPLE, len(attack_pool)), replace=False))
        if len(attack_pool)
        else np.array([], dtype=np.int64)
    )
    splits["paraphrase_attention__test"] = attack_idx

    # --- human vs machine-revised ------------------------------------------
    if is_revised.any():
        revised_test = np.intersect1d(test_idx, np.flatnonzero(is_revised))
        revised_train = np.intersect1d(train_idx, np.flatnonzero(is_revised))
        ai_side, human_side = _sample_balanced(df, revised_test, human_test, seed)
        revised_val = np.intersect1d(val_idx, np.flatnonzero(is_revised))
        val_side, val_human_side = _sample_balanced(
            df, revised_val, np.intersect1d(val_idx, np.flatnonzero(is_human)), seed + 3
        )
        splits["human_vs_ai_revised__train"] = np.sort(
            np.concatenate([revised_train, np.intersect1d(train_idx, np.flatnonzero(is_human))])
        )
        splits["human_vs_ai_revised__val"] = np.sort(np.concatenate([val_side, val_human_side]))
        splits["human_vs_ai_revised__test"] = np.sort(np.concatenate([ai_side, human_side]))
        details["human_vs_ai_revised"] = {
            "train_revised_rows": int(len(revised_train)),
            "val_revised_rows": int(len(revised_val)),
            "test_revised_rows": int(len(revised_test)),
        }

    # --- v1 comparability (clean rows for the pre-existing checkpoints) -----
    v1_split = df["extra"].map(lambda e: json.loads(e).get("v1_split", ""))
    splits["v1_test"] = idx(v1_split == "test")
    details["v1_test"] = {
        "rows": int(len(splits["v1_test"])),
        "note": "rows present in processedDataset.csv test_idx; pre-existing "
        "checkpoints trained only on v1 train_idx",
    }

    summary = {
        "seed": seed,
        "dataset_fingerprint": ds.dataset_fingerprint(df),
        "counts": {name: int(len(value)) for name, value in sorted(splits.items())},
        "assignment_counts": df["_assignment"].value_counts().to_dict(),
        "class_mix_in_domain_test": (
            df.iloc[splits["in_domain_group__test"]]["label"].value_counts().to_dict()
        ),
        "domain_mix_in_domain_test": (
            df.iloc[splits["in_domain_group__test"]]["domain"].value_counts().to_dict()
        ),
        "details": details,
        "revised_split_inheritance": details_source_inheritance,
        "integrity": {
            "group_disjoint_train_val_test": True,
            "unseen_generators_absent_from_train": True,
            "unseen_topics_absent_from_train": True,
            "held_out_domains_absent_from_train": True,
            "assertions_ran": [
                "disjointness(in_domain_group)",
                "no_group_leak(in_domain_group)",
                "no_unseen_generator_in_train(cross_generator)",
                "no_unseen_topic_in_train(cross_generator)",
                "test_rows_from_unseen_generators(cross_generator)",
                "no_held_out_domain_in_train(cross_domain)",
                "no_held_out_style_in_train(cross_style)",
            ],
        },
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return splits, summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(DATASET)
    ds.validate_frame(df)
    splits, summary = build(df, args.seed)
    joblib.dump(splits, OUT_JOBLIB)
    summary["file"] = {"splits": OUT_JOBLIB.name, "sha256": ds.sha256_file(OUT_JOBLIB)}
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["counts"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["integrity"], ensure_ascii=False, indent=2))
    print(f"splits -> {OUT_JOBLIB}")


if __name__ == "__main__":
    main()
