"""Fine-tune Laya (open-weights System One) on the Persian detection data.

Standalone adaptation of the Kaggle 2xT4 / Apple-Silicon MPS recipe for this
project and a CPU box:

1. base checkpoint: the ``multilingual`` subfolder of ``convaiinnovations/laya``
   (mmBERT-base, 100+ languages) -- the English root checkpoint cannot read
   Persian script at all (its own model card: "English only on root"), so every
   artifact lives under ``models/laya_multilingual`` and the fine-tuned export
   under ``models/laya_persian`` (both gitignored, paths from aitextd.paths).
2. training data: ``training/system_one_data/system_one_{train,test}.jsonl``
   (the same GPT-2-distilled soft targets train_system_one.py consumes), not the
   English LocalLLaMA/typed-decisions benchmark.
3. questions: the exact ``QUESTIONS`` dict the web app and the online eval
   harness send (choice ``source``, score ``machine_likeness``, noul
   ``needs_review``), so laya learns to answer production-shaped prompts and its
   numbers stay comparable with the local System One model and the Jev relay.

Training is the shipped RLCD loop (strictly proper scoring reward + soft CE),
identical maths to the reference recipe. CPU defaults: the encoder is frozen and
detached (only the decision head trains -- mmBERT-base at 12 cores is slow), a
stratified subsample of records, max_len 512. ``--unfrozen-layers N`` unfreezes
the last N encoder layers for a proper full fine-tune on a GPU.

After training: temperature calibration is fitted per question type on held-out
calibration records (Laya ships over-confident -- fit it on your own data before
trusting probabilities), the checkpoint is evaluated on the held-out test split
with the same battery as the online eval (acc / F1 / ROC-AUC / ECE / score MAE /
noul AUC), and everything is written to ``models/laya_persian`` in laya's own
layout, loadable with ``laya.load("models/laya_persian")``.

The model is intentionally NOT registered in models.json yet -- it is not part
of the serving stack until we wire it in.

The ``laya`` package itself is not a project dependency yet; it must be
importable (pip install laya, or add it to pyproject deps).

Run from the project root:
    python training/train_laya.py                      # CPU head-only defaults
    python training/train_laya.py --epochs 1 --records 800
    python training/train_laya.py --device cuda --unfrozen-layers 4 --micro-batch 4 --grad-accum 32
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from itertools import pairwise
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

try:
    from laya.agent import _fix_tokenizer_config
    from laya.common import (
        QTYPES,
        TEMP_MAX,
        TEMP_MIN,
        build_model,
        build_sequence,
        proper_reward,
        render_options,
        state_room,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        f"cannot import laya ({exc}); install it (pip install laya) or put the "
        "laya package on PYTHONPATH before running this script"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from aitextd import paths
from aitextd.constants import SYSTEM_ONE_CHOICE_CANDIDATES as CHOICE_CANDIDATES
from aitextd.constants import SYSTEM_ONE_SCORE_LEVELS as SCORE_LEVELS
from eval_online_systemone import QUESTIONS
from split import RANDOM_SEED

BASE_REPO_ID = "convaiinnovations/laya"
BASE_SUBFOLDER = "multilingual"

# Question id -> (targets key in the JSONL record, System One primitive type).
QUESTION_TARGETS = {
    "source": ("source", "choice"),
    "machine_likeness": ("ai_level", "score"),
    "needs_review": ("needs_review", "noul"),
}
MAX_SCORE_LEVEL = len(SCORE_LEVELS) - 1  # human=0 .. AI=4, as in the eval harness

torch.set_num_threads(12)


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def subsample(records, n, seed):
    """Deterministic stratified subset -- the source splits are already
    balanced, but a plain prefix would silently skew re-runs at other sizes."""
    if n >= len(records):
        return records
    idx, _ = train_test_split(
        range(len(records)), train_size=n, random_state=seed,
        stratify=[r["label"] for r in records],
    )
    return [records[i] for i in sorted(idx)]


def prepare_base(base_dir):
    """Ensure models/laya_multilingual holds a flat laya checkpoint directory
    (rl_agent_config.json + model.safetensors + encoder/ + tokenizer/) so
    laya.load()/laya.agent can consume it without a subfolder argument."""
    base_dir = Path(base_dir)
    if not (base_dir / "model.safetensors").exists():
        import shutil

        from huggingface_hub import snapshot_download

        print(f"Downloading {BASE_REPO_ID} [{BASE_SUBFOLDER}] to {base_dir} ...")
        base_dir.parent.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_download(BASE_REPO_ID, allow_patterns=[f"{BASE_SUBFOLDER}/*"])
        shutil.copytree(Path(snapshot) / BASE_SUBFOLDER, base_dir, dirs_exist_ok=True)
    _fix_tokenizer_config(str(base_dir))
    return str(base_dir)


def internal_question(qid):
    """QUESTIONS entry (public shape) -> laya's internal {t, ins, crit} dict."""
    q = QUESTIONS[qid]
    crit = q.get("criteria")
    if q["type"] == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    return {"t": q["type"], "ins": q["instructions"], "crit": crit}


def target_distribution(record, qid):
    """Record soft target -> probability row in render_options order."""
    key, qtype = QUESTION_TARGETS[qid]
    value = record["targets"][key]
    if qtype == "choice":
        row = [float(value[c]) for c in CHOICE_CANDIDATES]
    elif qtype == "score":
        row = [float(p) for p in value]
    else:  # noul renders options as [false, true]; needs_review is P(true)
        p = float(value)
        row = [1.0 - p, p]
    total = sum(row)
    if total <= 0:
        row = [1.0 / len(row)] * len(row)
    else:
        row = [x / total for x in row]
    return row


def build_items(tokenizer, internal, records, qids, max_len, head_max_len):
    """One laya training item per (record, question) pair.

    The state is serialized exactly like the serving path -- {"text": ...} --
    and pre-clamped to a generous character budget per question so tokenizing a
    50k-character blog post does not dominate preprocessing; build_sequence
    applies the real token clamp afterwards."""
    items = []
    skipped = 0
    for record in records:
        text = str(record["state"]["text"])
        for qid in qids:
            q = internal[qid]
            room = state_room(tokenizer, q, max_len, head_max_len)
            char_budget = max(128, room * 4)  # ~4 chars/token at the 256k vocab
            state = {"text": text[:char_budget]}
            target = target_distribution(record, qid)
            sequence, markers = build_sequence(tokenizer, state, q, max_len, head_max_len)
            if len(markers) != len(render_options(q)):
                skipped += 1
                continue
            items.append(
                {
                    "ids": sequence,
                    "markers": markers,
                    "qtype": QTYPES[q["t"]],
                    "target": target,
                    "label": int(record["label"]),
                    "qid": qid,
                }
            )
    return items, skipped


def items_cache_key(base_dir, max_len, head_max_len, data_dir, counts, seed):
    digest = hashlib.sha1(
        json.dumps(
            {
                "base_dir": str(Path(base_dir).resolve()),
                "max_len": max_len,
                "head_max_len": head_max_len,
                "questions": QUESTIONS,
                "counts": counts,
                "seed": seed,
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return {"key": digest, "data_dir": str(data_dir)}


def prepare_items(base_dir, tokenizer, cfg, args):
    """Build (or load cached) train / calib / test item lists.

    The calibration split is carved from whole records, never single items --
    the reference notebook split per item and so scored the same document in
    both training and calibration."""
    data_dir = paths.SYSTEM_ONE_DATA_DIR
    cache_dir = Path(args.items_dir)
    cache_meta = cache_dir / "laya_items.meta.json"
    cache_key = items_cache_key(
        base_dir, args.max_len, cfg["head_max_len"], data_dir,
        counts=(args.records, args.calib_records, args.test_records), seed=args.seed,
    )
    paths_per_split = {
        name: cache_dir / f"laya_items_{name}.pt"
        for name in ("train", "calib", "test")
    }
    if not args.force_preprocess and cache_meta.exists():
        try:
            cached = json.loads(cache_meta.read_text())
        except (OSError, json.JSONDecodeError):
            cached = None
        if cached == cache_key and all(p.exists() for p in paths_per_split.values()):
            print(f"Using cached laya items: {cache_dir}")
            return {name: torch.load(p, map_location="cpu", weights_only=False)
                    for name, p in paths_per_split.items()}
        print("Laya item cache key changed (or files missing); rebuilding.")

    qids = list(QUESTION_TARGETS)
    internal = {qid: internal_question(qid) for qid in qids}

    train_records = subsample(load_jsonl(data_dir / "system_one_train.jsonl"),
                              args.records, RANDOM_SEED)
    calib_records = subsample(load_jsonl(data_dir / "system_one_train.jsonl"),
                              args.calib_records, RANDOM_SEED + 1)
    train_records = [r for r in train_records if r["id"] not in {c["id"] for c in calib_records}]
    test_records = subsample(load_jsonl(data_dir / "system_one_test.jsonl"),
                             args.test_records, RANDOM_SEED)

    cache_dir.mkdir(parents=True, exist_ok=True)
    items = {}
    for name, records in (("train", train_records), ("calib", calib_records), ("test", test_records)):
        start = time.time()
        items[name], skipped = build_items(tokenizer, internal, records, qids,
                                           args.max_len, cfg["head_max_len"])
        print(f"{name}: {len(items[name])} items from {len(records)} records "
              f"(skipped={skipped}) in {time.time() - start:.0f}s")
        torch.save(items[name], paths_per_split[name])
    cache_meta.write_text(json.dumps(cache_key, indent=2))
    return items


def collate(items, pad_id):
    """Pad a list of laya items into model inputs (same layout as laya's own
    collate, kept local so the batch carries our extra meta keys)."""
    batch_size = len(items)
    seq_len = max(len(item["ids"]) for item in items)
    kmax = max(len(item["markers"]) for item in items)
    input_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
    attention = torch.zeros((batch_size, seq_len), dtype=torch.long)
    marker_pos = torch.zeros((batch_size, kmax), dtype=torch.long)
    marker_mask = torch.zeros((batch_size, kmax), dtype=torch.bool)
    target = torch.zeros((batch_size, kmax), dtype=torch.float32)

    for i, item in enumerate(items):
        length = len(item["ids"])
        input_ids[i, :length] = torch.tensor(item["ids"], dtype=torch.long)
        attention[i, :length] = 1
        k = len(item["markers"])
        marker_pos[i, :k] = torch.tensor(item["markers"], dtype=torch.long)
        marker_mask[i, :k] = True
        target[i, :len(item["target"])] = torch.tensor(item["target"], dtype=torch.float32)

    return (
        input_ids,
        attention,
        marker_pos,
        marker_mask,
        target,
        torch.tensor([item["qtype"] for item in items], dtype=torch.long),
    )


def forward_batch(model, chunk, tokenizer, device):
    ids, attention, positions, mask, target, qtype = collate(chunk, tokenizer.pad_token_id)
    return (
        ids.to(device), attention.to(device), positions.to(device),
        mask.to(device), target.to(device), qtype.to(device),
    )


def fit_temperature(samples):
    """MLE temperature (LBFGS over log T) for one question type's calibration
    logits/targets pairs; clamped to laya's [TEMP_MIN, TEMP_MAX] the same way
    the runtime clamps shipped temperatures."""
    if len(samples) < 10:
        return 1.0
    kmax = max(len(logits) for logits, _ in samples)
    logits = torch.full((len(samples), kmax), -1e4)
    targets = torch.zeros((len(samples), kmax))
    for i, (values, target) in enumerate(samples):
        logits[i, :len(values)] = torch.as_tensor(values)
        targets[i, :len(target)] = torch.as_tensor(target, dtype=torch.float32)

    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(targets * torch.log_softmax(logits / log_temperature.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_temperature.exp(), TEMP_MIN, TEMP_MAX).item())


def save_checkpoint(model, tokenizer, cfg, output_dir, epoch, final=False):
    path = Path(output_dir) if final else Path(output_dir) / "checkpoint_latest"
    path.mkdir(parents=True, exist_ok=True)
    weights = {
        name: value.detach().half().cpu().contiguous()
        for name, value in model.state_dict().items()
    }
    from safetensors.torch import save_file

    save_file(weights, str(path / "model.safetensors"))
    model.encoder.config.save_pretrained(path / "encoder")
    tokenizer.save_pretrained(path / "tokenizer")
    with open(path / "checkpoint_meta.json", "w") as f:
        json.dump({"epoch": epoch, "final": final}, f, indent=2)
    with open(path / "rl_agent_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def evaluate(model, tokenizer, items, temperatures, device, batch_size=16):
    """Held-out metrics on the production questions, comparable with
    training/eval_online_systemone.py (AI class = positive)."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        mean_absolute_error,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    model.eval()
    rows = []  # (qtype, k, scaled probs)
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            ids, attention, positions, mask, _target, qtype = forward_batch(
                model, chunk, tokenizer, device
            )
            logits, _ = model(ids, attention, positions, mask, qtype, detach_encoder=True)
            for i, item in enumerate(chunk):
                k = len(item["markers"])
                z = logits[i, :k].float().cpu().numpy() / temperatures[item["qtype"]]
                p = np.exp(z - z.max())
                rows.append((item["qtype"], p, item["target"], item["label"]))

    choice = [(p, t, y) for qt, p, t, y in rows if qt == QTYPES["choice"]]
    score = [(p, t, y) for qt, p, t, y in rows if qt == QTYPES["score"]]
    noul = [(p, t, y) for qt, p, t, y in rows if qt == QTYPES["noul"]]

    y = np.array([y for _, _, y in choice])
    ai_prob = np.array([p[1] for p, _, _ in choice])
    pred = (ai_prob >= 0.5).astype(int)
    metrics = {
        "choiceSamples": len(choice),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precisionAI": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recallAI": round(float(recall_score(y, pred, zero_division=0)), 4),
        "f1AI": round(float(f1_score(y, pred, zero_division=0)), 4),
        "f1Macro": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
        "rocAuc": round(float(roc_auc_score(y, ai_prob)), 4) if len(set(y)) > 1 else None,
        "prAuc": round(float(average_precision_score(y, ai_prob)), 4) if len(set(y)) > 1 else None,
        "brier": round(float(np.mean((ai_prob - y) ** 2)), 4),
        "scoreMae": round(
            float(mean_absolute_error(
                [MAX_SCORE_LEVEL * y for _, _, y in score],
                [float((np.arange(len(p)) * p).sum()) for p, _, _ in score],
            )), 4) if score else None,
        "noulAuc": round(float(roc_auc_score([y for _, _, y in noul],
                                             [p[1] for _, p, _ in noul])), 4)
        if len(noul) > 1 and len(set(y for _, _, y in noul)) > 1 else None,
    }
    # 10-bin ECE on P(ai) -- same definition as the online eval harness.
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in pairwise(edges):
        m = (ai_prob > lo) & (ai_prob <= hi) if lo > 0 else (ai_prob >= lo) & (ai_prob <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - ai_prob[m].mean())
    metrics["ece10"] = round(float(ece), 4)
    return metrics


def train(args, base_dir, items_by_split, device):
    with open(Path(base_dir) / "rl_agent_config.json") as f:
        cfg = json.load(f)
    cfg["max_len"] = args.max_len
    cfg.setdefault("head_max_len", 256)

    tokenizer = AutoTokenizer.from_pretrained(Path(base_dir) / "tokenizer")
    model = build_model(cfg, encoder_dir=Path(base_dir) / "encoder")
    model.load_state_dict(load_file(str(Path(base_dir) / "model.safetensors")), strict=True)
    model.float()

    if args.unfrozen_layers > 0:
        # ModernBert-based backbone keeps its transformer stack in encoder.layers.
        encoder_layers = model.encoder.layers
        for p in model.encoder.parameters():
            p.requires_grad = False
        for layer in encoder_layers[-args.unfrozen_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        detach_encoder = False
        if not args.no_checkpointing:
            model.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.head_checkpointing = True
            cfg["gradient_checkpointing"] = True
    else:
        # Head-only (CPU default): freeze AND detach the encoder so backward
        # never traverses the 22-layer backbone -- ~3x faster per step.
        for p in model.encoder.parameters():
            p.requires_grad = False
        detach_encoder = True

    model.to(device).train()

    all_train = items_by_split["train"]
    order = list(range(len(all_train)))
    random.Random(args.seed).shuffle(order)
    train_items = [all_train[i] for i in sorted(order)]
    calib_items = items_by_split["calib"]

    encoder_params = [p for n, p in model.named_parameters()
                      if "encoder." in n and p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if "encoder." not in n and p.requires_grad]
    param_groups = [{"params": head_params, "lr": args.lr_head}]
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": args.lr_encoder})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    updates = max(1, math.ceil(len(train_items) / args.micro_batch / args.grad_accum) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=updates, eta_min=1e-6)

    mode = f"full encoder (last {args.unfrozen_layers} layers)" if encoder_params \
        else "head-only (encoder detached)"
    print(f"Device: {device}; mode: {mode}")
    print(f"Training items: {len(train_items)}; calibration items: {len(calib_items)}")
    print(f"micro_batch={args.micro_batch}; grad_accum={args.grad_accum}; epochs={args.epochs}; "
          f"max_len={args.max_len}")

    train_start = time.time()
    for epoch in range(args.epochs):
        random.Random(RANDOM_SEED + epoch).shuffle(train_items)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        n_batches = 0
        sigma = 0.4 + (0.1 - 0.4) * epoch / max(1, args.epochs - 1)

        for start in range(0, len(train_items), args.micro_batch):
            chunk = train_items[start:start + args.micro_batch]
            ids, attention, positions, mask, target, qtype = forward_batch(
                model, chunk, tokenizer, device
            )

            logits, activation = model(
                ids, attention, positions, mask, qtype, detach_encoder=detach_encoder
            )
            logits = logits.float()
            k = mask.sum(-1, keepdim=True).float()

            # RLCD: zero-mean Gaussian noise on the reported distribution.
            eps = torch.randn((4, *logits.shape), device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            noisy_logits = logits.detach().unsqueeze(0) + eps
            probabilities = torch.softmax(noisy_logits.masked_fill(~mask, -1e4), -1)

            with torch.no_grad():
                reward = proper_reward(
                    probabilities, target.unsqueeze(0), qtype, mask, w_sph=0.75, w_rps=1.0
                )
                advantage = reward - reward.mean(0, keepdim=True)
                advantage = advantage / (advantage.std() + 1e-6)

            logp = -(((noisy_logits - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)
            loss_rl = -(advantage * logp).mean()
            loss_ce = -(
                target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)
            ).sum(-1).mean()
            loss = (loss_rl + loss_ce + 0.0 * activation.sum()) / args.grad_accum
            loss.backward()

            n_batches += 1
            if n_batches % args.grad_accum == 0 or start + args.micro_batch >= len(train_items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * args.grad_accum
            if n_batches % 20 == 0:
                elapsed = time.time() - train_start
                per_step = elapsed / n_batches
                remaining = (len(train_items) / args.micro_batch - n_batches) * per_step
                print(
                    f"epoch {epoch + 1}/{args.epochs}, step {n_batches}, "
                    f"loss={loss.item() * args.grad_accum:.4f}, "
                    f"{per_step:.1f}s/step eta={remaining / 60:.0f}min",
                    flush=True,
                )

        print(f"Epoch {epoch + 1}/{args.epochs} complete; avg_loss={total_loss / max(1, n_batches):.4f}")
        save_checkpoint(model, tokenizer, cfg, args.output_dir, epoch + 1)

    print("Running temperature calibration on held-out records ...")
    model.eval()
    samples = [[] for _ in range(3)]
    with torch.no_grad():
        for start in range(0, len(calib_items), args.micro_batch):
            chunk = calib_items[start:start + args.micro_batch]
            ids, attention, positions, mask, target, qtype = forward_batch(
                model, chunk, tokenizer, device
            )
            logits, _ = model(ids, attention, positions, mask, qtype, detach_encoder=True)
            for i, item in enumerate(chunk):
                samples[item["qtype"]].append(
                    (logits[i, :len(item["markers"])].float().cpu(), item["target"])
                )
    temperatures = [fit_temperature(group) if group else 1.2 for group in samples]

    print("Evaluating on the held-out test split ...")
    metrics = evaluate(model, tokenizer, items_by_split["test"], temperatures, device)
    print("Final metrics:", json.dumps(metrics, ensure_ascii=False))

    cfg.update(
        {
            "fine_tuned": True,
            "model_name": "laya-persian",
            "temperature": temperatures,
            "training": {
                "epochs_completed": args.epochs,
                "train_items": len(train_items),
                "calib_items": len(calib_items),
                "test_items": len(items_by_split["test"]),
                "max_len": args.max_len,
                "mode": "full" if encoder_params else "head_only",
                "base": f"{BASE_REPO_ID} [{BASE_SUBFOLDER}]",
                "fine_tuned_from_checkpoint": True,
                "data": "training/system_one_data (GPT-2-distilled soft targets)",
            },
        }
    )
    cfg.pop("temperature_by_options", None)
    save_checkpoint(model, tokenizer, cfg, args.output_dir, args.epochs, final=True)
    (Path(args.output_dir) / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Temperatures: {[round(t, 4) for t in temperatures]}")
    print(f"Model saved to {args.output_dir}")
    print(f"Total time: {(time.time() - train_start) / 60:.1f} min")


def choose_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA unavailable; using CPU instead.")
        return torch.device("cpu")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print("Warning: MPS unavailable; using CPU instead.")
        return torch.device("cpu")
    return torch.device(requested)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya on the Persian System One data")
    parser.add_argument("--base-dir", default=str(paths.LAYA_BASE_DIR))
    parser.add_argument("--init-from", default=None,
                        help="start from an earlier fine-tuned laya checkpoint directory "
                        "(e.g. progressive unfreezing from a head-only run)")
    parser.add_argument("--output-dir", default=str(paths.LAYA_OUTPUT_DIR))
    parser.add_argument("--items-dir", default=str(paths.LAYA_ITEMS_DIR))
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--records", type=int, default=1200, help="train records (excl. calibration)")
    parser.add_argument("--calib-records", type=int, default=200, help="records for temperature fit")
    parser.add_argument("--test-records", type=int, default=400, help="held-out eval records")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--unfrozen-layers", type=int, default=0,
                        help="0 = head-only (CPU default); N = fine-tune the last N encoder layers")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = choose_device(args.device)

    base_dir = prepare_base(args.base_dir)
    if args.init_from:
        if not (Path(args.init_from) / "model.safetensors").exists():
            raise SystemExit(f"--init-from has no model.safetensors: {args.init_from}")
        base_dir = str(args.init_from)
        print(f"Initializing weights from {base_dir} (base checkpoint for "
              f"tokenizer/config: {args.base_dir})")
    with open(Path(base_dir) / "rl_agent_config.json") as f:
        base_cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(Path(base_dir) / "tokenizer")
    items_by_split = prepare_items(base_dir, tokenizer, base_cfg, args)
    train(args, base_dir, items_by_split, device)


if __name__ == "__main__":
    main()
