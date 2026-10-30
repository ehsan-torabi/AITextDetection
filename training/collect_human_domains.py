"""Collect the additional human-written Persian domains (review requirement).

The v1 dataset had a single human domain (blogs). A multi-domain human side is
required before any cross-domain claim can be made, so this script streams
public, licence-tagged corpora from the Hugging Face Hub, applies the same
filters and PII scrubbing to each, and writes one CSV per domain under
``data/human/`` together with a provenance record.

Sources (id -> domain, license):

- blogs.csv (already local)          -> blog          (project corpus)
- pourmand1376/isna-news             -> news          (Apache-2.0)
- wikimedia/wikipedia (20231101.fa)  -> encyclopedia  (CC-BY-SA-4.0)
- moali-mkh-2000/PersianTwitter...   -> social_media  (ODbL, PII-scrubbed)
- PerSets/iran-legal-persian-qa      -> legal_admin   (CC)
- Maral/conversational-persian-...   -> translated_subtitles (CC-BY-4.0, origin=translated)

Every text passes a PII scrubber (emails, URLs, @handles, phone numbers) and a
minimum-length / Persian-script filter. Nothing is invented, paraphrased or
"cleaned up" beyond that: the raw surface is what the detectors must face.

Run from the project root:
    python training/collect_human_domains.py [--only news,social_media] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from aitextd import dataset as ds
from aitextd import paths

OUT_DIR = paths.DATA_DIR / "human"
PROVENANCE = OUT_DIR / "provenance.json"

PERSIAN_RANGE = re.compile(r"[\u0600-\u06ff]")


# --- cleaning helpers -------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+|t\.co/\S+)")
_HANDLE_RE = re.compile(r"(?<!\w)@[\w\u0600-\u06ff_]+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?98|0)?9\d{9}(?!\d)")
_LONG_ID_RE = re.compile(r"\b\d{10,}\b")


def scrub_pii(text: str) -> tuple[str, bool]:
    """Remove direct identifiers. Returns (scrubbed, changed)."""
    original = text
    text = _EMAIL_RE.sub(" [EMAIL] ", text)
    text = _URL_RE.sub(" [URL] ", text)
    text = _HANDLE_RE.sub(" [USER] ", text)
    text = _PHONE_RE.sub(" [PHONE] ", text)
    text = _LONG_ID_RE.sub(" [ID] ", text)
    return text, text != original


def persian_ratio(text: str) -> float:
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return 0.0
    return sum(1 for c in letters if PERSIAN_RANGE.match(c)) / len(letters)


def strip_wiki_markup(text: str) -> str:
    """Light-only markup removal: templates, refs, links and emphasis runs.
    Deliberately conservative -- the aim is readable prose, not a rewrite."""
    text = re.sub(r"\{\{[^{}]*\}\}", " ", text)
    text = re.sub(r"<ref[^>]*>.*?</ref>", " ", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]{1,40}>", " ", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)
    text = text.replace("'''", "").replace("''", "")
    return text


def keep_text(
    text: str,
    *,
    min_chars: int = 200,
    max_chars: int = 20000,
    min_persian_ratio: float = 0.85,
    reject_markup_ratio: float = 0.02,
) -> bool:
    text = text.strip()
    if not (min_chars <= len(text) <= max_chars):
        return False
    if persian_ratio(text) < min_persian_ratio:
        return False
    markup = text.count("{{") + text.count("[[") + text.count("|")
    if markup and markup / max(len(text) / 50, 1) > reject_markup_ratio * 50:
        return False
    return True


# --- source registry --------------------------------------------------------


def _rows_isna(limit: int, scan_cap: int):
    from datasets import load_dataset

    stream = load_dataset(
        "pourmand1376/isna-news", split="train", streaming=True
    )
    scanned = 0
    kept = 0
    for row in stream:
        scanned += 1
        if scanned > scan_cap or kept >= limit:
            break
        body = re.sub(r"\s+", " ", str(row.get("TEXT") or "")).strip()
        if not keep_text(body, min_chars=250):
            continue
        extra = {}
        try:
            meta = json.loads(row.get("METADATA") or "{}")
            extra = {"title": meta.get("title", ""), "date": meta.get("jalali-time", "")}
        except json.JSONDecodeError:
            pass
        kept += 1
        yield {
            "text": body,
            "author_group": "journalist",
            "topic_id": "",
            "extra": json.dumps(extra, ensure_ascii=False),
        }


def _rows_wikipedia(limit: int, scan_cap: int):
    from datasets import load_dataset

    stream = load_dataset(
        "wikimedia/wikipedia", "20231101.fa", split="train", streaming=True
    )
    scanned = 0
    kept = 0
    for row in stream:
        scanned += 1
        if scanned > scan_cap or kept >= limit:
            break
        title = str(row.get("title") or "")
        if title.startswith(("فهرست", "رده", "ویکی‌پدیا:", "الگو:", "صفحهٔ اصلی")):
            continue
        body = strip_wiki_markup(str(row.get("text") or ""))
        body = re.sub(r"[ \t]+", " ", body)
        body = re.sub(r"\n{2,}", "\n", body).strip()
        if not keep_text(body, min_chars=400, max_chars=12000):
            continue
        kept += 1
        yield {
            "text": body,
            "author_group": "wikipedia_editor",
            "topic_id": "",
            "extra": json.dumps({"title": title, "url": row.get("url", "")}, ensure_ascii=False),
        }


def _rows_twitter(limit: int, scan_cap: int):
    from datasets import load_dataset

    stream = load_dataset(
        "moali-mkh-2000/PersianTwitterDataset-SentimentAnalysis",
        split="train",
        streaming=True,
    )
    scanned = 0
    kept = 0
    for row in stream:
        scanned += 1
        if scanned > scan_cap or kept >= limit:
            break
        body = re.sub(r"\s+", " ", str(row.get("Tweets") or "")).strip()
        if not keep_text(body, min_chars=40, max_chars=600, min_persian_ratio=0.7):
            continue
        kept += 1
        yield {
            "text": body,
            "author_group": "tweeter",
            "topic_id": "",
            "extra": json.dumps({"sentiment": row.get("Label", "")}, ensure_ascii=False),
        }


def _rows_legal(limit: int, scan_cap: int):
    from datasets import load_dataset

    stream = load_dataset("PerSets/iran-legal-persian-qa", split="train", streaming=True)
    scanned = 0
    kept = 0
    for row in stream:
        scanned += 1
        if scanned > scan_cap or kept >= limit:
            break
        tags = ",".join(row.get("tags") or [])
        body = re.sub(r"\s+", " ", str(row.get("questionBody") or "")).strip()
        if keep_text(body, min_chars=80, max_chars=4000, min_persian_ratio=0.8):
            kept += 1
            yield {
                "text": body,
                "author_group": "legal_asker",
                "topic_id": tags,
                "extra": json.dumps({"qid": row.get("id", "")}, ensure_ascii=False),
            }
        for answer in (row.get("answers") or [])[:3]:
            if kept >= limit:
                break
            answer = re.sub(r"\s+", " ", str(answer or "")).strip()
            if not keep_text(answer, min_chars=80, max_chars=4000, min_persian_ratio=0.8):
                continue
            kept += 1
            yield {
                "text": answer,
                "author_group": "legal_expert",
                "topic_id": tags,
                "extra": json.dumps({"qid": row.get("id", "")}, ensure_ascii=False),
            }


def _rows_subtitles(limit: int, scan_cap: int):
    """Subtitles arrive one line per row; a document is one episode."""
    from datasets import load_dataset

    stream = load_dataset(
        "Maral/conversational-persian-subtitles", split="train", streaming=True
    )
    episodes: dict[tuple[str, str], list[tuple[str, str]]] = {}
    scanned = 0
    for row in stream:
        scanned += 1
        if scanned > scan_cap:
            break
        key = (str(row.get("movie_name") or ""), str(row.get("episode_id") or ""))
        line = str(row.get("subtitle_text") or "").strip()
        if line:
            episodes.setdefault(key, []).append((str(row.get("start_time") or ""), line))
    kept = 0
    for (movie, episode), lines in sorted(episodes.items()):
        if kept >= limit:
            break
        lines.sort(key=lambda item: item[0])
        body = re.sub(r"\s+", " ", " ".join(line for _, line in lines)).strip()
        if not keep_text(body, min_chars=400, max_chars=20000, min_persian_ratio=0.7):
            continue
        kept += 1
        yield {
            "text": body,
            "author_group": "subtitle_translator",
            "topic_id": "",
            "extra": json.dumps({"movie": movie, "episode": episode}, ensure_ascii=False),
        }


def _rows_blog(limit: int, scan_cap: int):
    """The existing local blog corpus. Read from processedDataset.csv rather
    than blogs.csv: those are exactly the v1 human rows, which keeps the v2
    blog slice comparable with the already-trained checkpoints and avoids a
    second 2 GB scan."""
    df = pd.read_csv(paths.PROCESSED_CSV)
    human = df.loc[df["aiGenerated"] == 0, "text"]
    kept = 0
    for body in human:
        if kept >= limit:
            break
        body = re.sub(r"\s+", " ", str(body)).strip()
        if not keep_text(body, min_chars=250, max_chars=20000, min_persian_ratio=0.9):
            continue
        kept += 1
        yield {
            "text": body,
            "author_group": "blogger",
            "topic_id": "",
            "extra": "{}",
        }


SOURCES = {
    "news": dict(
        fn=_rows_isna,
        source="pourmand1376/isna-news",
        license="apache-2.0",
        origin="persian",
        default_limit=3000,
        scan_cap_factor=90,
    ),
    "encyclopedia": dict(
        fn=_rows_wikipedia,
        source="wikimedia/wikipedia:20231101.fa",
        license="cc-by-sa-4.0",
        origin="persian",
        default_limit=3000,
        scan_cap_factor=25,
    ),
    "social_media": dict(
        fn=_rows_twitter,
        source="moali-mkh-2000/PersianTwitterDataset-SentimentAnalysis",
        license="odbl",
        origin="persian",
        default_limit=3000,
        scan_cap_factor=10,
    ),
    "legal_admin": dict(
        fn=_rows_legal,
        source="PerSets/iran-legal-persian-qa",
        license="cc",
        origin="persian",
        default_limit=3000,
        scan_cap_factor=60,
    ),
    "translated_subtitles": dict(
        fn=_rows_subtitles,
        source="Maral/conversational-persian-subtitles",
        license="cc-by-4.0",
        origin="translated",
        default_limit=400,
        scan_cap_factor=120,
    ),
    "blog": dict(
        fn=_rows_blog,
        source="blogs.csv (project corpus, via processedDataset.csv)",
        license="see project README",
        origin="persian",
        default_limit=12000,
        scan_cap_factor=2,
    ),
}


def collect(domain: str, limit: int) -> dict:
    spec = SOURCES[domain]
    scanned_cap = max(limit * spec["scan_cap_factor"], limit + 500)
    print(f"[{domain}] streaming {spec['source']} (limit={limit}) ...")
    rows = []
    for row in spec["fn"](limit, scanned_cap):
        text, changed = scrub_pii(row["text"])
        row["text"] = text
        row["pii_scrubbed"] = int(changed)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no rows collected for {domain}")

    out = pd.DataFrame(
        {
            "doc_id": [ds.stable_doc_id(t) for t in df["text"]],
            "text": df["text"],
            "domain": domain,
            "source": spec["source"],
            "license": spec["license"],
            "origin": spec["origin"],
            "author_group": df["author_group"],
            "topic_id": df.get("topic_id", ""),
            "pii_scrubbed": df["pii_scrubbed"],
            "extra": df.get("extra", "{}"),
            "label": ds.HUMAN,
            "label_binary": 0,
            "generator": "human",
            "provider": "",
            "prompt_id": "",
            "prompt_style": "",
            "revision_type": "none",
            "paraphrase_method": "",
            "non_native_flag": 0,
        }
    )
    # The subtitle slice is translated into Persian, so the translation
    # language itself is a (documented) non-native-prose signal.
    if spec["origin"] == "translated":
        out["non_native_flag"] = 1

    out["length_words"] = out["text"].map(lambda t: len(t.split()))
    out["length_bucket"] = out["length_words"].map(ds.length_bucket)
    out["formality"] = out["text"].map(ds.formality_label)

    out = out[[c for c in ds.COLUMNS if c in out.columns]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{domain}.csv"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    record = {
        "domain": domain,
        "source": spec["source"],
        "license": spec["license"],
        "origin": spec["origin"],
        "rows": int(len(out)),
        "scanned_cap": scanned_cap,
        "unique_doc_ids": int(out["doc_id"].nunique()),
        "pii_scrubbed_rows": int(out["pii_scrubbed"].sum()),
        "length_words_median": float(out["length_words"].median()),
        "formality_mix": out["formality"].value_counts().to_dict(),
        "file": csv_path.name,
        "file_sha256": ds.sha256_file(csv_path),
        "collected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"[{domain}] wrote {len(out)} rows -> {csv_path}")
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated domain list")
    ap.add_argument("--limit", type=int, default=0, help="override rows per domain")
    args = ap.parse_args()

    domains = [d.strip() for d in args.only.split(",") if d.strip()] or list(SOURCES)
    provenance = {}
    if PROVENANCE.exists():
        provenance = json.loads(PROVENANCE.read_text(encoding="utf-8"))

    failures = {}
    for domain in domains:
        if domain not in SOURCES:
            raise SystemExit(f"unknown domain {domain!r}; known: {list(SOURCES)}")
        limit = args.limit or SOURCES[domain]["default_limit"]
        try:
            provenance[domain] = collect(domain, limit)
        except Exception as exc:  # keep going: one flaky source must not block the rest
            failures[domain] = f"{type(exc).__name__}: {exc}"
            print(f"[{domain}] FAILED: {failures[domain]}")

    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    provenance["_failures"] = failures
    PROVENANCE.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"provenance -> {PROVENANCE}")
    if failures:
        print("FAILED domains:", ", ".join(failures))


if __name__ == "__main__":
    main()
