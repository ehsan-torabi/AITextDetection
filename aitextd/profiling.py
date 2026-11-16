"""Deployment profiler for the aitextd serving stack.

Measures what a production deployment actually needs:

* disk           - on-disk size of models/, data/, models.json, web-app/
* per-model      - load time + peak RSS for each single model (isolated subprocess;
                   the incremental cost of adding one more model)
* full-stack     - one process calling store.init() (loads ALL 16 models at once,
                   exactly like production), then warmup requests; reports steady-state
                   RSS and per-endpoint latency so you can size RAM and workers/CPU.

Run from the project root:

    PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages:$PWD" \
    /home/ehsan/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11 \
    -m aitextd.profiling full-stack

All heavy imports happen inside subprocesses so the parent stays light. Numbers are
printed as JSON lines (one object per line) plus a human-readable summary.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = PROJECT_ROOT / ".venv" / "lib" / "python3.11" / "site-packages"
UV_PYTHON = "/home/ehsan/.local/share/uv/python/cpython-3.11-linux-x86_64-gnu/bin/python3.11"

GIB = 1024 ** 3
MIB = 1024 ** 2

# Linux ru_maxrss is reported in KiB (1024-byte units).
KIB = 1024


def _rss_gib(kib: float) -> float:
    return kib / (KIB * KIB)  # KiB -> GiB

# Representative Persian detection texts (short / medium / long).
SAMPLE_TEXTS = {
    "short": "این متن یک متن آزمایشی کوتاه است.",
    "medium": (
        "مقادیر زیادی از محتوای اینترنتی در حال حاضر توسط مدل‌های زبانی بزرگ تولید می‌شود. "
        "تشخیص متن تولیدشده توسط هوش مصنوعی از نوشته انسان یکی از چالش‌های امروز است."
    ),
    "long": None,  # filled below at import time from a paragraph if available
}

LONG_TEXT = (
    "مقادیر زیادی از محتوای اینترنتی در حال حاضر توسط مدل‌های زبانی بزرگ تولید می‌شود. "
    "تشخیص متن تولیدشده توسط هوش مصنوعی از نوشته انسان یکی از چالش‌های امروز است. "
    "این پروژه این مشکل را با یک لایه تصمیم (System One) و یک مجموعه weighted ensemble از "
    "چند مدل از جمله ParsBERT، GPT-2، BiLSTM، TextCNN و مدل‌های کلاسیک sklearn حل می‌کند. "
    "هر درخواست یک گزینه مدل می‌گیرد، پیش‌پردازش فارسی را اجرا می‌کند، ویژگی‌های زیربنایی را محاسبه می‌کند و "
    "احتمال تولیدشده توسط هوش مصنوعی را برمی‌گرداند. این سرویس برای اجرا روی CPU با چند Worker طراحی شده است."
)


def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SITE_PACKAGES) + os.pathsep + str(PROJECT_ROOT)
    return env


def _gib(num_bytes: float) -> str:
    return f"{num_bytes / GIB:.2f} GiB"


def _mib(num_bytes: float) -> str:
    return f"{num_bytes / MIB:.1f} MiB"


# ---------------------------------------------------------------- disk

def disk() -> list[dict]:
    """Walk models/, data/ and report per-file + total sizes (MiB)."""
    roots = [PROJECT_ROOT / "models", PROJECT_ROOT / "data"]
    files: list[dict] = []
    total = 0
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file():
                size = p.stat().st_size
                total += size
                files.append({
                    "path": str(p.relative_to(PROJECT_ROOT)),
                    "size_bytes": size,
                    "size": _mib(size),
                })
    # top-level config manifests
    for name in ("models.json",):
        p = PROJECT_ROOT / name
        if p.is_file():
            size = p.stat().st_size
            total += size
            files.append({
                "path": name,
                "size_bytes": size,
                "size": _mib(size),
            })
    files.sort(key=lambda f: f["size_bytes"], reverse=True)
    return {"disk": files, "total_bytes": total, "total": _gib(total)}


# ---------------------------------------------------------------- per-model (child)

def _per_model_child(model_id: str) -> dict:
    """Run in a subprocess: load ONE model, report time + peak RSS (+ import baseline)."""
    import aitextd  # noqa: F401  (import cost measured separately)
    from aitextd import loaders, paths

    import json as _json

    t0 = time.perf_counter()
    with open(paths.MODELS_JSON, encoding="utf-8") as f:
        md_list = _json.load(f)
    md = {m["id"]: m for m in md_list}[model_id]

    import joblib  # noqa: F401

    vectorizer = None
    if md.get("runtime") == "sklearn-tfidf":
        vectorizer = joblib.load(paths.TFIDF_VECTORIZER_JOBLIB)

    load_t0 = time.perf_counter()
    if md.get("runtime") == "sklearn-tfidf":
        bundle = loaders.load_sklearn_model(md, vectorizer)
    else:
        loader = loaders.LOADERS.get(md.get("runtime", "sklearn-tfidf"))
        bundle = loader(md)
    load_sec = time.perf_counter() - load_t0

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # Linux: KiB
    cpu_sec = (
        resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime
    )
    return {
        "model_id": model_id,
        "runtime": md.get("runtime"),
        "filename": md.get("filename"),
        "artifact_size_mib": _mib(
            (paths.MODELS_DIR / md.get("filename", "")).stat().st_size
        ) if md.get("filename") and (paths.MODELS_DIR / md["filename"]).is_file()
        else None,
        "load_sec": round(load_sec, 3),
        "peak_rss_gib": round(_rss_gib(peak_kb), 3),
        "cpu_sec_load": round(cpu_sec, 3),
    }


def per_model() -> dict:
    """Load each model in its own subprocess; print JSON lines + summary."""
    from aitextd import paths

    with open(paths.MODELS_JSON, encoding="utf-8") as f:
        model_ids = [m["id"] for m in json.load(f)]

    results = []
    for mid in model_ids:
        proc = subprocess.run(
            [UV_PYTHON, "-m", "aitextd.profiling", "--child", "per-model", mid],
            capture_output=True, text=True, env=_env(), check=True,
        )
        results.append(json.loads(proc.stdout.strip().splitlines()[-1]))

    results.sort(key=lambda r: r["peak_rss_gib"], reverse=True)
    total_peak = sum(r["peak_rss_gib"] for r in results)
    total_load = sum(r["load_sec"] for r in results)
    return {
        "per_model": results,
        "count": len(results),
        "sum_peak_rss_gib": round(total_peak, 3),
        "sum_load_sec": round(total_load, 3),
        "note": "isolated subprocesses overcount shared tokenizers; see full-stack for real deploy footprint",
    }


# ---------------------------------------------------------------- full-stack (child)

def _full_stack_child() -> dict:
    """One process: store.init() (loads everything), warmup requests, report steady RSS + latency."""
    import time as _time

    from aitextd import paths
    from aitextd.store import ModelStore

    t_init = _time.perf_counter()
    store = ModelStore()
    store.init()
    init_sec = _time.perf_counter() - t_init

    init_peak_gib = _rss_gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    # Warmup requests: cheap sklearn + heavy backbone + ensemble (runs every model)
    # plus the model-independent /details breakdown.
    texts = {
        "short": SAMPLE_TEXTS["short"],
        "medium": SAMPLE_TEXTS["medium"],
        "long": LONG_TEXT,
    }

    def _timed(fn):
        t0 = _time.perf_counter()
        out = fn()
        return out, _time.perf_counter() - t0

    latencies = []
    for label, text in texts.items():
        for model_id in ("svm", "gpt2_transformer", "parsbert", "ensemble"):
            _, sec = _timed(lambda: store.detection(text, model_id))
            latencies.append({"text": label, "model": model_id, "sec": round(sec, 4)})
    _, details_sec = _timed(lambda: store.detail_breakdown(SAMPLE_TEXTS["medium"]))

    steady_peak_gib = _rss_gib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    cpu_sec = (
        resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_SELF).ru_stime
    )

    return {
        "init_sec": round(init_sec, 3),
        "init_peak_rss_gib": round(init_peak_gib, 3),
        "steady_peak_rss_gib": round(steady_peak_gib, 3),
        "rss_growth_gib": round(steady_peak_gib - init_peak_gib, 3),
        "cpu_sec_total": round(cpu_sec, 3),
        "details_sec": round(details_sec, 4),
        "latencies": latencies,
    }


def full_stack() -> dict:
    proc = subprocess.run(
        [UV_PYTHON, "-m", "aitextd.profiling", "--child", "full-stack"],
        capture_output=True, text=True, env=_env(), check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------- main

def _child_dispatch():
    kind = sys.argv[2]
    if kind == "per-model":
        print(json.dumps(_per_model_child(sys.argv[3])))
    elif kind == "full-stack":
        print(json.dumps(_full_stack_child()))
    else:
        raise SystemExit(f"unknown child kind: {kind}")


def _summary(result: dict) -> None:
    if "disk" in result:
        d = result["disk"]
        print("== disk ==")
        for f in d["disk"][:12]:
            print(f"  {f['size']:>9}  {f['path']}")
        if len(d["disk"]) > 12:
            print(f"  ... ({len(d['disk'])-12} more files)")
        print(f"  TOTAL: {d['total']}")
    if "per_model" in result:
        pm = result["per_model"]
        print("== per-model load (isolated) ==")
        print(f"  {'rss':>8} {'load':>7}  model")
        for r in pm["per_model"]:
            print(f"  {r['peak_rss_gib']:>7.2f}GiB {r['load_sec']:>6.2f}s  {r['model_id']} ({r['runtime']})")
        print(f"  sum of isolated peaks: {pm['sum_peak_rss_gib']} GiB  (overcounts shared tokenizers)")
    if "full_stack" in result:
        fs = result["full_stack"]
        print("== full-stack (one process, all models) ==")
        print(f"  init time:      {fs['init_sec']}s")
        print(f"  RSS after init: {fs['init_peak_rss_gib']} GiB   (steady-state base)")
        print(f"  RSS after reqs: {fs['steady_peak_rss_gib']} GiB   (+{fs['rss_growth_gib']} GiB growth)")
        print(f"  /details latency: {fs['details_sec']}s")
        print("  per-request latency (single forward pass):")
        for l in fs["latencies"]:
            print(f"    {l['model']:<20} {l['text']:<7} {l['sec']}s")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _child_dispatch()
        return

    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    out: dict = {}
    if command in ("all", "disk"):
        out["disk"] = disk()
    if command in ("all", "per-model"):
        out["per_model"] = per_model()
    if command in ("all", "full-stack"):
        out["full_stack"] = full_stack()

    print(json.dumps(out))
    print()
    _summary(out)


if __name__ == "__main__":
    main()
