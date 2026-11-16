"""Entrypoint for the Persian AI-Text Detection API (port 8000).

The app itself lives in aitextd.api; this launcher keeps the established
run command (`python server.py`) working. Model artifacts are not stored in
git: pass --download-online to fetch the missing ones from the Hugging Face
repo (https://huggingface.co/ehsantorabi/aitextd-models) before startup.
"""

import argparse
import json
import shutil
import sys

import uvicorn

from aitextd import download, paths
from aitextd.api import app


def _gib(num_bytes: float) -> str:
    return f"{num_bytes / 1024**3:.1f} GiB"


def ensure_models(download_online: bool) -> None:
    """Verify models/ is serving-ready; with --download-online, fetch the
    missing artifacts from the Hugging Face repo first (resumable: rerunning
    continues an interrupted download)."""
    if not paths.MODELS_JSON.is_file():
        print(
            f"Model registry not found: {paths.MODELS_JSON}\n"
            "Run from the project checkout (or point AITEXTD_ROOT at one); the "
            "registry is tracked in git alongside server.py.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        missing = download.missing_paths()
    except (json.JSONDecodeError, KeyError) as exc:
        print(
            f"Model registry {paths.MODELS_JSON} is not a valid models.json registry ({exc}).",
            file=sys.stderr,
        )
        sys.exit(1)
    if not missing:
        return

    if not download_online:
        print(
            "Missing model artifacts:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + f"\nThe API cannot serve without them. Re-run with --download-online "
            f"to fetch them from https://huggingface.co/{download.HF_REPO_ID}.",
            file=sys.stderr,
        )
        sys.exit(1)

    report = download.missing_size_report(missing)
    free = shutil.disk_usage(paths.PROJECT_ROOT).free
    if report is not None:
        total, largest = report
        size_note = f" ({_gib(total)})"
        # The in-flight temp file plus the final copy transiently coexist, so
        # headroom must cover the total plus the largest single artifact.
        if total + largest > free:
            print(
                f"Need about {_gib(total)} for the missing artifacts but only "
                f"{_gib(free)} is free on {paths.PROJECT_ROOT}; free up space and "
                "retry.",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        size_note = ""
    print(f"Downloading {len(missing)} missing artifact(s){size_note} ...", flush=True)

    still_missing = download.download_missing(missing)
    if still_missing:
        print(
            "Still missing after download:\n"
            + "\n".join(f"  - {path}" for path in still_missing)
            + "\nEither the Hub is unreachable (a restricted network can point "
            "huggingface_hub at a mirror via HF_ENDPOINT) or these paths are absent "
            "from the repo. Re-run with --download-online to resume; "
            "already-fetched files are kept.",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the Persian AI-Text Detection API (port 8000)."
    )
    parser.add_argument(
        "--download-online",
        action="store_true",
        help="fetch missing model artifacts from the Hugging Face repo before startup",
    )
    args = parser.parse_args()

    ensure_models(args.download_online)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
