"""Fetch missing serving artifacts from the Hugging Face model repo.

The trained checkpoints live on the Hub (mirrors models/ one-to-one -- see
huggingface/upload_models.sh). The launcher (server.py --download-online)
uses this module to bring a fresh checkout up to a serving-ready state; the
huggingface_hub calls are resumable and write files atomically, so an
interrupted download leaves nothing corrupt and the next run continues.

Every function takes its inputs as parameters (with the serving defaults),
so the logic is testable without network access or real artifacts.
"""

import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from aitextd import paths

HF_REPO_ID = "ehsantorabi/aitextd-models"

# Repo-relative paths of the shared serving artifacts, needed regardless of
# which model a request selects: the TF-IDF vectorizer (ModelStore.init and
# the style features), the raw Persian GPT-2 LM (the perplexity signal on
# every request and the pytorch-gpt2 backbone) plus its pickled tokenizer
# fallback, and the exported GPT-2 / ParsBERT tokenizer folders the LLM
# runtimes prefer over the pickle.
SHARED_REPO_PATHS: tuple[str, ...] = (
    "tfidf_vectorizer.joblib",
    "gpt2-fa.mdl",
    "gpt2-fa-tokenizer.mdl",
    "gpt2_tokenizer/tokenizer.json",
    "gpt2_tokenizer/tokenizer_config.json",
    "parsbert_tokenizer/tokenizer.json",
    "parsbert_tokenizer/tokenizer_config.json",
)

# The fine-tuned Laya artifact is a checkpoint directory, not a single file;
# these are the files under it that laya.load() needs (checkpoint_latest/ is
# a redundant mid-training duplicate and is deliberately not published).
LAYA_DIR_FILES: tuple[str, ...] = (
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)


def registry_repo_paths(models_json: Path = paths.MODELS_JSON) -> tuple[str, ...]:
    """Repo-relative artifact paths for every registered model in models.json.

    Plain runtimes register one artifact file; the pytorch-laya runtime
    registers a checkpoint directory, expanded to the files laya.load()
    actually reads.
    """
    with open(models_json, encoding="utf-8") as f:
        metadata_list = json.load(f)
    repo_paths: list[str] = []
    for md in metadata_list:
        filename = md["filename"]
        if md.get("runtime", "sklearn-tfidf") == "pytorch-laya":
            repo_paths.extend(f"{filename.rstrip('/')}/{sub}" for sub in LAYA_DIR_FILES)
        else:
            repo_paths.append(filename)
    # Registered artifacts are shared by construction (several models can point
    # at one file), so keep first-seen order and drop duplicates.
    return tuple(dict.fromkeys(repo_paths))


def required_repo_paths(models_json: Path = paths.MODELS_JSON) -> tuple[str, ...]:
    """Every repo-relative path a serving-ready models/ directory must hold."""
    return SHARED_REPO_PATHS + registry_repo_paths(models_json)


def missing_paths(
    required: tuple[str, ...] | None = None, models_dir: Path = paths.MODELS_DIR
) -> list[str]:
    """The required paths that do not exist under models_dir.

    Pure existence check: the launcher stays fully offline without the
    --download-online flag. Files snapshot_download itself wrote are safe
    (it downloads to an .incomplete temp file and renames into place), but a
    file truncated by some other means counts as present here and would only
    surface as a load error.
    """
    if required is None:
        required = required_repo_paths()
    return [repo_path for repo_path in required if not (models_dir / repo_path).exists()]


def missing_size_report(missing: list[str], repo_id: str = HF_REPO_ID) -> tuple[int, int] | None:
    """(total bytes, largest single file) for the missing paths on the Hub.

    None if the Hub is unreachable. Paths that no longer exist on the repo
    are ignored by get_paths_info and simply surface as still-missing after
    the download attempt.
    """
    try:
        infos = HfApi().get_paths_info(repo_id, missing, expand=True)
        sizes = []
        for info in infos:
            lfs = getattr(info, "lfs", None)
            sizes.append(lfs.size if lfs is not None else getattr(info, "size", 0) or 0)
        if not sizes:
            return None
        return sum(sizes), max(sizes)
    except Exception:
        return None


def download_missing(
    missing: list[str],
    models_dir: Path = paths.MODELS_DIR,
    repo_id: str = HF_REPO_ID,
) -> list[str]:
    """Download exactly the missing paths from the Hub repo into models_dir.

    One resumable snapshot_download for the whole set; on any Hub failure the
    files that already landed are kept and the paths still missing are
    returned (an empty list means models/ is serving-ready).
    """
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(models_dir),
            allow_patterns=list(missing),
        )
    except Exception as exc:
        print(
            f"Hugging Face download failed ({exc}); keeping what landed",
            file=sys.stderr,
            flush=True,
        )
    return missing_paths(tuple(missing), models_dir)
