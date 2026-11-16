"""Fast tests for the Hugging Face artifact downloader.

No network and no real artifacts: the file mapping is checked against the
real models.json registry (tracked in git), and every Hub interaction is
faked at the module boundary.
"""

import json
from pathlib import Path
from types import SimpleNamespace

from aitextd import download, paths


def _registry() -> list[dict]:
    with open(paths.MODELS_JSON, encoding="utf-8") as f:
        return json.load(f)


def test_required_paths_cover_every_registered_model():
    required = set(download.required_repo_paths())
    for md in _registry():
        filename = md["filename"]
        if md.get("runtime", "sklearn-tfidf") == "pytorch-laya":
            for sub in download.LAYA_DIR_FILES:
                assert f"{filename}/{sub}" in required
        else:
            assert filename in required, filename


def test_required_paths_include_shared_serving_artifacts():
    required = set(download.required_repo_paths())
    assert set(download.SHARED_REPO_PATHS) <= required


def test_missing_paths_filters_by_existence(tmp_path: Path):
    required = ("a/one.pt", "b/two.joblib", "three.pt")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "one.pt").touch()
    assert download.missing_paths(required, tmp_path) == ["b/two.joblib", "three.pt"]


def test_download_missing_passes_exact_patterns_and_reports_still_missing(
    tmp_path: Path, monkeypatch
):
    missing = ["one.pt", "dir/two.pt", "never.pt"]
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        for repo_path in kwargs["allow_patterns"][:-1]:
            target = Path(kwargs["local_dir"]) / repo_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()

    monkeypatch.setattr(download, "snapshot_download", fake_snapshot_download)
    still_missing = download.download_missing(missing, tmp_path)

    assert seen["allow_patterns"] == missing
    assert seen["local_dir"] == str(tmp_path)
    assert still_missing == ["never.pt"]


def test_download_missing_keeps_files_when_hub_fails(tmp_path: Path, monkeypatch):
    def fake_snapshot_download(**kwargs):
        raise ConnectionError("hub unreachable")

    monkeypatch.setattr(download, "snapshot_download", fake_snapshot_download)
    missing = ["one.pt", "two.pt"]
    assert download.download_missing(missing, tmp_path) == missing


def test_registry_duplicate_filenames_are_deduped(tmp_path: Path):
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            [
                {"id": "a", "filename": "shared.pt"},
                {"id": "b", "filename": "shared.pt"},
            ]
        ),
        encoding="utf-8",
    )
    assert download.registry_repo_paths(registry) == ("shared.pt",)


def test_required_paths_snapshot():
    """Pin the full serving-artifact set: guards the registry expansion and
    the shared paths against accidental drift (any real change to what
    store.init()/loaders.py open must update this list deliberately)."""
    assert download.required_repo_paths() == (
        "tfidf_vectorizer.joblib",
        "gpt2-fa.mdl",
        "gpt2-fa-tokenizer.mdl",
        "gpt2_tokenizer/tokenizer.json",
        "gpt2_tokenizer/tokenizer_config.json",
        "parsbert_tokenizer/tokenizer.json",
        "parsbert_tokenizer/tokenizer_config.json",
        "lstmModel.pt",
        "textcnnModel.pt",
        "textcnnRlcdModel.pt",
        "attnBilstmModel.pt",
        "gpt2ClassifierModel.pt",
        "parsbertClassifierModel.pt",
        "systemOneModel.pt",
        "svModel.joblib",
        "logregModel.joblib",
        "nbModel.joblib",
        "rfModel.joblib",
        "gbModel.joblib",
        "laya_persian/model.safetensors",
        "laya_persian/rl_agent_config.json",
        "laya_persian/encoder/config.json",
        "laya_persian/tokenizer/tokenizer.json",
        "laya_persian/tokenizer/tokenizer_config.json",
        "mlpModel.pt",
        "charcnnModel.pt",
        "tinyTransformerModel.pt",
    )


def test_missing_size_report_prefers_lfs_size(monkeypatch):
    class FakeApi:
        def get_paths_info(self, repo_id, paths, expand):
            assert expand is True
            return [
                SimpleNamespace(size=134, lfs=SimpleNamespace(size=1000)),
                SimpleNamespace(size=500, lfs=None),
            ]

    monkeypatch.setattr(download, "HfApi", FakeApi)
    assert download.missing_size_report(["a.pt", "b.joblib"]) == (1500, 1000)


def test_missing_size_report_ignores_folder_entries(monkeypatch):
    """A repo path that resolves to a folder (or is gone) must not crash the
    size report; RepoFolder-like objects carry no lfs/size attributes."""

    class FakeApi:
        def get_paths_info(self, repo_id, paths, expand):
            return [SimpleNamespace(size=500, lfs=None), SimpleNamespace(path="x")]

    monkeypatch.setattr(download, "HfApi", FakeApi)
    assert download.missing_size_report(["b.joblib", "now-a-folder.pt"]) == (500, 500)


def test_missing_size_report_returns_none_when_hub_fails(monkeypatch):
    class FakeApi:
        def get_paths_info(self, repo_id, paths, expand):
            raise ConnectionError("hub unreachable")

    monkeypatch.setattr(download, "HfApi", FakeApi)
    assert download.missing_size_report(["a.pt"]) is None
