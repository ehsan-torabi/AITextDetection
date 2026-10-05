"""API surface tests with the heavy model loading monkeypatched out."""

import pytest
from fastapi.testclient import TestClient

from aitextd import api


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(
        api.store,
        "models_dict",
        {
            "svm": {"id": "svm", "name": "SVM", "runtime": "sklearn-tfidf", "filename": "x"},
        },
        raising=False,
    )
    monkeypatch.setattr(
        api.store,
        "detection",
        lambda text, model_id: {"isAI": False, "modelUsed": model_id},
    )
    return TestClient(api.app)


def test_detect_rejects_blank_text(client):
    response = client.post("/detect", json={"text": "   ", "modelId": "svm"})
    assert response.status_code == 400


def test_detect_returns_model_result(client):
    response = client.post("/detect", json={"text": "سلام دنیا", "modelId": "svm"})
    assert response.status_code == 200
    assert response.json() == {"isAI": False, "modelUsed": "svm"}


def test_models_endpoint_lists_ensemble_first_and_strips_internal_fields(client):
    response = client.get("/models")
    assert response.status_code == 200
    models = response.json()["models"]
    assert models[0]["id"] == "ensemble"
    assert models[1]["id"] == "svm"
    # runtime/filename are implementation details and must not leak.
    assert "runtime" not in models[1]
    assert "filename" not in models[1]


def test_details_rejects_blank_text(client):
    response = client.post("/details", json={"text": "   "})
    assert response.status_code == 400


def test_details_returns_shared_rows(client, monkeypatch):
    rows = [{"label": "entropy", "value": 0.5, "raw": "4.20 bits"}]
    monkeypatch.setattr(api.store, "detail_breakdown", lambda text: rows)
    response = client.post("/details", json={"text": "سلام دنیا"})
    assert response.status_code == 200
    assert response.json() == {"details": rows}
