"""Golden-output parity for the full serving stack.

The first run with no golden file captures one; every later run asserts that
detection() returns byte-identical output for the same inputs. This is the
refactor guardrail: any change in preprocessing, features, model loading, or
prediction that alters a served prediction fails here.
"""

import json

import pytest
from conftest import ALL_MODEL_IDS, GOLDEN_PATH, SAMPLE_AI_TEXT, SAMPLE_HUMAN_TEXT


@pytest.mark.slow
def test_detection_matches_golden(store):
    samples = {"human": SAMPLE_HUMAN_TEXT, "ai": SAMPLE_AI_TEXT}
    actual = {
        model_id: {
            text_name: store.detection(text, model_id).model_dump()
            for text_name, text in samples.items()
        }
        for model_id in ALL_MODEL_IDS
    }

    if not GOLDEN_PATH.exists():
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, ensure_ascii=False, indent=2))
        pytest.fail(
            f"No golden file existed; captured one at {GOLDEN_PATH}. Re-run to compare against it."
        )

    golden = json.loads(GOLDEN_PATH.read_text())
    assert set(golden) == set(actual)
    for model_id in ALL_MODEL_IDS:
        for text_name in samples:
            assert actual[model_id][text_name] == golden[model_id][text_name], (
                f"detection() output changed for model={model_id} text={text_name}"
            )


@pytest.mark.slow
def test_perplexity_matches_golden(store):
    perplexities = {
        "human": store.detection(SAMPLE_HUMAN_TEXT, "svm").model_dump(),
        "ai": store.detection(SAMPLE_AI_TEXT, "svm").model_dump(),
    }
    if not GOLDEN_PATH.exists():
        pytest.fail("run test_detection_matches_golden first to capture the golden file")
    golden = json.loads(GOLDEN_PATH.read_text())
    # The golden pins perplexity through the detail row, whose "raw" display
    # field is rounded to 2 decimals.
    for text_name in perplexities:
        detail = next(d for d in golden["svm"][text_name]["details"] if d["label"] == "perplexity")
        actual_detail = next(
            d for d in perplexities[text_name]["details"] if d["label"] == "perplexity"
        )
        assert actual_detail["value"] == detail["value"]
        assert actual_detail["raw"] == detail["raw"]
