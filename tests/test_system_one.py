"""Pin the System One typed-answer wrapping: probability ordering and policy tiers."""

from aitextd import system_one
from aitextd.constants import SYSTEM_ONE_CHOICE_CANDIDATES, SYSTEM_ONE_SCORE_LEVELS


def make_bundle(proceed=0.8, needs_review=0.6):
    return {"config": {"thresholds": {"proceed": proceed, "needsReview": needs_review}}}


def make_result(choice_probs, score_probs, noul_prob):
    return {
        "choiceProbs": choice_probs,
        "scoreProbs": score_probs,
        "scoreValue": sum(p * i for i, p in enumerate(score_probs)),
        "noulProbability": noul_prob,
    }


def test_choice_answer_follows_candidate_order():
    # index 1 == "ai_generated": a high second probability means AI.
    result = make_result([0.1, 0.9], [0.1, 0.15, 0.2, 0.25, 0.3], 0.2)
    answers = system_one.typed_answers(make_bundle(), result)
    assert answers.choice.candidates == SYSTEM_ONE_CHOICE_CANDIDATES
    assert answers.choice.answer == "ai_generated"
    assert answers.choice.answerFa == "تولیدشده توسط هوش مصنوعی"
    assert answers.choice.probabilities == [0.1, 0.9]


def test_score_answer_and_weighted_value():
    result = make_result([0.5, 0.5], [0.1, 0.2, 0.3, 0.25, 0.15], 0.5)
    answers = system_one.typed_answers(make_bundle(), result)
    assert answers.score.answer == "mixed_signals"  # argmax at index 2
    assert answers.score.answerFa == "سیگنال‌های ترکیبی"
    expected_value = round(0.1 * 0 + 0.2 * 1 + 0.3 * 2 + 0.25 * 3 + 0.15 * 4, 3)
    assert answers.score.value == expected_value
    assert answers.score.levels == SYSTEM_ONE_SCORE_LEVELS


def test_noul_answer_threshold():
    result = make_result([0.5, 0.5], [0.2] * 5, 0.85)
    answers = system_one.typed_answers(make_bundle(), result)
    assert answers.noul.answer is True
    assert answers.noul.answerFa == "بله"


def test_policy_tiers():
    high = make_result([0.05, 0.95], [0.2] * 5, 0.1)
    assert system_one.typed_answers(make_bundle(), high).policy.tier == "proceed"

    mid = make_result([0.3, 0.7], [0.2] * 5, 0.1)
    assert system_one.typed_answers(make_bundle(), mid).policy.tier == "needs_review"

    low = make_result([0.55, 0.45], [0.2] * 5, 0.1)
    assert system_one.typed_answers(make_bundle(), low).policy.tier == "low_confidence"


def test_policy_uses_checkpoint_thresholds():
    result = make_result([0.25, 0.75], [0.2] * 5, 0.1)
    # Default thresholds: 0.75 < 0.8 -> needs_review.
    assert system_one.typed_answers(make_bundle(), result).policy.tier == "needs_review"
    # Lower the proceed threshold: 0.75 >= 0.7 -> proceed.
    assert system_one.typed_answers(make_bundle(proceed=0.7), result).policy.tier == "proceed"
