"""System One decision model: typed-answer rendering and decision policy.

A System One model returns typed answers and calibrated probabilities
instead of generated text. This module turns the raw forward-pass output of
aitextd.architectures.SystemOneDecisionModel into the typed-answer block of
the API response -- one answer per typed question (choice / score / noul) --
plus the three-tier decision policy (proceed / needs review / low
confidence). The thresholds come from the checkpoint, but the gating itself
stays in code: the model never chooses its own next action.
"""

import numpy as np

from aitextd.constants import (
    SYSTEM_ONE_CHOICE_CANDIDATES,
    SYSTEM_ONE_CHOICE_CANDIDATES_FA,
    SYSTEM_ONE_QUESTIONS,
    SYSTEM_ONE_SCORE_LEVELS,
    SYSTEM_ONE_SCORE_LEVELS_FA,
)
from aitextd.schemas import (
    SystemOneChoiceAnswer,
    SystemOneNoulAnswer,
    SystemOnePolicyDecision,
    SystemOneScoreAnswer,
    SystemOneTypedAnswers,
)


def typed_answers(bundle, result: dict) -> SystemOneTypedAnswers:
    """Wrap the raw System One forward-pass output into the typed-answer response
    block. `bundle` must carry the checkpoint config (for the policy
    thresholds); `result` is the dict returned by predict_system_one."""
    choice_probs = result["choiceProbs"]
    choice_idx = int(np.argmax(choice_probs))
    score_probs = result["scoreProbs"]
    score_idx = int(np.argmax(score_probs))

    thresholds = bundle["config"].get("thresholds", {})
    proceed_thr = float(thresholds.get("proceed", 0.8))
    review_thr = float(thresholds.get("needsReview", 0.6))
    confidence = max(choice_probs)
    if confidence >= proceed_thr:
        policy = SystemOnePolicyDecision(
            tier="proceed",
            tierFa="ادامه",
            descriptionFa="اطمینان مدل بالاست؛ تصمیم خودکار قابل‌اعتماد است.",
        )
    elif confidence >= review_thr:
        policy = SystemOnePolicyDecision(
            tier="needs_review",
            tierFa="نیازمند بازبینی انسانی",
            descriptionFa="پاسخ مدل در ناحیه مرزی است؛ پیش از اقدام، بازبینی انسانی توصیه می‌شود.",
        )
    else:
        policy = SystemOnePolicyDecision(
            tier="low_confidence",
            tierFa="اطمینان پایین",
            descriptionFa="اطمینان مدل کم است؛ تصمیم خودکار توصیه نمی‌شود.",
        )

    return SystemOneTypedAnswers(
        choice=SystemOneChoiceAnswer(
            question=SYSTEM_ONE_QUESTIONS["choice"]["en"],
            questionFa=SYSTEM_ONE_QUESTIONS["choice"]["fa"],
            candidates=SYSTEM_ONE_CHOICE_CANDIDATES,
            candidateLabelsFa=SYSTEM_ONE_CHOICE_CANDIDATES_FA,
            probabilities=[round(p, 4) for p in choice_probs],
            answer=SYSTEM_ONE_CHOICE_CANDIDATES[choice_idx],
            answerFa=SYSTEM_ONE_CHOICE_CANDIDATES_FA[choice_idx],
        ),
        score=SystemOneScoreAnswer(
            question=SYSTEM_ONE_QUESTIONS["score"]["en"],
            questionFa=SYSTEM_ONE_QUESTIONS["score"]["fa"],
            levels=SYSTEM_ONE_SCORE_LEVELS,
            levelLabelsFa=SYSTEM_ONE_SCORE_LEVELS_FA,
            probabilities=[round(p, 4) for p in score_probs],
            value=round(result["scoreValue"], 3),
            answer=SYSTEM_ONE_SCORE_LEVELS[score_idx],
            answerFa=SYSTEM_ONE_SCORE_LEVELS_FA[score_idx],
        ),
        noul=SystemOneNoulAnswer(
            question=SYSTEM_ONE_QUESTIONS["noul"]["en"],
            questionFa=SYSTEM_ONE_QUESTIONS["noul"]["fa"],
            probability=round(result["noulProbability"], 4),
            answer=result["noulProbability"] >= 0.5,
            answerFa="بله" if result["noulProbability"] >= 0.5 else "خیر",
        ),
        policy=policy,
    )
