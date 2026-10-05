"""Pydantic response schemas for the detection API."""

from pydantic import BaseModel


class DetectionDetail(BaseModel):
    label: str
    labelFa: str
    value: float  # normalized 0..1
    raw: str
    description: str


class EnsembleVote(BaseModel):
    modelId: str
    label: str
    labelFa: str
    aiProbability: float
    weight: float


class SystemOneChoiceAnswer(BaseModel):
    question: str
    questionFa: str
    candidates: list[str]
    candidateLabelsFa: list[str]
    probabilities: list[float]
    answer: str
    answerFa: str


class SystemOneScoreAnswer(BaseModel):
    question: str
    questionFa: str
    levels: list[str]
    levelLabelsFa: list[str]
    probabilities: list[float]
    value: float  # probability-weighted value on the 0..(n-1) ordinal scale
    answer: str
    answerFa: str


class SystemOneNoulAnswer(BaseModel):
    question: str
    questionFa: str
    probability: float
    answer: bool
    answerFa: str


class SystemOnePolicyDecision(BaseModel):
    tier: str
    tierFa: str
    descriptionFa: str


class SystemOneTypedAnswers(BaseModel):
    choice: SystemOneChoiceAnswer
    score: SystemOneScoreAnswer
    noul: SystemOneNoulAnswer
    policy: SystemOnePolicyDecision


class DetectionResponse(BaseModel):
    isAI: bool
    confidence: float
    verdict: str
    verdictFa: str
    details: list[DetectionDetail]
    modelUsed: str
    textLength: int
    wordCount: int
    ensembleVotes: list[EnsembleVote] | None = None
    # Filled only for the System One decision model (system_one):
    # typed answers to its three questions instead of a plain probability.
    typedAnswers: SystemOneTypedAnswers | None = None
