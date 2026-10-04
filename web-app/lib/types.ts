export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
}

export interface ModelTraining {
  totalSamples: number;
  trainSamples: number;
  testSamples: number;
  featureCount: number;
  testSplitRatio: number;
  randomSeed: number;
  dataset: string;
}

export interface ModelConfusionMatrix {
  trueNegative: number;
  falsePositive: number;
  falseNegative: number;
  truePositive: number;
}

export type ModelType = "ml" | "dl" | "ensemble";

export interface Model {
  id: string;
  name: string;
  nameFa?: string;
  description: string;
  descriptionFa?: string;
  algorithm: string;
  type?: ModelType;
  version: string;
  metrics?: ModelMetrics;
  training?: ModelTraining;
  confusionMatrix?: ModelConfusionMatrix;
  ensembleWeight?: number;
}

export interface ModelsResponse {
  models: Model[];
}

export interface DetectionRequest {
  text: string;
  modelId: string;
}

export interface DetectionDetail {
  label: string;
  labelFa: string;
  value: number;
  raw: string;
  description: string;
}

export interface EnsembleVote {
  modelId: string;
  label: string;
  labelFa: string;
  aiProbability: number;
  weight: number;
}

export interface JevChoiceAnswer {
  question: string;
  questionFa: string;
  candidates: string[];
  candidateLabelsFa: string[];
  probabilities: number[];
  answer: string;
  answerFa: string;
}

export interface JevScoreAnswer {
  question: string;
  questionFa: string;
  levels: string[];
  levelLabelsFa: string[];
  probabilities: number[];
  value: number;
  answer: string;
  answerFa: string;
}

export interface JevNoulAnswer {
  question: string;
  questionFa: string;
  probability: number;
  answer: boolean;
  answerFa: string;
}

export interface JevPolicyDecision {
  tier: string;
  tierFa: string;
  descriptionFa: string;
}

export interface JevTypedAnswers {
  choice: JevChoiceAnswer;
  score: JevScoreAnswer;
  noul: JevNoulAnswer;
  policy: JevPolicyDecision;
}

export interface DetectionResponse {
  isAI: boolean;
  confidence: number;
  verdict: string;
  verdictFa: string;
  details: DetectionDetail[];
  modelUsed: string;
  textLength: number;
  wordCount: number;
  ensembleVotes?: EnsembleVote[] | null;
  typedAnswers?: JevTypedAnswers | null;
}
