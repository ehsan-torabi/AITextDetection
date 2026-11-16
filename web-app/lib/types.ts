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

/** Which engine(s) a detection run should use. */
export type DetectionMode = "local" | "online" | "compare";

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

export interface SystemOneChoiceAnswer {
  question: string;
  questionFa: string;
  candidates: string[];
  candidateLabelsFa: string[];
  probabilities: number[];
  answer: string;
  answerFa: string;
}

export interface SystemOneScoreAnswer {
  question: string;
  questionFa: string;
  levels: string[];
  levelLabelsFa: string[];
  probabilities: number[];
  value: number;
  answer: string;
  answerFa: string;
}

export interface SystemOneNoulAnswer {
  question: string;
  questionFa: string;
  probability: number;
  answer: boolean;
  answerFa: string;
}

export interface SystemOnePolicyDecision {
  tier: string;
  tierFa: string;
  descriptionFa: string;
}

export interface SystemOneTypedAnswers {
  choice: SystemOneChoiceAnswer;
  score: SystemOneScoreAnswer;
  noul: SystemOneNoulAnswer;
  policy: SystemOnePolicyDecision;
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
  typedAnswers?: SystemOneTypedAnswers | null;
}

// ---------------------------------------------------------------------------
// Online System One (TypeSafe) detection
// ---------------------------------------------------------------------------

/** Connection settings for the online System One provider, kept in
 * localStorage on the client; the API key never touches server storage. */
export interface TypeSafeProviderSettings {
  baseUrl: string;
  model: string;
  apiKey: string;
}

export interface TypeSafeUsage {
  inputTokens: number;
  outputTokens: number;
}

/** Online System One result. Extends DetectionResponse so the existing
 * result UI renders it directly; the extra fields carry online-specific
 * metadata (resolved model, token usage, latency). */
export interface TypeSafeDetectionResponse extends DetectionResponse {
  typedAnswers: SystemOneTypedAnswers;
  /** TypeSafe's choice-question confidence (distribution concentration). */
  choiceConfidence: number;
  requestedModel: string;
  resolvedModel?: string;
  usage: TypeSafeUsage | null;
  latencyMs: number;
}

/** Outcome of running one text through both the online System One model and
 * a local model; either side may be missing when its request failed. */
export interface OnlineLocalComparison {
  local: DetectionResponse | null;
  online: TypeSafeDetectionResponse | null;
  localError?: string;
  onlineError?: string;
}
