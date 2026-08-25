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

export interface Model {
  id: string;
  name: string;
  nameFa?: string;
  description: string;
  descriptionFa?: string;
  algorithm: string;
  version: string;
  metrics?: ModelMetrics;
  training?: ModelTraining;
  confusionMatrix?: ModelConfusionMatrix;
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

export interface DetectionResponse {
  isAI: boolean;
  confidence: number;
  verdict: string;
  verdictFa: string;
  details: DetectionDetail[];
  modelUsed: string;
  textLength: number;
  wordCount: number;
}
