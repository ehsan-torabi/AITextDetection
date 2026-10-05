"use client";

import * as React from "react";
import type {
  DetectionResponse,
  SystemOneTypedAnswers,
  TypeSafeDetectionResponse,
} from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  ShieldCheck,
  ShieldAlert,
  Bot,
  User,
  TrendingUp,
  Activity,
  Sparkles,
  BarChart3,
  BrainCircuit,
  Scale,
  ListChecks,
  Eye,
} from "lucide-react";

interface DetectionResultProps {
  result: DetectionResponse | TypeSafeDetectionResponse;
  /** Optional provenance label shown next to the card title, e.g. for the
   * online System One result in compare mode. */
  sourceLabel?: string;
}

const metricIcons: Record<string, React.ReactNode> = {
  entropy: <BrainCircuit className="size-4" />,
  sentence_variety: <Activity className="size-4" />,
  repetition: <Sparkles className="size-4" />,
  coherence: <TrendingUp className="size-4" />,
};

export function DetectionResult({ result, sourceLabel }: DetectionResultProps) {
  const online = "usage" in result ? (result as TypeSafeDetectionResponse) : null;
  const percentage = Math.round(result.confidence * 100);
  const circumference = 2 * Math.PI * 58;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const confidenceLevel =
    result.confidence > 0.8 ? "بالا" : result.confidence > 0.5 ? "متوسط" : "پایین";

  const confidenceVariant: "default" | "secondary" | "destructive" =
    result.confidence > 0.8 ? "destructive" : result.confidence > 0.5 ? "default" : "secondary";

  return (
    <div className="space-y-6">
      {/* Main Result */}
      <Card>
        <CardHeader className="pb-0">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">نتیجه تحلیل</CardTitle>
            <div className="flex items-center gap-1.5">
              {sourceLabel && (
                <Badge variant="outline" className="gap-1 text-[10px]">
                  {sourceLabel}
                </Badge>
              )}
              <Badge variant={confidenceVariant}>اطمینان: {confidenceLevel}</Badge>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col items-center gap-6 py-4 sm:flex-row sm:items-start">
            {/* Circular Progress - CSS transition only */}
            <div className="relative flex-shrink-0">
              <svg
                width="140"
                height="140"
                viewBox="0 0 140 140"
                className="[&_circle]:fill-none [&_circle]:stroke-[8px] [&_circle]:[stroke-linecap:round] [&_circle]:origin-center [&_circle]:-rotate-90"
              >
                <circle cx="70" cy="70" r="58" className="stroke-muted" />
                <circle
                  cx="70"
                  cy="70"
                  r="58"
                  className="transition-[stroke-dashoffset] duration-1000 ease-out"
                  stroke={result.isAI ? "oklch(0.65 0.2 18)" : "oklch(0.696 0.17 162)"}
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-bold tabular-nums">{percentage}%</span>
                <span className="text-xs text-muted-foreground mt-0.5">احتمال</span>
              </div>
            </div>

            {/* Verdict */}
            <div className="flex-1 space-y-4 text-center sm:text-right">
              <div className="flex items-center justify-center gap-3 sm:justify-start">
                {result.isAI ? (
                  <div className="flex items-center justify-center size-10 rounded-xl bg-destructive/10 text-destructive">
                    <Bot className="size-5" />
                  </div>
                ) : (
                  <div className="flex items-center justify-center size-10 rounded-xl bg-primary/10 text-primary">
                    <User className="size-5" />
                  </div>
                )}
                <div>
                  <h3 className="text-lg font-bold">{result.verdictFa}</h3>
                  <p className="text-sm text-muted-foreground">{result.verdict}</p>
                </div>
              </div>

              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-muted-foreground justify-center sm:justify-start">
                <span className="flex items-center gap-1.5">
                  {result.isAI ? (
                    <ShieldAlert className="size-3.5 text-destructive" />
                  ) : (
                    <ShieldCheck className="size-3.5 text-primary" />
                  )}
                  مدل: {online?.resolvedModel ?? result.modelUsed}
                </span>
                {online && (
                  <span className="tabular-nums" dir="ltr">
                    اطمینان انتخاب: {Math.round(online.choiceConfidence * 100)}٪
                  </span>
                )}
                {online?.usage && (
                  <span className="tabular-nums" dir="ltr">
                    {online.usage.inputTokens}/{online.usage.outputTokens} tokens ·{" "}
                    {(online.latencyMs / 1000).toFixed(1)}s
                  </span>
                )}
                <span className="text-border">|</span>
                <span>{result.wordCount} کلمه</span>
                <span className="text-border">|</span>
                <span>{result.textLength} کاراکتر</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* System One typed answers (choice / score / noul) */}
      {result.typedAnswers && <SystemOneAnswersPanel answers={result.typedAnswers} />}

      {/* Ensemble Breakdown */}
      {result.ensembleVotes && result.ensembleVotes.length > 0 && (
        <EnsembleBreakdown votes={result.ensembleVotes} />
      )}

      {/* Detail Metrics */}
      <div>
        <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
          <BarChart3 className="size-4 text-muted-foreground" />
          جزئیات تحلیل
        </h3>
        <div className="grid gap-3 sm:grid-cols-2">
          {result.details.map((detail) => (
            <MetricCard key={detail.label} detail={detail} />
          ))}
        </div>
      </div>
    </div>
  );
}

function SystemOneAnswersPanel({ answers }: { answers: SystemOneTypedAnswers }) {
  const policyVariant: "default" | "destructive" | "secondary" =
    answers.policy.tier === "proceed"
      ? "default"
      : answers.policy.tier === "needs_review"
        ? "destructive"
        : "secondary";

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <ListChecks className="size-4 text-muted-foreground" />
            پاسخ‌های نوع‌بندی‌شده (سیستم-یک)
          </CardTitle>
          <Badge variant={policyVariant}>{answers.policy.tierFa}</Badge>
        </div>
        <CardDescription className="text-xs leading-relaxed">
          مدل تصمیم سیستم-یک به جای تولید متن، سه سؤال نوع‌بندی‌شده را در یک گذر رو به جلو پاسخ
          می‌دهد: انتخاب، امتیاز و بازبینی انسانی.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {/* Choice question: distribution over candidate answers */}
        <section className="space-y-2">
          <h4 className="text-xs font-semibold text-muted-foreground">
            {answers.choice.questionFa}
          </h4>
          {answers.choice.candidates.map((candidate, i) => {
            const percentage = Math.round(answers.choice.probabilities[i] * 100);
            const isAi = candidate === "ai_generated";
            const isAnswer = answers.choice.answer === candidate;
            return (
              <div
                key={candidate}
                className={`space-y-1.5 rounded-lg p-2 ${
                  isAnswer ? "bg-primary/5 ring-1 ring-primary/20" : ""
                }`}
              >
                <div className="flex items-center justify-between gap-2 text-xs">
                  <span className="flex items-center gap-1.5 font-medium">
                    {isAi ? (
                      <Bot className="size-3.5 text-destructive" />
                    ) : (
                      <User className="size-3.5 text-primary" />
                    )}
                    {answers.choice.candidateLabelsFa[i]}
                    {isAnswer && (
                      <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
                        پاسخ
                      </Badge>
                    )}
                  </span>
                  <span
                    className={`font-bold tabular-nums ${
                      isAi ? "text-destructive" : "text-primary"
                    }`}
                  >
                    {percentage}٪
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ease-out ${
                      isAi ? "bg-destructive" : "bg-primary"
                    }`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            );
          })}
        </section>

        {/* Score question: probability-weighted value between levels */}
        <section className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <h4 className="text-xs font-semibold text-muted-foreground">
              {answers.score.questionFa}
            </h4>
            <span className="text-xs font-bold tabular-nums" dir="ltr">
              {answers.score.value.toFixed(2)} / {answers.score.levels.length - 1}
            </span>
          </div>
          <div className="flex gap-1.5">
            {answers.score.probabilities.map((prob, i) => (
              <div key={answers.score.levels[i]} className="flex-1 space-y-1">
                <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                  <div
                    className="h-full rounded-full bg-chart-3 transition-all duration-700 ease-out"
                    style={{ width: `${Math.round(prob * 100)}%` }}
                  />
                </div>
                <p
                  className={`text-[10px] leading-tight text-center ${
                    answers.score.answer === answers.score.levels[i]
                      ? "font-bold text-foreground"
                      : "text-muted-foreground"
                  }`}
                >
                  {answers.score.levelLabelsFa[i]}
                </p>
              </div>
            ))}
          </div>
          <p className="text-[11px] text-muted-foreground">
            نزدیک‌ترین سطح:{" "}
            <span className="font-medium text-foreground">{answers.score.answerFa}</span>
          </p>
        </section>

        {/* Noul question: yes/no review probability */}
        <section className="space-y-2">
          <div className="flex items-center justify-between gap-2">
            <h4 className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
              <Eye className="size-3.5" />
              {answers.noul.questionFa}
            </h4>
            <Badge variant={answers.noul.answer ? "destructive" : "outline"}>
              {answers.noul.answerFa}
            </Badge>
          </div>
          <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-700 ease-out ${
                answers.noul.answer ? "bg-destructive" : "bg-primary"
              }`}
              style={{ width: `${Math.round(answers.noul.probability * 100)}%` }}
            />
          </div>
          <p className="text-[11px] tabular-nums text-muted-foreground">
            احتمال: {Math.round(answers.noul.probability * 100)}٪
          </p>
        </section>

        {/* Decision policy */}
        <p className="flex items-start gap-1.5 text-[11px] leading-relaxed text-muted-foreground border-t border-border pt-3">
          <ShieldCheck className="size-3 mt-0.5 flex-shrink-0" />
          <span>
            <span className="font-medium text-foreground">{answers.policy.tierFa}:</span>{" "}
            {answers.policy.descriptionFa}
          </span>
        </p>
      </CardContent>
    </Card>
  );
}

function EnsembleBreakdown({ votes }: { votes: NonNullable<DetectionResponse["ensembleVotes"]> }) {
  const maxWeight = Math.max(...votes.map((v) => v.weight));
  const ranked = [...votes].sort((a, b) => b.weight - a.weight);

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm flex items-center gap-2">
          <Scale className="size-4 text-muted-foreground" />
          سهم هر مدل در تصمیم ترکیبی
        </CardTitle>
        <CardDescription className="text-xs leading-relaxed">
          احتمال هوش‌مصنوعی‌بودن به تشخیص هر مدل، به همراه وزن آن در میانگین وزن‌دار نهایی (مدل
          GPT-2 وزن بیشتری دارد چون معمولاً دقیق‌ترین است).
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-2.5">
        {ranked.map((vote) => {
          const percentage = Math.round(vote.aiProbability * 100);
          const weightPct = (vote.weight / maxWeight) * 100;
          const isTopWeight = vote.weight === maxWeight;

          return (
            <div
              key={vote.modelId}
              className={`space-y-1.5 rounded-lg p-2 ${
                isTopWeight ? "bg-primary/5 ring-1 ring-primary/20" : ""
              }`}
            >
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="flex items-center gap-1.5 font-medium">
                  {vote.labelFa}
                  <Badge
                    variant={isTopWeight ? "default" : "outline"}
                    className="h-4 px-1.5 text-[10px] tabular-nums"
                  >
                    وزن ×{vote.weight}
                  </Badge>
                </span>
                <span
                  className={`font-bold tabular-nums ${
                    vote.aiProbability >= 0.5 ? "text-destructive" : "text-primary"
                  }`}
                >
                  {percentage}٪ AI
                </span>
              </div>
              <div className="flex items-center gap-2">
                <div className="h-1.5 flex-1 rounded-full bg-muted overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ease-out ${
                      vote.aiProbability >= 0.5 ? "bg-destructive" : "bg-primary"
                    }`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <div
                  className="h-1.5 rounded-full bg-chart-3/60"
                  style={{ width: `${Math.max(weightPct * 0.25, 6)}px` }}
                  title={`وزن: ${vote.weight}`}
                />
              </div>
            </div>
          );
        })}
      </CardContent>
    </Card>
  );
}

function MetricCard({ detail }: { detail: DetectionResponse["details"][number] }) {
  const percentage = Math.round(detail.value * 100);

  const barColor =
    detail.value > 0.7 ? "bg-destructive" : detail.value > 0.4 ? "bg-chart-3" : "bg-primary";

  return (
    <Card size="sm">
      <CardHeader className="pb-1.5">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm flex items-center gap-2">
            <span className="text-muted-foreground">
              {metricIcons[detail.label] ?? <BarChart3 className="size-4" />}
            </span>
            {detail.labelFa}
          </CardTitle>
          <span className="text-base font-bold tabular-nums">{percentage}%</span>
        </div>
        <CardDescription className="text-xs leading-relaxed">
          {detail.description}
          {detail.raw && (
            <span className="block mt-1 font-medium text-foreground/70" dir="ltr">
              {detail.raw}
            </span>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-700 ease-out ${barColor}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </CardContent>
    </Card>
  );
}
