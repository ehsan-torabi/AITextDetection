"use client";

import * as React from "react";
import type { DetectionResponse } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
} from "@/components/ui/card";
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
} from "lucide-react";

interface DetectionResultProps {
  result: DetectionResponse;
}

const metricIcons: Record<string, React.ReactNode> = {
  entropy: <BrainCircuit className="size-4" />,
  sentence_variety: <Activity className="size-4" />,
  repetition: <Sparkles className="size-4" />,
  coherence: <TrendingUp className="size-4" />,
};

export function DetectionResult({ result }: DetectionResultProps) {
  const percentage = Math.round(result.confidence * 100);
  const circumference = 2 * Math.PI * 58;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  const confidenceLevel =
    result.confidence > 0.8
      ? "بالا"
      : result.confidence > 0.5
        ? "متوسط"
        : "پایین";

  const confidenceVariant: "default" | "secondary" | "destructive" =
    result.confidence > 0.8
      ? "destructive"
      : result.confidence > 0.5
        ? "default"
        : "secondary";

  return (
    <div className="space-y-6">
      {/* Main Result */}
      <Card>
        <CardHeader className="pb-0">
          <div className="flex items-center justify-between">
            <CardTitle className="text-base">نتیجه تحلیل</CardTitle>
            <Badge variant={confidenceVariant}>
              اطمینان: {confidenceLevel}
            </Badge>
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
                  stroke={
                    result.isAI
                      ? "oklch(0.65 0.2 18)"
                      : "oklch(0.696 0.17 162)"
                  }
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeDashoffset}
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-3xl font-bold tabular-nums">
                  {percentage}%
                </span>
                <span className="text-xs text-muted-foreground mt-0.5">
                  احتمال
                </span>
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
                  <p className="text-sm text-muted-foreground">
                    {result.verdict}
                  </p>
                </div>
              </div>

              <div className="flex items-center gap-4 text-sm text-muted-foreground justify-center sm:justify-start">
                <span className="flex items-center gap-1.5">
                  {result.isAI ? (
                    <ShieldAlert className="size-3.5 text-destructive" />
                  ) : (
                    <ShieldCheck className="size-3.5 text-primary" />
                  )}
                  مدل: {result.modelUsed}
                </span>
                <span className="text-border">|</span>
                <span>{result.wordCount} کلمه</span>
                <span className="text-border">|</span>
                <span>{result.textLength} کاراکتر</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

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

function MetricCard({
  detail,
}: {
  detail: DetectionResponse["details"][number];
}) {
  const percentage = Math.round(detail.value * 100);

  const barColor =
    detail.value > 0.7
      ? "bg-destructive"
      : detail.value > 0.4
        ? "bg-chart-3"
        : "bg-primary";

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
