"use client";

import * as React from "react";
import type {
  DetectionResponse,
  OnlineLocalComparison,
  TypeSafeDetectionResponse,
} from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Bot, CheckCircle2, Cloud, HardDrive, ListChecks, TriangleAlert, User } from "lucide-react";

interface OnlineComparisonProps {
  comparison: OnlineLocalComparison;
}

/** AI probability from a choice answer, located by candidate id rather than
 * position (both local and online use ["human_written", "ai_generated"]). */
function aiProbability(typedAnswers: NonNullable<DetectionResponse["typedAnswers"]>): number {
  const idx = typedAnswers.choice.candidates.indexOf("ai_generated");
  return typedAnswers.choice.probabilities[idx] ?? 0;
}

/** Side-by-side matrix comparing one text analyzed by the online System One
 * model and a local model: the two engines form columns, each metric a row,
 * so verdicts and typed answers align for direct comparison. */
export function OnlineComparison({ comparison }: OnlineComparisonProps) {
  const { local, online, localError, onlineError } = comparison;

  const agreement: "agree" | "disagree" | "unknown" =
    local && online ? (local.isAI === online.isAI ? "agree" : "disagree") : "unknown";

  const localTyped = local?.typedAnswers ?? null;
  const onlineTyped = online?.typedAnswers ?? null;
  const maxLevel = onlineTyped?.score.levels.length ?? localTyped?.score.levels.length ?? 5;

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <ListChecks className="size-4 text-muted-foreground" />
            مقایسه نتایج آنلاین و محلی
          </CardTitle>
          {agreement === "agree" && (
            <Badge variant="default" className="gap-1">
              <CheckCircle2 className="size-3" />
              نتیجه یکسان
            </Badge>
          )}
          {agreement === "disagree" && (
            <Badge variant="destructive" className="gap-1">
              <TriangleAlert className="size-3" />
              مغایرت در نتیجه
            </Badge>
          )}
        </div>
        <CardDescription className="text-xs leading-relaxed">
          همان متن توسط مدل آنلاین System One و مدل محلی انتخابی تحلیل شده است؛ معیارها در ستون‌های
          کنار هم برای مقایسه مستقیم ردیف شده‌اند.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {(localError || onlineError) && (
          <Alert variant={local && online ? "default" : "destructive"}>
            <TriangleAlert className="size-4" />
            <AlertTitle>{local && online ? "خطای جزئی" : "تحلیل ناموفق"}</AlertTitle>
            <AlertDescription>
              {onlineError && <p>سرویس آنلاین: {onlineError}</p>}
              {localError && <p>مدل محلی: {localError}</p>}
            </AlertDescription>
          </Alert>
        )}

        {/* Side-by-side matrix: metric labels on the right, engines as columns */}
        <div dir="rtl" className="overflow-hidden rounded-xl border border-border">
          {/* Column headers */}
          <div className="grid grid-cols-[0.8fr_1fr_1fr] gap-3 border-b bg-muted/50 px-3 py-2">
            <span />
            <SideHeader
              icon={<Cloud className="size-3.5" />}
              title="آنلاین — System One"
              sub={online ? (online.resolvedModel ?? online.requestedModel) : undefined}
            />
            <SideHeader
              icon={<HardDrive className="size-3.5" />}
              title="محلی"
              sub={local?.modelUsed}
            />
          </div>

          {/* Verdict row */}
          <MetricRow label="نتیجه">
            <VerdictCell result={online} error={onlineError} />
            <VerdictCell result={local} error={localError} />
          </MetricRow>

          {/* AI probability row */}
          <MetricRow label="احتمال هوش‌مصنوعی‌بودن">
            <ProbCell
              value={onlineTyped ? aiProbability(onlineTyped) : null}
              highlight={online?.isAI}
            />
            <ProbCell
              value={localTyped ? aiProbability(localTyped) : null}
              highlight={local?.isAI}
            />
          </MetricRow>

          {/* Confidence row */}
          <MetricRow label="اطمینان">
            <PercentCell value={online?.confidence ?? null} />
            <PercentCell value={local?.confidence ?? null} />
          </MetricRow>

          {/* Machine-likeness score row */}
          <MetricRow label="ماشین‌مانندبودن (امتیاز)">
            <ValueCell
              value={onlineTyped ? `${onlineTyped.score.value.toFixed(2)} / ${maxLevel - 1}` : null}
            />
            <ValueCell
              value={localTyped ? `${localTyped.score.value.toFixed(2)} / ${maxLevel - 1}` : null}
            />
          </MetricRow>

          {/* Human-review probability row */}
          <MetricRow label="احتمال نیاز به بازبینی">
            <ProbCell value={onlineTyped ? onlineTyped.noul.probability : null} tone="chart" />
            <ProbCell value={localTyped ? localTyped.noul.probability : null} tone="chart" />
          </MetricRow>
        </div>

        {online?.usage && (
          <p className="text-[10px] tabular-nums text-muted-foreground" dir="ltr">
            System One: {online.usage.inputTokens} in / {online.usage.outputTokens} out tokens ·{" "}
            {(online.latencyMs / 1000).toFixed(1)}s
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function SideHeader({ icon, title, sub }: { icon: React.ReactNode; title: string; sub?: string }) {
  return (
    <div className="flex min-w-0 items-center gap-1.5">
      <span className="flex items-center gap-1.5 text-xs font-semibold">
        {icon}
        {title}
      </span>
      {sub && (
        <Badge variant="outline" className="h-4 shrink-0 px-1.5 text-[10px]" dir="ltr">
          {sub}
        </Badge>
      )}
    </div>
  );
}

/** One matrix row: metric label (right column) + one cell per engine. */
function MetricRow({
  label,
  children,
}: {
  label: string;
  children: [React.ReactNode, React.ReactNode];
}) {
  const [onlineCell, localCell] = children;
  return (
    <div className="grid grid-cols-[0.8fr_1fr_1fr] items-center gap-3 border-b border-border px-3 py-2.5 last:border-b-0">
      <span className="text-[11px] leading-tight text-muted-foreground">{label}</span>
      {onlineCell}
      {localCell}
    </div>
  );
}

function VerdictCell({
  result,
  error,
}: {
  result?: DetectionResponse | TypeSafeDetectionResponse | null;
  error?: string;
}) {
  if (!result) {
    return <span className="text-[11px] text-destructive">{error ?? "—"}</span>;
  }
  return (
    <span className="flex items-center gap-1.5">
      {result.isAI ? (
        <span className="flex items-center justify-center size-6 shrink-0 rounded-md bg-destructive/10 text-destructive">
          <Bot className="size-3.5" />
        </span>
      ) : (
        <span className="flex items-center justify-center size-6 shrink-0 rounded-md bg-primary/10 text-primary">
          <User className="size-3.5" />
        </span>
      )}
      <span className="text-xs font-bold leading-tight">{result.verdictFa}</span>
    </span>
  );
}

function ProbCell({
  value,
  highlight,
  tone = "auto",
}: {
  value: number | null;
  highlight?: boolean;
  tone?: "auto" | "chart";
}) {
  if (value == null) return <span className="text-[11px] text-muted-foreground">—</span>;
  const percentage = Math.round(value * 100);
  const barTone = tone === "chart" ? "bg-chart-3" : highlight ? "bg-destructive" : "bg-primary";
  return (
    <span className="flex items-center gap-2">
      <span className="h-1.5 flex-1 rounded-full bg-muted overflow-hidden">
        <span
          className={`block h-full rounded-full transition-all duration-700 ease-out ${barTone}`}
          style={{ width: `${percentage}%` }}
        />
      </span>
      <span
        className={`w-10 shrink-0 text-left text-[11px] font-bold tabular-nums ${
          highlight ? "text-destructive" : ""
        }`}
      >
        {percentage}٪
      </span>
    </span>
  );
}

function PercentCell({ value }: { value: number | null }) {
  if (value == null) return <span className="text-[11px] text-muted-foreground">—</span>;
  return <span className="text-[11px] font-bold tabular-nums">{Math.round(value * 100)}٪</span>;
}

function ValueCell({ value }: { value: string | null }) {
  if (value == null) return <span className="text-[11px] text-muted-foreground">—</span>;
  return (
    <span className="text-[11px] font-bold tabular-nums" dir="ltr">
      {value}
    </span>
  );
}
