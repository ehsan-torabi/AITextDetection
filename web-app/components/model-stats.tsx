"use client";

import * as React from "react";
import type { Model, ModelMetrics } from "@/lib/types";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  BrainCircuit,
  Database,
  Gauge,
  Grid3x3,
  Info,
  Target,
} from "lucide-react";

function formatNumber(n: number): string {
  return new Intl.NumberFormat("fa-IR").format(n);
}

const metricRows: { key: keyof ModelMetrics; label: string }[] = [
  { key: "accuracy", label: "دقت (Accuracy)" },
  { key: "precision", label: "دقت مثبت (Precision)" },
  { key: "recall", label: "بازیابی (Recall)" },
  { key: "f1Score", label: "امتیاز F1" },
];

export function ModelStats({ model }: { model: Model | undefined }) {
  if (!model) return null;

  return (
    <Card size="sm">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2">
            <BrainCircuit className="size-4 text-muted-foreground" />
            مشخصات مدل: {model.nameFa ?? model.name}
          </CardTitle>
          <Badge variant="secondary">نسخه {model.version}</Badge>
        </div>
        <CardDescription className="text-xs leading-relaxed">
          {model.descriptionFa ?? model.description}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Performance Metrics */}
        {model.metrics && (
          <section className="space-y-2.5">
            <h4 className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
              <Gauge className="size-3.5" />
              عملکرد مدل روی داده‌های آزمون
            </h4>
            <div className="grid gap-x-6 gap-y-2 sm:grid-cols-2">
              {metricRows.map(({ key, label }) => (
                <MetricBar key={key} label={label} value={model.metrics![key]} />
              ))}
            </div>
          </section>
        )}

        {/* Training Info */}
        {model.training && (
          <section className="space-y-2">
            <h4 className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
              <Database className="size-3.5" />
              اطلاعات آموزش
            </h4>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs sm:grid-cols-3">
              <InfoItem
                icon={<Target className="size-3 text-muted-foreground" />}
                label="الگوریتم"
                value={model.algorithm}
              />
              <InfoItem
                label="کل نمونه‌ها"
                value={formatNumber(model.training.totalSamples)}
              />
              <InfoItem
                label="نمونه‌های آموزش"
                value={formatNumber(model.training.trainSamples)}
              />
              <InfoItem
                label="نمونه‌های آزمون"
                value={formatNumber(model.training.testSamples)}
              />
              <InfoItem
                label="ویژگی‌های TF-IDF"
                value={formatNumber(model.training.featureCount)}
              />
              <InfoItem
                label="نسبت داده آزمون"
                value={`${Math.round(model.training.testSplitRatio * 100)}%`}
              />
            </dl>
          </section>
        )}

        {/* Confusion Matrix */}
        {model.confusionMatrix && (
          <section className="space-y-2">
            <h4 className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
              <Grid3x3 className="size-3.5" />
              ماتریس درهم‌ریختگی (داده‌های آزمون)
            </h4>
            <ConfusionMatrixTable cm={model.confusionMatrix} />
          </section>
        )}

        <p className="flex items-start gap-1.5 text-[11px] leading-relaxed text-muted-foreground border-t border-border pt-2">
          <Info className="size-3 mt-0.5 flex-shrink-0" />
          معیارها از ارزیابی مدل روی {model.training
            ? formatNumber(model.training.testSamples)
            : ""}{" "}
          نمونه دیده‌نشده محاسبه شده‌اند.
        </p>
      </CardContent>
    </Card>
  );
}

function MetricBar({ label, value }: { label: string; value: number }) {
  const percentage = Math.round(value * 100);
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-bold tabular-nums">{percentage.toLocaleString("fa-IR")}٪</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full bg-primary transition-all duration-700 ease-out"
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function InfoItem({
  icon,
  label,
  value,
}: {
  icon?: React.ReactNode;
  label: string;
  value: string;
}) {
  return (
    <div className="flex flex-col gap-0.5 rounded-lg bg-muted/50 px-2.5 py-1.5">
      <dt className="flex items-center gap-1 text-[11px] text-muted-foreground">
        {icon}
        {label}
      </dt>
      <dd className="font-medium tabular-nums">{value}</dd>
    </div>
  );
}

function ConfusionMatrixTable({
  cm,
}: {
  cm: NonNullable<Model["confusionMatrix"]>;
}) {
  const total =
    cm.trueNegative + cm.falsePositive + cm.falseNegative + cm.truePositive;

  const cells = [
    { label: "تشخیص صحیح انسان (TN)", value: cm.trueNegative, good: true },
    { label: "اشتباه: انسان → هوش مصنوعی (FP)", value: cm.falsePositive, good: false },
    { label: "اشتباه: هوش مصنوعی → انسان (FN)", value: cm.falseNegative, good: false },
    { label: "تشخیص صحیح هوش مصنوعی (TP)", value: cm.truePositive, good: true },
  ];

  return (
    <div className="grid grid-cols-2 gap-2">
      {cells.map((cell) => (
        <div
          key={cell.label}
          className={`rounded-lg px-2.5 py-1.5 ring-1 ${
            cell.good
              ? "bg-primary/5 ring-primary/20"
              : "bg-destructive/5 ring-destructive/20"
          }`}
        >
          <div
            className={`text-base font-bold tabular-nums ${
              cell.good ? "text-primary" : "text-destructive"
            }`}
          >
            {formatNumber(cell.value)}
          </div>
          <div className="text-[11px] leading-tight text-muted-foreground">
            {cell.label}{" "}
            <span className="opacity-60">
              ({Math.round((cell.value / total) * 100)}٪)
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
