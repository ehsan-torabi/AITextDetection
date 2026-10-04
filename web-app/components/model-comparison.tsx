"use client";

import * as React from "react";
import type { Model } from "@/lib/types";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ArrowUpDown, LineChart } from "lucide-react";

interface ModelComparisonProps {
  models: Model[];
  selectedId?: string;
  onSelect?: (id: string) => void;
}

type SortKey = "accuracy" | "f1Score" | "precision" | "recall";

const sortOptions: { key: SortKey; label: string }[] = [
  { key: "accuracy", label: "دقت" },
  { key: "f1Score", label: "F1" },
  { key: "precision", label: "دقت مثبت" },
  { key: "recall", label: "بازیابی" },
];

export function ModelComparison({ models, selectedId, onSelect }: ModelComparisonProps) {
  const [sortKey, setSortKey] = React.useState<SortKey>("accuracy");

  const ranked = React.useMemo(() => {
    return models
      .filter((m) => m.metrics)
      .sort((a, b) => (b.metrics![sortKey] ?? 0) - (a.metrics![sortKey] ?? 0));
  }, [models, sortKey]);

  if (ranked.length === 0) return null;

  const maxValue = Math.max(...ranked.map((m) => m.metrics![sortKey]));

  return (
    <Card size="sm">
      <CardHeader>
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2">
            <LineChart className="size-4 text-muted-foreground" />
            مقایسه مدل‌ها
          </CardTitle>
        </div>
        <CardDescription className="text-xs leading-relaxed">
          عملکرد همه مدل‌های آموزش‌دیده روی داده‌های آزمون خودشان.
        </CardDescription>
        <div className="flex flex-wrap items-center gap-1.5 pt-1">
          <ArrowUpDown className="size-3 text-muted-foreground" />
          {sortOptions.map((opt) => (
            <button
              key={opt.key}
              type="button"
              onClick={() => setSortKey(opt.key)}
              className={`rounded-full px-2 py-0.5 text-[11px] transition-colors ${
                sortKey === opt.key
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:bg-muted/70"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </CardHeader>
      <CardContent className="space-y-2">
        {ranked.map((model, i) => {
          const value = model.metrics![sortKey];
          const percentage = Math.round(value * 100);
          const widthPct = maxValue > 0 ? (value / maxValue) * 100 : 0;
          const isSelected = model.id === selectedId;

          return (
            <button
              key={model.id}
              type="button"
              onClick={() => onSelect?.(model.id)}
              className={`w-full space-y-1 rounded-lg p-1.5 text-right transition-colors ${
                onSelect ? "cursor-pointer hover:bg-muted/50" : "cursor-default"
              } ${isSelected ? "bg-muted/70 ring-1 ring-primary/30" : ""}`}
            >
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="flex items-center gap-1.5 font-medium">
                  <span className="text-muted-foreground tabular-nums">{i + 1}.</span>
                  {model.nameFa ?? model.name}
                  {model.type && (
                    <Badge
                      variant={model.type === "dl" ? "default" : "outline"}
                      className="h-4 px-1.5 text-[10px]"
                    >
                      {model.type === "dl" ? "DL" : "ML"}
                    </Badge>
                  )}
                </span>
                <span className="font-bold tabular-nums">{percentage}٪</span>
              </div>
              <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-700 ease-out ${
                    model.type === "dl" ? "bg-chart-3" : "bg-primary"
                  }`}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
            </button>
          );
        })}
      </CardContent>
    </Card>
  );
}
