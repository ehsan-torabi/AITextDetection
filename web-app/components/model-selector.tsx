"use client";

import * as React from "react";
import type { Model } from "@/lib/types";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";

interface ModelSelectorProps {
  models: Model[];
  value: string;
  onValueChange: (value: string) => void;
  disabled?: boolean;
}

function TypeBadge({ type }: { type?: Model["type"] }) {
  if (!type) return null;
  const labels: Record<NonNullable<Model["type"]>, string> = {
    ensemble: "ترکیبی",
    dl: "یادگیری عمیق",
    ml: "یادگیری ماشین",
  };
  return (
    <Badge
      variant={type === "ensemble" ? "outline" : type === "dl" ? "default" : "secondary"}
      className="h-4 px-1.5 text-[10px]"
    >
      {labels[type]}
    </Badge>
  );
}

export function ModelSelector({
  models,
  value,
  onValueChange,
  disabled,
}: ModelSelectorProps) {
  const ensembleModel = models.find((m) => m.type === "ensemble");
  const mlModels = models.filter((m) => m.type !== "dl" && m.type !== "ensemble");
  const dlModels = models.filter((m) => m.type === "dl");

  return (
    <div className="space-y-2">
      <Label htmlFor="model-selector">مدل تشخیص</Label>
      <Select
        value={value}
        onValueChange={(v) => onValueChange(v ?? "")}
        disabled={disabled}
      >
        <SelectTrigger className="w-full" id="model-selector">
          <SelectValue placeholder="انتخاب مدل..." />
        </SelectTrigger>
        <SelectContent>
          {ensembleModel && (
            <>
              <SelectGroup>
                <SelectLabel>پیشنهادی</SelectLabel>
                <ModelOption model={ensembleModel} />
              </SelectGroup>
              <SelectSeparator />
            </>
          )}
          {mlModels.length > 0 && (
            <SelectGroup>
              <SelectLabel>یادگیری ماشین کلاسیک</SelectLabel>
              {mlModels.map((model) => (
                <ModelOption key={model.id} model={model} />
              ))}
            </SelectGroup>
          )}
          {mlModels.length > 0 && dlModels.length > 0 && <SelectSeparator />}
          {dlModels.length > 0 && (
            <SelectGroup>
              <SelectLabel>یادگیری عمیق</SelectLabel>
              {dlModels.map((model) => (
                <ModelOption key={model.id} model={model} />
              ))}
            </SelectGroup>
          )}
        </SelectContent>
      </Select>
    </div>
  );
}

function ModelOption({ model }: { model: Model }) {
  return (
    <SelectItem value={model.id}>
      <div className="flex items-center gap-2">
        <span className="font-medium">{model.nameFa ?? model.name}</span>
        {model.metrics && (
          <span className="text-xs text-muted-foreground">
            (دقت: {Math.round(model.metrics.accuracy * 100)}%)
          </span>
        )}
        <TypeBadge type={model.type} />
      </div>
    </SelectItem>
  );
}
