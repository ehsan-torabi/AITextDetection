"use client";

import * as React from "react";
import type { Model } from "@/lib/types";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";

interface ModelSelectorProps {
  models: Model[];
  value: string;
  onValueChange: (value: string) => void;
  disabled?: boolean;
}

export function ModelSelector({
  models,
  value,
  onValueChange,
  disabled,
}: ModelSelectorProps) {
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
          {models.map((model) => (
            <SelectItem key={model.id} value={model.id}>
              <div className="flex items-center gap-2">
                <span className="font-medium">{model.nameFa ?? model.name}</span>
                {model.metrics && (
                  <span className="text-xs text-muted-foreground">
                    (دقت: {Math.round(model.metrics.accuracy * 100)}%)
                  </span>
                )}
              </div>
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
