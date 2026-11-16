"use client";

import * as React from "react";
import type { DetectionMode } from "@/lib/types";
import { Cloud, HardDrive, Scale } from "lucide-react";
import { cn } from "@/lib/utils";

const MODES: { value: DetectionMode; label: string; hint?: string; icon: React.ReactNode }[] = [
  {
    value: "local",
    label: "مدل‌های محلی",
    hint: "ضمنی — روی سرور خودتان",
    icon: <HardDrive className="size-4" />,
  },
  {
    value: "online",
    label: "آنلاین (System One)",
    hint: "نیازمند کلید API سرویس TypeSafe",
    icon: <Cloud className="size-4" />,
  },
  { value: "compare", label: "مقایسه هر دو", hint: "هم‌زمان آنلاین و محلی", icon: <Scale className="size-4" /> },
];

interface DetectionModeSelectorProps {
  value: DetectionMode;
  onValueChange: (mode: DetectionMode) => void;
  disabled?: boolean;
}

export function DetectionModeSelector({
  value,
  onValueChange,
  disabled,
}: DetectionModeSelectorProps) {
  return (
    <div className="space-y-2">
      <span className="text-sm font-medium">روش تحلیل</span>
      <div
        role="radiogroup"
        aria-label="روش تحلیل"
        className="grid w-full grid-cols-3 gap-1 rounded-xl bg-muted p-1"
      >
        {MODES.map((mode) => {
          const selected = value === mode.value;
          return (
            <button
              key={mode.value}
              type="button"
              role="radio"
              aria-checked={selected}
              disabled={disabled}
              onClick={() => onValueChange(mode.value)}
              className={cn(
                "flex items-center justify-center gap-1.5 rounded-lg px-2 py-2 text-xs font-medium transition-all sm:text-sm",
                selected
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground",
                disabled && "cursor-not-allowed opacity-50",
              )}
            >
              {mode.icon}
              <div className="flex flex-col items-center justify-center">
                <span className="truncate">{mode.label}</span>
                {mode.hint && (
                  <span className="text-[10px] opacity-70">{mode.hint}</span>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
