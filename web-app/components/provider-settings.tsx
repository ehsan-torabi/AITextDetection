"use client";

import * as React from "react";
import type { TypeSafeProviderSettings } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";
import {
  Check,
  ChevronDown,
  Cloud,
  ExternalLink,
  KeyRound,
  RotateCcw,
  Settings2,
} from "lucide-react";

interface ProviderSettingsProps {
  settings: TypeSafeProviderSettings;
  configured: boolean;
  onSave: (next: TypeSafeProviderSettings) => void;
  onClear: () => void;
  disabled?: boolean;
  /** Controlled open state; omit to let the panel manage itself. */
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function ProviderSettings({
  settings,
  configured,
  onSave,
  onClear,
  disabled,
  open: openProp,
  onOpenChange,
}: ProviderSettingsProps) {
  const [openState, setOpenState] = React.useState(false);
  const open = openProp ?? openState;
  const setOpen = (v: boolean) => {
    setOpenState(v);
    onOpenChange?.(v);
  };
  const [draft, setDraft] = React.useState(settings);
  const [saved, setSaved] = React.useState(false);

  // Re-sync the draft when saved settings change externally (save/clear),
  // using the render-time adjustment pattern instead of an effect.
  const [lastSynced, setLastSynced] = React.useState(settings);
  if (lastSynced !== settings) {
    setLastSynced(settings);
    setDraft(settings);
  }

  // This panel renders inside the main detection <form>, so it must not nest
  // another <form>; saving is button-driven and Enter in the inputs is handled
  // explicitly so it neither submits nor triggers the outer form.
  function handleSave(e?: React.SyntheticEvent) {
    e?.preventDefault();
    onSave(draft);
    setSaved(true);
    window.setTimeout(() => setSaved(false), 2000);
  }

  function handleInputKeyDown(e: React.KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      e.preventDefault();
      handleSave();
    }
  }

  return (
    <div className="space-y-2">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="flex w-full items-center gap-2 text-xs text-muted-foreground transition-colors hover:text-foreground"
        disabled={disabled}
      >
        <ChevronDown className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`} />
        <Cloud className="size-3.5" />
        سرویس آنلاین System One
        {configured ? (
          <Badge variant="default" className="h-4 px-1.5 text-[10px]">
            پیکربندی شده
          </Badge>
        ) : (
          <Badge variant="outline" className="h-4 px-1.5 text-[10px]">
            بدون کلید API
          </Badge>
        )}
      </button>

      {open && (
        <Card size="sm">
          <CardContent>
            <div className="space-y-4" dir="ltr">
              <div className="space-y-2">
                <Label htmlFor="provider-api-key" className="flex items-center gap-1.5">
                  <KeyRound className="size-3.5 text-muted-foreground" />
                  API Key
                </Label>
                <Input
                  id="provider-api-key"
                  type="password"
                  autoComplete="off"
                  placeholder="sk-..."
                  value={draft.apiKey}
                  onChange={(e) => setDraft({ ...draft, apiKey: e.target.value })}
                  onKeyDown={handleInputKeyDown}
                  disabled={disabled}
                />
                <p className="text-[11px] leading-relaxed text-muted-foreground" dir="rtl">
                  کلید فقط در مرورگر شما (localStorage) ذخیره می‌شود و تنها هنگام تحلیل، همراه همان
                  درخواست به سرویس انتخابی ارسال می‌گردد.
                </p>
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="provider-base-url" className="flex items-center gap-1.5">
                    <Settings2 className="size-3.5 text-muted-foreground" />
                    Base URL
                  </Label>
                  <Input
                    id="provider-base-url"
                    type="url"
                    placeholder="https://api.typesafe.ai"
                    value={draft.baseUrl}
                    onChange={(e) => setDraft({ ...draft, baseUrl: e.target.value })}
                    onKeyDown={handleInputKeyDown}
                    disabled={disabled}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="provider-model">Model</Label>
                  <Input
                    id="provider-model"
                    type="text"
                    placeholder="jev-latest"
                    value={draft.model}
                    onChange={(e) => setDraft({ ...draft, model: e.target.value })}
                    onKeyDown={handleInputKeyDown}
                    disabled={disabled}
                  />
                </div>
              </div>

              <div className="flex items-center gap-2" dir="rtl">
                <Button
                  type="button"
                  size="sm"
                  className="rounded-lg"
                  onClick={() => handleSave()}
                  disabled={disabled}
                >
                  {saved ? (
                    <>
                      <Check className="size-3.5" />
                      ذخیره شد
                    </>
                  ) : (
                    "ذخیره تنظیمات"
                  )}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="rounded-lg"
                  onClick={onClear}
                  disabled={disabled}
                >
                  <RotateCcw className="size-3.5" />
                  بازنشانی
                </Button>
                <a
                  href="https://docs.typesafe.ai"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mr-auto flex items-center gap-1 text-xs text-muted-foreground underline-offset-2 hover:text-foreground hover:underline"
                >
                  مستندات TypeSafe
                  <ExternalLink className="size-3" />
                </a>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
