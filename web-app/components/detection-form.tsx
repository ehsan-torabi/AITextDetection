"use client";

import * as React from "react";
import type { DetectionResponse } from "@/lib/types";
import { useModels } from "@/lib/use-models";
import { Textarea } from "@/components/ui/textarea";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { ModelSelector } from "@/components/model-selector";
import { ModelStats } from "@/components/model-stats";
import { ModelComparison } from "@/components/model-comparison";
import { DetectionResult } from "@/components/detection-result";
import { Alert, AlertTitle, AlertDescription } from "@/components/ui/alert";
import { Send, Loader2, AlertCircle, RotateCcw, FileText, ChevronDown } from "lucide-react";

const MIN_TEXT_LENGTH = 20;

export function DetectionForm() {
  const [text, setText] = React.useState("");
  const [modelId, setModelId] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);
  const [result, setResult] = React.useState<DetectionResponse | null>(null);
  const [showComparison, setShowComparison] = React.useState(false);
  const { models, loading: modelsLoading, error: modelsError, refetch } = useModels();

  const effectiveModelId = modelId || models[0]?.id || "";
  const selectedModel = models.find((m) => m.id === effectiveModelId);

  const wordCount = React.useMemo(() => {
    const trimmed = text.trim();
    if (!trimmed) return 0;
    return trimmed.split(/\s+/).length;
  }, [text]);

  const charCount = text.length;
  const isValid = charCount >= MIN_TEXT_LENGTH && effectiveModelId !== "";

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!isValid) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, modelId: effectiveModelId }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || data.error || "خطا در ارسال درخواست");
      }

      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "خطایی رخ داد. لطفاً دوباره تلاش کنید.");
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setText("");
    setResult(null);
    setError(null);
  }

  return (
    <div className="space-y-8">
      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Textarea */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="detection-textarea" className="flex items-center gap-2">
              <FileText className="size-4 text-muted-foreground" />
              متن مورد بررسی
            </Label>
            <div className="flex items-center gap-3 text-xs text-muted-foreground tabular-nums">
              <span>{wordCount} کلمه</span>
              <span className="text-border">|</span>
              <span
                className={charCount > 0 && charCount < MIN_TEXT_LENGTH ? "text-destructive" : ""}
              >
                {charCount} کاراکتر
              </span>
            </div>
          </div>
          <Textarea
            id="detection-textarea"
            placeholder="متن خود را اینجا وارد کنید..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            className="min-h-[200px] resize-y text-base leading-8"
            dir="rtl"
            disabled={loading}
          />
          {charCount > 0 && charCount < MIN_TEXT_LENGTH && (
            <p className="text-xs text-destructive">
              حداقل {MIN_TEXT_LENGTH} کاراکتر وارد کنید ({MIN_TEXT_LENGTH - charCount} کاراکتر دیگر)
            </p>
          )}
        </div>

        {/* Model Selector */}
        {modelsError ? (
          <div className="space-y-2">
            <Label>مدل تشخیص</Label>
            <div className="flex items-center gap-3 rounded-xl border border-destructive/30 bg-destructive/5 px-3 py-2.5 text-sm text-destructive">
              <span>{modelsError}</span>
              <button
                type="button"
                onClick={refetch}
                className="mr-auto text-xs underline underline-offset-2 hover:text-destructive/80 transition-colors"
              >
                تلاش مجدد
              </button>
            </div>
          </div>
        ) : (
          <>
            <ModelSelector
              models={models}
              value={effectiveModelId}
              onValueChange={setModelId}
              disabled={loading || modelsLoading}
            />
            {selectedModel && !loading && <ModelStats model={selectedModel} />}

            {models.length > 1 && !loading && (
              <div className="space-y-2">
                <button
                  type="button"
                  onClick={() => setShowComparison((v) => !v)}
                  className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                >
                  <ChevronDown
                    className={`size-3.5 transition-transform ${
                      showComparison ? "rotate-180" : ""
                    }`}
                  />
                  مقایسه همه مدل‌ها ({models.length})
                </button>
                {showComparison && (
                  <ModelComparison
                    models={models}
                    selectedId={effectiveModelId}
                    onSelect={setModelId}
                  />
                )}
              </div>
            )}
          </>
        )}

        {/* Actions */}
        <div className="flex items-center gap-3 pt-1">
          <Button
            type="submit"
            disabled={!isValid || loading || modelsLoading}
            size="lg"
            className="flex-1 sm:flex-none sm:min-w-[160px] rounded-xl h-10 active:scale-[0.98] transition-all"
          >
            {loading ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                در حال تحلیل...
              </>
            ) : (
              <>
                <Send className="size-4" />
                تحلیل متن
              </>
            )}
          </Button>

          {(text.length > 0 || result) && (
            <Button
              type="button"
              variant="outline"
              size="lg"
              onClick={handleReset}
              disabled={loading}
              className="rounded-xl h-10 active:scale-[0.98] transition-all"
            >
              <RotateCcw className="size-4" />
              پاک کردن
            </Button>
          )}
        </div>
      </form>

      {/* Error */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="size-4" />
          <AlertTitle>خطا</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Loading */}
      {loading && (
        <div className="space-y-4">
          <div className="rounded-xl border border-border bg-card p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="size-10 rounded-xl bg-muted animate-pulse" />
              <div className="flex-1 space-y-2">
                <div className="h-4 w-2/3 rounded-lg bg-muted animate-pulse" />
                <div className="h-3 w-1/2 rounded-lg bg-muted animate-pulse" />
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {Array.from({ length: 4 }).map((_, i) => (
                <div key={i} className="h-20 rounded-xl bg-muted animate-pulse" />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Results */}
      {result && !loading && <DetectionResult result={result} />}
    </div>
  );
}
