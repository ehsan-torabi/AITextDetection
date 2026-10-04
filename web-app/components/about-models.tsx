"use client";

import { useModels } from "@/lib/use-models";
import { ModelComparison } from "@/components/model-comparison";
import { ModelStats } from "@/components/model-stats";
import { Loader2 } from "lucide-react";

export function AboutModels() {
  const { models, loading, error } = useModels();

  const realModels = models.filter((m) => m.type !== "ensemble");
  const classic = realModels.filter((m) => m.type === "ml");
  const deep = realModels.filter((m) => m.type === "dl");

  if (loading) {
    return (
      <div className="flex items-center gap-2 text-sm text-muted-foreground py-6">
        <Loader2 className="size-4 animate-spin" />
        در حال دریافت اطلاعات زنده مدل‌ها از سرور...
      </div>
    );
  }

  if (error || realModels.length === 0) {
    return (
      <p className="text-sm text-muted-foreground py-6">
        اطلاعات زنده مدل‌ها اکنون در دسترس نیست؛ برای مشاهدهٔ معیارها، سرور بک‌اند (
        <code className="text-xs">server.py</code>) باید در حال اجرا باشد.
      </p>
    );
  }

  return (
    <div className="space-y-6">
      <ModelComparison models={realModels} />

      {classic.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-muted-foreground">
            مدل‌های یادگیری ماشین کلاسیک ({classic.length})
          </h3>
          <div className="grid gap-4 sm:grid-cols-2">
            {classic.map((m) => (
              <ModelStats key={m.id} model={m} />
            ))}
          </div>
        </div>
      )}

      {deep.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-muted-foreground">
            مدل‌های یادگیری عمیق ({deep.length})
          </h3>
          <div className="grid gap-4 sm:grid-cols-2">
            {deep.map((m) => (
              <ModelStats key={m.id} model={m} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
