"use client";

import * as React from "react";
import type { Model, ModelsResponse } from "@/lib/types";

export function useModels() {
  const [models, setModels] = React.useState<Model[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    let active = true;

    (async () => {
      try {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error("خطا در دریافت لیست مدل‌ها");
        const data: ModelsResponse = await res.json();
        if (active) setModels(data.models ?? []);
      } catch {
        if (active)
          setError("خطا در دریافت لیست مدل‌ها. لطفاً دوباره تلاش کنید.");
      } finally {
        if (active) setLoading(false);
      }
    })();

    return () => {
      active = false;
    };
  }, []);

  const refetch = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/models");
      if (!res.ok) throw new Error("خطا در دریافت لیست مدل‌ها");
      const data: ModelsResponse = await res.json();
      setModels(data.models ?? []);
    } catch {
      setError("خطا در دریافت لیست مدل‌ها. لطفاً دوباره تلاش کنید.");
    } finally {
      setLoading(false);
    }
  }, []);

  return { models, loading, error, refetch };
}
