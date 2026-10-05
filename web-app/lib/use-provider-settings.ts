"use client";

import * as React from "react";
import type { TypeSafeProviderSettings } from "@/lib/types";
import {
  DEFAULT_PROVIDER_SETTINGS,
  clearProviderSettings,
  getProviderSnapshot,
  isProviderConfigured,
  saveProviderSettings,
  subscribeProvider,
} from "@/lib/provider-settings";

/** Client-side provider settings backed by localStorage, exposed through
 * useSyncExternalStore so same-tab updates and cross-tab `storage` events
 * both re-render. During hydration the server snapshot (defaults) is used. */
export function useProviderSettings() {
  const settings = React.useSyncExternalStore(
    subscribeProvider,
    getProviderSnapshot,
    () => DEFAULT_PROVIDER_SETTINGS,
  );

  const update = React.useCallback((next: TypeSafeProviderSettings) => {
    saveProviderSettings({
      baseUrl: next.baseUrl.trim() || DEFAULT_PROVIDER_SETTINGS.baseUrl,
      model: next.model.trim() || DEFAULT_PROVIDER_SETTINGS.model,
      apiKey: next.apiKey,
    });
  }, []);

  const clear = React.useCallback(() => {
    clearProviderSettings();
  }, []);

  return {
    settings,
    configured: isProviderConfigured(settings),
    update,
    clear,
  };
}
