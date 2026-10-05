import type { TypeSafeProviderSettings } from "@/lib/types";

export const PROVIDER_SETTINGS_KEY = "typesafe-provider-settings";
export const PROVIDER_CHANGE_EVENT = "typesafe-provider-settings-changed";

export const DEFAULT_PROVIDER_SETTINGS: TypeSafeProviderSettings = {
  baseUrl: "https://api.typesafe.ai",
  model: "jev-latest",
  apiKey: "",
};

export function isProviderConfigured(settings: TypeSafeProviderSettings | null): boolean {
  return Boolean(settings?.apiKey?.trim());
}

function parseSettings(raw: string | null): TypeSafeProviderSettings {
  try {
    if (!raw) return { ...DEFAULT_PROVIDER_SETTINGS };
    const parsed = JSON.parse(raw) as Partial<TypeSafeProviderSettings>;
    return {
      baseUrl: parsed.baseUrl?.trim() || DEFAULT_PROVIDER_SETTINGS.baseUrl,
      model: parsed.model?.trim() || DEFAULT_PROVIDER_SETTINGS.model,
      apiKey: parsed.apiKey ?? "",
    };
  } catch {
    return { ...DEFAULT_PROVIDER_SETTINGS };
  }
}

/** One-off read; outside a browser it returns the defaults. */
export function loadProviderSettings(): TypeSafeProviderSettings {
  if (typeof window === "undefined") return { ...DEFAULT_PROVIDER_SETTINGS };
  try {
    return parseSettings(window.localStorage.getItem(PROVIDER_SETTINGS_KEY));
  } catch {
    return { ...DEFAULT_PROVIDER_SETTINGS };
  }
}

export function saveProviderSettings(settings: TypeSafeProviderSettings): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(PROVIDER_SETTINGS_KEY, JSON.stringify(settings));
  } catch {
    // Private-mode browsers may block storage; settings just won't persist.
  }
  window.dispatchEvent(new Event(PROVIDER_CHANGE_EVENT));
}

export function clearProviderSettings(): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.removeItem(PROVIDER_SETTINGS_KEY);
  } catch {
    // Ignore storage failures the same way as save.
  }
  window.dispatchEvent(new Event(PROVIDER_CHANGE_EVENT));
}

// --- useSyncExternalStore plumbing -----------------------------------------
// getSnapshot must return a stable reference while the stored JSON is
// unchanged, so reads are cached by raw string.

let cachedRaw: string | null | undefined; // undefined = cache not filled yet
let cachedSnapshot: TypeSafeProviderSettings = DEFAULT_PROVIDER_SETTINGS;

export function getProviderSnapshot(): TypeSafeProviderSettings {
  if (typeof window === "undefined") return DEFAULT_PROVIDER_SETTINGS;
  let raw: string | null = null;
  try {
    raw = window.localStorage.getItem(PROVIDER_SETTINGS_KEY);
  } catch {
    raw = null;
  }
  if (raw !== cachedRaw) {
    cachedRaw = raw;
    cachedSnapshot = parseSettings(raw);
  }
  return cachedSnapshot;
}

export function subscribeProvider(onChange: () => void): () => void {
  if (typeof window === "undefined") return () => {};
  window.addEventListener("storage", onChange);
  window.addEventListener(PROVIDER_CHANGE_EVENT, onChange);
  return () => {
    window.removeEventListener("storage", onChange);
    window.removeEventListener(PROVIDER_CHANGE_EVENT, onChange);
  };
}

/** Normalize a base URL for joining with the API path (no trailing slash). */
export function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.trim().replace(/\/+$/, "");
}
