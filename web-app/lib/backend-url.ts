/** Base URL of the aitextd FastAPI backend. Override with BACKEND_URL env var. */
export const BACKEND_URL = (process.env.BACKEND_URL || "http://0.0.0.0:8000").replace(
  /\/+$/,
  "");

/** Build a full backend URL from a path (e.g. "/detect"). */
export function backendUrl(path: string): string {
  return `${BACKEND_URL}${path.startsWith("/") ? path : "/" + path}`;
}
