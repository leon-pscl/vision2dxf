import type { SessionState } from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      /* keep the status line */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function createSession(): Promise<string> {
  const r = await fetch(`${BASE}/api/session`, { method: "POST" });
  const { session_id } = await json<{ session_id: string }>(r);
  return session_id;
}

export async function fetchSession(sid: string): Promise<SessionState> {
  return json<SessionState>(await fetch(`${BASE}/api/session/${sid}`));
}

export async function runStep<T>(
  sid: string,
  step: number,
  payload: Record<string, unknown> = {},
  backend?: string,
): Promise<T> {
  const body: Record<string, unknown> = { payload };
  if (backend) body.backend = backend;
  const r = await fetch(`${BASE}/api/session/${sid}/step/${step}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return json<T>(r);
}

export async function fetchModels(): Promise<{ adapters: Record<string, string>; device: string }> {
  return json(await fetch(`${BASE}/api/models`));
}

export function exportYamlUrl(sid: string): string {
  return `${BASE}/api/session/${sid}/export.yaml`;
}

/** Read a File or Blob as a base64 data URL. The image stays in memory. */
export function readFileAsDataUrl(file: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(String(fr.result));
    fr.onerror = () => reject(fr.error ?? new Error("could not read the file"));
    fr.readAsDataURL(file);
  });
}

export function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error("could not decode the image"));
    img.src = src;
  });
}
