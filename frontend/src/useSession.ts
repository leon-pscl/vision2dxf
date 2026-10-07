import { useCallback, useEffect, useRef, useState } from "react";
import { createSession, fetchSession, runStep } from "./api";
import type { SessionState } from "./types";

export interface Session {
  sid: string | null;
  state: SessionState | null;
  busy: number;
  error: string | null;
  results: Record<string, Record<string, unknown>>;
  status: (step: number) => "pending" | "done" | "warning";
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  clearError: () => void;
  reload: () => Promise<void>;
}

export function useSession(): Session {
  const [sid, setSid] = useState<string | null>(null);
  const [state, setState] = useState<SessionState | null>(null);
  const [busy, setBusy] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const inFlight = useRef(0);

  useEffect(() => {
    createSession()
      .then(setSid)
      .catch((e: Error) => setError(e.message));
  }, []);

  const reload = useCallback(async () => {
    if (!sid) return;
    try {
      setState(await fetchSession(sid));
    } catch (e) {
      setError((e as Error).message);
    }
  }, [sid]);

  const run = useCallback(
    async <T,>(step: number, payload: Record<string, unknown> = {}, backend?: string) => {
      if (!sid) throw new Error("no session yet");
      inFlight.current += 1;
      setBusy(inFlight.current);
      try {
        const out = await runStep<T>(sid, step, payload, backend);
        setError(null);
        // refresh the authoritative status from the server
        await reload();
        return out;
      } catch (e) {
        setError((e as Error).message);
        throw e;
      } finally {
        inFlight.current -= 1;
        setBusy(inFlight.current);
      }
    },
    [sid, reload],
  );

  const status = useCallback(
    (step: number) => state?.status[String(step)]?.status ?? "pending",
    [state],
  );

  return {
    sid,
    state,
    busy,
    error,
    results: state?.results ?? {},
    status,
    run,
    clearError: () => setError(null),
    reload,
  };
}
