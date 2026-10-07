import { useCallback, useState } from "react";
import { LandmarkCanvas } from "../components/LandmarkCanvas";
import type { LandmarkPoint, LandmarkResult } from "../types";

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  result: LandmarkResult | null;
  confirmed: boolean;
  front: string;
  onUpdated: (r: LandmarkResult) => void;
}

type Update = { name: string; x: number; y: number };

export function Step05Landmarks({ run, busy, result, confirmed, front, onUpdated }: Props) {
  // optimistic local copy so dragging is smooth, reconciled on each response
  const [points, setPoints] = useState<Record<string, LandmarkPoint> | null>(
    result?.points ?? null,
  );
  const [pending, setPending] = useState<Record<string, Update>>({});

  const shown = points ?? result?.points ?? null;

  const push = useCallback(
    async (updates: Update[], confirm = false) => {
      // confirming with nothing queued still has to reach the server, or the
      // step 6 gate would never open
      if (!updates.length && !confirm) return;
      const merged = { ...pending, ...Object.fromEntries(updates.map((u) => [u.name, u])) };
      setPending(merged);

      // optimistic: mark it user-owned straight away so the colour follows the drag
      setPoints((prev) => {
        const base = prev ?? result?.points;
        if (!base) return prev;
        const next = { ...base };
        for (const u of updates) {
          if (next[u.name]) next[u.name] = { ...next[u.name], x: u.x, y: u.y, source: "user" };
        }
        return next;
      });

      try {
        const r = await run<LandmarkResult>(5, {
          ...(front ? { image: front } : {}),
          updates: Object.values(merged),
          ...(confirm ? { confirm: true } : {}),
        });
        setPoints(r.points);
        setPending({});
        onUpdated(r);
      } catch {
        setPending(merged); // keep the queued updates so the next call retries them
      }
    },
    [pending, result, front, run, onUpdated],
  );

  const counts = Object.values(shown ?? {}).reduce<Record<string, number>>((acc, p) => {
    acc[p.source] = (acc[p.source] ?? 0) + 1;
    return acc;
  }, {});

  return (
    <div className="content">
      <div className="stat-row" style={{ marginBottom: 12 }}>
        <div className="stat">
          <span className="label-muted">Points</span>
          <b>{shown ? Object.keys(shown).length : 0}</b>
        </div>
        {(["model", "heuristic", "user"] as const).map((s) => (
          <div className="stat" key={s}>
            <span className="label-muted">{s}</span>
            <b>{counts[s] ?? 0}</b>
          </div>
        ))}
        <div className="btn-row" style={{ marginLeft: "auto" }}>
          <button
            className="btn"
            disabled={busy || !front}
            onClick={() => run(5, front ? { image: front } : {})}
          >
            {busy ? "Running…" : shown ? "Reset points" : "Detect landmarks"}
          </button>
          <button
            className={confirmed ? "btn btn-ghost" : "btn"}
            disabled={busy || !shown}
            onClick={() => push(Object.values(pending), true)}
          >
            {confirmed ? "Confirmed" : "Confirm landmarks"}
          </button>
        </div>
      </div>

      {shown ? (
        <>
          <p className="hint">
            Drag a point to move it, or pick one below and click to place it. A moved point becomes
            yours. Steps 6 and later unlock once you confirm.
          </p>
          <LandmarkCanvas
            image={front}
            points={shown}
            disabled={busy}
            onChange={(updates) => push(updates)}
          />
        </>
      ) : (
        <p className="notice">
          {front
            ? "Detect the landmarks to place points on the photo."
            : "Upload a front image in step 2 first."}
        </p>
      )}
    </div>
  );
}
