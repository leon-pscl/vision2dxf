import type { SegmentationResult } from "../types";

interface Props {
  run: <T>(step: number, payload?: Record<string, unknown>, backend?: string) => Promise<T>;
  busy: boolean;
  result: SegmentationResult | null;
  front: string;
}

export function Step03Segmentation({ run, busy, result, front }: Props) {
  const hasImage = Boolean(front);

  const segment = async (box?: [number, number, number, number]) => {
    if (!front) return;
    await run(3, { image: front, ...(box ? { box } : {}) });
  };

  return (
    <div className="content">
      <div className="stat-row" style={{ marginBottom: 14 }}>
        <div className="stat">
          <span className="label-muted">Model</span>
          <b>{result?.model_name ?? "—"}</b>
        </div>
        <div className="stat">
          <span className="label-muted">Boundary quality</span>
          <b>{result ? result.boundary_quality.toFixed(3) : "—"}</b>
        </div>
        <div className="stat">
          <span className="label-muted">Mask opacity</span>
          <b>50%</b>
        </div>
        <div className="btn-row" style={{ marginLeft: "auto" }}>
          <button className="btn" disabled={!hasImage || busy} onClick={() => segment()}>
            {busy ? "Segmenting…" : "Segment"}
          </button>
          <button
            className="btn btn-ghost"
            disabled={!hasImage || busy}
            onClick={() => segment([0, 0, 100, 100])}
            title="Sends a box prompt, so the prompt path is exercised"
          >
            With box prompt
          </button>
        </div>
      </div>

      {!front && <p className="notice">Upload a front image in step 2 first.</p>}

      {front && (
        <div className="mask-stack">
          <img className="photo-under" src={front} alt="front view" />
          {result && (
            <img
              className="mask-over"
              src={`data:image/png;base64,${result.mask_png}`}
              alt="segmentation mask"
            />
          )}
        </div>
      )}
    </div>
  );
}
