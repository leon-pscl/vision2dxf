import { useMemo } from "react";
import { sanitiseSvg } from "../svg";
import type { PatternPiece } from "../types";

interface Props {
  pieces: PatternPiece[];
  showNotches?: boolean;
}

export function PatternGrid({ pieces, showNotches = false }: Props) {
  const clean = useMemo(() => pieces.map((p) => ({ ...p, svg: sanitiseSvg(p.svg) })), [pieces]);

  if (clean.length === 0) return <p className="notice">No pieces yet.</p>;

  return (
    <div className="pieces">
      {clean.map((p) => (
        <div className="piece" key={p.name}>
          <h4>{p.name}</h4>
          <div className="svg-holder" dangerouslySetInnerHTML={{ __html: p.svg }} />
          {(p.seam_allowance_cm !== null || showNotches) && (
            <div className="piece-meta">
              {p.seam_allowance_cm !== null && <span>SA {p.seam_allowance_cm} cm</span>}
              {showNotches && <span> · {p.notches.length} notches</span>}
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
