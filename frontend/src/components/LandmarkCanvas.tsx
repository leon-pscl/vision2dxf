import { useEffect, useRef, useState } from "react";
import { loadImage } from "../api";
import type { LandmarkPoint, LandmarksSource } from "../types";

const COLOUR: Record<LandmarksSource, string> = {
  model: "#1f6feb",
  heuristic: "#f08c00",
  user: "#2b8a3e",
};

/** Landmark names shown as toggles, and the label used when selected. */
export const GROUP_NAMES = [
  "nose",
  "shoulder_left",
  "shoulder_right",
  "elbow_left",
  "elbow_right",
  "wrist_left",
  "wrist_right",
  "neck_left",
  "neck_right",
  "bust",
  "chest",
  "waist",
  "hip",
  "crotch",
  "knee_left",
  "knee_right",
  "ankle_left",
  "ankle_right",
];

const R = 5;

interface Props {
  image: string;
  points: Record<string, LandmarkPoint>;
  onChange: (updates: { name: string; x: number; y: number }[]) => void;
  disabled?: boolean;
}

/**
 * Draggable landmark canvas. Points are coloured by source; a drag flips the
 * point to source="user" and reports the new pixel position upward.
 */
export function LandmarkCanvas({ image, points, onChange, disabled }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const draggingRef = useRef<string | null>(null);
  const sizeRef = useRef({ w: 0, h: 0 });
  const [selected, setSelected] = useState<string | null>("waist");
  const [hovered, setHovered] = useState<string | null>(null);

  // load once, then paint on every change
  useEffect(() => {
    let alive = true;
    loadImage(image)
      .then((img) => {
        if (!alive) return;
        imgRef.current = img;
        sizeRef.current = { w: img.naturalWidth, h: img.naturalHeight };
        const c = canvasRef.current;
        if (c) {
          c.width = img.naturalWidth;
          c.height = img.naturalHeight;
        }
        paint();
      })
      .catch(() => {
        /* the step shows a message; nothing to paint */
      });
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [image]);

  const paint = () => {
    const c = canvasRef.current;
    const img = imgRef.current;
    if (!c || !img) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, c.width, c.height);
    ctx.drawImage(img, 0, 0);

    for (const [name, p] of Object.entries(points)) {
      const active = name === selected || name === hovered;
      const colour = COLOUR[p.source];

      if (active) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, R + 4, 0, Math.PI * 2);
        ctx.strokeStyle = colour;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }

      ctx.beginPath();
      ctx.arc(p.x, p.y, R, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      if (active) {
        ctx.font = "11px ui-sans-serif, system-ui, sans-serif";
        const text = `${name.replace(/_/g, " ")} · ${p.source}`;
        const w = ctx.measureText(text).width;
        const ty = p.y < 20 ? p.y + 18 : p.y - 12;
        ctx.fillStyle = "rgba(255,255,255,0.92)";
        ctx.fillRect(p.x + 9, ty - 9, w + 8, 15);
        ctx.fillStyle = "#17181c";
        ctx.fillText(text, p.x + 13, ty + 2);
      }
    }
  };

  useEffect(paint); // eslint-disable-line react-hooks/exhaustive-deps

  // pointer position in image pixels, whatever the CSS scale is
  const toImage = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current!;
    const rect = c.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * c.width,
      y: ((e.clientY - rect.top) / rect.height) * c.height,
    };
  };

  const hitTest = (x: number, y: number): string | null => {
    let best: string | null = null;
    let bestD = R + 6;
    for (const [name, p] of Object.entries(points)) {
      const d = Math.hypot(p.x - x, p.y - y);
      if (d < bestD) {
        bestD = d;
        best = name;
      }
    }
    return best;
  };

  const onDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (disabled) return;
    const { x, y } = toImage(e);
    const name = hitTest(x, y) ?? (selected && points[selected] ? selected : null);
    if (!name) return;
    draggingRef.current = name;
    setSelected(name);
    e.currentTarget.setPointerCapture(e.pointerId);
    onChange([{ name, x, y }]);
  };

  const onMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const { x, y } = toImage(e);
    if (!draggingRef.current) {
      setHovered(hitTest(x, y));
      return;
    }
    onChange([{ name: draggingRef.current, x, y }]);
  };

  const onUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    draggingRef.current = null;
    if (e.currentTarget.hasPointerCapture(e.pointerId)) {
      e.currentTarget.releasePointerCapture(e.pointerId);
    }
  };

  // click-to-reposition fallback: pick a point, then click anywhere to place it
  const onClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (disabled || !selected) return;
    const rect = canvasRef.current!.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * canvasRef.current!.width;
    const y = ((e.clientY - rect.top) / rect.height) * canvasRef.current!.height;
    onChange([{ name: selected, x, y }]);
  };

  return (
    <div className="pane-main">
      <div className="lm-wrap">
        <canvas
          ref={canvasRef}
          onPointerDown={onDown}
          onPointerMove={onMove}
          onPointerUp={onUp}
          onPointerLeave={() => setHovered(null)}
          onClick={onClick}
          role="img"
          aria-label="Body landmarks. Drag a point, or pick one and click to place it."
        />
      </div>

      <div className="lm-key">
        {(["model", "heuristic", "user"] as LandmarksSource[]).map((s) => (
          <span key={s}>
            <i className={`dot ${s}`} /> {s}
          </span>
        ))}
      </div>

      <div className="field" style={{ width: "100%", maxWidth: 420 }}>
        <label className="label" htmlFor="lm-pick">
          Selected point
        </label>
        <select
          id="lm-pick"
          value={selected ?? ""}
          onChange={(e) => setSelected(e.target.value || null)}
          disabled={disabled}
        >
          <option value="">none</option>
          {GROUP_NAMES.filter((n) => n in points).map((n) => (
            <option key={n} value={n}>
              {n.replace(/_/g, " ")}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}
