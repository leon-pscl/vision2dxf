/**
 * Live smoke test against a running backend. Skipped when nothing is listening,
 * so `npm test` is safe to run anywhere.
 *
 *   1. backend/ : uvicorn main:app --port 8000
 *   2. frontend/: VITE_API_BASE=http://127.0.0.1:8000 npx vitest run
 */
import { beforeAll, describe, expect, it } from "vitest";
import { createSession, exportYamlUrl, fetchModels, fetchSession, runStep } from "./api";

let sid = "";
let front = "";
let side = "";

// Flat PNGs, embedded as literals so no image fixture is written to disk.
const FLAT_PNG_200x320 =
  "iVBORw0KGgoAAAANSUhEUgAAAMgAAAFACAIAAAB2rqTIAAADQklEQVR4nO3SQQkAIADAQPsnsYEZbGUJhyAHF2CPjb0mXDeeF/AlY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBYJY5EwFgljkTAWCWORMBaJA0LoL5beKrtFAAAAAElFTkSuQmCC";

// Same image at side-view proportions.
const FLAT_PNG_160x320 = (() => {
  const b = atob(FLAT_PNG_200x320);
  const u = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
  // patch the IHDR width: bytes 16..19
  const dv = new DataView(u.buffer);
  dv.setUint32(16, 160);
  // fix the IHDR CRC: tag "IHDR" at 12..15, crc at 29..32
  let crc = 0xffffffff;
  for (let i = 12; i <= 28; i++) {
    crc ^= u[i];
    for (let k = 0; k < 8; k++) crc = crc & 1 ? (crc >>> 1) ^ 0xedb88320 : crc >>> 1;
  }
  dv.setUint32(29, (crc ^ 0xffffffff) >>> 0);
  let s = "";
  for (let i = 0; i < u.length; i++) s += String.fromCharCode(u[i]);
  return btoa(s);
})();

async function live(): Promise<boolean> {
  try {
    await fetchModels();
    return true;
  } catch {
    return false;
  }
}

beforeAll(async () => {
  if (!(await live())) return;

  sid = await createSession();

  front = FLAT_PNG_200x320;
  side = FLAT_PNG_160x320;
});

describe("phase 1 pipeline over the wire", () => {
  it("the backend is reachable", async () => {
    if (!(await live())) return;
    const m = await fetchModels();
    expect(m.adapters.segmentation).toContain("adapters.");
    expect(["cpu", "cuda"]).toContain(m.device);
  });

  it("clicks through all ten steps", async () => {
    if (!(await live())) return;

    const spec = await runStep<Record<string, unknown>>(sid, 1, { garment_type: "polo_shirt" });
    expect(spec.is_mock).toBe(true);

    const cap = await runStep<Record<string, unknown>>(sid, 2, {
      front_image: front,
      side_image: side,
      height_cm: 178,
      weight_kg: 74,
    });
    expect(cap).toHaveProperty("distance_ok");

    const seg = await runStep<{ mask_png: string; boundary_quality: number }>(sid, 3, {
      image: front,
    });
    expect(seg.mask_png.length).toBeGreaterThan(100);
    expect(seg.boundary_quality).toBeGreaterThanOrEqual(0);

    const calib = await runStep<{ px_per_cm: Record<string, number> }>(sid, 4);
    expect(Object.keys(calib.px_per_cm)).toEqual(["front", "side"]);

    // step 6 is locked until step 5 is confirmed
    await expect(runStep(sid, 6)).rejects.toThrow(/confirm/i);

    const lm = await runStep<{ points: Record<string, { source: string }> }>(sid, 5, {
      confirm: true,
    });
    expect(lm.points.waist.source).toBe("heuristic");

    const dragged = await runStep<{ points: Record<string, { source: string; x: number }> }>(sid, 5, {
      updates: [{ name: "waist", x: 100, y: 120 }],
    });
    expect(dragged.points.waist.source).toBe("user");

    const reg = await runStep<{ measurements: Record<string, { value_cm: number }> }>(
      sid,
      6,
      {},
      "regression",
    );
    const bm = await runStep<{ measurements: Record<string, { value_cm: number }> }>(
      sid,
      6,
      {},
      "body_model",
    );
    // identical result type, intervals included
    expect(Object.keys(reg.measurements)).toEqual(Object.keys(bm.measurements));
    for (const k of Object.keys(reg.measurements)) {
      expect(typeof reg.measurements[k].value_cm).toBe("number");
    }

    const val = await runStep<{ yaml: string; flags: unknown[] }>(sid, 7);
    expect(val.yaml).toContain("measurements");

    const draft = await runStep<{ route: string; pieces: { name: string }[] }>(sid, 8, {
      route: "garmentcode",
    });
    expect(draft.route).toBe("garmentcode");
    expect(draft.pieces.length).toBeGreaterThan(0);

    const prod = await runStep<{ pieces: { notches: unknown[] }[] }>(sid, 9);
    expect(prod.pieces[0].notches.length).toBeGreaterThan(0);

    const drape = await runStep<{ toile_checklist: string[] }>(sid, 10);
    expect(drape.toile_checklist.length).toBeGreaterThan(0);

    const state = await fetchSession(sid);
    for (let n = 1; n <= 10; n++) expect(state.results[String(n)]).toBeDefined();
    expect(state.any_mock).toBe(true);
    expect(exportYamlUrl(sid)).toContain(sid);
  });
});
