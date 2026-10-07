/** Steps, in order. Titles are uppercase in the CSS. */
export const STEPS: { n: number; title: string }[] = [
  { n: 1, title: "Garment" },
  { n: 2, title: "Capture" },
  { n: 3, title: "Segmentation" },
  { n: 4, title: "Calibration" },
  { n: 5, title: "Landmarks" },
  { n: 6, title: "Measurement" },
  { n: 7, title: "Validation" },
  { n: 8, title: "Drafting" },
  { n: 9, title: "Production" },
  { n: 10, title: "Toile" },
];

/** Steps 6 to 10 are locked until the step 5 landmarks are confirmed. */
export function isLocked(step: number, confirmed: boolean, results: Record<string, unknown>): boolean {
  if (step < 6) return false;
  if (step === 6) return !confirmed;
  // 7 to 10 also need step 6 to have run
  return !results["6"];
}
