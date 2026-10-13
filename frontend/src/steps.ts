export const STEPS = [
  { id: 'upload', title: 'Measurements' },
  { id: 'definitions', title: 'Definitions' },
  { id: 'review', title: 'Review' },
  { id: 'configure', title: 'Configure' },
  { id: 'generate', title: 'Generate' },
  { id: 'cross-sections', title: 'Cross-sections' },
  { id: 'surface', title: 'Surface & seams' },
  { id: 'flattening', title: 'Flattening' },
  { id: 'pattern', title: 'Pattern' },
  { id: 'validation', title: 'Validation' },
  { id: 'export', title: 'Export' },
] as const

export type StepId = (typeof STEPS)[number]['id']

export function stepIndex(id: StepId): number {
  return STEPS.findIndex((s) => s.id === id)
}

export function firstIncompleteStep(ready: Record<StepId, boolean>): number {
  const index = STEPS.findIndex((s) => !ready[s.id])
  return index === -1 ? STEPS.length - 1 : index
}
