export const STEPS = [
  { id: 'upload', title: 'Upload YAML' },
  { id: 'review', title: 'Review measurements' },
  { id: 'configure', title: 'Garment & ease' },
  { id: 'generate', title: 'Generate' },
  { id: 'inspect', title: 'Inspect pattern' },
  { id: 'validation', title: 'Review validation' },
  { id: 'export', title: 'Export files' },
] as const

export type StepId = (typeof STEPS)[number]['id']

export function stepIndex(id: StepId): number {
  return STEPS.findIndex((s) => s.id === id)
}

export function firstIncompleteStep(ready: Record<StepId, boolean>): number {
  const index = STEPS.findIndex((s) => !ready[s.id])
  return index === -1 ? STEPS.length - 1 : index
}
