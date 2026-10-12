export const STEPS = [
  { id: 'upload', title: 'Upload YAML' },
  { id: 'definitions', title: 'Field definitions' },
  { id: 'review', title: 'Review measurements' },
  { id: 'base', title: 'Base block & differences' },
  { id: 'configure', title: 'Ease & limits' },
  { id: 'generate', title: 'Transform' },
  { id: 'compare', title: 'Compare base & target' },
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
