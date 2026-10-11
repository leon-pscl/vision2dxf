import { STEPS } from '../steps'

export function Stepper({
  current,
  reachable,
  onSelect,
}: {
  current: number
  reachable: (index: number) => boolean
  onSelect: (index: number) => void
}) {
  return (
    <nav className="stepper" aria-label="Workflow">
      <ol>
        {STEPS.map((step, index) => {
          const state = index === current ? 'current' : index < current ? 'done' : 'todo'
          return (
            <li key={step.id} className={`step step-${state}`}>
              <button
                type="button"
                onClick={() => onSelect(index)}
                disabled={!reachable(index)}
                aria-current={index === current ? 'step' : undefined}
              >
                <span className="step-index">{index + 1}</span>
                <span className="step-title">{step.title}</span>
              </button>
            </li>
          )
        })}
      </ol>
    </nav>
  )
}
