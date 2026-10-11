import type { ReactNode } from 'react'

export function Button({
  children,
  onClick,
  variant = 'default',
  disabled,
  type = 'button',
  title,
}: {
  children: ReactNode
  onClick?: () => void
  variant?: 'default' | 'primary' | 'ghost' | 'danger'
  disabled?: boolean
  type?: 'button' | 'submit'
  title?: string
}) {
  return (
    <button type={type} className={`btn btn-${variant}`} onClick={onClick} disabled={disabled} title={title}>
      {children}
    </button>
  )
}

export function NumberField({
  label,
  value,
  onChange,
  unit,
  hint,
  min,
  step = 1,
}: {
  label: string
  value: number | null
  onChange: (value: number | null) => void
  unit?: string
  hint?: string
  min?: number
  step?: number
}) {
  return (
    <label className="field">
      <span className="field-label">
        {label}
        {unit ? <em> {unit}</em> : null}
      </span>
      <input
        type="number"
        inputMode="decimal"
        value={value ?? ''}
        min={min}
        step={step}
        placeholder="auto"
        onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))}
      />
      {hint ? <small className="hint">{hint}</small> : null}
    </label>
  )
}

export function Select<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (value: T) => void
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value as T)}>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </label>
  )
}

export function Callout({
  tone = 'info',
  title,
  children,
}: {
  tone?: 'info' | 'warn' | 'error' | 'ok'
  title?: string
  children: ReactNode
}) {
  return (
    <div className={`callout callout-${tone}`} role={tone === 'error' ? 'alert' : undefined}>
      {title ? <strong>{title}</strong> : null}
      <div>{children}</div>
    </div>
  )
}

export function Toggle({
  checked,
  onChange,
  label,
}: {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
}) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} />
      <span>{label}</span>
    </label>
  )
}

export function Badge({ tone, children }: { tone: 'ok' | 'warn' | 'error' | 'neutral'; children: ReactNode }) {
  return <span className={`badge badge-${tone}`}>{children}</span>
}

export function Spinner({ label }: { label: string }) {
  return (
    <p className="spinner" role="status" aria-live="polite">
      {label}
    </p>
  )
}
