import { Badge, Button, NumberField, Select } from '../components/ui'
import type { AppConfig, Settings } from '../types'

export function ConfigureStep({
  settings,
  onChange,
  config,
  assumptions,
}: {
  settings: Settings
  onChange: (settings: Settings) => void
  config: AppConfig | null
  assumptions: { field: string | null; code: string; message: string }[]
}) {
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) => onChange({ ...settings, [key]: value })
  const preset = config?.easePresets[settings.ease_preset]
  const paperSizes = Object.keys(config?.paper.sizes_mm ?? { A0: [] })

  return (
    <div className="stack">
      <h2>Garment and ease</h2>

      <div className="grid-2">
        <Select
          label="Garment type"
          value={settings.garment_type}
          options={[{ value: 'basic_bodice', label: 'Basic bodice (front + back)' }]}
          onChange={(v) => set('garment_type', v)}
        />
        <Select
          label="Fit preset"
          value={settings.ease_preset}
          options={Object.keys(config?.easePresets ?? { fitted: {} }).map((k) => ({ value: k, label: k.replace('_', ' ') }))}
          onChange={(v) => set('ease_preset', v)}
        />
      </div>

      <fieldset>
        <legend>Ease (mm, total girth)</legend>
        <div className="grid-3">
          <NumberField label="Bust ease" unit="mm" value={settings.bust_ease_mm ?? preset?.bust_mm ?? null} onChange={(v) => set('bust_ease_mm', v)} />
          <NumberField label="Waist ease" unit="mm" value={settings.waist_ease_mm ?? preset?.waist_mm ?? null} onChange={(v) => set('waist_ease_mm', v)} />
          <NumberField label="Hip ease" unit="mm" value={settings.hip_ease_mm ?? preset?.hip_mm ?? null} onChange={(v) => set('hip_ease_mm', v)} />
        </div>
        <small className="hint">Leave a field empty to use the preset value.</small>
      </fieldset>

      <fieldset>
        <legend>Seam allowance by edge role (mm)</legend>
        <div className="grid-3">
          {Object.entries(config?.seamAllowances ?? {}).map(([role, value]) => (
            <NumberField
              key={role}
              label={role.replace('_', ' ')}
              unit="mm"
              value={settings.seam_allowances[role] ?? value}
              onChange={(v) => onChange({ ...settings, seam_allowances: { ...settings.seam_allowances, [role]: v ?? value } })}
            />
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend>Drafting overrides</legend>
        <p className="muted">
          Used when a measurement cannot be trusted. Every override is recorded as an assumption in
          <code> generation-manifest.json</code>.
        </p>
        <div className="grid-3">
          <NumberField label="Shoulder slope" unit="deg" min={0} step={0.5} value={settings.shoulder_slope_deg} onChange={(v) => set('shoulder_slope_deg', v)} hint="measured: shoulder_incl" />
          <NumberField label="Armscye depth" unit="mm" min={0} value={settings.armscye_depth_mm} onChange={(v) => set('armscye_depth_mm', v)} hint="measured: armscye_depth" />
          <NumberField label="Bust apex level" unit="mm" min={0} value={settings.bust_apex_y_mm} onChange={(v) => set('bust_apex_y_mm', v)} hint="measured: vert_bust_line" />
          <NumberField label="Dart intake" unit="mm" min={0} value={settings.dart_intake_mm} onChange={(v) => set('dart_intake_mm', v)} hint="default: projection x ratio" />
          <Select
            label="Cutting line join"
            value={settings.cutting_join ?? 'miter'}
            options={[{ value: 'miter', label: 'miter' }, { value: 'bevel', label: 'bevel' }, { value: 'round', label: 'round' }]}
            onChange={(v) => set('cutting_join', v)}
          />
          <Select
            label="Paper size"
            value={settings.paper_size ?? config?.paper.default_size ?? 'A0'}
            options={paperSizes.map((s) => ({ value: s, label: s }))}
            onChange={(v) => set('paper_size', v)}
          />
        </div>
      </fieldset>

      {assumptions.length > 0 ? (
        <ul className="tight assumptions">
          {assumptions.map((a, i) => (
            <li key={i}>
              <Badge tone="warn">{a.code}</Badge> {a.message}
            </li>
          ))}
        </ul>
      ) : null}

      <div className="row">
        <Button variant="ghost" onClick={() => onChange({
          ...settings,
          bust_ease_mm: null,
          waist_ease_mm: null,
          hip_ease_mm: null,
          shoulder_slope_deg: null,
          armscye_depth_mm: null,
          bust_apex_y_mm: null,
          dart_intake_mm: null,
        })}>
          Clear overrides
        </Button>
      </div>
    </div>
  )
}
