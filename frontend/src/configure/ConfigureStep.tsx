import { Badge, Button, Callout, NumberField, Select, Toggle } from '../components/ui'
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
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) =>
    onChange({ ...settings, [key]: value })
  const preset = config?.easePresets[settings.ease_preset]
  const paperSizes = Object.keys(config?.paper.sizes_mm ?? { A0: [] })
  const envelope = config?.transformation.envelope

  return (
    <div className="stack">
      <h2>Ease and transformation limits</h2>

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
          options={Object.keys(config?.easePresets ?? { fitted: {} }).map((k) => ({
            value: k,
            label: k.replace('_', ' '),
          }))}
          onChange={(v) => set('ease_preset', v)}
        />
      </div>

      <fieldset>
        <legend>Ease (mm, total girth)</legend>
        <p className="muted">
          Ease is applied as a difference from the ease the base block already carries, so keeping the
          block's own preset leaves its fit alone. Changing it widens or narrows the side seam.
        </p>
        <div className="grid-3">
          <NumberField label="Bust ease" unit="mm" value={settings.bust_ease_mm ?? preset?.bust_mm ?? null} onChange={(v) => set('bust_ease_mm', v)} />
          <NumberField label="Waist ease" unit="mm" value={settings.waist_ease_mm ?? preset?.waist_mm ?? null} onChange={(v) => set('waist_ease_mm', v)} />
          <NumberField label="Hip ease" unit="mm" value={settings.hip_ease_mm ?? preset?.hip_mm ?? null} onChange={(v) => set('hip_ease_mm', v)} hint="no hip edge on a torso block" />
        </div>
        <small className="hint">Leave a field empty to use the preset value.</small>
      </fieldset>

      <fieldset>
        <legend>Transformation strength</legend>
        <p className="muted">
          The envelope is the range of differences this block has been tested across. Beyond it the
          result is an extrapolation, and the confirmation is recorded in the manifest. A target more
          than {envelope?.stop_factor ?? 2}× past the range is refused however it is confirmed.
        </p>
        <Toggle
          label="Allow a target outside the tested envelope"
          checked={settings.confirm_envelope}
          onChange={(on) => set('confirm_envelope', on)}
        />
        {envelope ? (
          <details>
            <summary>Tested ranges</summary>
            <table className="rules">
              <thead>
                <tr>
                  <th scope="col">Measurement</th>
                  <th scope="col">Max difference</th>
                  <th scope="col">Max %</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(envelope.measurements).map(([name, limits]) => (
                  <tr key={name}>
                    <th scope="row"><code>{name}</code></th>
                    <td className="nowrap">
                      {limits.max_abs_mm !== undefined ? `${limits.max_abs_mm} mm` : `${limits.max_abs_deg}°`}
                    </td>
                    <td className="nowrap">{limits.max_pct !== undefined ? `${limits.max_pct}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        ) : null}
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
              onChange={(v) =>
                onChange({
                  ...settings,
                  seam_allowances: { ...settings.seam_allowances, [role]: v ?? value },
                })
              }
            />
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend>Output</legend>
        <div className="grid-3">
          <Select
            label="Cutting line join"
            value={settings.cutting_join ?? 'miter'}
            options={[
              { value: 'miter', label: 'miter' },
              { value: 'bevel', label: 'bevel' },
              { value: 'round', label: 'round' },
            ]}
            onChange={(v) => set('cutting_join', v)}
          />
          <Select
            label="Paper size"
            value={settings.paper_size ?? config?.paper.default_size ?? 'A0'}
            options={paperSizes.map((s) => ({ value: s, label: s }))}
            onChange={(v) => set('paper_size', v)}
          />
          <Select
            label="Paper orientation"
            value={settings.paper_orientation ?? config?.paper.default_orientation ?? 'landscape'}
            options={[
              { value: 'landscape', label: 'landscape' },
              { value: 'portrait', label: 'portrait' },
            ]}
            onChange={(v) => set('paper_orientation', v)}
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

      {config ? (
        <Callout tone="info" title="Solver">
          <p>
            Method <code>{config.transformation.method}</code>, configuration v
            {config.transformation.version}, influence map v
            {config.transformation.influenceMapVersion}. Landmark residual tolerance{' '}
            <code>{String(config.transformation.solver.landmark_residual_tol_mm)}</code> mm.
          </p>
        </Callout>
      ) : null}

      <div className="row">
        <Button
          variant="ghost"
          onClick={() =>
            onChange({
              ...settings,
              bust_ease_mm: null,
              waist_ease_mm: null,
              hip_ease_mm: null,
              seam_allowances: {},
              confirm_envelope: false,
            })
          }
        >
          Clear overrides
        </Button>
      </div>
    </div>
  )
}
