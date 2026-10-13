import { Badge, Button, Callout, NumberField, Select, Toggle } from '../components/ui'
import type { AppConfig, Settings } from '../types'

export function ConfigureStep({
  settings,
  onChange,
  config,
}: {
  settings: Settings
  onChange: (settings: Settings) => void
  config: AppConfig | null
}) {
  const set = <K extends keyof Settings>(key: K, value: Settings[K]) =>
    onChange({ ...settings, [key]: value })
  const preset = config?.easePresets[settings.ease_preset]
  const paperSizes = Object.keys(config?.paper.sizes_mm ?? { A0: [] })

  return (
    <div className="stack">
      <h2>Reconstruction and garment settings</h2>
      <p className="muted">
        These settings control the measured proxy surface, its garment ease, the fixed seam
        network, flattening, and production output. The first release is limited to a basic torso.
      </p>

      <Callout tone={settings.confirm_units ? 'ok' : 'warn'} title="Measurement units">
        The measurement input is interpreted in centimetres and converted to millimetres.
        Confirm the unit before generation.
        <Toggle
          label="I confirm the uploaded measurement values are in centimetres"
          checked={settings.confirm_units}
          onChange={(checked) => set('confirm_units', checked)}
        />
      </Callout>

      <fieldset>
        <legend>Proxy torso</legend>
        <div className="grid-3">
          <Select
            label="Garment scope"
            value={settings.garment_type}
            options={[{ value: 'basic_torso', label: 'Basic torso' }]}
            onChange={(v) => set('garment_type', v)}
          />
          <Select
            label="Cross-section model"
            value={settings.section_model}
            options={[
              { value: 'superellipse', label: 'Superellipse (assumed exponent)' },
              { value: 'ellipse', label: 'Ellipse' },
            ]}
            onChange={(v) => set('section_model', v)}
          />
          <NumberField
            label="Superellipse exponent"
            value={settings.section_exponent ?? (settings.section_model === 'ellipse' ? 2 : config?.crossSection.exponent ?? 2.3)}
            min={2}
            step={0.1}
            onChange={(v) => set('section_exponent', v)}
            hint={settings.section_model === 'ellipse' ? 'Ellipse uses n = 2.' : 'A configurable shape assumption.'}
          />
          <NumberField
            label="Ring resolution"
            value={settings.cross_section_resolution ?? config?.mesh.cross_section_resolution ?? 48}
            min={16}
            step={4}
            onChange={(v) => set('cross_section_resolution', v)}
          />
          <NumberField
            label="Longitudinal segments per level"
            value={settings.longitudinal_segments ?? config?.mesh.longitudinal_segments ?? 4}
            min={1}
            onChange={(v) => set('longitudinal_segments', v)}
          />
        </div>
      </fieldset>

      <fieldset>
        <legend>Garment ease</legend>
        <div className="grid-2">
          <Select
            label="Fit preset"
            value={settings.ease_preset}
            options={Object.keys(config?.easePresets ?? { fitted: {} }).map((name) => ({
              value: name,
              label: name.replace(/_/g, ' '),
            }))}
            onChange={(v) => set('ease_preset', v)}
          />
          <Badge tone="neutral">Ease is added to circumference, not as a uniform offset</Badge>
        </div>
        <div className="grid-3">
          <NumberField label="Bust ease" unit="mm" value={settings.bust_ease_mm ?? preset?.bust_mm ?? null} onChange={(v) => set('bust_ease_mm', v)} />
          <NumberField label="Waist ease" unit="mm" value={settings.waist_ease_mm ?? preset?.waist_mm ?? null} onChange={(v) => set('waist_ease_mm', v)} />
          <NumberField label="Hip ease" unit="mm" value={settings.hip_ease_mm ?? preset?.hip_mm ?? null} onChange={(v) => set('hip_ease_mm', v)} hint="Hip section is disabled unless configured." />
          <NumberField label="Armhole ease" unit="mm" value={settings.armhole_ease_mm ?? preset?.armhole_mm ?? null} onChange={(v) => set('armhole_ease_mm', v)} />
        </div>
      </fieldset>

      <fieldset>
        <legend>Seams and flattening</legend>
        <div className="grid-3">
          <Select
            label="Predetermined seam network"
            value={settings.seam_preset}
            options={[{ value: config?.seams.preset ?? 'basic_torso_v1', label: config?.seams.preset ?? 'basic_torso_v1' }]}
            onChange={(v) => set('seam_preset', v)}
          />
          <Select
            label="Flattening method"
            value={settings.flattening_method}
            options={[{ value: 'lscm_arap', label: 'LSCM initialization + ARAP refinement' }]}
            onChange={(v) => set('flattening_method', v)}
          />
          <NumberField label="Front neckline drop" unit="mm" value={settings.front_neck_drop_mm ?? config?.seams.front_neck_drop_mm ?? 70} min={0} onChange={(v) => set('front_neck_drop_mm', v)} />
          <NumberField label="Back neckline drop" unit="mm" value={settings.back_neck_drop_mm ?? config?.seams.back_neck_drop_mm ?? 20} min={0} onChange={(v) => set('back_neck_drop_mm', v)} />
        </div>
        <details>
          <summary>Flattening objective weights</summary>
          <p className="muted">Weights are recorded with the generated diagnostics. Flipped triangles remain a hard rejection.</p>
          <div className="grid-3">
            {([
              ['w_distortion', 'Distortion'],
              ['w_boundary', 'Boundary'],
              ['w_seam', 'Seam'],
              ['w_grain', 'Grain'],
              ['w_flip', 'Flip penalty'],
            ] as const).map(([key, label]) => (
              <NumberField
                key={key}
                label={label}
                value={settings[key] ?? config?.flattening.objective[key] ?? 0}
                min={0}
                step={0.01}
                onChange={(v) => set(key, v)}
              />
            ))}
          </div>
        </details>
      </fieldset>

      <fieldset>
        <legend>Production output</legend>
        <div className="grid-3">
          <Select
            label="Cutting-line join"
            value={settings.cutting_join ?? 'miter'}
            options={['miter', 'bevel', 'round'].map((value) => ({ value, label: value }))}
            onChange={(v) => set('cutting_join', v)}
          />
          <Select
            label="Paper size"
            value={settings.paper_size ?? config?.paper.default_size ?? 'A0'}
            options={paperSizes.map((size) => ({ value: size, label: size }))}
            onChange={(v) => set('paper_size', v)}
          />
          <Select
            label="Paper orientation"
            value={settings.paper_orientation ?? config?.paper.default_orientation ?? 'landscape'}
            options={['landscape', 'portrait'].map((value) => ({ value, label: value }))}
            onChange={(v) => set('paper_orientation', v)}
          />
        </div>
        <h3>Seam allowances (mm)</h3>
        <div className="grid-3">
          {Object.entries(config?.seamAllowances ?? {}).map(([role, value]) => (
            <NumberField
              key={role}
              label={role.replace(/_/g, ' ')}
              unit="mm"
              value={settings.seam_allowances[role] ?? value}
              min={0}
              onChange={(v) => onChange({
                ...settings,
                seam_allowances: { ...settings.seam_allowances, [role]: v ?? value },
              })}
            />
          ))}
        </div>
      </fieldset>

      {config ? (
        <Callout tone="info" title={`Reconstruction configuration v${config.version}`}>
          <p>
            Configured ring resolution {config.mesh.cross_section_resolution}; longitudinal
            resolution {config.mesh.longitudinal_segments}; flattening method{' '}
            <code>{config.flattening.method}</code>.
          </p>
        </Callout>
      ) : null}

      <Button
        variant="ghost"
        onClick={() => onChange({
          ...settings,
          section_exponent: null,
          cross_section_resolution: null,
          longitudinal_segments: null,
          bust_ease_mm: null,
          waist_ease_mm: null,
          hip_ease_mm: null,
          armhole_ease_mm: null,
          front_neck_drop_mm: null,
          back_neck_drop_mm: null,
          w_distortion: null,
          w_boundary: null,
          w_seam: null,
          w_grain: null,
          w_flip: null,
          seam_allowances: {},
        })}
      >
        Restore configured defaults
      </Button>
    </div>
  )
}
