export const BEAUTY_PARAMETERS = ['slim_face', 'chin', 'eyes', 'nose', 'mouth', 'eye_bright', 'under_eye', 'lip_tint', 'blush', 'smooth', 'whiten', 'brightness', 'warmth'] as const
export const NONNEGATIVE_BEAUTY_PARAMETERS = new Set<string>(['eye_bright', 'under_eye', 'lip_tint', 'blush', 'smooth', 'whiten'])
export const DEFAULT_BEAUTY_COLORS = { lip: '#c4485f', blush: '#ff9f80' }
export type BeautyParameter = typeof BEAUTY_PARAMETERS[number]
export type BeautyParameters = Record<BeautyParameter, number>
export interface BeautySource { source_type: 'input' | 'output' | 'temp'; file_path: string }
export interface BeautyPreset { colors: { lip: string; blush: string }; source_mode?: 'picker' | 'image'; version: 1; parameters: BeautyParameters; source: BeautySource | null }
export interface BeautyGeometry {
  handles: { center: [number, number]; radius: number; key: BeautyParameter; delta: [number, number]; scale: number }[]
  regions: { center: [number, number]; radii: [number, number] }[]
  features?: { eyes: [number, number][][]; lip_outer: [number, number][]; lip_inner: [number, number][] }
  accents?: { center: [number, number]; radii: [number, number] }[]
  axis?: [number, number]
  blur?: number
}
export interface BeautyPreview { source?: BeautySource | null; input_connected?: boolean; filename: string; type: 'temp'; geometry: BeautyGeometry; face_found: boolean }
export const DEFAULT_BEAUTY_PRESET: BeautyPreset = {
  version: 1, source: null, colors: { ...DEFAULT_BEAUTY_COLORS },
  parameters: { slim_face: 0, chin: 0, eyes: 0, nose: 0, mouth: 0, eye_bright: 0, under_eye: 0, lip_tint: 0, blush: 0, smooth: 0, whiten: 0, brightness: 0, warmth: 0 },
}
export function normalizeBeautyPreset(value: unknown): BeautyPreset {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid face beauty preset')
  const record = value as Partial<BeautyPreset>
  if (record.version !== undefined && record.version !== 1) throw new Error('Unsupported face beauty preset version')
  const parameters = { ...DEFAULT_BEAUTY_PRESET.parameters }
  if (record.parameters !== undefined && (!record.parameters || typeof record.parameters !== 'object' || Array.isArray(record.parameters))) throw new Error('Invalid face beauty parameters')
  for (const key of BEAUTY_PARAMETERS) {
    const raw = record.parameters?.[key] === undefined ? 0 : record.parameters[key]
    if (typeof raw !== 'number' || !Number.isFinite(raw)) throw new Error(`Invalid face beauty parameter: ${key}`)
    parameters[key] = Math.min(1, Math.max(NONNEGATIVE_BEAUTY_PARAMETERS.has(key) ? 0 : -1, raw))
  }
  const colors = { ...DEFAULT_BEAUTY_COLORS }
  if (record.colors !== undefined) {
    if (!record.colors || typeof record.colors !== 'object' || Array.isArray(record.colors)) throw new Error('Invalid face beauty colors')
    for (const key of ['lip', 'blush'] as const) {
      const color = record.colors[key] === undefined ? colors[key] : record.colors[key]
      if (typeof color !== 'string' || !/^#[0-9a-f]{6}$/i.test(color)) throw new Error(`Invalid face beauty color: ${key}`)
      colors[key] = color.toLowerCase()
    }
  }
  const source = record.source ?? null
  if (source !== null && (typeof source !== 'object' || Array.isArray(source) || !['input', 'output', 'temp'].includes(source.source_type) || typeof source.file_path !== 'string')) throw new Error('Invalid face beauty image source')
  if (record.source_mode !== undefined && record.source_mode !== 'picker' && record.source_mode !== 'image') throw new Error('Invalid face beauty source mode')
  return { version: 1, parameters, colors, source, ...(record.source_mode ? { source_mode: record.source_mode } : {}) }
}
