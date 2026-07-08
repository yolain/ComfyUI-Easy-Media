export const MODEL_MISSING_EVENT = 'easy-media-model-missing'

export interface MissingModelInfo {
  name: string
  display_name: string
  filename: string
  directory: string
  path: string
  url: string
  urls?: string[]
}

export class MissingModelError extends Error {
  model: MissingModelInfo

  constructor(message: string, model: MissingModelInfo) {
    super(message)
    this.name = 'MissingModelError'
    this.model = model
  }
}

function isMissingModelInfo(value: unknown): value is MissingModelInfo {
  if (!value || typeof value !== 'object') return false
  const model = value as Partial<Record<keyof MissingModelInfo, unknown>>
  return (
    typeof model.name === 'string'
    && typeof model.display_name === 'string'
    && typeof model.filename === 'string'
    && typeof model.directory === 'string'
    && typeof model.path === 'string'
    && typeof model.url === 'string'
    && (model.urls === undefined || (Array.isArray(model.urls) && model.urls.every((url) => typeof url === 'string')))
  )
}

export function parseMissingModelPayload(value: unknown): MissingModelInfo | null {
  return isMissingModelInfo(value) ? value : null
}

export function throwIfMissingModelResponse(payload: unknown): void {
  if (!payload || typeof payload !== 'object') return
  const response = payload as { error?: unknown; model_missing?: unknown }
  const model = parseMissingModelPayload(response.model_missing)
  if (!model) return
  const message = typeof response.error === 'string'
    ? response.error
    : `${model.display_name} model is not installed.`
  throw new MissingModelError(message, model)
}

export async function downloadEasyMediaModel(modelName: string): Promise<MissingModelInfo> {
  const response = await fetch('/easy-media/models/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: modelName }),
  })
  let payload: unknown
  try {
    payload = await response.json()
  } catch (error) {
    throw new Error(`Model download returned invalid JSON: ${String(error)}`)
  }
  if (!response.ok) {
    const message = payload && typeof payload === 'object' && 'error' in payload
      ? String((payload as { error: unknown }).error)
      : `Model download failed (${response.status})`
    throw new Error(message)
  }
  if (!payload || typeof payload !== 'object') {
    throw new Error('Invalid model download response')
  }
  const model = parseMissingModelPayload((payload as { model?: unknown }).model)
  if (!model) throw new Error('Invalid model download response')
  return model
}
