import { PROMPT_ENHANCER_MODELS } from '@/lib/prompt-enhancer-account'

export interface PromptEnhancerNodeWidget {
  name?: string
  value?: unknown
}

export interface PromptEnhancerAccountNode {
  onConfigure?: (serialisedNode: unknown) => void
  onExecuted?: (output: unknown) => void
  onWidgetChanged?: (name: string, value: unknown, oldValue: unknown, widget: unknown) => void
  widgets?: PromptEnhancerNodeWidget[]
}

function unwrapWidgetValue(value: unknown): unknown {
  let current = value
  while (Array.isArray(current) && current.length === 1) current = current[0]
  return current
}

function readObjectString(value: unknown, key: string): string {
  const unwrapped = unwrapWidgetValue(value)
  if (!unwrapped || typeof unwrapped !== 'object') return ''
  const nested = unwrapWidgetValue((unwrapped as Record<string, unknown>)[key])
  return typeof nested === 'string' ? nested : ''
}

export function readPromptEnhancerAccountSelection(
  node: Pick<PromptEnhancerAccountNode, 'widgets'>,
): { apiKey: string; modelName: string } {
  const widgets = node.widgets ?? []
  const rootModelValue = widgets.find((widget) => widget.name === 'model')?.value
  const modelValue = unwrapWidgetValue(
    widgets.find((widget) => widget.name === 'model.model')?.value ?? rootModelValue,
  )
  const apiKeyValue = unwrapWidgetValue(
    widgets.find((widget) => widget.name === 'model.apikey')?.value
      ?? widgets.find((widget) => widget.name === 'apikey')?.value,
  )
  return {
    modelName: typeof modelValue === 'string'
      ? modelValue
      : readObjectString(rootModelValue, 'model') || PROMPT_ENHANCER_MODELS.minimax,
    apiKey: typeof apiKeyValue === 'string'
      ? apiKeyValue
      : readObjectString(rootModelValue, 'apikey'),
  }
}
