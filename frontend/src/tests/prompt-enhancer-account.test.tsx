import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  PromptEnhancerAccountBar,
  PromptEnhancerAccountWidget,
} from '@/components/widgets/PromptEnhancerAccountBar'
import {
  clearPromptEnhancerBalanceCache,
  getPromptEnhancerAccountProvider,
  PROMPT_ENHANCER_MODELS,
} from '@/lib/prompt-enhancer-account'
import {
  readPromptEnhancerAccountSelection,
} from '@/lib/prompt-enhancer-account-node'

const runningHubResponse = {
  amount: 25.5,
  currency: 'CNY',
}

describe('prompt enhancer API account management', () => {
  beforeEach(() => {
    clearPromptEnhancerBalanceCache()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('keeps provider balance, console, and API Key endpoints in one config', () => {
    const runningHub = getPromptEnhancerAccountProvider(PROMPT_ENHANCER_MODELS.runningHubDoubao)
    expect(runningHub.balanceRequest?.url).toBe('/easy-media/prompt-enhancer/runninghub-balance')
    expect(runningHub.balanceRequest?.refreshIntervalMs).toBe(30_000)
    expect(runningHub.balancePageUrl).toBeTruthy()
    expect(runningHub.apiKeyPageUrl).toBeTruthy()

    const minimax = getPromptEnhancerAccountProvider(PROMPT_ENHANCER_MODELS.minimax)
    expect(minimax.balanceRequest).toBeUndefined()
    expect(minimax.balancePageUrl).toBeTruthy()
    expect(minimax.apiKeyPageUrl).toBeTruthy()
  })

  it('reads the selected dynamic-combo model and API key widgets', () => {
    expect(readPromptEnhancerAccountSelection({
      widgets: [
        { name: 'model', value: [PROMPT_ENHANCER_MODELS.runningHubGlm] },
        { name: 'model.apikey', value: ['rh-secret'] },
      ],
    })).toEqual({
      modelName: PROMPT_ENHANCER_MODELS.runningHubGlm,
      apiKey: 'rh-secret',
    })

    expect(readPromptEnhancerAccountSelection({
      widgets: [{
        name: 'model',
        value: [{ model: [PROMPT_ENHANCER_MODELS.runningHubDoubao], apikey: ['nested-secret'] }],
      }],
    })).toEqual({
      modelName: PROMPT_ENHANCER_MODELS.runningHubDoubao,
      apiKey: 'nested-secret',
    })
  })

  it('updates the backend-declared widget when the sibling DynamicCombo changes', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => runningHubResponse,
    } as Response)
    const node: {
      onWidgetChanged?: (name: string, value: unknown, oldValue: unknown, widget: unknown) => void
      widgets: Array<{ name: string; value: unknown }>
    } = {
      widgets: [{ name: 'model', value: [PROMPT_ENHANCER_MODELS.minimax] }],
    }
    render(
      <PromptEnhancerAccountWidget
        app={{ ui: { settings: { settingsValues: { 'Comfy.Locale': 'zh-CN' } } } } as never}
        inputName="api_account"
        node={node}
        onChange={vi.fn()}
        value=""
        widget={{} as never}
      />,
    )
    expect(screen.getByText('去查看余额')).not.toBeNull()

    node.widgets = [
      { name: 'model', value: [PROMPT_ENHANCER_MODELS.runningHubDoubao] },
      { name: 'model.apikey', value: ['rh-secret'] },
    ]
    act(() => node.onWidgetChanged?.('model', node.widgets[0].value, null, node.widgets[0]))
    expect(await screen.findByText(/余额：/)).not.toBeNull()
  })

  it('loads RunningHub balance and rate-limits execution and provider-switch refreshes', async () => {
    let now = 1_000
    vi.spyOn(Date, 'now').mockImplementation(() => now)
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => runningHubResponse,
    } as Response)

    const { rerender } = render(
      <PromptEnhancerAccountBar
        apiKey="rh-secret"
        locale="zh-CN"
        modelName={PROMPT_ENHANCER_MODELS.runningHubDoubao}
        refreshToken={0}
      />,
    )
    const balanceBadge = await screen.findByText(/余额：/)
    expect(balanceBadge.className).toContain('bg-secondary')
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(
      '/easy-media/prompt-enhancer/runninghub-balance',
      expect.objectContaining({ body: JSON.stringify({ api_key: 'rh-secret' }) }),
    )

    rerender(
      <PromptEnhancerAccountBar
        apiKey="rh-secret"
        locale="zh-CN"
        modelName={PROMPT_ENHANCER_MODELS.runningHubDoubao}
        refreshToken={1}
      />,
    )
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    rerender(
      <PromptEnhancerAccountBar
        apiKey="another-key"
        locale="zh-CN"
        modelName={PROMPT_ENHANCER_MODELS.minimax}
        refreshToken={1}
      />,
    )
    expect(await screen.findByText('去查看余额')).not.toBeNull()
    rerender(
      <PromptEnhancerAccountBar
        apiKey="rh-secret"
        locale="zh-CN"
        modelName={PROMPT_ENHANCER_MODELS.runningHubGlm}
        refreshToken={1}
      />,
    )
    expect(await screen.findByText(/余额：/)).not.toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(1)

    now += 30_001
    rerender(
      <PromptEnhancerAccountBar
        apiKey="rh-secret"
        locale="zh-CN"
        modelName={PROMPT_ENHANCER_MODELS.runningHubGlm}
        refreshToken={2}
      />,
    )
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
  })

  it('asks the local proxy to use config.yaml when the node API key is empty', async () => {
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => runningHubResponse,
    } as Response)
    render(
      <PromptEnhancerAccountBar
        apiKey=""
        locale="zh-CN"
        modelName={PROMPT_ENHANCER_MODELS.runningHubDoubao}
      />,
    )

    expect(await screen.findByText(/余额：/)).not.toBeNull()
    expect(fetchMock).toHaveBeenCalledWith(
      '/easy-media/prompt-enhancer/runninghub-balance',
      expect.objectContaining({ body: JSON.stringify({ api_key: '' }) }),
    )
  })

  it('opens the configured balance and API Key pages for providers without a balance API', () => {
    const openMock = vi.spyOn(globalThis, 'open').mockImplementation(() => null)
    const minimax = getPromptEnhancerAccountProvider(PROMPT_ENHANCER_MODELS.minimax)
    render(
      <PromptEnhancerAccountBar
        apiKey=""
        locale="zh-CN"
        modelName={PROMPT_ENHANCER_MODELS.minimax}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: /去查看余额/ }))
    fireEvent.click(screen.getByRole('button', { name: /API Key/ }))
    expect(openMock).toHaveBeenNthCalledWith(1, minimax.balancePageUrl, '_blank', 'noopener,noreferrer')
    expect(openMock).toHaveBeenNthCalledWith(2, minimax.apiKeyPageUrl, '_blank', 'noopener,noreferrer')
  })

  it('rate-limits failed balance requests during the same cooldown', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('network unavailable'))
    const { rerender } = render(
      <PromptEnhancerAccountBar
        apiKey="rh-secret"
        locale="zh-CN"
        modelName={PROMPT_ENHANCER_MODELS.runningHubDoubao}
        refreshToken={0}
      />,
    )
    expect(await screen.findByText('余额：--')).not.toBeNull()

    rerender(
      <PromptEnhancerAccountBar
        apiKey="rh-secret"
        locale="zh-CN"
        modelName={PROMPT_ENHANCER_MODELS.runningHubDoubao}
        refreshToken={1}
      />,
    )
    await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
  })
})
