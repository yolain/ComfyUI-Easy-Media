import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { installWorkflowSubmission, runWorkflowSubmission, type WorkflowSubmission } from '@/lib/workflow-submission'

function setup() {
  const current = { path: 'workflows/original.json' }
  let graph: WorkflowSubmission['workflow'] = { id: 'original', nodes: [], links: [], version: 0.4 }
  const app = {
    processingQueue: false,
    extensionManager: { workflow: { activeWorkflow: current, isBusy: false }, toast: { add: vi.fn() } },
    loadGraphData: vi.fn(async (workflow: typeof graph) => { graph = workflow }),
    rootGraph: { serialize: vi.fn(() => graph) },
    canvas: { setDirty: vi.fn() },
    api: {
      clientId: 'socket-1',
      queuePrompt: vi.fn().mockResolvedValue({ prompt_id: 'prompt-1' }),
      fetchApi: vi.fn(),
    },
    queuePrompt: vi.fn(async () => {
      await app.api.queuePrompt(0, { workflow: graph, output: {} })
      return true
    }),
  }
  const job: WorkflowSubmission = {
    request_id: 'request-1', claim_token: 'token', mode: 'new_tab', name: 'Demo', auto_queue: true,
    workflow: { id: 'source', nodes: [], links: [], version: 0.4 },
  }
  return { app, nativeApp: app as unknown as ComfyApp, job, current }
}

afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks(); vi.unstubAllGlobals() })

describe('UI workflow submission', () => {
  it('uses valid workflow UUIDs on HTTP pages without crypto.randomUUID', async () => {
    vi.stubGlobal('crypto', { getRandomValues: crypto.getRandomValues.bind(crypto) })
    const { app, nativeApp, job } = setup()
    expect(await runWorkflowSubmission(nativeApp, job)).toMatchObject({ status: 'queued' })
    expect(app.rootGraph.serialize().id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
  })

  it('opens the visible workflow before invoking native Run once and restores the observer', async () => {
    const { app, nativeApp, job } = setup()
    const original = app.api.queuePrompt
    expect(await runWorkflowSubmission(nativeApp, job)).toEqual({ status: 'queued', prompt_id: 'prompt-1' })
    expect(app.loadGraphData).toHaveBeenCalledWith(expect.objectContaining({ nodes: [], links: [] }), true, true, 'Demo-request-1')
    expect(app.loadGraphData.mock.invocationCallOrder[0]).toBeLessThan(app.queuePrompt.mock.invocationCallOrder[0])
    expect(app.queuePrompt).toHaveBeenCalledExactlyOnceWith(0, 1)
    expect(app.api.queuePrompt).toBe(original)
    expect(job.workflow.id).toBe('source')
  })

  it('replaces the current tab without queueing when requested', async () => {
    const { app, nativeApp, job, current } = setup()
    expect(await runWorkflowSubmission(nativeApp, { ...job, mode: 'replace', auto_queue: false })).toEqual({ status: 'loaded' })
    expect(app.loadGraphData).toHaveBeenCalledWith(expect.anything(), true, true, current)
    expect(app.queuePrompt).not.toHaveBeenCalled()
  })

  it('does not overwrite a graph while native submission is busy', async () => {
    const { app, nativeApp, job } = setup()
    app.processingQueue = true
    expect(await runWorkflowSubmission(nativeApp, job)).toMatchObject({ status: 'failed' })
    expect(app.loadGraphData).not.toHaveBeenCalled()
    expect(app.queuePrompt).not.toHaveBeenCalled()
  })

  it('does not queue when loading silently fails', async () => {
    const { app, nativeApp, job } = setup()
    app.loadGraphData.mockImplementation(async () => {})
    expect(await runWorkflowSubmission(nativeApp, job)).toMatchObject({ status: 'failed' })
    expect(app.queuePrompt).not.toHaveBeenCalled()
  })

  it('blocks a workflow switched by the user during the native queue operation', async () => {
    const { app, nativeApp, job } = setup()
    const original = app.api.queuePrompt
    app.queuePrompt.mockImplementation(async () => {
      await app.api.queuePrompt(0, { workflow: { id: 'other' }, output: {} })
      return true
    })
    expect(await runWorkflowSubmission(nativeApp, job)).toMatchObject({ status: 'failed' })
    expect(original).not.toHaveBeenCalled()
    expect(app.api.queuePrompt).toBe(original)
  })

  it('preserves accepted prompt_id even when a native post-queue hook fails', async () => {
    const { app, nativeApp, job } = setup()
    const queue = app.queuePrompt.getMockImplementation()!
    app.queuePrompt.mockImplementation(async () => { await queue(); throw new Error('Widget hook failed') })
    expect(await runWorkflowSubmission(nativeApp, job)).toEqual({ status: 'queued', prompt_id: 'prompt-1' })
  })

  it('reports unknown for a lost queue response and never retries execution', async () => {
    const { app, nativeApp, job } = setup()
    const original = app.api.queuePrompt.mockRejectedValue(new TypeError('Network error'))
    expect(await runWorkflowSubmission(nativeApp, job)).toEqual({ status: 'unknown', error: 'Network error' })
    expect(original).toHaveBeenCalledTimes(1)
    expect(app.api.queuePrompt).toBe(original)
  })

  it('retries only the acknowledgement if result delivery fails', async () => {
    vi.useFakeTimers()
    vi.spyOn(console, 'error').mockImplementation(() => {})
    const { app, nativeApp, job } = setup()
    app.api.fetchApi.mockImplementation(async (path: string) => {
      if (path.endsWith('/result')) throw new TypeError('Disconnected')
      return { ok: true, json: async () => ({ job }) }
    })
    const stop = installWorkflowSubmission(nativeApp)
    await vi.advanceTimersByTimeAsync(8000)
    stop()
    expect(app.queuePrompt).toHaveBeenCalledTimes(1)
    expect(app.api.fetchApi.mock.calls.filter(([path]) => path.endsWith('/result')).length).toBeGreaterThan(1)
    expect(app.api.fetchApi.mock.calls.filter(([path]) => path.endsWith('/poll'))).toHaveLength(1)
  })
})
