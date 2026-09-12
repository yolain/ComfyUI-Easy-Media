import { act, fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  SAMPLING_PREVIEW_EVENT,
  SamplingPreviewWidget,
  type SamplingPreviewPayload,
} from '@/components/widgets/SamplingPreviewWidget'

describe('SamplingPreviewWidget', () => {
  const listeners = new Map<string, (event: CustomEvent<unknown>) => void>()
  const api = {
    addCustomEventListener: vi.fn((name: string, listener: (event: CustomEvent<unknown>) => void) => {
      listeners.set(name, listener)
    }),
    removeCustomEventListener: vi.fn((name: string) => listeners.delete(name)),
  }
  const rawApp = {
    api,
    ui: { settings: { settingsValues: { 'Comfy.Locale': 'en' } } },
  }
  const app = rawApp as never

  beforeEach(() => {
    listeners.clear()
    vi.clearAllMocks()
    rawApp.ui.settings.settingsValues['Comfy.Locale'] = 'en'
    vi.stubGlobal('requestAnimationFrame', vi.fn(() => 1))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true,
      value: vi.fn(() => 'blob:preview'),
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true,
      value: vi.fn(),
    })
  })

  function publish(overrides: Partial<SamplingPreviewPayload> = {}) {
    const payload: SamplingPreviewPayload = {
      node_id: '42',
      prompt_id: 'prompt-1',
      image: btoa('frame-1'),
      mime: 'image/jpeg',
      step: 3,
      total: 8,
      fps: 24,
      frame_count: 60,
      segment_index: 1,
      sampling_pass: 'second',
      ...overrides,
    }
    act(() => listeners.get(SAMPLING_PREVIEW_EVENT)?.(new CustomEvent(
      SAMPLING_PREVIEW_EVENT,
      { detail: payload },
    )))
  }

  it('shows metadata in the playback bar without playback text or percentage', () => {
    const onVisibilityChange = vi.fn()
    render(
      <SamplingPreviewWidget
        app={app}
        nodeId="42"
        onVisibilityChange={onVisibilityChange}
      />,
    )

    publish()

    expect(screen.getByAltText('Sampling preview').getAttribute('src')).toBe('blob:preview')
    expect(screen.getByText('Segment 2 · Second pass · 3/8')).not.toBeNull()
    const progress = screen.getByRole('progressbar', {
      name: 'Sampling preview playback progress',
    })
    expect(progress.getAttribute('aria-valuemax')).toBe('100')
    expect(screen.queryByText('Preview playback')).toBeNull()
    expect(screen.queryByText(/%$/)).toBeNull()
    expect(onVisibilityChange).toHaveBeenLastCalledWith(true)
  })

  it('pauses and resumes from either the image or progress bar', () => {
    render(<SamplingPreviewWidget app={app} nodeId="42" />)
    publish()

    const pauseControls = screen.getAllByRole('button', { name: 'Pause sampling preview' })
    fireEvent.click(pauseControls[0])
    expect(screen.getAllByRole('button', { name: 'Resume sampling preview' })).toHaveLength(2)

    fireEvent.click(screen.getAllByRole('button', { name: 'Resume sampling preview' })[1])
    expect(screen.getAllByRole('button', { name: 'Pause sampling preview' })).toHaveLength(2)
  })

  it('freezes an animated WebP fallback when browser frame decoding is unavailable', () => {
    const drawImage = vi.fn()
    vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ drawImage } as never)
    render(<SamplingPreviewWidget app={app} nodeId="42" />)
    publish({ mime: 'image/webp' })

    const image = screen.getByAltText('Sampling preview') as HTMLImageElement
    Object.defineProperties(image, {
      complete: { configurable: true, value: true },
      naturalWidth: { configurable: true, value: 320 },
      naturalHeight: { configurable: true, value: 180 },
    })
    fireEvent.click(screen.getAllByRole('button', { name: 'Pause sampling preview' })[0])

    const frozenFrame = screen.getByTestId('sampling-preview-frozen-frame') as HTMLCanvasElement
    expect(drawImage).toHaveBeenCalledWith(image, 0, 0)
    expect(frozenFrame.hidden).toBe(false)
    expect(frozenFrame.width).toBe(320)
    expect(frozenFrame.height).toBe(180)

    publish({ image: btoa('new-preview'), mime: 'image/webp', step: 4 })
    expect(drawImage).toHaveBeenCalledTimes(1)
    expect(frozenFrame.hidden).toBe(false)
    expect(screen.getByText('Segment 2 · Second pass · 3/8')).not.toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: 'Resume sampling preview' })[1])
    expect(frozenFrame.hidden).toBe(true)
    expect(screen.getByText('Segment 2 · Second pass · 4/8')).not.toBeNull()
  })

  it('hides the image while keeping its placeholder and show control', () => {
    render(<SamplingPreviewWidget app={app} nodeId="42" />)
    publish()

    fireEvent.click(screen.getByRole('button', { name: 'Hide sampling preview' }))
    expect(screen.queryByAltText('Sampling preview')).toBeNull()
    expect(screen.getByTestId('sampling-preview-placeholder')).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Show sampling preview' })).not.toBeNull()
    expect(screen.getByText('Segment 2 · Second pass · 3/8')).not.toBeNull()
  })

  it('ignores preview events targeting another node', () => {
    render(<SamplingPreviewWidget app={app} nodeId="42" />)

    publish({ node_id: '99' })

    expect(screen.queryByAltText('Sampling preview')).toBeNull()
  })

  it('renders a display-only preview without controls for pointer-transparent overlays', () => {
    render(<SamplingPreviewWidget app={app} nodeId="42" interactive={false} />)
    publish()

    expect(screen.getByAltText('Sampling preview')).not.toBeNull()
    expect(screen.queryByRole('button')).toBeNull()
    expect(screen.getByRole('progressbar')).not.toBeNull()
  })

  it('keeps only the visibility icon interactive in a canvas-transparent overlay', () => {
    render(
      <SamplingPreviewWidget
        app={app}
        nodeId="42"
        interactive={false}
        visibilityControlOnly
      />,
    )
    publish()

    const hideButton = screen.getByRole('button', { name: 'Hide sampling preview' })
    expect(hideButton.className).toContain('pointer-events-auto')
    expect(screen.queryByRole('button', { name: 'Pause sampling preview' })).toBeNull()
    fireEvent.click(hideButton)
    expect(screen.getByRole('button', { name: 'Show sampling preview' })).not.toBeNull()
  })

  it('reads the LiteGraph node id when the event arrives, after node creation', () => {
    let assignedNodeId = -1
    render(<SamplingPreviewWidget app={app} nodeId={() => assignedNodeId} />)
    assignedNodeId = 42

    publish()

    expect(screen.getByAltText('Sampling preview')).not.toBeNull()
  })

  it('accepts the canvas display node id used by expanded graph execution', () => {
    render(<SamplingPreviewWidget app={app} nodeId="42" />)

    publish({ node_id: '42.3.1.87', display_node_id: '42' })

    expect(screen.getByAltText('Sampling preview')).not.toBeNull()
  })

  it('uses the current ComfyUI locale', () => {
    rawApp.ui.settings.settingsValues['Comfy.Locale'] = 'zh'
    render(<SamplingPreviewWidget app={app} nodeId="42" />)

    publish()

    expect(screen.getByText('片段 2 · 二采 · 3/8')).not.toBeNull()
    expect(screen.getByRole('button', { name: '隐藏采样预览' })).not.toBeNull()
  })

  it('clears stale preview when the project starts a new execution', () => {
    const onVisibilityChange = vi.fn()
    const view = render(
      <SamplingPreviewWidget
        app={app}
        nodeId="42"
        executionRevision={0}
        onVisibilityChange={onVisibilityChange}
      />,
    )
    publish()
    expect(screen.getByAltText('Sampling preview')).not.toBeNull()

    view.rerender(
      <SamplingPreviewWidget
        app={app}
        nodeId="42"
        executionRevision={1}
        onVisibilityChange={onVisibilityChange}
      />,
    )

    expect(screen.queryByAltText('Sampling preview')).toBeNull()
    expect(onVisibilityChange).toHaveBeenLastCalledWith(false)
  })

  it('ignores queued preview frames from a previous prompt', () => {
    render(<SamplingPreviewWidget app={app} nodeId="42" />)
    act(() => listeners.get('execution_start')?.(new CustomEvent(
      'execution_start',
      { detail: { prompt_id: 'prompt-2' } },
    )))

    publish({ prompt_id: 'prompt-1' })
    expect(screen.queryByAltText('Sampling preview')).toBeNull()

    publish({ prompt_id: 'prompt-2' })
    expect(screen.getByAltText('Sampling preview')).not.toBeNull()
  })

  it('keeps the completed preview when an unrelated prompt starts', () => {
    render(<SamplingPreviewWidget app={app} nodeId="42" />)
    publish({ prompt_id: 'prompt-1' })

    act(() => listeners.get('execution_start')?.(new CustomEvent(
      'execution_start',
      { detail: { prompt_id: 'unrelated-prompt' } },
    )))

    expect(screen.getByAltText('Sampling preview')).not.toBeNull()
  })

  it('uses the requested half-frame count and does not restart on every sampling step', () => {
    let nextFrame: FrameRequestCallback | undefined
    vi.mocked(requestAnimationFrame).mockImplementation((callback) => {
      nextFrame = callback
      return 1
    })
    vi.spyOn(performance, 'now').mockReturnValue(0)
    render(<SamplingPreviewWidget app={app} nodeId="42" />)
    publish({ frame_count: 60, fps: 24 })

    act(() => nextFrame?.(1000))
    const progress = screen.getByRole('progressbar', {
      name: 'Sampling preview playback progress',
    })
    expect(progress.getAttribute('aria-valuenow')).toBe('40')

    publish({ image: btoa('new-preview'), step: 4, frame_count: 60, fps: 24 })

    expect(progress.getAttribute('aria-valuenow')).toBe('40')
  })

  it('plays every frame from the legacy backend payload until ComfyUI restarts', () => {
    let nextFrame: FrameRequestCallback | undefined
    let urlIndex = 0
    vi.mocked(requestAnimationFrame).mockImplementation((callback) => {
      nextFrame = callback
      return 1
    })
    vi.mocked(URL.createObjectURL).mockImplementation(() => `blob:preview-${++urlIndex}`)
    vi.spyOn(performance, 'now').mockReturnValue(0)
    render(<SamplingPreviewWidget app={app} nodeId="42" />)
    publish({
      images: [btoa('frame-1'), btoa('frame-2')],
      frame_count: 2,
      fps: 2,
    })

    expect(screen.getByAltText('Sampling preview').getAttribute('src')).toBe('blob:preview-2')
    act(() => nextFrame?.(600))
    expect(screen.getByAltText('Sampling preview').getAttribute('src')).toBe('blob:preview-3')
  })

  it('holds the currently displayed addressable frame while paused', () => {
    let nextFrame: FrameRequestCallback | undefined
    let urlIndex = 0
    vi.mocked(requestAnimationFrame).mockImplementation((callback) => {
      nextFrame = callback
      return 1
    })
    vi.mocked(URL.createObjectURL).mockImplementation(() => `blob:preview-${++urlIndex}`)
    vi.spyOn(performance, 'now').mockReturnValue(0)
    render(<SamplingPreviewWidget app={app} nodeId="42" />)
    publish({
      images: [btoa('frame-1'), btoa('frame-2')],
      frame_count: 2,
      fps: 2,
    })
    act(() => nextFrame?.(600))
    const pausedSource = screen.getByAltText('Sampling preview').getAttribute('src')
    expect(pausedSource).toBe('blob:preview-3')

    fireEvent.click(screen.getAllByRole('button', { name: 'Pause sampling preview' })[0])
    publish({
      image: btoa('new-preview'),
      images: [btoa('new-frame-1'), btoa('new-frame-2')],
      frame_count: 2,
      fps: 2,
      step: 4,
    })

    expect(screen.getByAltText('Sampling preview').getAttribute('src')).toBe(pausedSource)
    expect(screen.getByText('Segment 2 · Second pass · 3/8')).not.toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: 'Resume sampling preview' })[1])
    expect(screen.getByAltText('Sampling preview').getAttribute('src')).not.toBe(pausedSource)
    expect(screen.getByText('Segment 2 · Second pass · 4/8')).not.toBeNull()
  })
})
