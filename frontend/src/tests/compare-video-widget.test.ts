import { createElement } from 'react'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  CompareVideoWidget,
  mergeWatchOutputHistorySelections,
  parseCompareVideoPayload,
} from '@/components/widgets/compareVideoWidget'
import { suppressCompareVideoDefaultPreview } from '@/lib/compare-video-node'

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: () => null,
}))

const source = { filename: 'source.mp4', type: 'temp' as const }
const output = { filename: 'output.mp4', type: 'temp' as const }

function createEventApi() {
  const listeners = new Map<string, Set<(event: CustomEvent<unknown>) => void>>()
  return {
    addEventListener(type: string, callback: (event: CustomEvent<unknown>) => void) {
      const callbacks = listeners.get(type) ?? new Set()
      callbacks.add(callback)
      listeners.set(type, callbacks)
    },
    removeEventListener(type: string, callback: (event: CustomEvent<unknown>) => void) {
      listeners.get(type)?.delete(callback)
    },
    emit(type: string, detail: unknown) {
      const callbacks = listeners.get(type)
      if (!callbacks) return
      const event = { detail } as CustomEvent<unknown>
      for (const callback of callbacks) callback(event)
    },
    listenerCount(type: string) {
      return listeners.get(type)?.size ?? 0
    },
  }
}

function recentVideo(path: string, mtime: number, sourceType: 'output' | 'temp' = 'output') {
  return {
    type: 'file' as const,
    name: path.split('/').pop() ?? path,
    path,
    url: `/view?filename=${encodeURIComponent(path.split('/').pop() ?? path)}&type=${sourceType}&subfolder=`,
    size: 100,
    mtime,
    source_type: sourceType,
  }
}

describe('compare video payload parsing', () => {
  it('reads a direct compare_videos payload', () => {
    expect(parseCompareVideoPayload({
      compare_videos: { source, output, frame_count: 12 },
    })).toMatchObject({ source, output, frame_count: 12 })
  })

  it('reads a payload nested in ComfyUI executed output', () => {
    expect(parseCompareVideoPayload({
      node: 8,
      output: { compare_videos: { source, output, fps: 24 } },
    })).toMatchObject({ source, output, fps: 24 })
  })

  it('reads a payload nested in a ui object', () => {
    expect(parseCompareVideoPayload({
      ui: { compare_videos: { source, duration: 1.5 } },
    })).toMatchObject({ source, duration: 1.5 })
  })

  it('accepts list-wrapped compare_videos values', () => {
    expect(parseCompareVideoPayload({
      output: { compare_videos: [{ source, output }] },
    })).toMatchObject({ source, output })
  })
})

describe('watch output history selection merge', () => {
  it('keeps a single recent video in the output slot', () => {
    expect(mergeWatchOutputHistorySelections(
      { source: null, output: null },
      [recentVideo('renders/latest.mp4', 200)],
    )).toEqual({
      source: null,
      output: { source_type: 'output', file_path: 'renders/latest.mp4' },
    })
  })

  it('puts the previous video in source and the newest video in output', () => {
    expect(mergeWatchOutputHistorySelections(
      { source: null, output: null },
      [
        recentVideo('renders/latest.mp4', 300),
        recentVideo('renders/previous.mp4', 200),
      ],
    )).toEqual({
      source: { source_type: 'output', file_path: 'renders/previous.mp4' },
      output: { source_type: 'output', file_path: 'renders/latest.mp4' },
    })
  })

  it('shifts the previous output into source when a new video arrives', () => {
    expect(mergeWatchOutputHistorySelections(
      {
        source: { source_type: 'output', file_path: 'renders/previous.mp4' },
        output: { source_type: 'output', file_path: 'renders/older.mp4' },
      },
      [
        recentVideo('renders/latest.mp4', 400),
        recentVideo('renders/previous.mp4', 300),
        recentVideo('renders/older.mp4', 200),
      ],
    )).toEqual({
      source: { source_type: 'output', file_path: 'renders/previous.mp4' },
      output: { source_type: 'output', file_path: 'renders/latest.mp4' },
    })
  })

  it('deduplicates the same name and path and preserves a manual slot', () => {
    expect(mergeWatchOutputHistorySelections(
      {
        source: { source_type: 'output', file_path: 'renders/manual.mp4' },
        output: null,
      },
      [
        recentVideo('renders/manual.mp4', 300),
        recentVideo('renders/latest.mp4', 200),
      ],
      new Set(['source']),
    )).toEqual({
      source: { source_type: 'output', file_path: 'renders/manual.mp4' },
      output: { source_type: 'output', file_path: 'renders/latest.mp4' },
    })
  })

  it('does not add the same history video to both comparison slots', () => {
    expect(mergeWatchOutputHistorySelections(
      { source: null, output: null },
      [
        recentVideo('renders/same.mp4', 300),
        recentVideo('renders/same.mp4', 200),
      ],
    )).toEqual({
      source: null,
      output: { source_type: 'output', file_path: 'renders/same.mp4' },
    })
  })
})

describe('compare video node preview', () => {
  it('hides only the default ComfyUI output preview', () => {
    const originalDraw = vi.fn()
    const compareNodeType = { prototype: { onDrawBackground: originalDraw } }
    const otherNodeType = { prototype: {} }

    suppressCompareVideoDefaultPreview(compareNodeType, { name: 'easy compareVideos' })
    suppressCompareVideoDefaultPreview(otherNodeType, { name: 'easy saveVideo' })

    expect(compareNodeType.prototype).toMatchObject({ hideOutputImages: true })
    expect(otherNodeType.prototype).not.toHaveProperty('hideOutputImages')

    const defaultPreview = { name: 'video-preview' }
    const compareWidget = { name: 'compare_video' }
    const videoContainer = document.createElement('div')
    videoContainer.append(document.createElement('video'))
    const node = Object.assign(Object.create(compareNodeType.prototype), {
      widgets: [compareWidget, defaultPreview],
      videoContainer,
      imgs: [document.createElement('video')],
      removeWidget(widget: { name?: string }) {
        this.widgets = this.widgets.filter((item: { name?: string }) => item !== widget)
      },
    })

    node.onDrawBackground()

    expect(originalDraw).not.toHaveBeenCalled()
    expect(node.widgets).toEqual([compareWidget])
    expect(videoContainer.childElementCount).toBe(0)
    expect(node.videoContainer).toBeUndefined()
    expect(node.imgs).toBeUndefined()
  })
})

describe('CompareVideoWidget', () => {
  beforeEach(() => {
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
  })
  function widgetProps(
    node: object,
    onChange = vi.fn(),
    value: {
      save_output: boolean
      filename_prefix: string
      watch_output_history: boolean
      compare_mode?: 'source' | 'compare' | 'side-by-side' | 'output'
      source?: { source_type: 'input' | 'output' | 'temp' | 'url'; file_path?: string; url?: string } | null
      output?: { source_type: 'input' | 'output' | 'temp' | 'url'; file_path?: string; url?: string } | null
    } = { save_output: false, filename_prefix: 'ComfyUI', watch_output_history: false },
    api: ReturnType<typeof createEventApi> = createEventApi(),
  ) {
    return {
      app: {
        api,
        canvas: { ds: { scale: 1 } },
        ui: { settings: { settingsValues: {} } },
      } as never,
      node,
      value,
      onChange,
      inputName: 'compare_video',
      widget: {} as never,
    }
  }

  it('starts with audio enabled and uses output audio for comparison', () => {
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      unobserve() {}
      disconnect() {}
    })
    const node = {
      id: 1,
      __easyMediaCompareVideos: { source, output },
    }
    render(createElement(CompareVideoWidget, widgetProps(node)))

    const videos = document.querySelectorAll('video')
    expect(videos).toHaveLength(2)
    expect(videos[0].muted).toBe(true)
    expect(videos[1].muted).toBe(false)
    expect(screen.getByRole('button', { name: 'Mute audio' })).not.toBeNull()
  })

  it('shows source and output in equal-width A/B panels when both videos are available', () => {
    render(createElement(CompareVideoWidget, widgetProps({
      id: 9,
      __easyMediaCompareVideos: { source, output },
    })))

    fireEvent.click(screen.getByRole('button', { name: 'Compare' }))

    expect(screen.getByRole('button', { name: 'A/B' })).not.toBeNull()
    const sourcePanel = document.querySelector('[data-compare-video-panel="source"]')
    const outputPanel = document.querySelector('[data-compare-video-panel="output"]')
    expect(sourcePanel?.className).toContain('w-1/2')
    expect(sourcePanel?.className).toContain('right-auto')
    expect(outputPanel?.className).toContain('w-1/2')
    expect(outputPanel?.className).toContain('left-1/2')
    expect((outputPanel as HTMLVideoElement | null)?.style.clipPath).toBe('')
  })

  it('does not offer A/B mode unless both videos are available', () => {
    render(createElement(CompareVideoWidget, widgetProps({
      id: 10,
      __easyMediaCompareVideos: { source },
    })))

    expect(screen.queryByRole('button', { name: 'A/B' })).toBeNull()
    expect(document.querySelector('[data-compare-video-panel]')).toBeNull()
  })

  it('shows source and output media pickers before any videos are available', () => {
    render(createElement(CompareVideoWidget, widgetProps({ id: 11 })))

    expect(screen.getByRole('button', { name: 'Select source' })).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Select output' })).not.toBeNull()
  })

  it('edits save settings in the empty state', () => {
    const onChange = vi.fn()
    render(createElement(CompareVideoWidget, widgetProps({ id: 2 }, onChange)))

    const saveOutputCheckbox = screen.getByRole('checkbox', { name: 'Save output video' })
    expect(saveOutputCheckbox.getAttribute('aria-checked')).toBe('false')
    fireEvent.click(saveOutputCheckbox)
    expect(onChange).toHaveBeenCalledWith({
      compare_mode: 'compare',
      save_output: true,
      filename_prefix: 'ComfyUI',
      watch_output_history: false,
      source: null,
      output: null,
    })

    fireEvent.change(screen.getByRole('textbox', { name: 'Save prefix' }), { target: { value: 'renders/final' } })
    expect(onChange).toHaveBeenCalledWith({
      compare_mode: 'compare',
      save_output: false,
      filename_prefix: 'renders/final',
      watch_output_history: false,
      source: null,
      output: null,
    })
  })

  it('turns off save output and starts watching output history when watch mode is enabled', () => {
    const onChange = vi.fn()
    render(createElement(CompareVideoWidget, widgetProps({ id: 12 }, onChange)))

    fireEvent.click(screen.getByRole('checkbox', { name: 'Watch output history' }))

    expect(onChange).toHaveBeenCalledWith({
      compare_mode: 'compare',
      save_output: false,
      filename_prefix: 'ComfyUI',
      watch_output_history: true,
      source: null,
      output: null,
    })
  })

  it('disables save output while watch output history is enabled', () => {
    render(createElement(CompareVideoWidget, widgetProps(
      { id: 13 },
      vi.fn(),
      {
        save_output: false,
        filename_prefix: 'ComfyUI',
        watch_output_history: true,
      },
    )))

    const saveOutputCheckbox = screen.getByRole('checkbox', { name: 'Save output video' })
    expect((saveOutputCheckbox as HTMLButtonElement).disabled).toBe(true)
  })

  it('clears previously selected videos when switching into watch output history mode', () => {
    const onChange = vi.fn()
    render(createElement(CompareVideoWidget, widgetProps(
      { id: 17 },
      onChange,
      {
        save_output: true,
        filename_prefix: 'ComfyUI',
        watch_output_history: false,
        source: { source_type: 'input', file_path: 'source.mp4' },
        output: { source_type: 'output', file_path: 'output.mp4' },
      },
    )))

    fireEvent.click(screen.getByRole('button', { name: 'Output settings' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'Watch output history' }))

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      compare_mode: 'compare',
      save_output: false,
      watch_output_history: true,
      source: null,
      output: null,
    }))
  })

  it('allows clearing a displayed slot while watching output history', () => {
    const onChange = vi.fn()
    render(createElement(CompareVideoWidget, widgetProps(
      { id: 18 },
      onChange,
      {
        save_output: false,
        filename_prefix: 'ComfyUI',
        watch_output_history: true,
        source: { source_type: 'output', file_path: 'renders/source.mp4' },
        output: { source_type: 'output', file_path: 'renders/output.mp4' },
      },
    )))

    const clearSource = screen.getByRole('button', { name: 'Clear source video' })
    expect((clearSource as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(clearSource)

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      compare_mode: 'compare',
      watch_output_history: true,
      source: null,
      output: { source_type: 'output', file_path: 'renders/output.mp4' },
    }))
  })

  it('does not populate watch output history when first enabled', () => {
    const onChange = vi.fn()
    const api = createEventApi()

    render(createElement(CompareVideoWidget, widgetProps(
      { id: 20 },
      onChange,
      {
        save_output: false,
        filename_prefix: 'ComfyUI',
        watch_output_history: true,
      },
      api,
    )))

    expect(api.listenerCount('executed')).toBeGreaterThan(0)
    expect(api.listenerCount('execution_success')).toBeGreaterThan(0)
    expect(onChange).not.toHaveBeenCalled()
    expect(document.querySelectorAll('video')).toHaveLength(0)
  })

  it('clears stale executed payload when leaving watch output history mode with empty slots', () => {
    const node = {
      id: 22,
      __easyMediaCompareVideos: { source, output },
    }
    const onChange = vi.fn()
    const { rerender } = render(createElement(CompareVideoWidget, widgetProps(
      node,
      onChange,
      {
        save_output: false,
        filename_prefix: 'ComfyUI',
        watch_output_history: true,
        source: null,
        output: null,
      },
    )))

    expect(document.querySelectorAll('video')).toHaveLength(0)

    rerender(createElement(CompareVideoWidget, widgetProps(
      node,
      onChange,
      {
        save_output: false,
        filename_prefix: 'ComfyUI',
        watch_output_history: false,
        source: null,
        output: null,
      },
    )))

    expect(document.querySelectorAll('video')).toHaveLength(0)
  })

  it('fills the previous output into source and newest output into output after a successful execution', async () => {
    const onChange = vi.fn()
    const api = createEventApi()

    render(createElement(CompareVideoWidget, widgetProps(
      { id: 14 },
      onChange,
      {
        save_output: false,
        filename_prefix: 'ComfyUI',
        watch_output_history: true,
      },
      api,
    )))

    await waitFor(() => expect(api.listenerCount('execution_success')).toBeGreaterThan(0))
    api.emit('execution_start', {})
    api.emit('executed', {
      node: 1,
      output: {
        video: [
          { filename: 'previous.mp4', subfolder: 'renders', type: 'output' },
          { filename: 'latest.mp4', subfolder: 'renders', type: 'output' },
        ],
      },
    })
    api.emit('execution_success', {})

    await waitFor(() => expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      source: { source_type: 'output', file_path: 'renders/previous.mp4' },
      output: { source_type: 'output', file_path: 'renders/latest.mp4' },
    })))
  })

  it('puts a single successful output into the output slot while watching', async () => {
    const onChange = vi.fn()
    const api = createEventApi()

    render(createElement(CompareVideoWidget, widgetProps(
      { id: 15 },
      onChange,
      {
        save_output: false,
        filename_prefix: 'ComfyUI',
        watch_output_history: true,
      },
      api,
    )))

    await waitFor(() => expect(api.listenerCount('execution_success')).toBeGreaterThan(0))
    api.emit('execution_start', {})
    api.emit('executed', {
      node: 2,
      output: { video: [{ filename: 'only.mp4', subfolder: '', type: 'output' }] },
    })
    api.emit('execution_success', {})

    await waitFor(() => expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      source: null,
      output: { source_type: 'output', file_path: 'only.mp4' },
    })))
  })

  it('accepts temp preview videos from the successful execution while watching', async () => {
    const onChange = vi.fn()
    const api = createEventApi()

    render(createElement(CompareVideoWidget, widgetProps(
      { id: 16 },
      onChange,
      {
        save_output: false,
        filename_prefix: 'ComfyUI',
        watch_output_history: true,
      },
      api,
    )))

    await waitFor(() => expect(api.listenerCount('execution_success')).toBeGreaterThan(0))
    api.emit('execution_start', {})
    api.emit('executed', {
      node: 3,
      output: { images: [{ filename: 'preview_output_abc.mp4', subfolder: '', type: 'temp' }] },
    })
    api.emit('execution_success', {})

    await waitFor(() => expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      source: null,
      output: { source_type: 'temp', file_path: 'preview_output_abc.mp4' },
    })))
  })

  it('preserves the selected compare mode after a successful execution', async () => {
    const onChange = vi.fn()
    const api = createEventApi()

    render(createElement(CompareVideoWidget, widgetProps(
      { id: 19 },
      onChange,
      {
        save_output: false,
        filename_prefix: 'ComfyUI',
        watch_output_history: true,
        compare_mode: 'side-by-side',
      },
      api,
    )))

    await waitFor(() => expect(api.listenerCount('execution_success')).toBeGreaterThan(0))
    api.emit('execution_start', {})
    api.emit('executed', {
      node: 4,
      output: {
        video: [
          { filename: 'previous.mp4', subfolder: 'renders', type: 'output' },
          { filename: 'latest.mp4', subfolder: 'renders', type: 'output' },
        ],
      },
    })
    api.emit('execution_success', {})

    await waitFor(() => expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      compare_mode: 'side-by-side',
      source: { source_type: 'output', file_path: 'renders/previous.mp4' },
      output: { source_type: 'output', file_path: 'renders/latest.mp4' },
    })))
  })

  it('shows output settings and download controls after preview', () => {
    const onChange = vi.fn()
    let clickedAnchor: HTMLAnchorElement | null = null
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function captureAnchor(this: HTMLAnchorElement) {
      clickedAnchor = this
    })
    render(createElement(CompareVideoWidget, widgetProps({
      id: 3,
      __easyMediaCompareVideos: { source, output },
    }, onChange)))

    expect(screen.getByRole('button', { name: 'Download output video' })).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Output settings' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Save prefix' }), { target: { value: 'compare' } })
    expect(onChange).toHaveBeenCalledWith({
      compare_mode: 'compare',
      save_output: false,
      filename_prefix: 'compare',
      watch_output_history: false,
      source: null,
      output: null,
    })

    fireEvent.click(screen.getByRole('button', { name: 'Download output video' }))
    expect(anchorClick).toHaveBeenCalledOnce()
    expect(clickedAnchor).not.toBeNull()
    expect((clickedAnchor as HTMLAnchorElement | null)?.download).toBe('output.mp4')
    expect((clickedAnchor as HTMLAnchorElement | null)?.href).toContain('/view?filename=output.mp4')
    anchorClick.mockRestore()
  })

  it('pins the toolbar to the bottom and reveals it on hover or keyboard focus', () => {
    render(createElement(CompareVideoWidget, widgetProps({
      id: 5,
      __easyMediaCompareVideos: { source, output },
    })))

    const toolbar = document.querySelector('[data-compare-video-toolbar]')
    expect(toolbar).not.toBeNull()
    expect(toolbar?.className).toContain('absolute')
    expect(toolbar?.className).toContain('bottom-0')
    expect(toolbar?.className).toContain('opacity-0')
    expect(toolbar?.className).toContain('transition-[opacity,transform]')
    expect(toolbar?.className).toContain('group-hover:opacity-100')
    expect(toolbar?.className).toContain('group-focus-within:opacity-100')
  })

  it('switches from one row to two rows below 420px', () => {
    render(createElement(CompareVideoWidget, widgetProps({
      id: 8,
      __easyMediaCompareVideos: { source, output, duration: 10 },
    })))

    const toolbar = document.querySelector('[data-compare-video-toolbar]')
    const seekRow = document.querySelector('[data-compare-video-seek]')
    const playbackControls = document.querySelector('[data-compare-video-playback]')
    const actionsRow = document.querySelector('[data-compare-video-actions]')
    const volumeSlider = document.querySelector('[data-compare-video-volume]')

    expect(toolbar?.className).toContain('@container/compare-video-controls')
    expect(toolbar?.className).toContain('grid-cols-[auto_minmax(0,1fr)_auto]')
    expect(toolbar?.className).toContain('@max-[420px]/compare-video-controls:grid-cols-[minmax(0,1fr)_auto]')
    expect(seekRow?.className).toContain('grid-cols-[auto_minmax(4rem,1fr)_auto]')
    expect(seekRow?.className).toContain('@max-[420px]/compare-video-controls:col-span-2')
    expect(seekRow?.className).toContain('@max-[420px]/compare-video-controls:row-start-1')
    expect(playbackControls?.className).toContain('@max-[420px]/compare-video-controls:row-start-2')
    expect(actionsRow?.className).toContain('@max-[420px]/compare-video-controls:row-start-2')
    expect(volumeSlider?.className).not.toContain('@max-[420px]/compare-video-controls:hidden')
    expect(screen.getByRole('slider', { name: 'Video time' })).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Mute audio' })).not.toBeNull()
  })

  it('uses output metadata duration when source and output durations differ', () => {
    render(createElement(CompareVideoWidget, widgetProps({
      id: 6,
      __easyMediaCompareVideos: { source, output, duration: 2 },
    })))

    const videos = document.querySelectorAll('video')
    Object.defineProperty(videos[0], 'duration', { configurable: true, value: 10 })
    Object.defineProperty(videos[1], 'duration', { configurable: true, value: 2 })
    fireEvent.loadedMetadata(videos[0])
    fireEvent.loadedMetadata(videos[1])

    expect(screen.getByText('0:02')).not.toBeNull()
    expect(screen.queryByText('0:10')).toBeNull()
  })

  it('uses output playback time as the comparison clock', () => {
    render(createElement(CompareVideoWidget, widgetProps({
      id: 7,
      __easyMediaCompareVideos: { source, output, duration: 10 },
    })))

    const videos = document.querySelectorAll('video')
    Object.defineProperty(videos[0], 'currentTime', { configurable: true, writable: true, value: 8 })
    Object.defineProperty(videos[1], 'currentTime', { configurable: true, writable: true, value: 1 })
    fireEvent.timeUpdate(videos[0])

    expect(screen.queryByText('0:08')).toBeNull()
    fireEvent.timeUpdate(videos[1])
    expect(screen.getByText('0:01')).not.toBeNull()
  })

  it('hides the download control when no output video is connected', () => {
    render(createElement(CompareVideoWidget, widgetProps({
      id: 4,
      __easyMediaCompareVideos: { source },
    })))

    expect(screen.queryByRole('button', { name: 'Download output video' })).toBeNull()
  })
})
