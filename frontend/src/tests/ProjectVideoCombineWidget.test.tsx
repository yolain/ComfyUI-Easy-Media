import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ProjectVideoCombineWidget } from '@/components/widgets/ProjectVideoCombineWidget'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import type { ProjectData } from '@/types/project'

vi.mock('@/hooks/use-canvas-scale', () => ({ useCanvasScale: () => 1 }))
vi.mock('@/hooks/use-element-width', () => ({ useElementWidth: () => 480 }))
vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({ MediaSelector: () => null }))

const projectData: ProjectData = {
  project_name: 'demo',
  width: 1280,
  height: 720,
  frame_rate: 24,
  auto_combine: true,
  clips: [{
    id: 'segment-0',
    index: 0,
    file_path: 'easy_media/projects/demo/video_0_1.mp4',
    file_name: 'video_0_1.mp4',
    media_revision: '1000000001',
    source_start_frame: 0,
    source_end_frame: 120,
    source_frame_count: 120,
    continuity_mode: 'shot',
    enabled: true,
  }],
}

function widgetProps(overrides: Partial<ReactWidgetProps<ProjectData>> = {}) {
  const toast = { add: vi.fn() }
  const dialog = { confirm: vi.fn().mockResolvedValue(true) }
  const workflow = {
    activeWorkflow: { key: 'workflow-a' },
    $subscribe: vi.fn().mockReturnValue(vi.fn()),
  }
  const api = {
    fetchApi: vi.fn(),
    getQueue: vi.fn().mockResolvedValue({ Running: [], Pending: [] }),
    queuePrompt: vi.fn().mockResolvedValue(undefined),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addCustomEventListener: vi.fn(),
    removeCustomEventListener: vi.fn(),
  }
  const node = {
    id: 8,
    graph: { links: { 7: { origin_id: 3, target_id: 8 }, 9: { origin_id: 8, target_id: 10 } } },
  }
  const graphToPrompt = vi.fn().mockResolvedValue({
    workflow: { nodes: [] },
    output: {
      '3': { inputs: { project_name: 'demo' }, class_type: 'easy multitrackProject' },
      '8': {
        inputs: { project_name: ['3', 0], project_data: JSON.stringify(projectData) },
        class_type: 'easy multitrackProjectVideoCombine',
      },
      '10': {
        inputs: { video: ['8', 0], filename_prefix: ['8', 1] },
        class_type: 'easy saveVideo',
      },
    },
  })
  const props = {
    value: projectData,
    onChange: vi.fn(),
    inputName: 'project_data',
    node,
    widget: {},
    app: {
      api,
      graphToPrompt,
      ui: { settings: { settingsValues: { 'Comfy.Locale': 'en' } } },
      extensionManager: { toast, dialog, workflow },
    },
    ...overrides,
  } as unknown as ReactWidgetProps<ProjectData>
  return { props, api, toast, dialog, workflow, graphToPrompt }
}

beforeEach(() => {
  Object.defineProperty(document, 'visibilityState', {
    configurable: true,
    value: 'visible',
  })
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
})

describe('ProjectVideoCombineWidget', () => {
  function renderComparison() {
    let animationFrame: FrameRequestCallback | undefined
    vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => {
      animationFrame = callback
      return 1
    }))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
    const clip = {
      ...projectData.clips[0],
      source_start_frame: 24,
      video_files: [projectData.clips[0], {
        file_path: 'easy_media/projects/demo/video_0_2.mp4',
        file_name: 'video_0_2.mp4',
        source_frame_count: 72,
      }],
    }
    const second = {
      ...projectData.clips[0], id: 'segment-1', index: 1,
      file_path: 'easy_media/projects/demo/video_1_1.mp4', file_name: 'video_1_1.mp4',
    }
    const { props } = widgetProps({ value: { ...projectData, clips: [clip, second] } })
    const result = render(<ProjectVideoCombineWidget {...props} />)
    const trigger = screen.getByRole('button', { name: 'Select up to two videos for segment 1' })
    fireEvent.click(trigger)
    fireEvent.click(screen.getByRole('checkbox', { name: 'video_0_2.mp4' }))
    fireEvent.click(trigger)
    const child = within(result.container.querySelector('[data-compare-video-playback]') as HTMLElement)
    const mainPlay = screen.getAllByRole('button', { name: 'Play' }).find((button) => !button.closest('[data-compare-video-playback]')) as HTMLButtonElement
    return { ...result, child, mainPlay, tick: (time: number) => act(() => animationFrame?.(time)) }
  }

  it('plays comparisons from the project trim and continues into the next segment', () => {
    const { container, child, mainPlay, tick } = renderComparison()
    const videos = [...container.querySelectorAll('video')]
    expect(mainPlay.disabled).toBe(false)
    expect(videos[0].currentTime).toBe(1)
    expect(videos[1].currentTime).toBe(1)
    fireEvent.click(mainPlay)
    const started = performance.now()
    expect(child.getByRole('button', { name: 'Pause' })).not.toBeNull()
    expect(videos.every((video) => !video.loop)).toBe(true)
    tick(started + 2000)
    expect(videos[0].currentTime).toBeCloseTo(3, 0)
    expect(videos[1].currentTime).toBeCloseTo(3, 0)

    const pause = vi.spyOn(videos[0], 'pause')
    tick(started + 4200)
    expect(pause).toHaveBeenCalled()
    expect(container.querySelector('[data-compare-video-playback]')).toBeNull()
    const next = container.querySelector('video')!
    expect(next.src).toContain('video_1_1.mp4')
    expect(vi.mocked(next.play).mock.contexts).toContain(next)
    expect(screen.getByRole('button', { name: 'Pause' })).not.toBeNull()
  })

  it('keeps local comparison playback independent while the project is paused', () => {
    const { container, child, mainPlay } = renderComparison()
    const videos = [...container.querySelectorAll('video')]
    fireEvent.click(child.getByRole('button', { name: 'Play' }))
    expect(mainPlay.getAttribute('aria-label')).toBe('Play')
    videos[1].currentTime = 2
    fireEvent.timeUpdate(videos[1])
    expect(videos[0].currentTime).toBe(2)
    expect(container.querySelector('.text-gradient')?.textContent).toBe('00:00:00')
    fireEvent.ended(videos[1])
    expect(container.querySelector('[data-compare-video-playback]')).not.toBeNull()
    expect(mainPlay.getAttribute('aria-label')).toBe('Play')
  })

  it('does not restart a shorter comparison video or rewind the parent timeline when it ends', () => {
    const { container, mainPlay, tick } = renderComparison()
    const videos = [...container.querySelectorAll('video')]
    Object.defineProperty(videos[1], 'duration', { configurable: true, value: 3 })
    fireEvent.click(mainPlay)
    const started = performance.now()
    tick(started + 2500)
    expect(videos[1].currentTime).toBe(3)
    const play = vi.spyOn(videos[1], 'play')
    play.mockClear()
    fireEvent.ended(videos[1])
    expect(play).not.toHaveBeenCalled()
    expect(videos[0].currentTime).toBeGreaterThan(3)
    expect(mainPlay.getAttribute('aria-label')).toBe('Pause')
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Video time' }), { key: 'Home' })
    expect(videos[1].currentTime).toBe(1)
    expect(play).toHaveBeenCalled()
    tick(started + 4200)
    expect(container.querySelector('video')?.src).toContain('video_1_1.mp4')
  })

  it('pauses both clocks from the child during parent playback, then allows local playback', () => {
    const { child, mainPlay } = renderComparison()
    fireEvent.click(mainPlay)
    fireEvent.click(child.getByRole('button', { name: 'Pause' }))
    expect(mainPlay.getAttribute('aria-label')).toBe('Play')
    fireEvent.click(child.getByRole('button', { name: 'Play' }))
    expect(mainPlay.getAttribute('aria-label')).toBe('Play')
    fireEvent.click(mainPlay)
    fireEvent.click(mainPlay)
    expect(child.getByRole('button', { name: 'Play' })).not.toBeNull()
  })

  it('refreshes the selected project and clears the stale preview first', async () => {
    const { props, api } = widgetProps({ value: { ...projectData, clips: [] } })
    api.fetchApi.mockResolvedValue({ ok: true, json: async () => projectData })

    render(<ProjectVideoCombineWidget {...props} />)

    await waitFor(() => expect(props.onChange).toHaveBeenLastCalledWith(projectData))
    expect(api.fetchApi).toHaveBeenCalledWith('/easy-media/project?project_name=demo')
  })

  it('shows an externally assigned project name before the project list is loaded', () => {
    const { props } = widgetProps({ value: { ...projectData, project_name: 'assigned-project' } })

    render(<ProjectVideoCombineWidget {...props} />)

    expect(screen.getByRole('combobox', { name: 'Select project' }).textContent).toContain('assigned-project')
  })

  it('treats a missing default project manifest as an empty project without an error toast', async () => {
    const defaultData = { ...projectData, project_name: 'default' }
    const { props, api, toast } = widgetProps({ value: defaultData })
    api.fetchApi.mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ error: 'H3 project manifest was not found' }),
    })

    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Refresh project' }))

    await waitFor(() => expect(api.fetchApi).toHaveBeenCalledWith(
      '/easy-media/project?project_name=default',
    ))
    await waitFor(() => expect(props.onChange).toHaveBeenCalledWith({
      project_name: 'default',
      width: 0,
      height: 0,
      frame_rate: 24,
      clips: [],
      auto_combine: true,
    }))
    expect(toast.add).not.toHaveBeenCalled()
  })

  it('still reports a missing manifest for a named project', async () => {
    const { props, api, toast } = widgetProps()
    api.fetchApi.mockResolvedValue({
      ok: false,
      status: 404,
      json: async () => ({ error: 'H3 project manifest was not found' }),
    })

    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Refresh project' }))

    await waitFor(() => expect(toast.add).toHaveBeenCalledWith(expect.objectContaining({
      severity: 'error',
      summary: 'Project refresh failed',
    })))
  })

  it('confirms and deletes the selected project through the ComfyUI API', async () => {
    const { props, api, toast, dialog } = widgetProps()
    api.fetchApi.mockResolvedValue({
      ok: true,
      json: async () => ({ project_name: 'demo', deleted: true }),
    })

    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Delete project' }))

    await waitFor(() => expect(dialog.confirm).toHaveBeenCalledWith({
      title: 'Delete project?',
      message: expect.stringContaining('demo'),
    }))
    await waitFor(() => expect(api.fetchApi).toHaveBeenCalledWith(
      '/easy-media/project?project_name=demo',
      { method: 'DELETE' },
    ))
    expect(props.onChange).toHaveBeenCalledWith({
      project_name: 'default',
      width: 0,
      height: 0,
      frame_rate: 24,
      clips: [],
      auto_combine: true,
    })
    expect(toast.add).toHaveBeenCalledWith(expect.objectContaining({
      severity: 'success',
      summary: 'Project deleted',
    }))
  })

  it('does not call the delete API when confirmation is declined', async () => {
    const { props, api, dialog } = widgetProps()
    dialog.confirm.mockResolvedValue(false)

    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Delete project' }))

    await waitFor(() => expect(dialog.confirm).toHaveBeenCalledTimes(1))
    expect(api.fetchApi).not.toHaveBeenCalled()
  })

  it('explains that deleting default clears its files but keeps the folder', async () => {
    const defaultData = { ...projectData, project_name: 'default' }
    const { props, api, dialog } = widgetProps({ value: defaultData })
    api.fetchApi.mockResolvedValue({
      ok: true,
      json: async () => ({ project_name: 'default', deleted: true }),
    })

    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Delete project' }))

    await waitFor(() => expect(dialog.confirm).toHaveBeenCalledWith(expect.objectContaining({
      message: expect.stringContaining('default folder will be kept'),
    })))
    await waitFor(() => expect(api.fetchApi).toHaveBeenCalledWith(
      '/easy-media/project?project_name=default',
      { method: 'DELETE' },
    ))
  })

  it('keeps the current preview during same-project refresh and updates its revision', async () => {
    let successHandler: (() => void) | undefined
    const refreshedData: ProjectData = {
      ...projectData,
      clips: [{ ...projectData.clips[0], media_revision: '1000000002' }],
    }
    const { props, api } = widgetProps()
    api.addEventListener.mockImplementation((name: string, handler: () => void) => {
      if (name === 'execution_success') successHandler = handler
    })
    api.fetchApi.mockResolvedValue({ ok: true, json: async () => refreshedData })

    const { container } = render(<ProjectVideoCombineWidget {...props} />)
    expect(container.querySelector('video')?.getAttribute('src')).toContain('v=1000000001')

    await act(async () => successHandler?.())

    expect(props.onChange).not.toHaveBeenCalledWith(expect.objectContaining({ clips: [] }))
    await waitFor(() => expect(props.onChange).toHaveBeenCalledWith(refreshedData))
  })

  it('switches to the project from a sampling refresh event', async () => {
    let refreshHandler: ((event: CustomEvent<unknown>) => void) | undefined
    const refreshedData: ProjectData = {
      ...projectData,
      project_name: 'next-project',
      clips: [{ ...projectData.clips[0], media_revision: '1000000002' }],
    }
    const { props, api } = widgetProps()
    api.addCustomEventListener.mockImplementation((name: string, handler: () => void) => {
      if (name === 'easy_multitrack_project_refresh') refreshHandler = handler
    })
    api.fetchApi.mockResolvedValue({ ok: true, json: async () => refreshedData })

    render(<ProjectVideoCombineWidget {...props} />)
    await act(async () => refreshHandler?.(new CustomEvent('easy_multitrack_project_refresh', {
      detail: { project_name: 'next-project', phase: 'before', segment_index: 0 },
    })))

    expect(api.fetchApi).toHaveBeenCalledWith('/easy-media/project?project_name=next-project')
    expect(props.onChange).toHaveBeenCalledWith(expect.objectContaining({
      project_name: 'next-project',
      clips: [],
    }))
    await waitFor(() => expect(props.onChange).toHaveBeenCalledWith(refreshedData))
  })

  it('does not queue manual combine while ComfyUI has queued work', async () => {
    const { props, api, toast } = widgetProps()
    api.getQueue.mockResolvedValue({ Running: [{ id: 'running' }], Pending: [] })

    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Combine' }))

    await waitFor(() => expect(toast.add).toHaveBeenCalledWith(expect.objectContaining({
      severity: 'error',
      summary: 'Project combine failed',
    })))
    expect(api.queuePrompt).not.toHaveBeenCalled()
  })

  it('queues only the combine node and downstream save nodes', async () => {
    const { props, api, graphToPrompt } = widgetProps()

    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Combine' }))

    await waitFor(() => expect(api.queuePrompt).toHaveBeenCalledTimes(1))
    expect(graphToPrompt).toHaveBeenCalledTimes(1)
    const queued = api.queuePrompt.mock.calls[0][1]
    expect(Object.keys(queued.output)).toEqual(['8', '10'])
    expect(JSON.parse(queued.output['8'].inputs.project_data)).toEqual(projectData)
    expect(queued.output['10'].inputs).toEqual({ video: ['8', 0], filename_prefix: ['8', 1] })
  })

  it('queues downstream nodes after a reloaded graph restores links as a Map', async () => {
    const { props, api } = widgetProps()
    const reloadedNode = {
      id: 8,
      graph: {
        links: new Map([
          [7, { origin_id: 3, target_id: 8 }],
          [9, { origin_id: 8, target_id: 10 }],
        ]),
      },
    }

    render(<ProjectVideoCombineWidget {...props} node={reloadedNode} />)
    fireEvent.click(screen.getByRole('button', { name: 'Combine' }))

    await waitFor(() => expect(api.queuePrompt).toHaveBeenCalledTimes(1))
    expect(Object.keys(api.queuePrompt.mock.calls[0][1].output)).toEqual(['8', '10'])
  })

  it('removes stale external input links omitted from the API prompt after reload', async () => {
    const { props, api, graphToPrompt } = widgetProps()
    const reloadedNode = {
      id: 8,
      graph: {
        links: new Map([
          [7, { origin_id: 3, target_id: 8 }],
          [9, { origin_id: 8, target_id: 10 }],
          [11, { origin_id: 11, target_id: 10 }],
        ]),
        _nodes: [{ id: 3 }, { id: 8 }, { id: 10 }, { id: 11 }],
      },
    }
    graphToPrompt.mockResolvedValue({
      workflow: { nodes: [{ id: 3 }, { id: 8 }, { id: 10 }, { id: 11 }] },
      output: {
        '8': {
          inputs: { project_name: ['3', 0], project_data: JSON.stringify(projectData) },
          class_type: 'easy multitrackProjectVideoCombine',
        },
        '10': {
          inputs: {
            input_mode: {
              input_mode: 'video',
              video: ['8', 0],
              audio: ['11', 0],
            },
            filename_prefix: ['8', 1],
          },
          class_type: 'easy saveVideo',
        },
      },
    })

    render(<ProjectVideoCombineWidget {...props} node={reloadedNode} />)
    fireEvent.click(screen.getByRole('button', { name: 'Combine' }))

    await waitFor(() => expect(api.queuePrompt).toHaveBeenCalledTimes(1))
    expect(api.queuePrompt.mock.calls[0][1].output['10'].inputs).toEqual({
      input_mode: { input_mode: 'video', video: ['8', 0] },
      filename_prefix: ['8', 1],
    })
  })

  it('serializes the auto combine checkbox through project data', () => {
    const { props } = widgetProps()
    render(<ProjectVideoCombineWidget {...props} />)

    fireEvent.click(screen.getByRole('checkbox', { name: 'Auto combine' }))

    expect(props.onChange).toHaveBeenCalledWith({ ...projectData, auto_combine: false })
  })

  it('exposes project playback to the shared sync play action', async () => {
    const { props } = widgetProps()
    render(<ProjectVideoCombineWidget {...props} />)

    await act(async () => {
      await (props.node as typeof props.node & { __easyMediaSyncPlay?: (startAt: number) => void })
        .__easyMediaSyncPlay?.(performance.now())
    })

    expect(screen.getByRole('button', { name: 'Pause' })).not.toBeNull()
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalled()
  })

  it('toggles preview audio without changing project data', () => {
    const { props } = widgetProps()
    const { container } = render(<ProjectVideoCombineWidget {...props} />)
    const video = container.querySelector('video') as HTMLVideoElement

    expect(video.muted).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Mute preview audio' }))

    expect(video.muted).toBe(true)
    expect(screen.getByRole('button', { name: 'Unmute preview audio' })).not.toBeNull()
    expect(props.onChange).not.toHaveBeenCalled()
  })

  it('accepts the shared sync play mute decision', async () => {
    const { props } = widgetProps()
    const { container } = render(<ProjectVideoCombineWidget {...props} />)

    await act(async () => {
      await (props.node as typeof props.node & {
        __easyMediaSyncPlay?: (startAt: number, muted?: boolean) => void
      }).__easyMediaSyncPlay?.(performance.now(), true)
    })

    expect((container.querySelector('video') as HTMLVideoElement).muted).toBe(true)
    expect(screen.getByRole('button', { name: 'Unmute preview audio' })).not.toBeNull()
  })

  it('seeks the preview video when the playhead moves within the same clip', () => {
    const { props } = widgetProps()
    const { container } = render(<ProjectVideoCombineWidget {...props} />)
    const video = container.querySelector('video') as HTMLVideoElement
    const ruler = container.querySelector('.cursor-col-resize') as HTMLDivElement
    vi.spyOn(ruler, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      width: 480,
      right: 480,
      top: 0,
      bottom: 24,
      height: 24,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    })

    fireEvent.mouseDown(ruler, { clientX: 230 })

    expect(video.currentTime).toBeGreaterThan(2)
    fireEvent.mouseUp(window)
  })

  it('keeps the preloaded next video element mounted when crossing a clip boundary', () => {
    const secondClip = {
      ...projectData.clips[0],
      id: 'segment-1',
      index: 1,
      file_path: 'easy_media/projects/demo/video_1_1.mp4',
      file_name: 'video_1_1.mp4',
      media_revision: '1000000002',
      source_start_frame: 24,
    }
    const { props } = widgetProps({
      value: { ...projectData, clips: [projectData.clips[0], secondClip] },
    })
    const { container } = render(<ProjectVideoCombineWidget {...props} />)
    const videosBefore = [...container.querySelectorAll('video')]
    const preloadedSecondVideo = videosBefore[1]
    expect(preloadedSecondVideo.currentTime).toBeCloseTo(1)

    fireEvent.click(screen.getByRole('button', { name: 'Select up to two videos for segment 2' }))

    const videosAfter = [...container.querySelectorAll('video')]
    expect(videosAfter[0]).toBe(preloadedSecondVideo)
    expect(videosAfter[0].getAttribute('aria-hidden')).toBe('false')
  })

  it('loops playback back to frame zero after reaching the project end', () => {
    let animationFrame: FrameRequestCallback | undefined
    vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => {
      animationFrame = callback
      return 1
    }))
    vi.stubGlobal('cancelAnimationFrame', vi.fn())
    const { props } = widgetProps()
    const { container } = render(<ProjectVideoCombineWidget {...props} />)

    fireEvent.click(screen.getByRole('button', { name: 'Play' }))
    const clockStart = performance.now()
    act(() => animationFrame?.(clockStart + 2000))
    expect(container.querySelector('.text-gradient')?.textContent).not.toBe('00:00:00')

    act(() => animationFrame?.(clockStart + 6000))

    expect(container.querySelector('.text-gradient')?.textContent).toBe('00:00:00')
    expect(screen.getByRole('button', { name: 'Pause' })).not.toBeNull()
  })

  it('stops preview playback when the browser page becomes hidden', () => {
    const { props } = widgetProps()
    render(<ProjectVideoCombineWidget {...props} />)

    fireEvent.click(screen.getByRole('button', { name: 'Play' }))
    expect(screen.getByRole('button', { name: 'Pause' })).not.toBeNull()

    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      value: 'hidden',
    })
    fireEvent(document, new Event('visibilitychange'))

    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Play' })).not.toBeNull()
  })

  it('stops only this project preview when the active ComfyUI workflow tab changes', () => {
    let workflowSubscriber: ((mutation: unknown, state: {
      activeWorkflow: { key: string }
    }) => void) | undefined
    const { props, workflow } = widgetProps()
    workflow.$subscribe.mockImplementation((subscriber) => {
      workflowSubscriber = subscriber
      return vi.fn()
    })
    render(<ProjectVideoCombineWidget {...props} />)

    fireEvent.click(screen.getByRole('button', { name: 'Play' }))
    expect(screen.getByRole('button', { name: 'Pause' })).not.toBeNull()

    act(() => {
      workflow.activeWorkflow = { key: 'workflow-b' }
      workflowSubscriber?.({}, { activeWorkflow: workflow.activeWorkflow })
    })

    expect(HTMLMediaElement.prototype.pause).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Play' })).not.toBeNull()
  })

  it('uses one-based labels and shows the continuity mode', () => {
    const { props } = widgetProps()

    render(<ProjectVideoCombineWidget {...props} />)

    expect(screen.getByText('Segment 1')).not.toBeNull()
    expect(screen.getByText('Shot')).not.toBeNull()
    expect(screen.queryByRole('combobox', { name: 'Select a file for segment 1' })).toBeNull()
  })

  it('keeps timeline content mounted while hiding and showing the track area', () => {
    const { props } = widgetProps()
    render(<ProjectVideoCombineWidget {...props} />)
    const segmentLabel = screen.getByText('Segment 1')

    fireEvent.click(screen.getByRole('button', { name: 'Hide timeline' }))

    expect(segmentLabel.isConnected).toBe(true)
    expect(screen.getByText('Segment 1')).toBe(segmentLabel)

    fireEvent.click(screen.getByRole('button', { name: 'Show timeline' }))

    expect(screen.getByText('Segment 1')).toBe(segmentLabel)
  })

  it('allows selecting two videos for comparison and promotes the remaining video for output', () => {
    const alternatePath = 'easy_media/projects/demo/video_0_2.mp4'
    const data: ProjectData = {
      ...projectData,
      clips: [{
        ...projectData.clips[0],
        video_files: [
          {
            file_path: projectData.clips[0].file_path,
            file_name: projectData.clips[0].file_name,
            media_revision: projectData.clips[0].media_revision,
            source_frame_count: 120,
          },
          {
            file_path: alternatePath,
            file_name: 'video_0_2.mp4',
            media_revision: '1000000002',
            source_frame_count: 96,
          },
          {
            file_path: 'easy_media/projects/demo/video_0_3.mp4',
            file_name: 'video_0_3.mp4',
            media_revision: '1000000003',
            source_frame_count: 72,
          },
        ],
      }],
    }
    const { props } = widgetProps({ value: data })

    const { container } = render(<ProjectVideoCombineWidget {...props} />)
    const fileSelect = screen.getByRole('button', { name: 'Select up to two videos for segment 1' })
    expect(fileSelect.className).toContain('justify-center')
    expect(fileSelect.firstElementChild?.className).toContain('flex-col')
    fireEvent.click(fileSelect)
    const original = screen.getByRole('checkbox', { name: 'video_0_1.mp4' })
    const alternate = screen.getByRole('checkbox', { name: 'video_0_2.mp4' })
    expect(original.getAttribute('aria-checked')).toBe('true')
    expect(alternate.getAttribute('aria-checked')).toBe('false')

    fireEvent.click(alternate)

    expect(container.querySelectorAll('video')).toHaveLength(2)
    expect(screen.getByRole('button', { name: 'A/B' })).not.toBeNull()
    expect((screen.getByRole('checkbox', { name: 'video_0_3.mp4' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.queryByRole('button', { name: 'Select source' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Select output' })).toBeNull()
    expect(props.onChange).not.toHaveBeenCalled()
    expect((screen.getByRole('button', { name: 'Delete video_0_3.mp4' }) as HTMLButtonElement).disabled).toBe(false)

    fireEvent.click(original)

    expect(props.onChange).toHaveBeenCalledWith({
      ...data,
      clips: [expect.objectContaining({
        file_path: alternatePath,
        file_name: 'video_0_2.mp4',
        media_revision: '1000000002',
        source_end_frame: 96,
        source_frame_count: 96,
      })],
    })
  })

  it('cancels deletion without changing selection or bubbling the delete click', async () => {
    const { props, api, dialog } = widgetProps()
    dialog.confirm.mockResolvedValue(false)
    const bubbled = vi.fn()
    render(<div onClick={bubbled}><ProjectVideoCombineWidget {...props} /></div>)
    fireEvent.click(screen.getByRole('button', { name: 'Select up to two videos for segment 1' }))
    bubbled.mockClear()

    const deleteButton = screen.getByRole('button', { name: 'Delete video_0_1.mp4' })
    fireEvent.pointerDown(deleteButton)
    fireEvent.click(deleteButton)

    await waitFor(() => expect(dialog.confirm).toHaveBeenCalledWith({
      title: 'Delete generated clip?',
      message: expect.stringContaining('video_0_1.mp4'),
    }))
    expect(bubbled).not.toHaveBeenCalled()
    expect(api.fetchApi).not.toHaveBeenCalled()
    expect(props.onChange).not.toHaveBeenCalled()
    expect(screen.getByRole('checkbox', { name: 'video_0_1.mp4' }).getAttribute('aria-checked')).toBe('true')
  })

  it('deletes the selected version and keeps the remaining comparison video and trim', async () => {
    const alternate = {
      file_path: 'easy_media/projects/demo/video_0_2.mp4',
      file_name: 'video_0_2.mp4',
      source_frame_count: 96,
    }
    const clip = { ...projectData.clips[0], source_start_frame: 12, video_files: [projectData.clips[0], alternate] }
    const { props, api } = widgetProps({ value: { ...projectData, clips: [clip], auto_combine: false } })
    api.fetchApi.mockResolvedValue({
      ok: true,
      json: async () => ({ ...projectData, updated_at: 123, clips: [{ ...clip, ...alternate, video_files: [alternate] }] }),
    })
    const { rerender, container } = render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Select up to two videos for segment 1' }))
    fireEvent.click(screen.getByRole('checkbox', { name: 'video_0_2.mp4' }))
    expect(container.querySelectorAll('video')).toHaveLength(2)
    fireEvent.click(screen.getByRole('button', { name: 'Delete video_0_1.mp4' }))

    await waitFor(() => expect(props.onChange).toHaveBeenCalledWith(expect.objectContaining({
      auto_combine: false,
      updated_at: 123,
      clips: [expect.objectContaining({ ...alternate, source_start_frame: 12, source_end_frame: 96, video_files: [alternate] })],
    })))
    expect(api.fetchApi).toHaveBeenCalledWith('/easy-media/project/video', {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ project_name: 'demo', segment_index: 0, file_path: clip.file_path }),
    })
    const nextData = vi.mocked(props.onChange).mock.calls.at(-1)![0]
    rerender(<ProjectVideoCombineWidget {...props} value={nextData} />)
    expect(screen.queryByRole('checkbox', { name: 'video_0_1.mp4' })).toBeNull()
    expect(container.querySelectorAll('video')).toHaveLength(1)
  })

  it('removes the last generated clip after confirmation', async () => {
    const { props, api } = widgetProps()
    api.fetchApi.mockResolvedValue({ ok: true, json: async () => ({ ...projectData, clips: [], updated_at: 123 }) })
    const { rerender } = render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Select up to two videos for segment 1' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete video_0_1.mp4' }))

    await waitFor(() => expect(props.onChange).toHaveBeenCalledWith({ ...projectData, clips: [], updated_at: 123 }))
    rerender(<ProjectVideoCombineWidget {...props} value={{ ...projectData, clips: [], updated_at: 123 }} />)
    expect(screen.queryByText('Segment 1')).toBeNull()
  })

  it.each([0, 1])('returns to normal preview after deleting comparison video %s', async (deletedIndex) => {
    const files = [projectData.clips[0], {
      file_path: 'easy_media/projects/demo/video_0_2.mp4',
      file_name: 'video_0_2.mp4',
      source_frame_count: 96,
    }]
    const initial = { ...projectData, clips: [{ ...projectData.clips[0], video_files: files }] }
    const remaining = files[1 - deletedIndex]
    const { props, api } = widgetProps({ value: initial })
    api.fetchApi.mockResolvedValue({
      ok: true,
      json: async () => ({ ...initial, clips: [{ ...initial.clips[0], ...remaining, video_files: [remaining] }] }),
    })
    function Widget() {
      const [value, setValue] = useState<ProjectData>(initial)
      return <ProjectVideoCombineWidget {...props} value={value} onChange={(next) => {
        props.onChange(next)
        setValue(next)
      }} />
    }
    const { container } = render(<Widget />)
    fireEvent.click(screen.getByRole('button', { name: 'Select up to two videos for segment 1' }))
    fireEvent.click(screen.getByRole('checkbox', { name: files[1].file_name }))
    const controls = container.querySelector('[data-compare-video-playback]') as HTMLElement
    const comparisonVideos = [...container.querySelectorAll('video')]
    fireEvent.click(within(controls).getByRole('button', { name: 'Play' }))
    fireEvent.click(screen.getByRole('button', { name: `Delete ${files[deletedIndex].file_name}` }))

    await waitFor(() => expect(props.onChange).toHaveBeenCalled())
    await waitFor(() => expect(container.querySelector('[data-compare-video-playback]')).toBeNull())
    expect(screen.queryByRole('button', { name: 'A/B' })).toBeNull()
    expect(container.querySelectorAll('video')).toHaveLength(1)
    const preview = container.querySelector('video')!
    expect(preview.src).toContain(remaining.file_name)
    expect(preview.getAttribute('aria-hidden')).toBe('false')
    for (const video of comparisonVideos) {
      expect(vi.mocked(HTMLMediaElement.prototype.pause).mock.contexts).toContain(video)
    }
    fireEvent.click(screen.getByRole('button', { name: 'Play' }))
    expect(vi.mocked(HTMLMediaElement.prototype.play).mock.contexts).toContain(preview)
    expect(screen.getByRole('button', { name: 'Pause' })).not.toBeNull()
  })

  it('keeps project data unchanged and reports a deletion failure', async () => {
    const { props, api, toast } = widgetProps()
    api.fetchApi.mockResolvedValue({ ok: false, json: async () => ({ error: 'Could not save project' }) })
    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Select up to two videos for segment 1' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete video_0_1.mp4' }))

    await waitFor(() => expect(toast.add).toHaveBeenCalledWith(expect.objectContaining({
      severity: 'error', summary: 'Clip deletion failed', detail: 'Could not save project',
    })))
    expect(props.onChange).not.toHaveBeenCalled()
    expect((screen.getByRole('button', { name: 'Delete video_0_1.mp4' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it.each([404, 405])('explains that the backend must be restarted for a plain-text HTTP %s response', async (status) => {
    const { props, api, toast } = widgetProps()
    api.fetchApi.mockResolvedValue(new Response(`${status}: Not Found`, { status }))
    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Select up to two videos for segment 1' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete video_0_1.mp4' }))

    await waitFor(() => expect(toast.add).toHaveBeenCalledWith(expect.objectContaining({
      severity: 'error', detail: expect.stringContaining('Restart ComfyUI'),
    })))
    expect(props.onChange).not.toHaveBeenCalled()
    expect(api.fetchApi).toHaveBeenCalledTimes(1)
  })

  it.each([
    { body: '<html>Bad Gateway</html>', status: 502 },
    { body: '<html>Fallback page</html>', status: 200 },
    { body: '{"error":"Unexpected payload"}', status: 200 },
    { body: '{"project_name":"other","clips":[]}', status: 200 },
  ])('preserves project state for an invalid deletion response: $status $body', async ({ body, status }) => {
    const { props, api, toast } = widgetProps()
    api.fetchApi.mockResolvedValue(new Response(body, { status }))
    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Select up to two videos for segment 1' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete video_0_1.mp4' }))

    await waitFor(() => expect(toast.add).toHaveBeenCalledWith(expect.objectContaining({
      severity: 'error', detail: expect.stringContaining(`invalid project response (HTTP ${status})`),
    })))
    expect(props.onChange).not.toHaveBeenCalled()
  })

  it('preserves a JSON 404 error from the registered deletion endpoint', async () => {
    const { props, api, toast } = widgetProps()
    api.fetchApi.mockResolvedValue(new Response(JSON.stringify({ error: 'Project manifest was not found' }), { status: 404 }))
    render(<ProjectVideoCombineWidget {...props} />)
    fireEvent.click(screen.getByRole('button', { name: 'Select up to two videos for segment 1' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete video_0_1.mp4' }))

    await waitFor(() => expect(toast.add).toHaveBeenCalledWith(expect.objectContaining({
      severity: 'error', detail: 'Project manifest was not found',
    })))
    expect(props.onChange).not.toHaveBeenCalled()
  })
})
