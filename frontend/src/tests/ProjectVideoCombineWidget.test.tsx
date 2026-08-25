import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ProjectVideoCombineWidget } from '@/components/widgets/ProjectVideoCombineWidget'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import type { ProjectData } from '@/types/project'

vi.mock('@/hooks/use-canvas-scale', () => ({ useCanvasScale: () => 1 }))
vi.mock('@/hooks/use-element-width', () => ({ useElementWidth: () => 480 }))

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
      extensionManager: { toast },
    },
    ...overrides,
  } as unknown as ReactWidgetProps<ProjectData>
  return { props, api, toast, graphToPrompt }
}

beforeEach(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue()
  vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
})

describe('ProjectVideoCombineWidget', () => {
  it('refreshes the selected project and clears the stale preview first', async () => {
    const { props, api } = widgetProps({ value: { ...projectData, clips: [] } })
    api.fetchApi.mockResolvedValue({ ok: true, json: async () => projectData })

    render(<ProjectVideoCombineWidget {...props} />)

    await waitFor(() => expect(props.onChange).toHaveBeenLastCalledWith(projectData))
    expect(api.fetchApi).toHaveBeenCalledWith('/easy-media/project?project_name=demo')
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

  it('serializes the auto combine checkbox through project data', () => {
    const { props } = widgetProps()
    render(<ProjectVideoCombineWidget {...props} />)

    fireEvent.click(screen.getByRole('checkbox', { name: 'Auto combine' }))

    expect(props.onChange).toHaveBeenCalledWith({ ...projectData, auto_combine: false })
  })
})
