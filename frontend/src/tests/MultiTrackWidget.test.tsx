import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { createDefaultTrackData } from '@/lib/multitrack-utils'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import type { TrackData } from '@/types/multitrack'
import { MultiTrackWidget } from '@/components/widgets/MultiTrackWidget'

vi.mock('@/hooks/use-canvas-scale', () => ({
  useCanvasScale: () => 1,
}))

vi.mock('@/hooks/use-element-width', () => ({
  useElementWidth: () => 480,
}))

vi.mock('@/lib/audio-utils', () => ({
  loadBrowserAudioMetadata: vi.fn().mockResolvedValue({ duration: 2 }),
}))

vi.mock('@/components/widgets/multitrack/PreviewArea', () => ({
  PreviewArea: () => <div data-testid="preview-area" />,
}))

vi.mock('@/components/widgets/multitrack/MultiTrackRuler', () => ({
  MultiTrackRuler: () => <div data-testid="multitrack-ruler" />,
}))

vi.mock('@/components/widgets/multitrack/TrackArea', () => ({
  TrackArea: ({ data, node, app, onCloneTaskSegment, onAddTrack, onAddAudio, onSmartSplit, onSmartSplitTasks, onResizeSegment, cutMode }: {
    data: TrackData
    node: unknown
    app: unknown
    onCloneTaskSegment: (trackId: string, segmentId: string) => void
    onAddTrack: (type: 'audio') => void
    onAddAudio: (trackId: string, filePath: string, sourceType: 'input', previewUrl?: string) => void
    onSmartSplit: (segmentId: string) => void
    onSmartSplitTasks: (segmentId: string) => void
    onResizeSegment: (segmentId: string, edge: 'start' | 'end', nextTime: number) => void
    cutMode: boolean
  }) => {
    const taskTrack = data.tracks.find((track) => track.type === 'task')
    const audioTrack = data.tracks.find((track) => track.type === 'audio')
    const segment = taskTrack?.segments[0]
    const videoSegment = data.tracks.find((track) => track.type === 'video')?.segments[0]
    return (
      <div data-testid="multitrack-track-area" data-cut-mode={cutMode}>
        <div data-testid="audio-slot-context">{node && app ? 'connected' : 'missing'}</div>
        {taskTrack && segment ? (
          <button type="button" onClick={() => onCloneTaskSegment(taskTrack.id, segment.id)}>
            clone task
          </button>
        ) : null}
        <button type="button" onClick={() => onAddTrack('audio')}>add audio track</button>
        {audioTrack ? (
          <button type="button" onClick={() => onAddAudio(audioTrack.id, 'audio.wav', 'input')}>add audio segment</button>
        ) : null}
        {videoSegment ? (
          <button type="button" onClick={() => onSmartSplit(videoSegment.id)}>smart split video</button>
        ) : null}
        {videoSegment ? (
          <button type="button" onClick={() => onSmartSplitTasks(videoSegment.id)}>smart split tasks</button>
        ) : null}
        {videoSegment ? (
          <button type="button" onClick={() => onResizeSegment(videoSegment.id, 'start', 2)}>trim video start</button>
        ) : null}
        {videoSegment ? (
          <button type="button" onClick={() => onResizeSegment(videoSegment.id, 'end', 8)}>trim video end</button>
        ) : null}
        {audioTrack ? (
          <button
            type="button"
            onClick={() => onAddAudio(audioTrack.id, '__slot__:audio', 'input', '/view?filename=voice.wav&type=input&subfolder=')}
          >
            add audio slot segment
          </button>
        ) : null}
      </div>
    )
  },
}))

vi.mock('@/components/widgets/multitrack/MultiTrackToolbar', () => ({
  MultiTrackToolbar: ({ onToggleTimeline, onToggleCutMode, cutMode }: {
    onToggleTimeline: () => void
    onToggleCutMode: () => void
    cutMode: boolean
  }) => (
    <div>
      <button type="button" onClick={onToggleTimeline}>toggle timeline</button>
      <button type="button" data-cut-mode-toggle onClick={onToggleCutMode} aria-pressed={cutMode}>toggle cut</button>
    </div>
  ),
}))

function widgetProps(): ReactWidgetProps<TrackData> {
  return {
    value: createDefaultTrackData(),
    onChange: vi.fn(),
    inputName: 'tracks',
    node: {},
    widget: {} as ReactWidgetProps<TrackData>['widget'],
    app: {
      ui: { settings: { settingsValues: {} } },
    } as ReactWidgetProps<TrackData>['app'],
  }
}

describe('MultiTrackWidget', () => {
  it.each([
    ['trim video start', 2, 10],
    ['trim video end', 0, 8],
  ])('syncs a matching task when using %s', (buttonName, expectedStart, expectedEnd) => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-matching',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    data.tracks[1].segments = [{
      id: 'video-main',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video', duration: 10 },
    }]
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: buttonName }))

    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks[0].segments[0]).toMatchObject({
      start_frame: expectedStart,
      end_frame: expectedEnd,
    })
  })

  it('animates the ruler and track region to zero height when toggled', () => {
    render(<MultiTrackWidget {...widgetProps()} />)

    const timelinePanel = screen.getByTestId('multitrack-timeline-panel')
    expect(timelinePanel.className).toContain('grid-rows-[1fr]')
    expect(timelinePanel.className).toContain('transition-[grid-template-rows]')

    fireEvent.click(screen.getByRole('button', { name: 'toggle timeline' }))

    expect(timelinePanel.className).toContain('grid-rows-[0fr]')
    expect(timelinePanel.getAttribute('aria-hidden')).toBe('true')
  })

  it.each([
    {
      name: 'keeps the current total length when cloned tasks still fit',
      taskRanges: [[0, 2], [2, 4]],
      expectedTotalLength: 10,
    },
    {
      name: 'extends the total length only when cloned tasks exceed it',
      taskRanges: [[0, 4], [4, 9]],
      expectedTotalLength: 13,
    },
  ])('$name', ({ taskRanges, expectedTotalLength }) => {
    const data = createDefaultTrackData()
    data.total_length = 10
    data.tracks[0].segments = taskRanges.map(([startFrame, endFrame], index) => ({
      id: `task-${index}`,
      start_frame: startFrame,
      end_frame: endFrame,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default', text: `Task ${index}` },
    }))
    data.tracks[1].segments = [{
      id: 'video',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video' },
    }]
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'clone task' }))

    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({
      total_length: expectedTotalLength,
    }))
  })

  it('adds an audio track and serializes audio segments with metadata duration', async () => {
    const onAddTrackChange = vi.fn()
    render(<MultiTrackWidget {...widgetProps()} onChange={onAddTrackChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'add audio track' }))
    const addedTrackData = onAddTrackChange.mock.lastCall?.[0] as TrackData
    const audioTrack = addedTrackData.tracks.at(-1)
    expect(audioTrack).toMatchObject({ name: 'Audio 0', type: 'audio', muted: false, solo: false, volume_db: 0 })

    const onAddSegmentChange = vi.fn()
    render(<MultiTrackWidget {...widgetProps()} value={addedTrackData} onChange={onAddSegmentChange} />)
    fireEvent.click(screen.getAllByRole('button', { name: 'add audio segment' }).at(-1)!)
    await waitFor(() => expect(onAddSegmentChange).toHaveBeenCalled())
    const updated = onAddSegmentChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks.at(-1)?.segments[0]).toMatchObject({
      start_frame: 0,
      end_frame: 48,
      content: { media_type: 'audio', file_path: 'audio.wav', duration: 2, volume_db: 0 },
    })
  })

  it('provides the ComfyUI graph context to audio tracks', () => {
    const props = widgetProps()
    props.node = {
      inputs: [{ name: 'audio', type: 'AUDIO', link: 7 }],
    } as ReactWidgetProps<TrackData>['node']
    props.app = {
      ...props.app,
      graph: {
        links: { 7: { origin_id: 3, origin_slot: 0 } },
        getNodeById: () => ({
          type: 'LoadAudio',
          outputs: [{ shape: 0 }],
          widgets_values: ['voice.wav'],
        }),
      },
    } as unknown as ReactWidgetProps<TrackData>['app']

    render(<MultiTrackWidget {...props} />)

    expect(screen.getByTestId('audio-slot-context').textContent).toBe('connected')
  })

  it('uses the connected audio filename to load slot metadata and preview the segment', async () => {
    const data = createDefaultTrackData()
    data.tracks.push({
      id: 'audio-track',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      locked: false,
      segments: [],
    })
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'add audio slot segment' }))

    await waitFor(() => expect(onChange).toHaveBeenCalled())
    expect((onChange.mock.lastCall?.[0] as TrackData).tracks.at(-1)?.segments[0]).toMatchObject({
      end_frame: 48,
      content: {
        source_type: 'slot',
        slot_name: 'audio',
        file_name: 'audio',
        url: '/view?filename=voice.wav&type=input&subfolder=',
        duration: 2,
      },
    })
  })

  it('covers the whole widget while smart split is pending and clears it on success', async () => {
    const data = createDefaultTrackData()
    data.total_length = 240
    data.tracks[1].segments = [{
      id: 'video',
      start_frame: 0,
      end_frame: 240,
      color: data.tracks[1].color,
      content: { media_type: 'video', source_type: 'input', file_path: 'clip.mp4' },
    }]
    const onChange = vi.fn()
    let resolveFetch: ((response: Response) => void) | undefined
    const fetchMock = vi.fn((_input: RequestInfo | URL, _init?: RequestInit) => new Promise<Response>((resolve) => {
      resolveFetch = resolve
    }))
    vi.stubGlobal('fetch', fetchMock)

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'smart split video' }))

    expect(await screen.findByTestId('smart-split-overlay')).not.toBeNull()
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({ fps: 24 })
    resolveFetch?.({
      ok: true,
      status: 200,
      json: async () => ({
        ranges: [[0, 120], [120, 240]],
      }),
    } as Response)

    await waitFor(() => expect(screen.queryByTestId('smart-split-overlay')).toBeNull())
    expect(onChange).toHaveBeenCalledOnce()
    vi.unstubAllGlobals()
  })

  it('does not request task-only smart split when no task range matches', () => {
    const data = createDefaultTrackData()
    data.tracks[1].segments = [{
      id: 'video', start_frame: 0, end_frame: 240, color: data.tracks[1].color,
      content: { media_type: 'video', source_type: 'input', file_path: 'clip.mp4' },
    }]
    const toastAdd = vi.fn()
    const props = widgetProps()
    props.app = {
      ...props.app,
      extensionManager: { toast: { add: toastAdd } },
    } as unknown as ReactWidgetProps<TrackData>['app']
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    render(<MultiTrackWidget {...props} value={data} />)
    fireEvent.click(screen.getByRole('button', { name: 'smart split tasks' }))

    expect(fetchMock).not.toHaveBeenCalled()
    expect(toastAdd).toHaveBeenCalledWith(expect.objectContaining({ severity: 'warn' }))
    vi.unstubAllGlobals()
  })

  it('exits cut mode when clicking outside the track area', () => {
    render(<MultiTrackWidget {...widgetProps()} />)
    fireEvent.click(screen.getByRole('button', { name: 'toggle cut' }))
    expect(screen.getByTestId('multitrack-track-area').getAttribute('data-cut-mode')).toBe('true')

    fireEvent.click(screen.getByTestId('preview-area'))

    expect(screen.getByTestId('multitrack-track-area').getAttribute('data-cut-mode')).toBe('false')
  })
})
