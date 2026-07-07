import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createDefaultTrackData } from '@/lib/multitrack-utils'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import type { TrackData } from '@/types/multitrack'
import { MultiTrackWidget } from '@/components/widgets/MultiTrackWidget'
import { loadBrowserVideoMetadata } from '@/lib/video-utils'

vi.mock('@/hooks/use-canvas-scale', () => ({
  useCanvasScale: () => 1,
}))

vi.mock('@/hooks/use-element-width', () => ({
  useElementWidth: () => 480,
}))

vi.mock('@/lib/audio-utils', () => ({
  loadBrowserAudioMetadata: vi.fn().mockResolvedValue({ duration: 2 }),
}))

vi.mock('@/lib/video-utils', () => ({
  loadBrowserVideoMetadata: vi.fn(),
}))

vi.mock('@/components/widgets/multitrack/PreviewArea', () => ({
  PreviewArea: ({ selectedSegment, onSelectedSegmentContentChange, onGenerateSubtitleSpeech }: {
    selectedSegment: { trackType: string; segment: TrackData['tracks'][number]['segments'][number] } | null
    onSelectedSegmentContentChange: (patch: unknown) => void
    onGenerateSubtitleSpeech?: (
      segment: TrackData['tracks'][number]['segments'][number],
      settings: {
        model: 'VoxCPM2'
        prompt: string
        cfg: number
        steps: number
        referenceAudio: string
        referenceAudioSourceType: 'input'
      },
    ) => Promise<void>
  }) => (
    <div data-testid="preview-area">
      {selectedSegment?.trackType === 'subtitle' ? (
        <button
          type="button"
          onClick={() => onSelectedSegmentContentChange({
            subtitle_style: {
              font_size: 18,
              color: '#ff0000',
              outline_color: '#000000',
              background_color: 'rgba(0, 0, 0, 0.5)',
              background_opacity: 0.7,
              x: 0.2,
              y: 0.75,
              width: 0.6,
            },
          })}
        >
          update selected subtitle style
        </button>
      ) : null}
      {selectedSegment?.trackType === 'subtitle' ? (
        <button
          type="button"
          onClick={() => onGenerateSubtitleSpeech?.(selectedSegment.segment, {
            model: 'VoxCPM2',
            prompt: 'calm',
            cfg: 2.4,
            steps: 13,
            referenceAudio: 'voice.wav',
            referenceAudioSourceType: 'input',
          })}
        >
          generate subtitle speech
        </button>
      ) : null}
    </div>
  ),
}))

vi.mock('@/components/widgets/multitrack/MultiTrackRuler', () => ({
  MultiTrackRuler: ({ onSeek }: { onSeek: (time: number) => void }) => (
    <button type="button" data-testid="multitrack-ruler" onClick={() => onSeek(5)}>
      seek frame 5
    </button>
  ),
}))

vi.mock('@/components/widgets/multitrack/TrackArea', () => ({
  TrackArea: ({ data, node, app, onCloneTaskSegment, onSplitTaskSegment, onAddTrack, onAddVideo, onAddAudio, onAddSubtitleSegment, onSmartSplit, onSmartSplitTasks, onRecognizeSubtitles, onResizeSegment, onResizeSegmentPreview, onMoveSegment, onTrackAudioSettingsChange, selectedSegmentIds, onSelectSegment, onSelectSegments, cutMode }: {
    data: TrackData
    node: unknown
    app: unknown
    onCloneTaskSegment: (trackId: string, segmentId: string) => void
    onSplitTaskSegment: (segmentId: string) => void
    onAddTrack: (type: 'audio') => void
    onAddVideo: (trackId: string, filePath: string, sourceType: 'input', startFrame?: number) => void
    onAddAudio: (trackId: string, filePath: string, sourceType: 'input', previewUrl?: string) => void
    onAddSubtitleSegment: (trackId: string) => void
    onSmartSplit: (segmentId: string) => void
    onSmartSplitTasks: (segmentId: string) => void
    onRecognizeSubtitles: (segmentId: string) => void
    onResizeSegment: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
    onResizeSegmentPreview: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
    onMoveSegment: (segmentId: string, targetTrackId: string, nextStartTime: number) => void
    onTrackAudioSettingsChange: (trackId: string, patch: { muted?: boolean; solo?: boolean }) => void
    selectedSegmentIds: Set<string>
    onSelectSegment: (segmentId: string) => void
    onSelectSegments: (segmentIds: string[]) => void
    cutMode: boolean
  }) => {
    const taskTrack = data.tracks.find((track) => track.type === 'task')
    const audioTrack = data.tracks.find((track) => track.type === 'audio')
    const segment = taskTrack?.segments[0]
    const videoSegment = data.tracks.find((track) => track.type === 'video')?.segments[0]
    const firstVideoTrack = data.tracks.find((track) => track.type === 'video')
    const subtitleTrack = data.tracks.find((track) => track.type === 'subtitle')
    const subtitleSegment = data.tracks.find((track) => track.type === 'subtitle')?.segments[0]
    return (
      <div data-testid="multitrack-track-area" data-cut-mode={cutMode}>
        <div data-testid="selected-segment-count">{selectedSegmentIds.size}</div>
        <div data-testid="audio-slot-context">{node && app ? 'connected' : 'missing'}</div>
        {taskTrack && segment ? (
          <button type="button" onClick={() => onCloneTaskSegment(taskTrack.id, segment.id)}>
            clone task
          </button>
        ) : null}
        {taskTrack && segment ? (
          <button type="button" onClick={() => onSplitTaskSegment(segment.id)}>
            split task
          </button>
        ) : null}
        <button type="button" onClick={() => onAddTrack('audio')}>add audio track</button>
        <button
          type="button"
          onClick={() => onAddVideo(data.tracks[1].id, 'inserted.mp4', 'input', 0)}
        >
          insert video segment
        </button>
        {audioTrack ? (
          <button type="button" onClick={() => onAddAudio(audioTrack.id, 'audio.wav', 'input')}>add audio segment</button>
        ) : null}
        {firstVideoTrack ? (
          <button type="button" onClick={() => onTrackAudioSettingsChange(firstVideoTrack.id, { solo: !firstVideoTrack.solo })}>
            toggle first video solo
          </button>
        ) : null}
        {audioTrack ? (
          <button type="button" onClick={() => onTrackAudioSettingsChange(audioTrack.id, { solo: !audioTrack.solo })}>
            toggle first audio solo
          </button>
        ) : null}
        {videoSegment ? (
          <button type="button" onClick={() => onSmartSplit(videoSegment.id)}>smart split video</button>
        ) : null}
        {videoSegment ? (
          <button type="button" onClick={() => onRecognizeSubtitles(videoSegment.id)}>recognize video subtitles</button>
        ) : null}
        {subtitleTrack ? (
          <button type="button" onClick={() => onAddSubtitleSegment(subtitleTrack.id)}>add subtitle segment</button>
        ) : null}
        {videoSegment ? (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              onSelectSegment(videoSegment.id)
            }}
          >
            select video segment
          </button>
        ) : null}
        {subtitleSegment ? (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              onSelectSegment(subtitleSegment.id)
            }}
          >
            select subtitle segment
          </button>
        ) : null}
        {videoSegment && segment ? (
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              onSelectSegments([videoSegment.id, segment.id])
            }}
          >
            select video and task
          </button>
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
        {data.tracks[1]?.segments.length >= 2 ? (
          <button
            type="button"
            onClick={() => onMoveSegment(data.tracks[1].segments[0].id, data.tracks[1].id, data.tracks[1].segments[1].end_frame)}
          >
            move first video after second
          </button>
        ) : null}
        {videoSegment ? (
          <button
            type="button"
            onClick={() => {
              onResizeSegmentPreview(videoSegment.id, 'end', 9)
              onResizeSegmentPreview(videoSegment.id, 'end', 8)
            }}
          >
            preview trim video end
          </button>
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
  MultiTrackToolbar: ({
    onToggleTimeline,
    canDelete,
    onDeleteSelected,
    onCutAtCurrentTime,
    canTrimLeft,
    canTrimRight,
    onTrimLeftAtCurrentTime,
    onTrimRightAtCurrentTime,
    canUndo,
    canRedo,
    onUndo,
    onRedo,
  }: {
    onToggleTimeline: () => void
    canDelete: boolean
    onDeleteSelected: () => void
    onCutAtCurrentTime: () => void
    canTrimLeft: boolean
    canTrimRight: boolean
    onTrimLeftAtCurrentTime: () => void
    onTrimRightAtCurrentTime: () => void
    canUndo: boolean
    canRedo: boolean
    onUndo: () => void
    onRedo: () => void
  }) => (
    <div>
      <button type="button" onClick={onToggleTimeline}>toggle timeline</button>
      <button type="button" disabled={!canDelete} onClick={onDeleteSelected}>delete selected</button>
      <button
        type="button"
        onClick={(event) => {
          event.stopPropagation()
          onCutAtCurrentTime()
        }}
      >
        cut current time
      </button>
      <button type="button" disabled={!canTrimLeft} onClick={onTrimLeftAtCurrentTime}>trim left current time</button>
      <button type="button" disabled={!canTrimRight} onClick={onTrimRightAtCurrentTime}>trim right current time</button>
      <button type="button" onClick={onUndo} disabled={!canUndo}>undo history</button>
      <button type="button" onClick={onRedo} disabled={!canRedo}>redo history</button>
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
  beforeEach(() => {
    vi.mocked(loadBrowserVideoMetadata).mockResolvedValue({ duration: 1, width: 1920, height: 1080 })
  })

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

  it('records a single history entry when resizing after transient previews', () => {
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
    const props = widgetProps()
    const view = render(<MultiTrackWidget {...props} value={data} onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'preview trim video end' }))
    expect(onChange).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'trim video end' }))
    expect(onChange).toHaveBeenCalledOnce()
    const resized = onChange.mock.lastCall?.[0] as TrackData
    expect(resized.tracks[1].segments[0].end_frame).toBe(8)

    view.rerender(<MultiTrackWidget {...props} value={resized} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'undo history' }))
    const undone = onChange.mock.lastCall?.[0] as TrackData
    expect(undone.tracks[1].segments[0].end_frame).toBe(10)
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

  it('undoes and redoes multitrack widget changes without using canvas history', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-0',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    const onChange = vi.fn()
    const props = widgetProps()
    const view = render(<MultiTrackWidget {...props} value={data} onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'clone task' }))
    const cloned = onChange.mock.lastCall?.[0] as TrackData
    expect(cloned.tracks[0].segments).toHaveLength(2)

    view.rerender(<MultiTrackWidget {...props} value={cloned} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'undo history' }))
    const undone = onChange.mock.lastCall?.[0] as TrackData
    expect(undone.tracks[0].segments).toHaveLength(1)
    expect(undone.tracks[0].segments[0].id).toBe('task-0')

    view.rerender(<MultiTrackWidget {...props} value={undone} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'redo history' }))
    expect(onChange.mock.lastCall?.[0]).toEqual(cloned)
  })

  it('splits a task segment from the context dialog', () => {
    const data = createDefaultTrackData()
    data.frame_rate = 16
    data.total_length = 160
    data.tracks[0].segments = [{
      id: 'task-0',
      start_frame: 0,
      end_frame: 160,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default', text: 'Prompt' },
    }]
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'split task' }))
    const input = screen.getByRole('spinbutton', { name: 'Split seconds' })
    expect((input as HTMLInputElement).value).toBe('5')
    fireEvent.change(input, { target: { value: '5' } })
    fireEvent.blur(input)
    expect((input as HTMLInputElement).value).toBe('5')
    fireEvent.click(screen.getByRole('button', { name: 'Split' }))

    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks[0].segments).toHaveLength(2)
    expect(updated.tracks[0].segments.map((segment) => [segment.start_frame, segment.end_frame])).toEqual([
      [0, 80],
      [80, 160],
    ])
    expect(updated.tracks[0].segments.every((segment) => segment.content.text === 'Prompt')).toBe(true)
  })

  it('resets local redo history when the canvas restores a different widget value', () => {
    const initial = createDefaultTrackData()
    initial.tracks[0].segments = [{
      id: 'task-0',
      start_frame: 0,
      end_frame: 10,
      color: initial.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    const onChange = vi.fn()
    const props = widgetProps()
    const view = render(<MultiTrackWidget {...props} value={initial} onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'clone task' }))
    const cloned = onChange.mock.lastCall?.[0] as TrackData
    view.rerender(<MultiTrackWidget {...props} value={cloned} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'undo history' }))
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'redo history' }).disabled).toBe(false)

    const canvasRestored = {
      ...initial,
      frame_rate: 30,
    }
    view.rerender(<MultiTrackWidget {...props} value={canvasRestored} onChange={onChange} />)

    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'redo history' }).disabled).toBe(true)
  })

  it('keeps local history after dragging video segments to reorder them', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-0',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    data.tracks[1].segments = [
      {
        id: 'video-0',
        start_frame: 0,
        end_frame: 10,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 10 },
      },
      {
        id: 'video-1',
        start_frame: 10,
        end_frame: 20,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 10 },
      },
    ]
    data.total_length = 20
    const onChange = vi.fn()
    const props = widgetProps()
    const view = render(<MultiTrackWidget {...props} value={data} onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'clone task' }))
    const withHistory = onChange.mock.lastCall?.[0] as TrackData
    view.rerender(<MultiTrackWidget {...props} value={withHistory} onChange={onChange} />)
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'undo history' }).disabled).toBe(false)

    fireEvent.click(screen.getByRole('button', { name: 'move first video after second' }))
    const moved = onChange.mock.lastCall?.[0] as TrackData
    expect(moved.tracks[1].segments.map((segment) => segment.id)).toEqual(['video-1', 'video-0'])
    view.rerender(<MultiTrackWidget {...props} value={moved} onChange={onChange} />)

    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'undo history' }).disabled).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'undo history' }))
    const undoneMove = onChange.mock.lastCall?.[0] as TrackData
    expect(undoneMove.tracks[1].segments.map((segment) => segment.id)).toEqual(['video-0', 'video-1'])
  })

  it.each([
    {
      name: 'keeps the five-second minimum when cloned tasks still fit',
      taskRanges: [[0, 2], [2, 4]],
      expectedTotalLength: 120,
    },
    {
      name: 'extends the total length when cloned tasks exceed five seconds',
      taskRanges: [[0, 60], [60, 100]],
      expectedTotalLength: 160,
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

  it('solos a video track by muting other video and audio tracks, then restores them when solo is cleared', () => {
    const data = createDefaultTrackData()
    data.tracks[1].id = 'video-main'
    data.tracks[1].muted = false
    data.tracks[1].solo = false
    data.tracks.push(
      {
        id: 'video-secondary',
        name: 'Video 1',
        type: 'video',
        color: 'var(--secondary)',
        muted: false,
        solo: false,
        locked: false,
        segments: [],
      },
      {
        id: 'audio-main',
        name: 'Audio 0',
        type: 'audio',
        color: 'var(--highlight)',
        muted: false,
        solo: false,
        locked: false,
        segments: [],
      },
    )
    const onChange = vi.fn()
    const props = widgetProps()
    const view = render(<MultiTrackWidget {...props} value={data} onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'toggle first video solo' }))
    const soloed = onChange.mock.lastCall?.[0] as TrackData
    expect(soloed.tracks.find((track) => track.id === 'video-main')).toMatchObject({ muted: false, solo: true })
    expect(soloed.tracks.find((track) => track.id === 'video-secondary')).toMatchObject({ muted: true, solo: false })
    expect(soloed.tracks.find((track) => track.id === 'audio-main')).toMatchObject({ muted: true, solo: false })

    view.rerender(<MultiTrackWidget {...props} value={soloed} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'toggle first video solo' }))
    const restored = onChange.mock.lastCall?.[0] as TrackData
    expect(restored.tracks.find((track) => track.id === 'video-main')).toMatchObject({ muted: false, solo: false })
    expect(restored.tracks.find((track) => track.id === 'video-secondary')).toMatchObject({ muted: false, solo: false })
    expect(restored.tracks.find((track) => track.id === 'audio-main')).toMatchObject({ muted: false, solo: false })
  })

  it('solos an audio track by muting video tracks', () => {
    const data = createDefaultTrackData()
    data.tracks[1].id = 'video-main'
    data.tracks[1].muted = false
    data.tracks[1].solo = false
    data.tracks.push({
      id: 'audio-main',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      solo: false,
      locked: false,
      segments: [],
    })
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'toggle first audio solo' }))

    const soloed = onChange.mock.lastCall?.[0] as TrackData
    expect(soloed.tracks.find((track) => track.id === 'audio-main')).toMatchObject({ muted: false, solo: true })
    expect(soloed.tracks.find((track) => track.id === 'video-main')).toMatchObject({ muted: true, solo: false })
  })

  it('keeps matching task ranges aligned when inserting a video before existing segments', async () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-existing',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    data.tracks[1].segments = [{
      id: 'video-existing',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video', duration: 10 / data.frame_rate },
    }]
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'insert video segment' }))

    await waitFor(() => expect(onChange).toHaveBeenCalled())
    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks[0].segments.map((segment) => [segment.start_frame, segment.end_frame])).toEqual([
      [0, 24],
      [24, 34],
    ])
  })

  it('merges an asynchronously added video into the latest widget value', async () => {
    let resolveMetadata: ((metadata: { duration: number, width: number, height: number }) => void) | undefined
    vi.mocked(loadBrowserVideoMetadata).mockImplementation(() => new Promise((resolve) => {
      resolveMetadata = resolve
    }))
    const initial = createDefaultTrackData()
    const latest: TrackData = { ...initial, tracks: [...initial.tracks, {
      id: 'audio-added-while-loading',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      locked: false,
      segments: [],
    }] }
    const onChange = vi.fn()
    const props = widgetProps()
    const view = render(<MultiTrackWidget {...props} value={initial} onChange={onChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'insert video segment' }))
    view.rerender(<MultiTrackWidget {...props} value={latest} onChange={onChange} />)
    resolveMetadata?.({ duration: 1, width: 1920, height: 1080 })

    await waitFor(() => expect(onChange).toHaveBeenCalled())
    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks.some((track) => track.id === 'audio-added-while-loading')).toBe(true)
    expect(updated.tracks[1].segments).toHaveLength(1)
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

  it('applies selected subtitle style changes to every segment on the same subtitle track', () => {
    const data = createDefaultTrackData()
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [
        {
          id: 'subtitle-a',
          start_frame: 0,
          end_frame: 24,
          color: '#9D4937',
          content: {
            media_type: 'subtitle',
            text: 'First line',
            subtitle_style: {
              font_size: 12,
              color: '#ffffff',
              outline_color: '#000000',
              background_color: 'rgba(0, 0, 0, 0)',
              background_opacity: 0.7,
              x: 0.15,
              y: 0.8,
              width: 0.7,
            },
          },
        },
        {
          id: 'subtitle-b',
          start_frame: 24,
          end_frame: 48,
          color: '#9D4937',
          content: {
            media_type: 'subtitle',
            text: 'Second line',
            subtitle_style: {
              font_size: 10,
              color: '#00ff00',
              background_color: 'transparent',
              background_opacity: 0.7,
              x: 0.3,
              y: 0.7,
              width: 0.5,
            },
          },
        },
      ],
    })
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'select subtitle segment' }))
    fireEvent.click(screen.getByRole('button', { name: 'update selected subtitle style' }))

    const updated = onChange.mock.lastCall?.[0] as TrackData
    const subtitleTrack = updated.tracks.find((track) => track.id === 'subtitle-track')
    expect(subtitleTrack?.segments.map((segment) => segment.content.subtitle_style)).toEqual([
      {
        font_size: 18,
        color: '#ff0000',
        outline_color: '#000000',
        background_color: 'rgba(0, 0, 0, 0.5)',
        background_opacity: 0.7,
        x: 0.2,
        y: 0.75,
        width: 0.6,
      },
      {
        font_size: 18,
        color: '#ff0000',
        outline_color: '#000000',
        background_color: 'rgba(0, 0, 0, 0.5)',
        background_opacity: 0.7,
        x: 0.2,
        y: 0.75,
        width: 0.6,
      },
    ])
    expect(subtitleTrack?.segments.map((segment) => segment.content.text)).toEqual(['First line', 'Second line'])
  })

  it('adds a five-second default subtitle segment to a subtitle track', () => {
    const data = createDefaultTrackData()
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [],
    })
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'add subtitle segment' }))

    const updated = onChange.mock.lastCall?.[0] as TrackData
    const segment = updated.tracks.find((track) => track.id === 'subtitle-track')?.segments[0]
    expect(segment).toMatchObject({
      start_frame: 0,
      end_frame: 120,
      color: '#9D4937',
      content: {
        media_type: 'subtitle',
        text: '默认文字',
        subtitle_style: {
          font_size: 12,
          color: '#ffffff',
          outline_color: '#000000',
          background_color: 'rgba(0, 0, 0, 0)',
          background_opacity: 0.7,
          x: 0.125,
          y: 0.8,
          width: 0.75,
        },
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

  it('covers the widget while recognizing subtitles and appends a subtitle track', async () => {
    const data = createDefaultTrackData()
    data.total_length = 240
    data.tracks[1].segments = [{
      id: 'video',
      start_frame: 24,
      end_frame: 120,
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
    fireEvent.click(screen.getByRole('button', { name: 'recognize video subtitles' }))

    expect(await screen.findByTestId('subtitle-recognition-overlay')).not.toBeNull()
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      media_type: 'video',
      fps: 24,
    })
    resolveFetch?.({
      ok: true,
      status: 200,
      json: async () => ({
        segments: [{ start: 0, end: 1, text: 'Hello' }],
      }),
    } as Response)

    await waitFor(() => expect(screen.queryByTestId('subtitle-recognition-overlay')).toBeNull())
    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks.find((track) => track.type === 'subtitle')?.segments[0]).toMatchObject({
      start_frame: 24,
      end_frame: 48,
      color: '#9D4937',
      content: {
        media_type: 'subtitle',
        text: 'Hello',
        subtitle_style: expect.objectContaining({ font_size: 12 }),
      },
    })
    vi.unstubAllGlobals()
  })

  it('generates selected subtitle speech and adds the output to an audio track', async () => {
    const data = createDefaultTrackData()
    data.total_length = 120
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 0',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [{
        id: 'subtitle',
        start_frame: 24,
        end_frame: 48,
        color: '#9D4937',
        content: { media_type: 'subtitle', text: 'Hello 123' },
      }],
    })
    const onChange = vi.fn()
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        file_path: 'easy_media/Hello.wav',
        source_type: 'output',
        absolute_path: '/tmp/output/easy_media/Hello.wav',
        message: 'Audio generated and saved to /tmp/output/easy_media/Hello.wav',
        duration: 1,
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'select subtitle segment' }))
    fireEvent.click(screen.getByRole('button', { name: 'generate subtitle speech' }))

    await waitFor(() => expect(onChange).toHaveBeenCalled())
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      text: 'Hello 123',
      model: 'VoxCPM2',
      prompt: 'calm',
      cfg: 2.4,
      steps: 13,
      reference_audio_path: 'voice.wav',
      reference_audio_source_type: 'input',
    })
    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks.find((track) => track.type === 'audio')?.segments[0]).toMatchObject({
      start_frame: 24,
      end_frame: 48,
      content: {
        media_type: 'audio',
        source_type: 'output',
        file_path: 'easy_media/Hello.wav',
      },
    })
    vi.unstubAllGlobals()
  })

  it('shows missing model actions and downloads the requested model', async () => {
    const data = createDefaultTrackData()
    data.total_length = 240
    data.tracks[1].segments = [{
      id: 'video',
      start_frame: 0,
      end_frame: 240,
      color: data.tracks[1].color,
      content: { media_type: 'video', source_type: 'input', file_path: 'clip.mp4' },
    }]
    const model = {
      name: 'omnishotcut',
      display_name: 'OmniShotCut',
      filename: 'OmniShotCut_ckpt.pth',
      directory: '/ComfyUI/models/checkpoints',
      path: '/ComfyUI/models/checkpoints/OmniShotCut_ckpt.pth',
      url: 'https://huggingface.co/uva-cv-lab/OmniShotCut/resolve/main/OmniShotCut_ckpt.pth',
    }
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({
        ok: false,
        status: 428,
        json: async () => ({
          error: 'OmniShotCut model is not installed.',
          model_missing: model,
        }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ ok: true, model }),
      } as Response)
    vi.stubGlobal('fetch', fetchMock)

    render(<MultiTrackWidget {...widgetProps()} value={data} />)
    fireEvent.click(screen.getByRole('button', { name: 'smart split video' }))

    expect(await screen.findByTestId('missing-model-overlay')).not.toBeNull()
    expect(screen.getByText('OmniShotCut is missing')).not.toBeNull()
    expect(screen.getByText(/installed under checkpoints/)).not.toBeNull()
    expect(screen.getByText('/ComfyUI/models/checkpoints/OmniShotCut_ckpt.pth')).not.toBeNull()
    expect(screen.queryByRole('button', { name: 'Exit' })).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Auto download' }))

    await waitFor(() => expect(screen.queryByTestId('missing-model-overlay')).toBeNull())
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/easy-media/models/download', expect.objectContaining({
      method: 'POST',
      body: JSON.stringify({ model_name: 'omnishotcut' }),
    }))
    vi.unstubAllGlobals()
  })

  it('opens the source URL for manual model download', async () => {
    const data = createDefaultTrackData()
    data.tracks[1].segments = [{
      id: 'video',
      start_frame: 0,
      end_frame: 240,
      color: data.tracks[1].color,
      content: { media_type: 'video', source_type: 'input', file_path: 'clip.mp4' },
    }]
    const model = {
      name: 'omnishotcut',
      display_name: 'OmniShotCut',
      filename: 'OmniShotCut_ckpt.pth',
      directory: '/ComfyUI/models/checkpoints',
      path: '/ComfyUI/models/checkpoints/OmniShotCut_ckpt.pth',
      url: 'https://huggingface.co/uva-cv-lab/OmniShotCut/resolve/main/OmniShotCut_ckpt.pth',
    }
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: false,
      status: 428,
      json: async () => ({ error: 'missing', model_missing: model }),
    }))
    const openMock = vi.fn()
    vi.stubGlobal('open', openMock)

    render(<MultiTrackWidget {...widgetProps()} value={data} />)
    fireEvent.click(screen.getByRole('button', { name: 'smart split video' }))
    fireEvent.click(await screen.findByRole('button', { name: 'Manual download' }))

    expect(openMock).toHaveBeenCalledWith(model.url, '_blank', 'noopener,noreferrer')
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

  it('cuts all active track segments at the current time when no segment is selected', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    data.tracks[1].segments = [{
      id: 'video-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video', duration: 10 },
    }]
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByTestId('multitrack-ruler'))
    fireEvent.click(screen.getByRole('button', { name: 'cut current time' }))

    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks[0].segments).toHaveLength(2)
    expect(updated.tracks[0].segments.map((segment) => [segment.start_frame, segment.end_frame])).toEqual([[0, 5], [5, 10]])
    expect(updated.tracks[1].segments).toHaveLength(2)
    expect(updated.tracks[1].segments.map((segment) => [segment.start_frame, segment.end_frame])).toEqual([[0, 5], [5, 10]])
  })

  it('cuts only selected segments at the current time', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    data.tracks[1].segments = [{
      id: 'video-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video', duration: 10 },
    }]
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByTestId('multitrack-ruler'))
    fireEvent.click(screen.getByRole('button', { name: 'select video segment' }))
    fireEvent.click(screen.getByRole('button', { name: 'cut current time' }))

    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks[0].segments).toHaveLength(1)
    expect(updated.tracks[1].segments).toHaveLength(2)
    expect(updated.tracks[1].segments.map((segment) => [segment.start_frame, segment.end_frame])).toEqual([[0, 5], [5, 10]])
  })

  it.each([
    ['trim left current time', 'start_frame', 5],
    ['trim right current time', 'end_frame', 5],
  ] as const)('trims all active track segments from %s when no segment is selected', (buttonName, frameKey, expectedFrame) => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    data.tracks[1].segments = [{
      id: 'video-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video', duration: 10 },
    }]
    const audioTrack = data.tracks.find((track) => track.type === 'audio')
    if (audioTrack) {
      audioTrack.segments = [{
        id: 'audio-active',
        start_frame: 0,
        end_frame: 10,
        color: audioTrack.color,
        content: { media_type: 'audio', duration: 10 },
      }]
    }
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByTestId('multitrack-ruler'))
    const trimButton = screen.getByRole('button', { name: buttonName })
    expect((trimButton as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(trimButton)

    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks[0].segments[0][frameKey]).toBe(expectedFrame)
    expect(updated.tracks[1].segments[0][frameKey]).toBe(expectedFrame)
    const updatedAudioTrack = updated.tracks.find((track) => track.type === 'audio')
    if (updatedAudioTrack) expect(updatedAudioTrack.segments[0][frameKey]).toBe(expectedFrame)
  })

  it('trims a selected video segment and keeps the matching task aligned', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    data.tracks[1].segments = [{
      id: 'video-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video', duration: 10 },
    }]
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByTestId('multitrack-ruler'))
    fireEvent.click(screen.getByRole('button', { name: 'select video segment' }))
    fireEvent.click(screen.getByRole('button', { name: 'trim left current time' }))

    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks[0].segments[0].start_frame).toBe(5)
    expect(updated.tracks[1].segments[0].start_frame).toBe(5)
  })

  it('clears selected segment ids after deleting selected segments', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    data.tracks[1].segments = [{
      id: 'video-active',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video', duration: 10 },
    }]
    const onChange = vi.fn()

    render(<MultiTrackWidget {...widgetProps()} value={data} onChange={onChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'select video and task' }))
    expect(screen.getByTestId('selected-segment-count').textContent).toBe('2')

    fireEvent.click(screen.getByRole('button', { name: 'delete selected' }))

    expect(screen.getByTestId('selected-segment-count').textContent).toBe('0')
    expect(screen.getByRole<HTMLButtonElement>('button', { name: 'delete selected' }).disabled).toBe(true)
    const updated = onChange.mock.lastCall?.[0] as TrackData
    expect(updated.tracks.flatMap((track) => track.segments.map((segment) => segment.id))).not.toContain('video-active')
  })
})
