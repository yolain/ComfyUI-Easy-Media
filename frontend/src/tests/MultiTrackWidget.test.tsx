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
  TrackArea: ({ data, node, app, onCloneTaskSegment, onAddTrack, onAddAudio }: {
    data: TrackData
    node: unknown
    app: unknown
    onCloneTaskSegment: (trackId: string, segmentId: string) => void
    onAddTrack: (type: 'audio') => void
    onAddAudio: (trackId: string, filePath: string, sourceType: 'input', previewUrl?: string) => void
  }) => {
    const taskTrack = data.tracks.find((track) => track.type === 'task')
    const audioTrack = data.tracks.find((track) => track.type === 'audio')
    const segment = taskTrack?.segments[0]
    return (
      <div data-testid="multitrack-track-area">
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
  MultiTrackToolbar: ({ onToggleTimeline }: { onToggleTimeline: () => void }) => (
    <button type="button" onClick={onToggleTimeline}>toggle timeline</button>
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
})
