import { act, createEvent, fireEvent, render, screen } from '@testing-library/react'
import type { MouseEvent } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { TrackArea } from '@/components/widgets/multitrack/TrackArea'
import { createDefaultTrackData, getMultiTrackTrackHeight } from '@/lib/multitrack-utils'

vi.mock('@/components/widgets/multitrack/VideoTrack', () => ({
  VideoTrack: ({ track, onDragPreviewChange }: {
    track: { id: string; name: string }
    onDragPreviewChange: (segmentId: string, nextStartTime: number, clientY: number) => void
  }) => (
    <div data-testid="video-track" data-multitrack-track-id={track.id}>
      <button
        type="button"
        draggable
        data-multitrack-track-drag-handle={track.id}
        aria-label={`Reorder ${track.name}`}
      />
      <button
        data-testid="preview-video-drag"
        onClick={() => onDragPreviewChange('video-first', 4, 50)}
      />
    </div>
  ),
}))

vi.mock('@/components/widgets/multitrack/AudioTrack', () => ({
  AudioTrack: ({ track, node, app }: {
    track: { id: string; name: string }
    node: unknown
    app: unknown
  }) => (
    <div data-testid="audio-track" data-multitrack-track-id={track.id}>
      <button
        type="button"
        draggable
        data-multitrack-track-drag-handle={track.id}
        aria-label={`Reorder ${track.name}`}
      />
      {node && app ? 'connected' : 'missing'}
    </div>
  ),
}))

vi.mock('@/components/widgets/multitrack/MultiTrackSegmentBlock', () => ({
  MultiTrackSegmentBlock: ({
    segment,
    onDoubleClick,
  }: {
    segment: { id: string; start_frame: number; end_frame: number }
    onDoubleClick?: (segmentId: string, event: MouseEvent) => void
  }) => (
    <div
      data-testid={`segment-${segment.id}`}
      data-start-frame={segment.start_frame}
      data-end-frame={segment.end_frame}
      onDoubleClick={(event) => onDoubleClick?.(segment.id, event)}
    />
  ),
}))

describe('TrackArea track controls', () => {
  it('matches the task track height to film tracks in overview mode', () => {
    expect(getMultiTrackTrackHeight('task', true)).toBe(getMultiTrackTrackHeight('video', true))
    expect(getMultiTrackTrackHeight('task', false)).toBeLessThan(getMultiTrackTrackHeight('video', false))
  })

  it('swaps non-task tracks by dragging their left-side handles', () => {
    const data = createDefaultTrackData()
    const videoTrack = data.tracks.find((track) => track.type === 'video')!
    const audioTrack = {
      ...videoTrack,
      id: 'audio-track',
      name: 'Audio 0',
      type: 'audio' as const,
      segments: [],
    }
    data.tracks.push(audioTrack)
    const onReorderTrack = vi.fn()
    const transferData = new Map<string, string>()
    const dataTransfer = {
      files: [],
      items: [],
      types: [],
      effectAllowed: 'none',
      dropEffect: 'none',
      setData: (type: string, value: string) => transferData.set(type, value),
      getData: (type: string) => transferData.get(type) ?? '',
    }

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onReorderTrack={onReorderTrack}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    fireEvent.dragStart(screen.getByRole('button', { name: `Reorder ${videoTrack.name}` }), { dataTransfer })
    fireEvent.dragOver(screen.getByTestId('audio-track'), { dataTransfer })
    fireEvent.drop(screen.getByTestId('audio-track'), { dataTransfer })

    expect(onReorderTrack).toHaveBeenCalledWith(videoTrack.id, audioTrack.id)

    fireEvent.dragStart(screen.getByRole('button', { name: `Reorder ${audioTrack.name}` }), { dataTransfer })
    fireEvent.dragOver(screen.getByTestId(`track-row-${data.tracks[0].id}`), { dataTransfer })
    fireEvent.drop(screen.getByTestId(`track-row-${data.tracks[0].id}`), { dataTransfer })

    expect(onReorderTrack).toHaveBeenCalledTimes(1)
  })

  it('adds a task segment using the exact internal gap range', () => {
    const data = createDefaultTrackData()
    const taskTrack = data.tracks.find((track) => track.type === 'task')!
    taskTrack.segments = [
      { id: 'first', start_frame: 0, end_frame: 24, color: taskTrack.color, content: { media_type: 'none' } },
      { id: 'second', start_frame: 72, end_frame: 96, color: taskTrack.color, content: { media_type: 'none' } },
    ]
    data.total_length = 120
    const onAddTaskSegment = vi.fn()

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={onAddTaskSegment}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByTestId('track-gap-add-24-72'))
    expect(onAddTaskSegment).toHaveBeenCalledWith(taskTrack.id, 24, 72)
  })

  it('renders the add-track bar with video, audio, and subtitle enabled below their limits', () => {
    const onAddTrack = vi.fn()
    render(
      <TooltipProvider>
        <TrackArea
          data={createDefaultTrackData()}
          width={480}
          currentTime={0}
          snapEnabled={true}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={onAddTrack}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    const addTrackLabel = screen.getByText('Add track:')
    expect(addTrackLabel.parentElement?.className).toContain('border-b')
    expect((screen.getByRole('button', { name: 'Add video track' }) as HTMLButtonElement).disabled).toBe(false)
    expect((screen.getByRole('button', { name: 'Add subtitle track' }) as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Add video track' }))
    expect(onAddTrack).toHaveBeenCalledWith('video')
    fireEvent.click(screen.getByRole('button', { name: 'Add audio track' }))
    expect(onAddTrack).toHaveBeenCalledWith('audio')
    fireEvent.click(screen.getByRole('button', { name: 'Add subtitle track' }))
    expect(onAddTrack).toHaveBeenCalledWith('subtitle')
  })

  it('disables video, audio, and subtitle track creation after three tracks of each type', () => {
    const data = createDefaultTrackData()
    data.tracks.push(
      {
        id: 'video-b',
        name: 'Video 1',
        type: 'video',
        color: 'var(--highlight)',
        muted: false,
        locked: false,
        segments: [],
      },
      {
        id: 'video-c',
        name: 'Video 2',
        type: 'video',
        color: 'var(--highlight)',
        muted: false,
        locked: false,
        segments: [],
      },
      {
        id: 'audio-a',
        name: 'Audio 0',
        type: 'audio',
        color: 'var(--highlight)',
        muted: false,
        locked: false,
        segments: [],
      },
      {
        id: 'audio-b',
        name: 'Audio 1',
        type: 'audio',
        color: 'var(--highlight)',
        muted: false,
        locked: false,
        segments: [],
      },
      {
        id: 'audio-c',
        name: 'Audio 2',
        type: 'audio',
        color: 'var(--highlight)',
        muted: false,
        locked: false,
        segments: [],
      },
      {
        id: 'subtitle-a',
        name: 'Subtitle 1',
        type: 'subtitle',
        color: '#9D4937',
        muted: false,
        locked: false,
        segments: [],
      },
      {
        id: 'subtitle-b',
        name: 'Subtitle 2',
        type: 'subtitle',
        color: '#9D4937',
        muted: false,
        locked: false,
        segments: [],
      },
      {
        id: 'subtitle-c',
        name: 'Subtitle 3',
        type: 'subtitle',
        color: '#9D4937',
        muted: false,
        locked: false,
        segments: [],
      },
    )
    const onAddTrack = vi.fn()

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled={true}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={onAddTrack}
          onAddSubtitleSegment={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    expect((screen.getByRole('button', { name: 'Add video track' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Add audio track' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Add subtitle track' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Add video track' }))
    fireEvent.click(screen.getByRole('button', { name: 'Add audio track' }))
    fireEvent.click(screen.getByRole('button', { name: 'Add subtitle track' }))
    expect(onAddTrack).not.toHaveBeenCalled()
  })

  it('passes ComfyUI graph context to each audio track', () => {
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

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled={true}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{ inputs: [] }}
          app={{ graph: {} }}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
            onAddSubtitleSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    expect(screen.getByTestId('audio-track').textContent).toBe('connected')
  })

  it('shows horizontal subtitle controls after the last subtitle segment and adds subtitle segments', () => {
    const data = createDefaultTrackData()
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [{
        id: 'subtitle-first',
        start_frame: 0,
        end_frame: 24,
        color: '#9D4937',
        content: { media_type: 'subtitle', text: 'Existing' },
      }],
    })
    const onAddSubtitleSegment = vi.fn()
    const onDeleteTrack = vi.fn()

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled={true}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onAddSubtitleSegment={onAddSubtitleSegment}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={onDeleteTrack}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    const addButton = screen.getByRole('button', { name: 'Add subtitle' })
    const addControls = addButton.parentElement
    expect(addControls?.className).toContain('flex-row')
    expect((addControls as HTMLElement).style.left).toBe('86.80000000000001px')
    const visibilityButton = screen.getByRole('button', { name: 'Hide Subtitle 1' })
    expect(screen.queryByRole('button', { name: 'Delete Subtitle 1' })).toBeNull()
    expect(visibilityButton.parentElement).toBe(addControls)

    fireEvent.click(addButton)
    expect(onAddSubtitleSegment).toHaveBeenCalledWith('subtitle-track')

    expect(onDeleteTrack).not.toHaveBeenCalled()
  })

  it('requests preview subtitle editing after double clicking a track subtitle segment', () => {
    const data = createDefaultTrackData()
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [{
        id: 'subtitle-first',
        start_frame: 12,
        end_frame: 36,
        color: '#9D4937',
        content: { media_type: 'subtitle', text: 'Existing' },
      }],
    })
    const onSelectSegment = vi.fn()
    const onEditSubtitleSegment = vi.fn()

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled={true}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
            onAddSubtitleSegment={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onSelectSegment={onSelectSegment}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onEditSubtitleSegment={onEditSubtitleSegment}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    fireEvent.doubleClick(screen.getByTestId('segment-subtitle-first'))

    expect(onSelectSegment).toHaveBeenCalledWith('subtitle-first')
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(screen.queryByTestId('subtitle-track-editor')).toBeNull()
    expect(onEditSubtitleSegment).toHaveBeenCalledWith('subtitle-first')
  })

  it('previews matching task segments together with a dragged video segment', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [
      {
        id: 'task-first',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
      {
        id: 'task-second',
        start_frame: 2,
        end_frame: 5,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
    ]
    data.tracks[1].segments = [
      {
        id: 'video-first',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 2 },
      },
      {
        id: 'video-second',
        start_frame: 2,
        end_frame: 5,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 3 },
      },
    ]

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled={false}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByTestId('preview-video-drag'))

    expect(screen.getByTestId('segment-task-first').getAttribute('data-start-frame')).toBe('5')
    expect(screen.getByTestId('segment-task-second').getAttribute('data-end-frame')).toBe('5')
  })

  it('selects segments across tracks with a marquee drag', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [{
      id: 'task-first',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[0].color,
      content: { media_type: 'none', task_mode: 'default' },
    }]
    data.tracks[1].segments = [{
      id: 'video-first',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video', duration: 10 },
    }]
    const onSelectSegments = vi.fn()
    const onClearSelection = vi.fn()

    render(
      <TooltipProvider>
        <div data-testid="outer-clear-zone" onClick={onClearSelection}>
          <TrackArea
            data={data}
            width={480}
            currentTime={0}
            snapEnabled={true}
            canvasScale={1}
            selectedSegmentIds={new Set()}
            node={{}}
            app={{}}
            onAddVideo={vi.fn()}
            onAddAudio={vi.fn()}
            onReplaceAudio={vi.fn()}
            onAddTrack={vi.fn()}
            onReplaceVideo={vi.fn()}
            onAddTaskSegment={vi.fn()}
            onAddSubtitleSegment={vi.fn()}
            onSelectSegment={vi.fn()}
            onSelectSegments={onSelectSegments}
            onClearSelection={onClearSelection}
            onDeleteSegment={vi.fn()}
            onDeleteTrack={vi.fn()}
            onTrackAudioSettingsChange={vi.fn()}
            onDistributeTaskSegments={vi.fn()}
            onCloneTaskSegment={vi.fn()}
            onResizeSegment={vi.fn()}
            onResizeSegmentPreview={vi.fn()}
            onMoveSegment={vi.fn()}
            onSmartSplit={vi.fn()}
            onSmartSplitTasks={vi.fn()}
            cutMode={false}
            onCutSegment={vi.fn()}
          />
        </div>
      </TooltipProvider>,
    )

    const area = document.querySelector('[data-multitrack-track-area]') as HTMLDivElement
    vi.spyOn(area, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      top: 200,
      width: 480,
      height: 111,
      right: 580,
      bottom: 311,
      x: 100,
      y: 200,
      toJSON: () => ({}),
    })

    fireEvent.mouseDown(area, { button: 0, clientX: 120, clientY: 202 })
    fireEvent.mouseMove(window, { clientX: 190, clientY: 292 })
    expect(document.querySelector('.bg-primary\\/10')).toBeTruthy()
    fireEvent.mouseUp(window, { clientX: 190, clientY: 292 })

    expect(onSelectSegments).toHaveBeenCalledWith(['task-first', 'video-first'])
  })

  it('keeps selecting when an add-track marquee drag leaves above the track area', () => {
    const data = createDefaultTrackData()
    data.tracks[1].segments = [{
      id: 'video-first',
      start_frame: 0,
      end_frame: 10,
      color: data.tracks[1].color,
      content: { media_type: 'video', duration: 10 },
    }]
    const onSelectSegments = vi.fn()
    const onClearSelection = vi.fn()

    render(
      <TooltipProvider>
        <div data-testid="outer-clear-zone" onClick={onClearSelection}>
          <TrackArea
            data={data}
            width={480}
            currentTime={0}
            snapEnabled={true}
            canvasScale={1}
            selectedSegmentIds={new Set()}
            node={{}}
            app={{}}
            onAddVideo={vi.fn()}
            onAddAudio={vi.fn()}
            onReplaceAudio={vi.fn()}
            onAddTrack={vi.fn()}
            onReplaceVideo={vi.fn()}
            onAddTaskSegment={vi.fn()}
            onAddSubtitleSegment={vi.fn()}
            onSelectSegment={vi.fn()}
            onSelectSegments={onSelectSegments}
            onClearSelection={onClearSelection}
            onDeleteSegment={vi.fn()}
            onDeleteTrack={vi.fn()}
            onTrackAudioSettingsChange={vi.fn()}
            onDistributeTaskSegments={vi.fn()}
            onCloneTaskSegment={vi.fn()}
            onResizeSegment={vi.fn()}
            onResizeSegmentPreview={vi.fn()}
            onMoveSegment={vi.fn()}
            onSmartSplit={vi.fn()}
            onSmartSplitTasks={vi.fn()}
            cutMode={false}
            onCutSegment={vi.fn()}
          />
        </div>
      </TooltipProvider>,
    )

    const area = document.querySelector('[data-multitrack-track-area]') as HTMLDivElement
    vi.spyOn(area, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      top: 200,
      width: 480,
      height: 118,
      right: 580,
      bottom: 318,
      x: 100,
      y: 200,
      toJSON: () => ({}),
    })

    fireEvent.mouseDown(area, { button: 0, clientX: 190, clientY: 313 })
    fireEvent.mouseMove(window, { clientX: 120, clientY: 180 })
    fireEvent.mouseUp(window, { clientX: 120, clientY: 180 })
    fireEvent.click(screen.getByTestId('outer-clear-zone'))

    expect(onSelectSegments).toHaveBeenCalledWith(['video-first'])
    expect(onClearSelection).not.toHaveBeenCalled()
  })

  it('shows a loading state while an external file is uploading', async () => {
    const data = createDefaultTrackData()
    const onAddVideo = vi.fn()
    let resolveUpload!: (response: {
      ok: boolean
      json: () => Promise<{ name: string; subfolder: string }>
    }) => void
    const upload = new Promise<{
      ok: boolean
      json: () => Promise<{ name: string; subfolder: string }>
    }>((resolve) => {
      resolveUpload = resolve
    })
    vi.stubGlobal('fetch', vi.fn().mockReturnValue(upload))

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={onAddVideo}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    const area = document.querySelector('[data-multitrack-track-area]') as HTMLDivElement
    vi.spyOn(area, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      top: 200,
      width: 480,
      height: 118,
      right: 580,
      bottom: 318,
      x: 100,
      y: 200,
      toJSON: () => ({}),
    })
    const file = new File(['video'], 'clip.mp4', { type: 'video/mp4' })
    const dataTransfer = {
      files: [file],
      items: [{ kind: 'file', type: 'video/mp4', getAsFile: () => file }],
      types: ['Files'],
      dropEffect: 'none',
    }
    const drop = createEvent.drop(area)
    Object.defineProperties(drop, {
      clientX: { value: 220 },
      clientY: { value: 245 },
      dataTransfer: { value: dataTransfer },
    })

    fireEvent(area, drop)

    expect(screen.getByText('Uploading clip.mp4...')).toBeTruthy()
    expect(onAddVideo).not.toHaveBeenCalled()

    await act(async () => {
      resolveUpload({
        ok: true,
        json: () => Promise.resolve({ name: 'clip.mp4', subfolder: 'uploads' }),
      })
      await upload
    })

    await vi.waitFor(() => {
      expect(screen.queryByText('Uploading clip.mp4...')).toBeNull()
      expect(onAddVideo).toHaveBeenCalledWith(data.tracks[1].id, 'uploads/clip.mp4', 'input', 27)
    })
  })

  it('uploads external audio and video files at canvas-scaled pointer positions', async () => {
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
    const onAddAudio = vi.fn()
    const onAddVideo = vi.fn()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ name: 'clip.wav', subfolder: 'uploads' }),
    }))

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled={true}
          canvasScale={0.5}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={onAddVideo}
          onAddAudio={onAddAudio}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    const area = document.querySelector('[data-multitrack-track-area]') as HTMLDivElement
    vi.spyOn(area, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      top: 200,
      width: 240,
      height: 111,
      right: 340,
      bottom: 311,
      x: 100,
      y: 200,
      toJSON: () => ({}),
    })
    const file = new File(['audio'], 'clip.wav', { type: 'audio/wav' })

    const dataTransfer = {
      files: [],
      items: [{ kind: 'file', type: 'audio/wav', getAsFile: () => null }],
      types: ['Files'],
      dropEffect: 'none',
    }
    const dragOver = createEvent.dragOver(area)
    Object.defineProperties(dragOver, {
      clientX: { value: 220 },
      clientY: { value: 250 },
      dataTransfer: { value: dataTransfer },
    })
    fireEvent(area, dragOver)
    expect(dragOver.defaultPrevented).toBe(true)
    expect(screen.getByTestId('external-media-drop-slot')).toBeTruthy()
    dataTransfer.files = [file] as never[]
    dataTransfer.items = [{ kind: 'file', type: 'audio/wav', getAsFile: () => file }] as never[]
    const drop = createEvent.drop(area)
    Object.defineProperties(drop, {
      clientX: { value: 220 },
      clientY: { value: 250 },
      dataTransfer: { value: dataTransfer },
    })
    await act(async () => {
      fireEvent(area, drop)
    })

    await vi.waitFor(() => {
      expect(onAddAudio).toHaveBeenCalledWith('audio-track', 'uploads/clip.wav', 'input', undefined, 63)
    })

    const videoFile = new File(['video'], 'clip.mp4', { type: 'video/mp4' })
    const videoDataTransfer = {
      files: [videoFile],
      items: [{ kind: 'file', type: 'video/mp4', getAsFile: () => videoFile }],
      types: ['Files'],
      dropEffect: 'none',
    }
    const videoDrop = createEvent.drop(area)
    Object.defineProperties(videoDrop, {
      clientX: { value: 220 },
      clientY: { value: 225 },
      dataTransfer: { value: videoDataTransfer },
    })
    await act(async () => {
      fireEvent(area, videoDrop)
    })

    await vi.waitFor(() => {
      expect(onAddVideo).toHaveBeenCalledWith(data.tracks[1].id, 'uploads/clip.wav', 'input', 63)
    })
  })

  it('uploads an external image and creates a task segment containing it', async () => {
    const data = createDefaultTrackData()
    const taskTrack = data.tracks.find((track) => track.type === 'task')!
    taskTrack.segments = []
    const onAddTaskSegment = vi.fn()
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ name: 'reference.png', subfolder: 'uploads' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={onAddTaskSegment}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    const area = document.querySelector('[data-multitrack-track-area]') as HTMLDivElement
    vi.spyOn(area, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      top: 200,
      width: 480,
      height: 118,
      right: 580,
      bottom: 318,
      x: 100,
      y: 200,
      toJSON: () => ({}),
    })
    const imageFile = new File(['image'], 'reference.png', { type: 'image/png' })
    const dataTransfer = {
      files: [imageFile],
      items: [{ kind: 'file', type: 'image/png', getAsFile: () => imageFile }],
      types: ['Files'],
      dropEffect: 'none',
    }
    const dragOver = createEvent.dragOver(area)
    Object.defineProperties(dragOver, {
      clientX: { value: 220 },
      clientY: { value: 210 },
      dataTransfer: { value: dataTransfer },
    })
    fireEvent(area, dragOver)

    expect(dragOver.defaultPrevented).toBe(true)
    expect(screen.getByTestId('external-media-drop-slot')).toBeTruthy()

    const drop = createEvent.drop(area)
    Object.defineProperties(drop, {
      clientX: { value: 220 },
      clientY: { value: 210 },
      dataTransfer: { value: dataTransfer },
    })
    await act(async () => {
      fireEvent(area, drop)
    })

    await vi.waitFor(() => {
      expect(onAddTaskSegment).toHaveBeenCalledWith(taskTrack.id, 27, undefined, [
        expect.objectContaining({
          source_type: 'input',
          file_path: 'uploads/reference.png',
          file_name: 'reference.png',
        }),
      ])
    })
    expect(fetchMock).toHaveBeenCalledWith('/upload/image', expect.objectContaining({ method: 'POST' }))
  })

  it('reads an external SRT file dropped onto a subtitle track at the pointer frame', async () => {
    const data = createDefaultTrackData()
    data.total_length = 240
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [],
    })
    const onImportSubtitles = vi.fn()
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          snapEnabled
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onReplaceAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onImportSubtitles={onImportSubtitles}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onSelectSegments={vi.fn()}
          onClearSelection={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    const area = document.querySelector('[data-multitrack-track-area]') as HTMLDivElement
    vi.spyOn(area, 'getBoundingClientRect').mockReturnValue({
      left: 100,
      top: 200,
      width: 480,
      height: 148,
      right: 580,
      bottom: 348,
      x: 100,
      y: 200,
      toJSON: () => ({}),
    })
    const srtText = '1\n00:00:01,000 --> 00:00:02,000\nHello\n'
    const subtitleFile = Object.assign(
      new File([srtText], 'captions.srt', { type: 'application/x-subrip' }),
      { text: () => Promise.resolve(srtText) },
    )
    const dataTransfer = {
      files: [subtitleFile],
      items: [{ kind: 'file', type: 'application/x-subrip', getAsFile: () => subtitleFile }],
      types: ['Files'],
      dropEffect: 'none',
    }
    const dragOver = createEvent.dragOver(area)
    Object.defineProperties(dragOver, {
      clientX: { value: 276 },
      clientY: { value: 300 },
      dataTransfer: { value: dataTransfer },
    })
    fireEvent(area, dragOver)

    expect(dragOver.defaultPrevented).toBe(true)
    expect(screen.getByTestId('external-media-drop-slot')).toBeTruthy()

    const drop = createEvent.drop(area)
    Object.defineProperties(drop, {
      clientX: { value: 276 },
      clientY: { value: 300 },
      dataTransfer: { value: dataTransfer },
    })
    await act(async () => {
      fireEvent(area, drop)
    })

    await vi.waitFor(() => {
      expect(onImportSubtitles).toHaveBeenCalledWith('subtitle-track', srtText, 88)
    })
    expect(fetchMock).not.toHaveBeenCalled()
  })
})
