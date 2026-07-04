import { createEvent, fireEvent, render, screen } from '@testing-library/react'
import type { MouseEvent } from 'react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { TrackArea } from '@/components/widgets/multitrack/TrackArea'
import { createDefaultTrackData } from '@/lib/multitrack-utils'

vi.mock('@/components/widgets/multitrack/VideoTrack', () => ({
  VideoTrack: ({ onDragPreviewChange }: {
    onDragPreviewChange: (segmentId: string, nextStartTime: number, clientY: number) => void
  }) => (
    <div data-testid="video-track">
      <button
        data-testid="preview-video-drag"
        onClick={() => onDragPreviewChange('video-first', 4, 50)}
      />
    </div>
  ),
}))

vi.mock('@/components/widgets/multitrack/AudioTrack', () => ({
  AudioTrack: ({ node, app }: { node: unknown; app: unknown }) => (
    <div data-testid="audio-track">{node && app ? 'connected' : 'missing'}</div>
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
  it('renders the add-track bar with audio and subtitle enabled', () => {
    const onAddTrack = vi.fn()
    render(
      <TooltipProvider>
        <TrackArea
          data={createDefaultTrackData()}
          width={480}
          currentTime={0}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
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
    expect((screen.getByRole('button', { name: 'Add video track' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Add subtitle track' }) as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(screen.getByRole('button', { name: 'Add audio track' }))
    expect(onAddTrack).toHaveBeenCalledWith('audio')
    fireEvent.click(screen.getByRole('button', { name: 'Add subtitle track' }))
    expect(onAddTrack).toHaveBeenCalledWith('subtitle')
  })

  it('disables audio and subtitle track creation after two tracks of each type', () => {
    const data = createDefaultTrackData()
    data.tracks.push(
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
    )
    const onAddTrack = vi.fn()

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
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

    expect((screen.getByRole('button', { name: 'Add audio track' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Add subtitle track' }) as HTMLButtonElement).disabled).toBe(true)
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
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{ inputs: [] }}
          app={{ graph: {} }}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
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
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
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
    const deleteButton = screen.getByRole('button', { name: 'Delete Subtitle 1' })
    expect(deleteButton.parentElement).toBe(addControls)

    fireEvent.click(addButton)
    expect(onAddSubtitleSegment).toHaveBeenCalledWith('subtitle-track')

    fireEvent.click(deleteButton)
    expect(onDeleteTrack).toHaveBeenCalledWith('subtitle-track')
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
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
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
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
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

    expect(screen.getByTestId('segment-task-first').getAttribute('data-start-frame')).toBe('3')
    expect(screen.getByTestId('segment-task-second').getAttribute('data-end-frame')).toBe('3')
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
            canvasScale={1}
            selectedSegmentIds={new Set()}
            node={{}}
            app={{}}
            onAddVideo={vi.fn()}
            onAddAudio={vi.fn()}
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
            canvasScale={1}
            selectedSegmentIds={new Set()}
            node={{}}
            app={{}}
            onAddVideo={vi.fn()}
            onAddAudio={vi.fn()}
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
          canvasScale={0.5}
          selectedSegmentIds={new Set()}
          node={{}}
          app={{}}
          onAddVideo={onAddVideo}
          onAddAudio={onAddAudio}
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
    fireEvent(area, drop)

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
    fireEvent(area, videoDrop)

    await vi.waitFor(() => {
      expect(onAddVideo).toHaveBeenCalledWith(data.tracks[1].id, 'uploads/clip.wav', 'input', 63)
    })
  })
})
