import type { ReactNode } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import type { MultiTrack } from '@/types/multitrack'
import { VideoTrack } from '@/components/widgets/multitrack/VideoTrack'
import { TooltipProvider } from '@/components/ui/tooltip'

vi.mock('@/lib/i18n', () => ({
  useT: () => (key: string) => key,
}))

vi.mock('@/components/ui/popover', () => ({
  Popover: ({ children }: { children: ReactNode }) => <>{children}</>,
  PopoverAnchor: ({ children }: { children: ReactNode }) => <>{children}</>,
  PopoverContent: ({ children }: { children: ReactNode }) => <>{children}</>,
  PopoverTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
}))

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: ({ value, defaultTab, slotItems = [], onChange }: {
    value: string
    defaultTab: string
    slotItems?: Array<{ value: string }>
    onChange: (value: string, source: 'input') => void
  }) => (
    <button
      type="button"
      data-testid="media-selector"
      data-value={value}
      data-default-tab={defaultTab}
      data-slot-items={slotItems.map((item) => item.value).join(',')}
      onClick={() => onChange(slotItems[0]?.value ?? 'new.mp4', 'input')}
    />
  ),
}))

vi.mock('@/components/widgets/multitrack/MultiTrackSegmentBlock', () => ({
  MultiTrackSegmentBlock: ({ segment, sharedReference, onSharedReferenceToggle, onDoubleClick }: {
    segment: { id: string }
    sharedReference?: boolean
    onSharedReferenceToggle?: (enabled: boolean) => void
    onDoubleClick?: (segmentId: string, event: React.MouseEvent) => void
  }) => (
    <>
      <button type="button" onDoubleClick={(event) => onDoubleClick?.(segment.id, event)}>
        {segment.id}
      </button>
      <button
        type="button"
        aria-label={`shared-${segment.id}`}
        aria-pressed={sharedReference}
        onClick={() => onSharedReferenceToggle?.(!sharedReference)}
      />
    </>
  ),
}))

function videoTrack(): MultiTrack {
  return {
    id: 'video-track',
    name: 'Video',
    type: 'video',
    color: 'var(--primary)',
    muted: false,
    locked: false,
    segments: [{
      id: 'video-segment',
      start_frame: 0,
      end_frame: 23,
      color: 'var(--primary)',
      content: {
        media_type: 'video',
        source_type: 'output',
        file_path: 'renders/clip.mp4',
        file_name: 'clip.mp4',
      },
    }],
  }
}

describe('VideoTrack', () => {
  it('adds a video using the exact internal gap range', () => {
    const track = videoTrack()
    track.segments[0].content.shared_reference = true
    track.segments = [
      { ...track.segments[0], id: 'first', start_frame: 0, end_frame: 24 },
      { ...track.segments[0], id: 'second', start_frame: 72, end_frame: 96 },
    ]
    const onAddVideo = vi.fn()
    const onSharedReferenceChange = vi.fn()

    render(
      <TooltipProvider>
        <VideoTrack
          track={track}
          totalLength={120}
          frameRate={24}
          width={480}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={null}
          app={null}
          onAddVideo={onAddVideo}
          onSelectSegment={vi.fn()}
          onDeleteSegment={vi.fn()}
          canDeleteTrack={false}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onSharedReferenceChange={onSharedReferenceChange}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onDragPreviewChange={vi.fn()}
          onDragPreviewEnd={vi.fn()}
          onReplaceVideo={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
          onCloneSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByTestId('track-gap-add-24-72'))
    fireEvent.click(screen.getAllByTestId('media-selector')[0])

    expect(onAddVideo).toHaveBeenCalledWith('video-track', 'new.mp4', 'input', 24, 72)
    expect(screen.getByRole('button', { name: 'shared-first' }).getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(screen.getByRole('button', { name: 'shared-first' }))
    expect(onSharedReferenceChange).toHaveBeenCalledWith('video-track', 'first', false)
  })

  it('preselects the current video when a segment is opened for replacement', () => {
    render(
      <TooltipProvider>
        <VideoTrack
        track={videoTrack()}
        totalLength={24}
        frameRate={24}
        width={480}
          canvasScale={1}
        selectedSegmentIds={new Set()}
        node={null}
        app={null}
        onAddVideo={vi.fn()}
        onSelectSegment={vi.fn()}
        onDeleteSegment={vi.fn()}
        canDeleteTrack={false}
        onDeleteTrack={vi.fn()}
        onTrackAudioSettingsChange={vi.fn()}
        onResizeSegment={vi.fn()}
        onResizeSegmentPreview={vi.fn()}
        onMoveSegment={vi.fn()}
        onDragPreviewChange={vi.fn()}
        onDragPreviewEnd={vi.fn()}
        onReplaceVideo={vi.fn()}
        onSmartSplit={vi.fn()}
        onSmartSplitTasks={vi.fn()}
        cutMode={false}
        onCutSegment={vi.fn()}
        onCloneSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    fireEvent.doubleClick(screen.getByRole('button', { name: 'video-segment' }), {
      clientX: 20,
      clientY: 10,
    })

    const selectors = screen.getAllByTestId('media-selector')
    const reselectSelector = selectors.at(-1)
    expect(reselectSelector?.getAttribute('data-value')).toBe('renders/clip.mp4')
    expect(reselectSelector?.getAttribute('data-default-tab')).toBe('outputs')
  })

  it('shows connected video inputs in the slot selector and preserves the slot on replacement', () => {
    const track = videoTrack()
    track.segments[0].content = {
      media_type: 'video',
      source_type: 'slot',
      slot_name: 'video1',
      file_name: 'video1',
    }
    const sourceNode = {
      outputs: [{ shape: 0, link: null }],
    }
    const node = { inputs: [{ name: 'video', type: 'VIDEO', link: 7 }] }
    const app = {
      graph: {
        links: { 7: { origin_id: 3, origin_slot: 0 } },
        getNodeById: () => sourceNode,
      },
    }
    const onReplaceVideo = vi.fn()

    render(
      <TooltipProvider>
        <VideoTrack
          track={track}
          totalLength={24}
          frameRate={24}
          width={480}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          node={node}
          app={app}
          onAddVideo={vi.fn()}
          onSelectSegment={vi.fn()}
          onDeleteSegment={vi.fn()}
          canDeleteTrack={false}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onDragPreviewChange={vi.fn()}
          onDragPreviewEnd={vi.fn()}
          onReplaceVideo={onReplaceVideo}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
          onCloneSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    fireEvent.doubleClick(screen.getByRole('button', { name: 'video-segment' }), {
      clientX: 20,
      clientY: 10,
    })
    const selector = screen.getAllByTestId('media-selector').at(-1)
    expect(selector?.getAttribute('data-value')).toBe('__slot__:video1')
    expect(selector?.getAttribute('data-default-tab')).toBe('slot')
    expect(selector?.getAttribute('data-slot-items')).toBe('__slot__:video')
  })
})
