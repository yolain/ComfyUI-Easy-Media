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
  MediaSelector: ({ value, defaultTab }: { value: string; defaultTab: string }) => (
    <div data-testid="media-selector" data-value={value} data-default-tab={defaultTab} />
  ),
}))

vi.mock('@/components/widgets/multitrack/MultiTrackSegmentBlock', () => ({
  MultiTrackSegmentBlock: ({ segment, onDoubleClick }: {
    segment: { id: string }
    onDoubleClick?: (segmentId: string, event: React.MouseEvent) => void
  }) => (
    <button type="button" onDoubleClick={(event) => onDoubleClick?.(segment.id, event)}>
      {segment.id}
    </button>
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
  it('preselects the current video when a segment is opened for replacement', () => {
    render(
      <TooltipProvider>
        <VideoTrack
        track={videoTrack()}
        totalLength={24}
        frameRate={24}
        width={480}
        canvasScale={1}
        selectedSegmentId={null}
        onAddVideo={vi.fn()}
        onSelectSegment={vi.fn()}
        onDeleteSegment={vi.fn()}
        canDeleteTrack={false}
        onDeleteTrack={vi.fn()}
        onTrackAudioSettingsChange={vi.fn()}
        onResizeSegment={vi.fn()}
        onMoveSegment={vi.fn()}
        onDragPreviewChange={vi.fn()}
        onDragPreviewEnd={vi.fn()}
        onReplaceVideo={vi.fn()}
        onSmartSplit={vi.fn()}
        onSmartSplitTasks={vi.fn()}
        cutMode={false}
        onCutSegment={vi.fn()}
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
})
