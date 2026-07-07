import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { SubtitleTrack } from '@/components/widgets/multitrack/SubtitleTrack'
import { TooltipProvider } from '@/components/ui/tooltip'
import type { MultiTrack } from '@/types/multitrack'

describe('SubtitleTrack', () => {
  it('hides delete and keeps visibility toggle after subtitle segments', () => {
    const track: MultiTrack = {
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [{
        id: 'subtitle-segment',
        start_frame: 80,
        end_frame: 100,
        color: '#9D4937',
        content: { media_type: 'subtitle', text: 'Line' },
      }],
    }

    const onTrackVisibilityChange = vi.fn()
    const { container } = render(
      <TooltipProvider>
        <SubtitleTrack
          track={track}
          totalLength={100}
          frameRate={24}
          width={200}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          onSelectSegment={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackVisibilityChange={onTrackVisibilityChange}
          onEditSubtitleSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onDragPreviewChange={vi.fn()}
          onDragPreviewEnd={vi.fn()}
        />
      </TooltipProvider>,
    )

    const addButton = screen.getByRole('button', { name: 'Add subtitle' })
    const visibilityButton = screen.getByRole('button', { name: 'Hide Subtitle 1' })
    const actionGroup = addButton.parentElement

    expect(screen.queryByRole('button', { name: 'Delete Subtitle 1' })).toBeNull()
    expect(actionGroup).toBe(visibilityButton.parentElement)
    expect(actionGroup?.className).toContain('flex-row')
    expect(actionGroup?.style.left).toBe('206px')
    expect(container.querySelector('.lucide-eye-off')).not.toBeNull()
    fireEvent.click(visibilityButton)
    expect(onTrackVisibilityChange).toHaveBeenCalledWith('subtitle-track', false)
  })

  it('shows the reveal icon and dims subtitle segments when the track is hidden', () => {
    const track: MultiTrack = {
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      visible: false,
      segments: [{
        id: 'subtitle-segment',
        start_frame: 0,
        end_frame: 50,
        color: '#9D4937',
        content: { media_type: 'subtitle', text: 'Hidden line' },
      }],
    }

    const { container } = render(
      <TooltipProvider>
        <SubtitleTrack
          track={track}
          totalLength={100}
          frameRate={24}
          width={200}
          canvasScale={1}
          selectedSegmentIds={new Set()}
          onSelectSegment={vi.fn()}
          onAddSubtitleSegment={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackVisibilityChange={vi.fn()}
          onEditSubtitleSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onResizeSegmentPreview={vi.fn()}
          onMoveSegment={vi.fn()}
          onDragPreviewChange={vi.fn()}
          onDragPreviewEnd={vi.fn()}
        />
      </TooltipProvider>,
    )

    expect(screen.getByRole('button', { name: 'Show Subtitle 1' })).not.toBeNull()
    expect(container.querySelector('.lucide-eye')).not.toBeNull()
    expect(screen.getByRole('button', { name: /Hidden line/ }).className).toContain('opacity-50')
  })
})
