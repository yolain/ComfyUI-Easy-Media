import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { SubtitleTrack } from '@/components/widgets/multitrack/SubtitleTrack'
import { TooltipProvider } from '@/components/ui/tooltip'
import type { MultiTrack } from '@/types/multitrack'

describe('SubtitleTrack', () => {
  it('keeps add and delete actions side by side after the last subtitle segment', () => {
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

    render(
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
    const deleteButton = screen.getByRole('button', { name: 'Delete Subtitle 1' })
    const actionGroup = addButton.parentElement

    expect(actionGroup).toBe(deleteButton.parentElement)
    expect(actionGroup?.className).toContain('flex-row')
    expect(actionGroup?.style.left).toBe('206px')
  })
})
