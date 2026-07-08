import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { TrackAudioControls } from '@/components/widgets/multitrack/TrackAudioControls'
import type { MultiTrack } from '@/types/multitrack'

describe('TrackAudioControls', () => {
  it('toggles mute and solo independently', () => {
    const track: MultiTrack = {
      id: 'audio',
      name: 'Audio 1',
      type: 'audio',
      color: 'var(--multitrack-audio-bg)',
      muted: false,
      solo: false,
      locked: false,
      segments: [],
    }
    const onChange = vi.fn()
    render(
      <TooltipProvider>
        <TrackAudioControls track={track} icon={<span>icon</span>} onChange={onChange} />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Mute Audio 1' }))
    fireEvent.click(screen.getByRole('button', { name: 'Solo Audio 1' }))
    expect(screen.getByRole('button', { name: 'Mute Audio 1' }).classList).toContain('bg-card')
    expect(screen.getByRole('button', { name: 'Mute Audio 1' }).classList).toContain('text-[8px]')
    expect(screen.getByRole('button', { name: 'Solo Audio 1' }).classList).toContain('bg-card')
    expect(screen.getByRole('button', { name: 'Solo Audio 1' }).classList).toContain('text-[8px]')
    expect(onChange).toHaveBeenNthCalledWith(1, { muted: true })
    expect(onChange).toHaveBeenNthCalledWith(2, { solo: true })
  })

  it('preserves selection only when controls belong to the selected track', () => {
    const track: MultiTrack = {
      id: 'audio',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--multitrack-audio-bg)',
      muted: false,
      locked: false,
      segments: [],
    }
    const parentClick = vi.fn()
    const { rerender } = render(
      <div onClick={parentClick}>
        <TooltipProvider>
          <TrackAudioControls track={track} icon={<span>icon</span>} preserveSelection onChange={vi.fn()} />
        </TooltipProvider>
      </div>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Mute Audio 0' }))
    expect(parentClick).not.toHaveBeenCalled()

    rerender(
      <div onClick={parentClick}>
        <TooltipProvider>
          <TrackAudioControls track={track} icon={<span>icon</span>} onChange={vi.fn()} />
        </TooltipProvider>
      </div>,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Mute Audio 0' }))
    expect(parentClick).toHaveBeenCalledTimes(1)
  })
})
