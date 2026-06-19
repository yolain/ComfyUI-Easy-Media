import { fireEvent, render, screen } from '@testing-library/react'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { MultiTrackToolbar } from '@/components/widgets/multitrack/MultiTrackToolbar'

vi.mock('@/lib/i18n', () => ({
  useT: () => (key: string) => key,
}))

beforeAll(() => {
  vi.stubGlobal('ResizeObserver', class {
    observe() {}
    unobserve() {}
    disconnect() {}
  })
})

afterAll(() => {
  vi.unstubAllGlobals()
})

function renderToolbar(timelineCollapsed: boolean, onToggleTimeline = vi.fn()) {
  return render(
    <MultiTrackToolbar
      currentTime={0}
      totalLength={24}
      frameRate={24}
      isPlaying={false}
      zoom={1}
      timelineCollapsed={timelineCollapsed}
      onPlayPause={vi.fn()}
      onZoomChange={vi.fn()}
      onToggleTimeline={onToggleTimeline}
      canDelete={false}
      onDeleteSelected={vi.fn()}
    />,
  )
}

describe('MultiTrackToolbar', () => {
  it('shows the reversed timeline toggle icons and handles clicks', () => {
    const onToggleTimeline = vi.fn()
    const { container, rerender } = renderToolbar(false, onToggleTimeline)

    const collapseButton = screen.getByRole('button', { name: 'multitrack.hideTimeline' })
    expect(container.querySelector('.lucide-minimize-2')).not.toBeNull()
    fireEvent.click(collapseButton)
    expect(onToggleTimeline).toHaveBeenCalledOnce()

    rerender(
      <MultiTrackToolbar
        currentTime={0}
        totalLength={24}
        frameRate={24}
        isPlaying={false}
        zoom={1}
        timelineCollapsed
        onPlayPause={vi.fn()}
        onZoomChange={vi.fn()}
        onToggleTimeline={onToggleTimeline}
        canDelete={false}
        onDeleteSelected={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: 'multitrack.showTimeline' })).not.toBeNull()
    expect(container.querySelector('.lucide-maximize-2')).not.toBeNull()
  })
})
