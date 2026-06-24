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
      cutMode={false}
      onToggleCutMode={vi.fn()}
    />,
  )
}

describe('MultiTrackToolbar', () => {
  it('toggles segment cut mode from the text cursor button', () => {
    const onToggleCutMode = vi.fn()
    render(
      <MultiTrackToolbar
        currentTime={0}
        totalLength={24}
        frameRate={24}
        isPlaying={false}
        zoom={1}
        timelineCollapsed={false}
        onPlayPause={vi.fn()}
        onZoomChange={vi.fn()}
        onToggleTimeline={vi.fn()}
        canDelete={false}
        onDeleteSelected={vi.fn()}
        cutMode
        onToggleCutMode={onToggleCutMode}
      />,
    )

    const button = screen.getByRole('button', { name: 'multitrack.cutMode' })
    expect(button.getAttribute('aria-pressed')).toBe('true')
    fireEvent.click(button)
    expect(onToggleCutMode).toHaveBeenCalledOnce()
  })

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
        cutMode={false}
        onToggleCutMode={vi.fn()}
      />,
    )

    expect(screen.getByRole('button', { name: 'multitrack.showTimeline' })).not.toBeNull()
    expect(container.querySelector('.lucide-maximize-2')).not.toBeNull()
  })

  it('uses a consistent icon size across toolbar controls', () => {
    const { container } = renderToolbar(false)

    const iconClasses = Array.from(container.querySelectorAll('svg')).map((icon) => icon.getAttribute('class') ?? '')

    expect(iconClasses.length).toBeGreaterThan(0)
    expect(iconClasses.every((className) => className.includes('h-3.5') && className.includes('w-3.5'))).toBe(true)
  })
})
