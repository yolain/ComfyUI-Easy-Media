import { fireEvent, render, screen } from '@testing-library/react'
import type { ComponentProps } from 'react'
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest'
import { MultiTrackToolbar } from '@/components/widgets/multitrack/MultiTrackToolbar'
import { TooltipProvider } from '@/components/ui/tooltip'

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

function renderToolbar(
  timelineCollapsed: boolean,
  overrides: Partial<ComponentProps<typeof MultiTrackToolbar>> = {},
) {
  return render(
    <TooltipProvider>
      <MultiTrackToolbar
        currentTime={0}
        totalLength={24}
        frameRate={24}
        isPlaying={false}
        zoom={1}
        snapEnabled
        timelineCollapsed={timelineCollapsed}
        onPlayPause={vi.fn()}
        onZoomChange={vi.fn()}
        onSnapEnabledChange={vi.fn()}
        onToggleTimeline={vi.fn()}
        canDelete={false}
        onDeleteSelected={vi.fn()}
        onCutAtCurrentTime={vi.fn()}
        canTrimCenter={false}
        canTrimLeft={false}
        canTrimRight={false}
        onTrimLeftAtCurrentTime={vi.fn()}
        onTrimRightAtCurrentTime={vi.fn()}
        canUndo={false}
        canRedo={false}
        onUndo={vi.fn()}
        onRedo={vi.fn()}
        {...overrides}
      />
    </TooltipProvider>,
  )
}

describe('MultiTrackToolbar', () => {
  it('cuts at the current time from the scissors button', () => {
    const onCutAtCurrentTime = vi.fn()
    const { rerender } = renderToolbar(false, { onCutAtCurrentTime, canTrimCenter: true })

    const button = screen.getByRole('button', { name: 'multitrack.cutMode' })
    expect(button.hasAttribute('aria-pressed')).toBe(false)
    fireEvent.click(button)
    expect(onCutAtCurrentTime).toHaveBeenCalledOnce()

    rerender(
      <TooltipProvider>
        <MultiTrackToolbar
          currentTime={0}
          totalLength={24}
          frameRate={24}
          isPlaying={false}
          zoom={1}
          snapEnabled
          timelineCollapsed={false}
          onPlayPause={vi.fn()}
          onZoomChange={vi.fn()}
          onSnapEnabledChange={vi.fn()}
          onToggleTimeline={vi.fn()}
          canDelete={false}
          onDeleteSelected={vi.fn()}
          onCutAtCurrentTime={onCutAtCurrentTime}
          canTrimCenter={false}
          canTrimLeft={false}
          canTrimRight={false}
          onTrimLeftAtCurrentTime={vi.fn()}
          onTrimRightAtCurrentTime={vi.fn()}
          canUndo={false}
          canRedo={false}
          onUndo={vi.fn()}
          onRedo={vi.fn()}
        />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'multitrack.cutMode' }))
    expect(onCutAtCurrentTime).toHaveBeenCalledOnce()
  })

  it('shows the reversed timeline toggle icons and handles clicks', () => {
    const onToggleTimeline = vi.fn()
    const { container, rerender } = renderToolbar(false, { onToggleTimeline })

    const collapseButton = screen.getByRole('button', { name: 'multitrack.hideTimeline' })
    expect(container.querySelector('.lucide-minimize-2')).not.toBeNull()
    fireEvent.click(collapseButton)
    expect(onToggleTimeline).toHaveBeenCalledOnce()

    rerender(
      <TooltipProvider>
        <MultiTrackToolbar
          currentTime={0}
          totalLength={24}
          frameRate={24}
          isPlaying={false}
          zoom={1}
          snapEnabled
          timelineCollapsed
          onPlayPause={vi.fn()}
          onZoomChange={vi.fn()}
          onSnapEnabledChange={vi.fn()}
          onToggleTimeline={onToggleTimeline}
          canDelete={false}
          onDeleteSelected={vi.fn()}
          onCutAtCurrentTime={vi.fn()}
          canTrimCenter={false}
          canTrimLeft={false}
          canTrimRight={false}
          onTrimLeftAtCurrentTime={vi.fn()}
          onTrimRightAtCurrentTime={vi.fn()}
          canUndo={false}
          canRedo={false}
          onUndo={vi.fn()}
          onRedo={vi.fn()}
        />
      </TooltipProvider>,
    )

    expect(screen.getByRole('button', { name: 'multitrack.showTimeline' })).not.toBeNull()
    expect(container.querySelector('.lucide-maximize-2')).not.toBeNull()
  })

  it('uses a consistent icon size across toolbar controls', () => {
    const { container } = renderToolbar(false)

    const iconClasses = Array.from(container.querySelectorAll('svg')).map((icon) => icon.getAttribute('class') ?? '')
    const iconButtonClasses = Array.from(container.querySelectorAll('button')).map((button) => button.getAttribute('class') ?? '')

    expect(iconClasses.length).toBeGreaterThan(0)
    expect(iconClasses.every((className) => className.includes('size-3.5'))).toBe(true)
    expect(iconButtonClasses.every((className) => className.includes('[&_svg]:size-3.5'))).toBe(true)
  })

  it('renders undo and redo history buttons with disabled states', () => {
    const onUndo = vi.fn()
    const onRedo = vi.fn()
    const { rerender } = renderToolbar(false, { canUndo: true, canRedo: false, onUndo, onRedo })

    fireEvent.click(screen.getByRole('button', { name: 'multitrack.undo' }))
    fireEvent.click(screen.getByRole('button', { name: 'multitrack.redo' }))
    expect(onUndo).toHaveBeenCalledOnce()
    expect(onRedo).not.toHaveBeenCalled()

    rerender(
      <TooltipProvider>
        <MultiTrackToolbar
          currentTime={0}
          totalLength={24}
          frameRate={24}
          isPlaying={false}
          zoom={1}
          snapEnabled
          timelineCollapsed={false}
          onPlayPause={vi.fn()}
          onZoomChange={vi.fn()}
          onSnapEnabledChange={vi.fn()}
          onToggleTimeline={vi.fn()}
          canDelete={false}
          onDeleteSelected={vi.fn()}
          onCutAtCurrentTime={vi.fn()}
          canTrimCenter={false}
          canTrimLeft={false}
          canTrimRight={false}
          onTrimLeftAtCurrentTime={vi.fn()}
          onTrimRightAtCurrentTime={vi.fn()}
          canUndo={false}
          canRedo
          onUndo={onUndo}
          onRedo={onRedo}
        />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'multitrack.redo' }))
    expect(onRedo).toHaveBeenCalledOnce()
  })

  it('toggles timeline snapping from the right toolbar group', () => {
    const onSnapEnabledChange = vi.fn()
    const { container, rerender } = renderToolbar(false, { snapEnabled: true, onSnapEnabledChange })

    const snapButton = screen.getByRole('button', { name: 'multitrack.timelineSnap' })
    expect(snapButton.getAttribute('aria-pressed')).toBe('true')
    expect(container.querySelector('.lucide-magnet')).not.toBeNull()
    fireEvent.click(snapButton)
    expect(onSnapEnabledChange).toHaveBeenCalledWith(false)

    rerender(
      <TooltipProvider>
        <MultiTrackToolbar
          currentTime={0}
          totalLength={24}
          frameRate={24}
          isPlaying={false}
          zoom={1}
          snapEnabled={false}
          timelineCollapsed={false}
          onPlayPause={vi.fn()}
          onZoomChange={vi.fn()}
          onSnapEnabledChange={onSnapEnabledChange}
          onToggleTimeline={vi.fn()}
          canDelete={false}
          onDeleteSelected={vi.fn()}
          onCutAtCurrentTime={vi.fn()}
          canTrimCenter={false}
          canTrimLeft={false}
          canTrimRight={false}
          onTrimLeftAtCurrentTime={vi.fn()}
          onTrimRightAtCurrentTime={vi.fn()}
          canUndo={false}
          canRedo={false}
          onUndo={vi.fn()}
          onRedo={vi.fn()}
        />
      </TooltipProvider>,
    )

    expect(screen.getByRole('button', { name: 'multitrack.timelineSnap' }).getAttribute('aria-pressed')).toBe('false')
  })
})
