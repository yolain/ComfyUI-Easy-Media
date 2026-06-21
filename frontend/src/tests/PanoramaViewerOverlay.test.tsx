import { fireEvent, render, screen } from '@testing-library/react'
import { TooltipProvider } from '@/components/ui/tooltip'
import { PanoramaViewerOverlay } from '@/components/widgets/panorama/PanoramaViewerOverlay'
import { DEFAULT_PANORAMA_VIEW } from '@/lib/panorama-camera'

vi.mock('@/components/widgets/panorama/PanoramaCanvas', () => ({
  PanoramaCanvas: ({ view, onViewChange, onAspectRatioChange }: {
    view: typeof DEFAULT_PANORAMA_VIEW
    onViewChange: (view: typeof DEFAULT_PANORAMA_VIEW) => void
    onAspectRatioChange: (aspectRatio: number) => void
  }) => (
    <button
      type="button"
      data-testid="mock-panorama-canvas"
      data-yaw={view.yaw}
      onClick={() => {
        onAspectRatioChange(2)
        onViewChange({ ...view, yaw: 45, pitch: 10 })
      }}
    >
      move camera
    </button>
  ),
}))

class ResizeObserverMock {
  observe() {}
  disconnect() {}
  unobserve() {}
}

function renderOverlay({
  savedView,
  onPanoramaViewChange = vi.fn(),
  onExit = vi.fn(),
}: {
  savedView?: unknown
  onPanoramaViewChange?: (view: typeof DEFAULT_PANORAMA_VIEW | undefined) => void
  onExit?: () => void
} = {}) {
  return {
    onPanoramaViewChange,
    onExit,
    ...render(
      <TooltipProvider>
        <div data-multitrack-preview-area>
          <PanoramaViewerOverlay
            imageUrl="/view?filename=pano.png&type=input"
            savedView={savedView}
            onPanoramaViewChange={onPanoramaViewChange}
            onExit={onExit}
          />
        </div>
        <button type="button">outside preview</button>
      </TooltipProvider>,
    ),
  }
}

describe('PanoramaViewerOverlay', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', ResizeObserverMock)
  })

  it('keeps camera edits as a draft until apply', () => {
    const { onPanoramaViewChange, onExit } = renderOverlay()

    fireEvent.click(screen.getByTestId('mock-panorama-canvas'))
    expect(onPanoramaViewChange).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole('button', { name: 'Apply view' }))
    expect(onPanoramaViewChange).toHaveBeenCalledWith({
      ...DEFAULT_PANORAMA_VIEW,
      yaw: 45,
      pitch: 10,
      aspect_ratio: 2,
    })
    expect(onExit).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('slider')).toBeNull()
  })

  it('cancels panorama metadata and exits', () => {
    const onPanoramaViewChange = vi.fn()
    const onExit = vi.fn()
    renderOverlay({
      savedView: { ...DEFAULT_PANORAMA_VIEW, yaw: 80, aspect_ratio: 1.5 },
      onPanoramaViewChange,
      onExit,
    })

    expect(screen.getByTestId('mock-panorama-canvas').getAttribute('data-yaw')).toBe('80')
    fireEvent.click(screen.getByRole('button', { name: 'Cancel panorama' }))

    expect(onPanoramaViewChange).toHaveBeenCalledWith(undefined)
    expect(onExit).toHaveBeenCalledTimes(1)
  })

  it('shows camera orientation information in the bottom-right HUD', () => {
    renderOverlay({ savedView: { ...DEFAULT_PANORAMA_VIEW, yaw: 159, pitch: 61, hfov: 95 } })

    const hud = screen.getByTestId('panorama-orientation-hud')
    const initialYAxisEnd = screen.getByTestId('panorama-axis-y').getAttribute('y2')
    expect(hud.className).toContain('bottom-2')
    expect(hud.className).toContain('right-2')
    expect(hud.textContent).toContain('159°')
    expect(hud.textContent).toContain('61°')
    expect(hud.textContent).toContain('95°')
    expect(hud.textContent).toContain('×0.63')
    expect(screen.getByTestId('panorama-axis-stage').className).toContain('bg-black')
    expect(screen.getByTestId('panorama-axis-x').getAttribute('style')).toContain('stroke: var(--destructive)')
    expect(screen.getByTestId('panorama-axis-y').getAttribute('style')).toContain('stroke: var(--highlight)')
    expect(screen.getByTestId('panorama-axis-z').getAttribute('style')).toContain('stroke: var(--panorama-axis-z)')
    expect(screen.getByText('X').getAttribute('style')).toContain('fill: var(--destructive)')
    expect(screen.getByText('Y').getAttribute('style')).toContain('fill: var(--highlight)')
    expect(screen.getByText('Z').getAttribute('style')).toContain('fill: var(--panorama-axis-z)')

    fireEvent.click(screen.getByTestId('mock-panorama-canvas'))
    expect(screen.getByTestId('panorama-axis-y').getAttribute('y2')).not.toBe(initialYAxisEnd)
  })

  it('exits with Escape or a pointer press outside PreviewArea', () => {
    const first = renderOverlay()
    fireEvent.keyDown(document, { key: 'Escape' })
    expect(first.onExit).toHaveBeenCalledTimes(1)
    first.unmount()

    const second = renderOverlay()
    fireEvent.pointerDown(screen.getByRole('button', { name: 'outside preview' }))
    expect(second.onExit).toHaveBeenCalledTimes(1)
  })

  it('reports invalid saved metadata while using the default draft', () => {
    renderOverlay({ savedView: { version: 2 } })

    expect(screen.getByText('Saved panorama view is invalid.')).not.toBeNull()
    expect(screen.getByTestId('mock-panorama-canvas').getAttribute('data-yaw')).toBe('0')
  })
})
