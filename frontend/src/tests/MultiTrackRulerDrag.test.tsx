import type { ReactNode } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { MultiTrackRuler } from '@/components/widgets/multitrack/MultiTrackRuler'

vi.mock('@/components/ui/context-menu', () => ({
  ContextMenu: ({ children }: { children: ReactNode }) => <>{children}</>,
  ContextMenuTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  ContextMenuContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  ContextMenuItem: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}))

vi.mock('@/components/ui/tooltip', () => ({
  Tooltip: ({ children }: { children: ReactNode }) => <>{children}</>,
  TooltipTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  TooltipContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}))

describe('MultiTrackRuler task marker dragging', () => {
  function renderRuler(onMoveTaskMarker: (markerId: string, frame: number) => void) {
    return render(
      <MultiTrackRuler
        totalLength={96}
        frameRate={24}
        width={300}
        canvasScale={1}
        currentTime={0}
        taskMarkers={[
          { id: 'marker-1', frame: 24 },
          { id: 'marker-2', frame: 48 },
        ]}
        selectedTaskMarkerId={null}
        onSeek={vi.fn()}
        onSelectTaskMarker={vi.fn()}
        onMoveTaskMarker={onMoveTaskMarker}
        onDeleteTaskMarker={vi.fn()}
      />,
    )
  }

  it('moves a task marker to the dragged frame', () => {
    const onMoveTaskMarker = vi.fn()
    const view = renderRuler(onMoveTaskMarker)

    const marker = screen.getByRole('button', { name: 'Task marker at frame 24' })
    const ruler = view.container.firstElementChild
    expect(marker.className).toContain('cursor-grab')
    fireEvent.mouseDown(marker, { clientX: 84 })
    expect(marker.className).toContain('cursor-grabbing')
    expect(ruler?.className).toContain('cursor-grabbing')
    fireEvent.mouseMove(globalThis, { clientX: 196 })
    fireEvent.mouseUp(globalThis, { clientX: 196 })

    expect(onMoveTaskMarker).toHaveBeenCalledWith('marker-1', 72)
    expect(ruler?.className).toContain('cursor-col-resize')
  })

  it('does not move a task marker onto an occupied frame', () => {
    const onMoveTaskMarker = vi.fn()
    renderRuler(onMoveTaskMarker)

    const marker = screen.getByRole('button', { name: 'Task marker at frame 24' })
    fireEvent.mouseDown(marker, { clientX: 84 })
    fireEvent.mouseMove(globalThis, { clientX: 140 })
    fireEvent.mouseUp(globalThis, { clientX: 140 })

    expect(onMoveTaskMarker).not.toHaveBeenCalled()
  })

  it('seeks to 12 seconds on a 20-second timeline using the rendered ruler width', () => {
    const onSeek = vi.fn()
    const view = render(
      <MultiTrackRuler
        totalLength={480}
        frameRate={24}
        width={480}
        canvasScale={1.25}
        currentTime={0}
        taskMarkers={[]}
        selectedTaskMarkerId={null}
        onSeek={onSeek}
        onSelectTaskMarker={vi.fn()}
        onMoveTaskMarker={vi.fn()}
        onDeleteTaskMarker={vi.fn()}
      />,
    )
    const ruler = view.container.firstElementChild as HTMLDivElement
    vi.spyOn(ruler, 'getBoundingClientRect').mockReturnValue({
      left: 0, right: 480, top: 0, bottom: 24, width: 480, height: 24, x: 0, y: 0,
      toJSON: () => ({}),
    })

    fireEvent.mouseDown(ruler, { clientX: 270.4 })
    fireEvent.mouseUp(globalThis, { clientX: 270.4 })

    expect(onSeek).toHaveBeenCalledWith(288)
  })
})
