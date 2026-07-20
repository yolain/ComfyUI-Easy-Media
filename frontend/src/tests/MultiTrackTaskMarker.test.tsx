import type { ReactNode } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { MultiTrackTaskMarker } from '@/components/widgets/multitrack/MultiTrackTaskMarker'

vi.mock('@/components/ui/context-menu', () => ({
  ContextMenu: ({ children }: { children: ReactNode }) => <>{children}</>,
  ContextMenuTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  ContextMenuContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  ContextMenuItem: ({ children, onClick }: { children: ReactNode; onClick?: () => void }) => (
    <button type="button" onClick={onClick}>{children}</button>
  ),
}))

vi.mock('@/components/ui/tooltip', () => ({
  Tooltip: ({ children }: { children: ReactNode }) => <>{children}</>,
  TooltipTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  TooltipContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}))

describe('MultiTrackTaskMarker', () => {
  it('selects a marker on the ruler and deletes it from the context menu', () => {
    const onSelect = vi.fn()
    const onDragStart = vi.fn()
    const onDelete = vi.fn()
    const onParentClick = vi.fn()
    render(
      <div onClick={onParentClick}>
        <MultiTrackTaskMarker
          marker={{ id: 'marker-1', frame: 48 }}
          markerNumber={1}
          frameRate={24}
          left={128}
          selected
          dragging={false}
          onSelect={onSelect}
          onDragStart={onDragStart}
          onDelete={onDelete}
        />
      </div>,
    )

    const marker = screen.getByRole('button', { name: 'Task marker at frame 48' })
    expect(marker.style.left).toBe('128px')
    expect(marker.style.color).toBe('var(--highlight)')
    expect(marker.getAttribute('aria-pressed')).toBe('true')
    expect(document.querySelector('.lucide-task-marker-icon')).not.toBeNull()
    expect(screen.getByText('Task 01')).not.toBeNull()
    expect(screen.getByText('00:02:00')).not.toBeNull()
    fireEvent.mouseDown(marker)
    fireEvent.click(marker)
    expect(onSelect).toHaveBeenCalledWith('marker-1')
    expect(onDragStart).toHaveBeenCalledWith('marker-1', 0)
    expect(onParentClick).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Delete task marker' }))
    expect(onDelete).toHaveBeenCalledWith('marker-1')
  })

  it('uses accumulated minutes instead of an hour field in the marker time', () => {
    render(
      <MultiTrackTaskMarker
        marker={{ id: 'marker-long', frame: (60 * 60 + 2) * 24 }}
        markerNumber={2}
        frameRate={24}
        left={128}
        selected={false}
        dragging={false}
        onSelect={vi.fn()}
        onDragStart={vi.fn()}
        onDelete={vi.fn()}
      />,
    )

    expect(screen.getByText('60:02:00')).not.toBeNull()
  })
})
