import type { ReactNode } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { MultiTrackSegmentBlock } from '@/components/widgets/multitrack/MultiTrackSegmentBlock'
import type { MultiTrackSegment, MultiTrackType } from '@/types/multitrack'

vi.mock('@/components/ui/context-menu', () => ({
  ContextMenu: ({ children }: { children: ReactNode }) => <>{children}</>,
  ContextMenuTrigger: ({ children }: { children: ReactNode }) => <>{children}</>,
  ContextMenuContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  ContextMenuItem: ({ children, onClick }: { children: ReactNode; onClick?: () => void }) => (
    <button type="button" onClick={onClick}>{children}</button>
  ),
}))

function segment(type: MultiTrackType): MultiTrackSegment {
  return {
    id: `${type}-segment`,
    start_frame: 0,
    end_frame: 5,
    color: 'var(--primary)',
    content: {
      media_type: type === 'video' ? 'video' : 'none',
      task_mode: type === 'task' ? 'default' : undefined,
    },
  }
}

function renderBlock(trackType: MultiTrackType) {
  const onDelete = vi.fn()
  const onDistribute = vi.fn()
  const onClone = vi.fn()
  render(
    <MultiTrackSegmentBlock
      trackType={trackType}
      segmentIndex={0}
      segment={segment(trackType)}
      totalLength={10}
      frameRate={24}
      areaWidth={200}
      canvasScale={1}
      selected={false}
      onSelect={vi.fn()}
      onDelete={onDelete}
      onDistribute={trackType === 'task' ? onDistribute : undefined}
      onClone={trackType === 'task' ? onClone : undefined}
      onResize={vi.fn()}
      onMove={vi.fn()}
    />,
  )
  return { onDelete, onDistribute, onClone }
}

describe('MultiTrackSegmentBlock context menu', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      disconnect() {}
    })
  })
  it('offers distribute, clone, and delete actions for task segments', () => {
    const { onDelete, onDistribute, onClone } = renderBlock('task')

    fireEvent.click(screen.getByRole('button', { name: 'Distribute segments evenly' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clone segment' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete segment' }))

    expect(onDistribute).toHaveBeenCalledOnce()
    expect(onClone).toHaveBeenCalledWith('task-segment')
    expect(onDelete).toHaveBeenCalledWith('task-segment')
  })

  it('keeps non-task segment menus limited to delete', () => {
    renderBlock('subtitle')

    expect(screen.queryByRole('button', { name: 'Distribute segments evenly' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Clone segment' })).toBeNull()
    expect(screen.getByRole('button', { name: 'Delete segment' })).not.toBeNull()
  })

  it('renders a waveform canvas for audio segments', () => {
    const { container } = render(
      <MultiTrackSegmentBlock
        trackType="audio"
        segmentIndex={0}
        segment={{
          ...segment('audio'),
          content: { media_type: 'audio' },
        }}
        totalLength={10}
        frameRate={24}
        areaWidth={200}
        canvasScale={1}
        selected={false}
        onSelect={vi.fn()}
        onDelete={vi.fn()}
        onResize={vi.fn()}
        onMove={vi.fn()}
      />,
    )
    const canvas = container.querySelector('canvas')
    expect(canvas).not.toBeNull()
    expect(canvas?.parentElement?.className).toContain('flex-1')
  })
})
