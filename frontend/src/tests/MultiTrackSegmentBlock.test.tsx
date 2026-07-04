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
      media_type: type === 'video' ? 'video' : type === 'audio' ? 'audio' : 'none',
      task_mode: type === 'task' ? 'default' : undefined,
    },
  }
}

function renderBlock(trackType: MultiTrackType) {
  const onDelete = vi.fn()
  const onDistribute = vi.fn()
  const onClone = vi.fn()
  const onSmartSplit = vi.fn()
  const onSmartSplitTasks = vi.fn()
  const onRecognizeSubtitles = vi.fn()
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
      onSmartSplit={trackType === 'video' ? onSmartSplit : undefined}
      onSmartSplitTasks={trackType === 'video' ? onSmartSplitTasks : undefined}
      onRecognizeSubtitles={trackType === 'video' || trackType === 'audio' ? onRecognizeSubtitles : undefined}
      onResize={vi.fn()}
      onResizePreview={vi.fn()}
      onMove={vi.fn()}
    />,
  )
  return { onDelete, onDistribute, onClone, onSmartSplit, onSmartSplitTasks, onRecognizeSubtitles }
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

  it('offers smart split for video segments', () => {
    const { onSmartSplit, onSmartSplitTasks } = renderBlock('video')

    fireEvent.click(screen.getByRole('button', { name: 'Smart split' }))
    fireEvent.click(screen.getByRole('button', { name: 'Smart split (tasks only)' }))

    expect(onSmartSplit).toHaveBeenCalledWith('video-segment')
    expect(onSmartSplitTasks).toHaveBeenCalledWith('video-segment')
  })

  it.each(['video', 'audio'] as const)('offers subtitle recognition for %s segments', (trackType) => {
    const { onRecognizeSubtitles } = renderBlock(trackType)

    fireEvent.click(screen.getByRole('button', { name: 'Recognize subtitles' }))

    expect(onRecognizeSubtitles).toHaveBeenCalledWith(`${trackType}-segment`)
  })

  it('cuts at the clicked frame without starting a drag', () => {
    const onCut = vi.fn()
    const { container } = render(
      <MultiTrackSegmentBlock
        trackType="video"
        segmentIndex={0}
        segment={segment('video')}
        totalLength={10}
        frameRate={24}
        areaWidth={200}
        canvasScale={1}
        selected={false}
        onSelect={vi.fn()}
        onDelete={vi.fn()}
        onResize={vi.fn()}
      onResizePreview={vi.fn()}
        onMove={vi.fn()}
        cutMode
        onCut={onCut}
      />,
    )
    const block = container.querySelector('[role="button"]') as HTMLElement
    vi.spyOn(block, 'getBoundingClientRect').mockReturnValue({
      left: 0, right: 200, top: 0, bottom: 50, width: 200, height: 50, x: 0, y: 0,
      toJSON: () => ({}),
    })

    fireEvent.mouseDown(block, { button: 0, clientX: 100 })
    fireEvent.click(block, { clientX: 100 })

    expect(block.style.cursor).toBe('text')
    expect(onCut).toHaveBeenCalledWith('video-segment', 3)
  })

  it('does not open the video replacement action when double-clicking in cut mode', () => {
    const onDoubleClick = vi.fn()
    const { container } = render(
      <MultiTrackSegmentBlock
        trackType="video"
        segmentIndex={0}
        segment={segment('video')}
        totalLength={10}
        frameRate={24}
        areaWidth={200}
        canvasScale={1}
        selected={false}
        onSelect={vi.fn()}
        onDelete={vi.fn()}
        onResize={vi.fn()}
      onResizePreview={vi.fn()}
        onMove={vi.fn()}
        onDoubleClick={onDoubleClick}
        cutMode
        onCut={vi.fn()}
      />,
    )

    fireEvent.doubleClick(container.querySelector('[role="button"]') as HTMLElement)

    expect(onDoubleClick).not.toHaveBeenCalled()
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
      onResizePreview={vi.fn()}
        onMove={vi.fn()}
      />,
    )
    const canvas = container.querySelector('canvas')
    expect(canvas).not.toBeNull()
    expect(canvas?.parentElement?.className).toContain('flex-1')
  })
})
