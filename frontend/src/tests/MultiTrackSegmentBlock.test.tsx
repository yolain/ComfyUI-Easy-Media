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
  ContextMenuSub: ({ children }: { children: ReactNode }) => <>{children}</>,
  ContextMenuSubContent: ({ children }: { children: ReactNode }) => <div>{children}</div>,
  ContextMenuSubTrigger: ({ children }: { children: ReactNode }) => <div>{children}</div>,
}))

vi.mock('@/components/widgets/timeline/AudioWaveform', () => ({
  AudioWaveform: ({ startRatio, endRatio }: { startRatio?: number; endRatio?: number }) => (
    <canvas data-start-ratio={startRatio} data-end-ratio={endRatio} />
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
  const onSplitTask = vi.fn()
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
      onClone={trackType === 'task' || trackType === 'subtitle' ? onClone : undefined}
      onSplitTask={trackType === 'task' ? onSplitTask : undefined}
      onSmartSplit={trackType === 'video' ? onSmartSplit : undefined}
      onSmartSplitTasks={trackType === 'video' ? onSmartSplitTasks : undefined}
      onRecognizeSubtitles={trackType === 'video' || trackType === 'audio' ? onRecognizeSubtitles : undefined}
      onResize={vi.fn()}
      onResizePreview={vi.fn()}
      onMove={vi.fn()}
    />,
  )
  return { onDelete, onDistribute, onClone, onSplitTask, onSmartSplit, onSmartSplitTasks, onRecognizeSubtitles }
}

describe('MultiTrackSegmentBlock context menu', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      disconnect() {}
    })
  })
  it('offers distribute, clone, split, and delete actions for task segments', () => {
    const { onDelete, onDistribute, onClone, onSplitTask } = renderBlock('task')

    fireEvent.click(screen.getByRole('button', { name: 'Distribute segments evenly' }))
    fireEvent.click(screen.getByRole('button', { name: 'Clone segment' }))
    fireEvent.click(screen.getByRole('button', { name: 'Split segment' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete segment' }))

    expect(onDistribute).toHaveBeenCalledOnce()
    expect(onClone).toHaveBeenCalledWith('task-segment')
    expect(onSplitTask).toHaveBeenCalledWith('task-segment')
    expect(onDelete).toHaveBeenCalledWith('task-segment')
  })

  it('hides the task index in marker mode and tiles images with prompt text in overview mode', () => {
    const { container } = render(
      <MultiTrackSegmentBlock
        trackType="task"
        segmentIndex={3}
        segment={{
          ...segment('task'),
          content: {
            media_type: 'none',
            task_mode: 'default',
            user_prompt: 'A prompt shown at the bottom',
            images: [{ id: 'image', source_type: 'input', file_path: 'reference.png' }],
          },
        }}
        totalLength={10}
        frameRate={24}
        areaWidth={200}
        canvasScale={1}
        selected={false}
        showTaskIndex={false}
        taskOverview
        onSelect={vi.fn()}
        onDelete={vi.fn()}
        onResize={vi.fn()}
        onResizePreview={vi.fn()}
        onMove={vi.fn()}
      />,
    )

    expect(screen.queryByText(/Task3/)).toBeNull()
    expect(screen.getByText('A prompt shown at the bottom')).not.toBeNull()
    expect(screen.queryByText('00:00:05')).toBeNull()
    expect(container.querySelectorAll('img').length).toBeGreaterThan(1)
    expect(container.querySelector('img')?.getAttribute('src')).toContain('reference.png')
  })

  it('clamps a text-only task prompt to three lines in overview mode without showing its duration', () => {
    render(
      <MultiTrackSegmentBlock
        trackType="task"
        segmentIndex={1}
        segment={{
          ...segment('task'),
          content: {
            media_type: 'none',
            task_mode: 'default',
            user_prompt: 'A long text-only task prompt',
          },
        }}
        totalLength={10}
        frameRate={24}
        areaWidth={200}
        canvasScale={1}
        selected={false}
        taskOverview
        onSelect={vi.fn()}
        onDelete={vi.fn()}
        onResize={vi.fn()}
        onResizePreview={vi.fn()}
        onMove={vi.fn()}
      />,
    )

    const prompt = screen.getByText('A long text-only task prompt')
    expect(prompt.className).toContain('line-clamp-3')
    expect(screen.queryByText('00:00:05')).toBeNull()
  })

  it('offers clone and delete actions for subtitle segments', () => {
    const { onClone, onDelete } = renderBlock('subtitle')

    expect(screen.queryByRole('button', { name: 'Distribute segments evenly' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'Split segment' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Clone segment' }))
    fireEvent.click(screen.getByRole('button', { name: 'Delete segment' }))

    expect(onClone).toHaveBeenCalledWith('subtitle-segment')
    expect(onDelete).toHaveBeenCalledWith('subtitle-segment')
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

    fireEvent.click(screen.getByRole('button', { name: 'Qwen3-ASR' }))
    fireEvent.click(screen.getByRole('button', { name: 'Whisper Large V3' }))

    expect(onRecognizeSubtitles).toHaveBeenCalledWith(`${trackType}-segment`, 'qwen3-asr')
    expect(onRecognizeSubtitles).toHaveBeenCalledWith(`${trackType}-segment`, 'whisper-large-v3')
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

  it('renders only the waveform range visible after trimming an audio segment', () => {
    const { container } = render(
      <MultiTrackSegmentBlock
        trackType="audio"
        segmentIndex={0}
        segment={{
          ...segment('audio'),
          start_frame: 48,
          end_frame: 144,
          origin_start_frame: 0,
          content: { media_type: 'audio', duration: 8 },
        }}
        totalLength={240}
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
    expect(canvas?.dataset.startRatio).toBe('0.25')
    expect(canvas?.dataset.endRatio).toBe('0.75')
  })
})
