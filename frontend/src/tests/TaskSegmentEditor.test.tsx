import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { LocaleContext } from '@/lib/i18n'
import { TaskSegmentEditor } from '@/components/widgets/multitrack/TaskSegmentEditor'
import { getMediaListStoreRevision } from '@/stores/media-list-store'
import type { MultiTrackSegment } from '@/types/multitrack'

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: () => null,
}))

function activateTab(name: string) {
  const tab = screen.getByRole('tab', { name })
  fireEvent.pointerDown(tab, { button: 0, ctrlKey: false })
  fireEvent.mouseDown(tab, { button: 0, ctrlKey: false })
  fireEvent.click(tab)
}

function taskSegment(): MultiTrackSegment {
  return {
    id: 'task-segment',
    start_frame: 0,
    end_frame: 3,
    color: 'var(--multitrack-task-bg)',
    content: {
      media_type: 'none',
      task_mode: 'default',
      text: 'Initial prompt',
      images: [
        { id: 'a', source_type: 'input', file_path: 'a.png', file_name: 'a.png' },
        { id: 'b', source_type: 'input', file_path: 'b.png', file_name: 'b.png' },
      ],
    },
  }
}

function secondTaskSegment(): MultiTrackSegment {
  return {
    ...taskSegment(),
    id: 'task-segment-2',
    start_frame: 3,
    end_frame: 6,
    content: {
      media_type: 'none',
      task_mode: 'default',
      text: 'Second prompt',
      images: [],
    },
  }
}

function videoSegment(
  startFrame: number,
  endFrame: number,
  sourceType: 'input' | 'preset' = 'input',
): MultiTrackSegment {
  return {
    id: `video-${startFrame}-${endFrame}-${sourceType}`,
    start_frame: startFrame,
    end_frame: endFrame,
    color: 'var(--multitrack-video-bg)',
    content: {
      media_type: 'video',
      source_type: sourceType,
      file_path: 'video.mp4',
    },
  }
}

describe('TaskSegmentEditor', () => {
  it('updates task mode and prompt text', () => {
    const onContentChange = vi.fn()
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={onContentChange} />)

    fireEvent.click(screen.getByRole('combobox'))
    fireEvent.click(screen.getByRole('option', { name: 'Reference (r2v)' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Prompt' }), {
      target: { value: 'New prompt' },
    })

    expect(onContentChange).toHaveBeenCalledWith({ task_mode: 'ref' })
    expect(onContentChange).toHaveBeenCalledWith({ text: 'New prompt' })
  })

  it('uses rv2v for reference mode when a non-preset video overlaps the task range', () => {
    render(
      <TaskSegmentEditor
        segment={taskSegment()}
        videoSegments={[videoSegment(2, 5)]}
        onContentChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('combobox'))

    expect(screen.getByRole('option', { name: 'Reference (rv2v)' })).not.toBeNull()
  })

  it.each([
    ['preset video', videoSegment(2, 5, 'preset')],
    ['video ending at the task start', videoSegment(-2, 0)],
    ['video starting at the task end', videoSegment(3, 5)],
  ])('uses r2v for reference mode with %s', (_caseName, video) => {
    render(
      <TaskSegmentEditor
        segment={taskSegment()}
        videoSegments={[video]}
        onContentChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('combobox'))

    expect(screen.getByRole('option', { name: 'Reference (r2v)' })).not.toBeNull()
  })

  it('uploads dropped image files and appends them to task images', async () => {
    const onContentChange = vi.fn()
    const initialCacheRevision = getMediaListStoreRevision()
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ name: 'first.png', subfolder: 'uploads' }) })
      .mockResolvedValueOnce({ ok: true, json: async () => ({ name: 'second.png', subfolder: '' }) })
    vi.stubGlobal('fetch', fetchMock)
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={onContentChange} />)

    fireEvent.drop(screen.getByTestId('task-image-drop-zone'), {
      dataTransfer: {
        files: [
          new File(['first'], 'first.png', { type: 'image/png' }),
          new File(['second'], 'second.png', { type: 'image/png' }),
        ],
        types: ['Files'],
      },
    })

    await waitFor(() => {
      expect(onContentChange).toHaveBeenLastCalledWith({
        images: [
          expect.objectContaining({ file_name: 'a.png' }),
          expect.objectContaining({ file_name: 'b.png' }),
          expect.objectContaining({ file_path: 'uploads/first.png', file_name: 'first.png' }),
          expect.objectContaining({ file_path: 'second.png', file_name: 'second.png' }),
        ],
      })
    })
    expect(getMediaListStoreRevision()).toBe(initialCacheRevision + 1)
  })

  it('highlights the image picker while image files are dragged over it', () => {
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />)

    const dropZone = screen.getByTestId('task-image-drop-zone')
    fireEvent.dragEnter(dropZone, {
      dataTransfer: {
        files: [new File(['first'], 'first.png', { type: 'image/png' })],
        types: ['Files'],
      },
    })

    expect(dropZone.className).toContain('border-primary')

    fireEvent.dragLeave(dropZone, { relatedTarget: document.body })

    expect(dropZone.className).not.toContain('border-primary')
  })

  it('previews and deletes images from the image grid actions', () => {
    const onContentChange = vi.fn()
    const onOpenPanorama = vi.fn()
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(null)
    render(
      <TaskSegmentEditor
        segment={taskSegment()}
        onContentChange={onContentChange}
        onOpenPanorama={onOpenPanorama}
      />,
    )

    expect(screen.getByTestId('task-image-a').className).toContain('bg-black')
    expect(screen.getByAltText('a.png').className).toContain('h-full')
    expect(screen.getByAltText('a.png').className).toContain('w-full')
    expect(screen.getByAltText('a.png').className).toContain('object-contain')
    expect(screen.getByTestId('task-image-actions-a').className).toContain('opacity-0')
    expect(screen.getByTestId('task-image-actions-a').className).toContain('right-1')
    expect(screen.getByTestId('task-image-actions-a').className).toContain('top-1')
    expect(screen.getByTestId('task-image-index-a').textContent).toBe('0')
    expect(screen.getByTestId('task-image-index-a').className).toContain('bottom-0')
    expect(screen.getByTestId('task-image-index-b').textContent).toBe('1')
    expect(screen.getAllByRole('button', { name: 'Preview image' })[0].className).toContain('cursor-pointer')
    expect(screen.getAllByRole('button', { name: '720° panorama preview' })[0].className).toContain('cursor-pointer')
    expect(screen.getAllByRole('button', { name: 'Delete image' })[0].className).toContain('cursor-pointer')
    expect(screen.getAllByRole('button', { name: 'Delete image' })[0].className).toContain('text-destructive')
    const panoramaButton = screen.getAllByRole('button', { name: '720° panorama preview' })[0]
    const previewButton = screen.getAllByRole('button', { name: 'Preview image' })[0]
    expect(panoramaButton.compareDocumentPosition(previewButton) & Node.DOCUMENT_POSITION_FOLLOWING).not.toBe(0)
    expect(panoramaButton.querySelector('path')?.getAttribute('d')).toContain('M1.48 7.624')

    fireEvent.click(screen.getAllByRole('button', { name: 'Preview image' })[0])
    expect(openSpy).toHaveBeenCalledWith('/view?filename=a.png&type=input&subfolder=', '_blank', 'noopener,noreferrer')

    fireEvent.click(screen.getAllByRole('button', { name: '720° panorama preview' })[0])
    expect(onOpenPanorama).toHaveBeenCalledWith('a')

    fireEvent.click(screen.getAllByRole('button', { name: 'Delete image' })[0])
    expect(onContentChange).toHaveBeenCalledWith({
      images: [expect.objectContaining({ id: 'b' })],
    })

    openSpy.mockRestore()
  })

  it('highlights the panorama icon when a view is applied', () => {
    const segment = taskSegment()
    segment.content.images![0].panorama_view = {
      version: 1,
      projection: 'equirectangular',
      yaw: 20,
      pitch: 5,
      hfov: 75,
      aspect_ratio: 1.5,
    }
    render(<TaskSegmentEditor segment={segment} onContentChange={vi.fn()} />)

    expect(screen.getAllByRole('button', { name: '720° panorama preview' })[0].className).toContain('text-highlight')
    expect(screen.getAllByRole('button', { name: '720° panorama preview' })[1].className).toContain('text-foreground')
    expect(screen.getByTestId('task-image-grid').className).toContain('w-full')
    expect(screen.getByTestId('panorama-image-preview-a').className).toContain('absolute')
  })

  it('reorders images by dragging one image onto another', () => {
    const onContentChange = vi.fn()
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={onContentChange} />)

    fireEvent.dragStart(screen.getByTestId('task-image-a'))
    fireEvent.drop(screen.getByTestId('task-image-b'))

    expect(onContentChange).toHaveBeenCalledWith({
      images: [
        expect.objectContaining({ id: 'b' }),
        expect.objectContaining({ id: 'a' }),
      ],
    })
  })

  it('shows a stable image overlay without changing the underlying grid layout', () => {
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />)

    const grid = screen.getByTestId('task-image-grid')
    const firstImage = screen.getByTestId('task-image-a')
    const secondImage = screen.getByTestId('task-image-b')

    expect(grid.className).toContain('grid-cols-2')
    expect(screen.queryByTestId('task-image-focus-preview')).toBeNull()
    fireEvent.mouseEnter(secondImage)
    const preview = screen.getByTestId('task-image-focus-preview')
    expect(grid.className).toContain('grid-cols-2')
    expect(grid.className).not.toContain('grid-cols-1')
    expect(preview.className).toContain('absolute')
    expect(preview.className).toContain('inset-3')
    expect(preview.className).toContain('pointer-events-none')
    expect(preview.className).toContain('bg-black')
    expect(preview.querySelector('img')?.className).toContain('object-contain')
    expect(secondImage.className).toContain('relative')
    expect(secondImage.className).toContain('cursor-pointer')
    expect(secondImage.className).toContain('opacity-0')
    expect(firstImage.className).toContain('opacity-0')

    fireEvent.mouseLeave(grid)
    expect(grid.className).toContain('grid-cols-2')
    expect(screen.queryByTestId('task-image-focus-preview')).toBeNull()
    expect(secondImage.className).toContain('relative')
    expect(secondImage.className).not.toContain('opacity-0')
  })

  it('uses locale messages for task mode and prompt labels', () => {
    render(
      <LocaleContext.Provider value="zh">
        <TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />
      </LocaleContext.Provider>,
    )

    expect(screen.getByRole('combobox').textContent).toContain('默认 (i2v)')
    expect(screen.getByRole('textbox', { name: '提示词' })).not.toBeNull()
    expect(screen.getByLabelText('任务图片拖放区域')).not.toBeNull()
  })

  it('loads system prompt options once and matches the current task state locally', async () => {
    const onContentChange = vi.fn()
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [
          { task_type: 't2v', system_prompt: 'Default text-to-video template' },
          { task_type: 'i2v', system_prompt: 'Default image-to-video template' },
          { task_type: 'r2v', system_prompt: 'Default reference-to-video template' },
          { task_type: 'rv2v', system_prompt: 'Default reference-edit template' },
        ],
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { rerender, unmount } = render(<TaskSegmentEditor segment={taskSegment()} onContentChange={onContentChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'System Prompt' }))
    const systemPrompt = await screen.findByRole('textbox', { name: 'System Prompt' }) as HTMLTextAreaElement
    await waitFor(() => {
      expect(systemPrompt.value).toBe('Default image-to-video template')
    })

    const resetSegment = {
      ...taskSegment(),
      content: {
        ...taskSegment().content,
        system_prompt: '',
      },
    }
    rerender(<TaskSegmentEditor segment={resetSegment} onContentChange={onContentChange} />)
    expect(systemPrompt.value).toBe('Default image-to-video template')

    const refSegment = {
      ...taskSegment(),
      content: {
        ...taskSegment().content,
        task_mode: 'ref' as const,
      },
    }
    rerender(<TaskSegmentEditor segment={refSegment} onContentChange={onContentChange} />)
    await waitFor(() => {
      expect(systemPrompt.value).toBe('Default reference-to-video template')
    })

    rerender(
      <TaskSegmentEditor
        segment={refSegment}
        videoSegments={[videoSegment(2, 5)]}
        onContentChange={onContentChange}
      />,
    )
    await waitFor(() => {
      expect(systemPrompt.value).toBe('Default reference-edit template')
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith('/easy-media/prompt/system-prompts')

    unmount()
    render(<TaskSegmentEditor segment={refSegment} onContentChange={onContentChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'System Prompt' }))
    const remountedSystemPrompt = await screen.findByRole('textbox', { name: 'System Prompt' }) as HTMLTextAreaElement
    expect(remountedSystemPrompt.value).toBe('Default reference-to-video template')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('edits a customized system prompt and highlights complete placeholders', () => {
    const onContentChange = vi.fn()
    const customizedSegment = {
      ...taskSegment(),
      content: {
        ...taskSegment().content,
        system_prompt: 'Create {subject} in {style}',
      },
    }
    const { container } = render(
      <TooltipProvider>
        <TaskSegmentEditor segment={customizedSegment} onContentChange={onContentChange} />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'System Prompt' }))
    const textarea = screen.getByRole('textbox', { name: 'System Prompt' }) as HTMLTextAreaElement
    const highlightedVariables = container.querySelectorAll('[data-system-prompt-variable="true"]')

    expect(textarea.readOnly).toBe(false)
    expect(textarea.style.color).toBe('transparent')
    expect(textarea.style.caretColor).toBe('var(--foreground)')
    expect(Array.from(highlightedVariables, (item) => item.textContent)).toEqual(['{subject}', '{style}'])

    fireEvent.change(textarea, { target: { value: 'Create {character}' } })

    expect(onContentChange).toHaveBeenCalledWith({ system_prompt: 'Create {character}' })
  })

  it('resets a customized system prompt with an icon button and tooltip', async () => {
    const onContentChange = vi.fn()
    const customizedSegment = {
      ...taskSegment(),
      content: {
        ...taskSegment().content,
        system_prompt: 'Customized system prompt',
      },
    }
    render(
      <TooltipProvider>
        <TaskSegmentEditor segment={customizedSegment} onContentChange={onContentChange} />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'System Prompt' }))
    const resetButton = screen.getByRole('button', { name: 'Reset system prompt' })

    expect(resetButton.textContent).toBe('')
    expect(resetButton.querySelector('svg.lucide-rotate-ccw')).not.toBeNull()

    fireEvent.focus(resetButton)
    const tooltip = await screen.findByRole('tooltip')
    expect(tooltip.textContent).toBe('Reset system prompt')

    fireEvent.click(resetButton)

    expect(onContentChange).toHaveBeenCalledWith({ system_prompt: '' })
  })

  it('does not show the system prompt reset button for a default system prompt', () => {
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'System Prompt' }))

    expect(screen.queryByRole('button', { name: 'Reset system prompt' })).toBeNull()
  })

  it('edits all task prompts in combined mode and hides image selection', () => {
    const onTrackSegmentsContentChange = vi.fn()
    render(
      <TaskSegmentEditor
        segment={taskSegment()}
        trackSegments={[taskSegment(), secondTaskSegment()]}
        onContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
      />,
    )

    activateTab('Combined')

    expect(screen.queryByLabelText('Task image drop zone')).toBeNull()
    expect(screen.queryByRole('button', { name: 'User Prompt' })).toBeNull()
    expect(screen.queryByRole('button', { name: 'System Prompt' })).toBeNull()
    expect(screen.getByText('Use | to separate segments. Changes apply in real-time.')).not.toBeNull()
    fireEvent.change(screen.getByRole('textbox', { name: 'Prompt' }), {
      target: { value: 'First rewritten|Second rewritten' },
    })

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([
      { segmentId: 'task-segment', patch: { text: 'First rewritten' } },
      { segmentId: 'task-segment-2', patch: { text: 'Second rewritten' } },
    ])
  })

  it('creates and evenly distributes task segments when combined pasted text has more parts', () => {
    const onTrackSegmentsChange = vi.fn()
    render(
      <TaskSegmentEditor
        segment={taskSegment()}
        trackSegments={[taskSegment(), secondTaskSegment()]}
        totalFrames={10}
        onContentChange={vi.fn()}
        onTrackSegmentsChange={onTrackSegmentsChange}
      />,
    )

    activateTab('Combined')
    const prompt = screen.getByRole('textbox', { name: 'Prompt' }) as HTMLTextAreaElement
    fireEvent.change(prompt, { target: { value: 'First pasted｜Second pasted|Third pasted' } })

    expect(prompt.value).toBe('First pasted｜Second pasted|Third pasted')
    const highlight = screen.getByTestId('combined-prompt-highlight')
    expect(highlight.querySelectorAll('[data-pipe="true"]')).toHaveLength(2)
    expect(highlight.querySelector('[data-pipe="true"]')?.className).toContain('text-highlight')
    const updated = onTrackSegmentsChange.mock.lastCall?.[0] as MultiTrackSegment[]
    expect(updated).toHaveLength(3)
    expect(updated.map(({ start_frame, end_frame, content }) => ({
      start_frame,
      end_frame,
      text: content.text,
    }))).toEqual([
      { start_frame: 0, end_frame: 4, text: 'First pasted' },
      { start_frame: 4, end_frame: 7, text: 'Second pasted' },
      { start_frame: 7, end_frame: 10, text: 'Third pasted' },
    ])
    expect(updated[0].id).toBe('task-segment')
    expect(updated[1].id).toBe('task-segment-2')
    expect(updated[2]).toMatchObject({
      content: { media_type: 'none', task_mode: 'default', images: [] },
    })

    fireEvent.change(prompt, { target: { value: '' } })
    const cleared = onTrackSegmentsChange.mock.lastCall?.[0] as MultiTrackSegment[]
    expect(cleared).toHaveLength(1)
    expect(cleared[0]).toMatchObject({
      id: 'task-segment',
      start_frame: 0,
      end_frame: 10,
      content: { text: '' },
    })
  })

  it('shows and edits the selected task duration in the center of the footer', () => {
    const onDurationChange = vi.fn()
    const { rerender } = render(
      <TaskSegmentEditor
        segment={secondTaskSegment()}
        trackSegments={[taskSegment(), secondTaskSegment()]}
        frameRate={24}
        onContentChange={vi.fn()}
        onDurationChange={onDurationChange}
      />,
    )

    expect(screen.getByText('Task 1').className).toContain('text-[10px]')
    expect(screen.getByText('Task 1').className).toContain('text-primary')
    expect(screen.getByText('00:00:03').className).toContain('text-[10px]')

    fireEvent.click(screen.getByRole('button', { name: 'Edit task duration' }))
    expect(screen.queryByText('Task 1')).toBeNull()
    expect(screen.queryByText('00:00:03')).toBeNull()
    const durationInput = screen.getByRole('textbox', { name: 'Duration' })
    expect(durationInput.className).toContain('tabular-nums')
    expect(durationInput.getAttribute('placeholder')).toBe('00:00:00')
    fireEvent.change(durationInput, { target: { value: '00:01:12' } })
    fireEvent.keyDown(durationInput, { key: 'Enter' })
    expect(onDurationChange).toHaveBeenCalledWith(1.5)

    const updatedSegment = { ...secondTaskSegment(), end_frame: 39 }
    rerender(
      <TaskSegmentEditor
        segment={updatedSegment}
        trackSegments={[taskSegment(), updatedSegment]}
        frameRate={24}
        onContentChange={vi.fn()}
        onDurationChange={onDurationChange}
      />,
    )
    expect(screen.getByText('00:01:12')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Edit task duration' }))
    const invalidDurationInput = screen.getByRole('textbox', { name: 'Duration' })
    fireEvent.change(invalidDurationInput, { target: { value: '00:61:00' } })
    fireEvent.blur(invalidDurationInput)
    expect(screen.getByText('00:01:12')).not.toBeNull()
    expect(onDurationChange).toHaveBeenCalledTimes(1)
  })

  it('uses compact preview editor sizing without rendering the empty image picker as a button element', () => {
    const emptyImageSegment = {
      ...taskSegment(),
      content: {
        ...taskSegment().content,
        images: [],
      },
    }
    render(<TaskSegmentEditor segment={emptyImageSegment} onContentChange={vi.fn()} />)

    expect(screen.getByTestId('task-image-drop-zone').className).toContain('aspect-square')
    expect(screen.getByRole('button', { name: 'Task image drop zone' }).tagName).not.toBe('BUTTON')
    expect(screen.getByRole('textbox', { name: 'Prompt' }).className).toContain('text-[10px]')
  })
})
