import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { LocaleContext } from '@/lib/i18n'
import { TaskSegmentEditor } from '@/components/widgets/multitrack/TaskSegmentEditor'
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
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(null)
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={onContentChange} />)

    expect(screen.getByTestId('task-image-actions-a').className).toContain('opacity-0')
    expect(screen.getByTestId('task-image-actions-a').className).toContain('bg-black/30')
    expect(screen.getAllByRole('button', { name: 'Delete image' })[0].className).toContain('text-destructive')

    fireEvent.click(screen.getAllByRole('button', { name: 'Preview image' })[0])
    expect(openSpy).toHaveBeenCalledWith('/view?filename=a.png&type=input&subfolder=', '_blank', 'noopener,noreferrer')

    fireEvent.click(screen.getAllByRole('button', { name: 'Delete image' })[0])
    expect(onContentChange).toHaveBeenCalledWith({
      images: [expect.objectContaining({ id: 'b' })],
    })

    openSpy.mockRestore()
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

  it('shows the selected task number in the center of the footer', () => {
    render(
      <TaskSegmentEditor
        segment={secondTaskSegment()}
        trackSegments={[taskSegment(), secondTaskSegment()]}
        onContentChange={vi.fn()}
      />,
    )

    expect(screen.getByText('Task 1').className).toContain('left-1/2')
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
