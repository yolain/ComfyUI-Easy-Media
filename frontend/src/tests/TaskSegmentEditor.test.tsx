import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { LocaleContext } from '@/lib/i18n'
import { TaskSegmentEditor } from '@/components/widgets/multitrack/TaskSegmentEditor'
import { getMediaListStoreRevision } from '@/stores/media-list-store'
import type { MultiTrack, MultiTrackSegment } from '@/types/multitrack'

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: (props: {
    value: string
    allowMultipleSelection?: boolean
    maxSelectionCount?: number
    slotItems?: Array<{ value: string }>
    onChange: (value: string, source: 'input') => void
  }) => (
    <div
      data-testid="media-selector-mock"
      data-value={props.value}
      data-multiple={String(props.allowMultipleSelection)}
      data-limit={String(props.maxSelectionCount)}
      data-slot-items={props.slotItems?.map((item) => item.value).join(',') ?? ''}
    >
      <button type="button" onClick={() => props.onChange('replacement.png', 'input')}>
        choose replacement
      </button>
    </div>
  ),
}))

vi.stubGlobal('ResizeObserver', class {
  observe() {}
  unobserve() {}
  disconnect() {}
})

function activateTab(name: string) {
  const tab = screen.getByRole('tab', { name })
  fireEvent.pointerDown(tab, { button: 0, ctrlKey: false })
  fireEvent.mouseDown(tab, { button: 0, ctrlKey: false })
  fireEvent.click(tab)
}

function inputEditable(element: HTMLElement, value: string) {
  element.textContent = value
  fireEvent.input(element)
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
      user_prompt: 'Initial prompt',
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
      user_prompt: 'Second prompt',
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

function mediaTrack(type: 'audio' | 'video', segments: MultiTrackSegment[]): MultiTrack {
  return {
    id: `${type}-track`,
    name: type === 'audio' ? 'Voice track' : 'Source video track',
    type,
    color: type === 'audio' ? 'var(--multitrack-audio-bg)' : 'var(--multitrack-video-bg)',
    muted: false,
    locked: false,
    segments,
  }
}

describe('TaskSegmentEditor', () => {
  it('updates task mode and prompt text', () => {
    const onContentChange = vi.fn()
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={onContentChange} />)

    fireEvent.click(screen.getByRole('combobox', { name: 'Task mode' }))
    fireEvent.click(screen.getByRole('option', { name: 'R2V' }))
    inputEditable(screen.getByRole('textbox', { name: 'Prompt' }), 'New prompt')

    expect(onContentChange).toHaveBeenCalledWith({ task_mode: 'ref' })
    expect(onContentChange).toHaveBeenCalledWith({ user_prompt: 'New prompt' })
  })

  it('shows and stores MiniMax continuity mode from the second task onward', () => {
    const onContentChange = vi.fn()
    const first = taskSegment()
    const second = secondTaskSegment()
    const { rerender } = render(
      <TaskSegmentEditor
        segment={second}
        trackSegments={[first, second]}
        format="MiniMax"
        onContentChange={onContentChange}
      />,
    )

    const continuitySelect = screen.getByRole('combobox', { name: 'Continuity mode' })
    expect(continuitySelect.className).toContain('w-28')
    expect(screen.getByRole('combobox', { name: 'Task mode' }).className).toContain('w-24')
    fireEvent.click(continuitySelect)
    fireEvent.click(screen.getByRole('option', { name: 'Context' }))
    expect(onContentChange).toHaveBeenCalledWith({ continuity_mode: 'context' })
    fireEvent.click(continuitySelect)
    fireEvent.click(screen.getByRole('option', { name: 'Swap Context' }))
    expect(onContentChange).toHaveBeenCalledWith({ continuity_mode: 'context_swap' })

    rerender(
      <TaskSegmentEditor
        segment={first}
        trackSegments={[first, second]}
        format="MiniMax"
        onContentChange={onContentChange}
      />,
    )
    expect(screen.queryByRole('combobox', { name: 'Continuity mode' })).toBeNull()

    rerender(
      <TaskSegmentEditor
        segment={second}
        trackSegments={[first, second]}
        format="Seedance"
        onContentChange={onContentChange}
      />,
    )
    expect(screen.queryByRole('combobox', { name: 'Continuity mode' })).toBeNull()
  })

  it('shows and stores reference image size only for reference tasks', () => {
    const onContentChange = vi.fn()
    const referenceSegment = taskSegment()
    referenceSegment.content.task_mode = 'ref'
    referenceSegment.content.ref_image_size = 'max'
    const { rerender } = render(
      <TaskSegmentEditor
        segment={referenceSegment}
        format="MiniMax"
        onContentChange={onContentChange}
      />,
    )

    const sizeSelect = screen.getByRole('combobox', { name: 'Reference image size' })
    expect(sizeSelect.textContent).toContain('Max')
    fireEvent.click(sizeSelect)
    fireEvent.click(screen.getByRole('option', { name: 'Match' }))
    expect(onContentChange).toHaveBeenCalledWith({ ref_image_size: 'match' })

    rerender(
      <TaskSegmentEditor
        segment={taskSegment()}
        format="MiniMax"
        onContentChange={onContentChange}
      />,
    )
    expect(screen.queryByRole('combobox', { name: 'Reference image size' })).toBeNull()
  })

  it('synchronizes dropdown changes across selected tasks while keeping the first continuity mode as shot', () => {
    const onContentChange = vi.fn()
    const onTrackSegmentsContentChange = vi.fn()
    const first = taskSegment()
    first.content.task_mode = 'ref'
    first.content.continuity_mode = 'shot'
    const second = secondTaskSegment()
    second.content.task_mode = 'ref'
    second.content.continuity_mode = 'shot'

    render(
      <TaskSegmentEditor
        segment={first}
        trackSegments={[first, second]}
        selectedSegments={[first, second]}
        format="MiniMax"
        onContentChange={onContentChange}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
      />,
    )

    fireEvent.click(screen.getByRole('combobox', { name: 'Continuity mode' }))
    fireEvent.click(screen.getByRole('option', { name: 'Context' }))
    expect(onTrackSegmentsContentChange).toHaveBeenLastCalledWith([
      { segmentId: first.id, patch: { continuity_mode: 'shot' } },
      { segmentId: second.id, patch: { continuity_mode: 'context' } },
    ])

    fireEvent.click(screen.getByRole('combobox', { name: 'Reference image size' }))
    fireEvent.click(screen.getByRole('option', { name: 'Max' }))
    expect(onTrackSegmentsContentChange).toHaveBeenLastCalledWith([
      { segmentId: first.id, patch: { ref_image_size: 'max' } },
      { segmentId: second.id, patch: { ref_image_size: 'max' } },
    ])

    fireEvent.click(screen.getByRole('combobox', { name: 'Task mode' }))
    fireEvent.click(screen.getByRole('option', { name: 'VI2V' }))
    expect(onTrackSegmentsContentChange).toHaveBeenLastCalledWith([
      { segmentId: first.id, patch: { task_mode: 'edit' } },
      { segmentId: second.id, patch: { task_mode: 'edit' } },
    ])
    expect(onContentChange).not.toHaveBeenCalled()
  })

  it('stores the A/B prompt selection, edits B independently, and shows its tooltip on hover', async () => {
    const onContentChange = vi.fn()
    const initialSegment = taskSegment()
    const { rerender } = render(
      <TooltipProvider delayDuration={0}>
        <TaskSegmentEditor segment={initialSegment} onContentChange={onContentChange} />
      </TooltipProvider>,
    )

    const tooltipText = 'New: The user prompt output follows your A/B selection (select A to output A, select B to output B). Use A/B to compare prompts, or to preserve both the original and reverse-engineered prompts.'
    fireEvent.pointerMove(screen.getByLabelText(tooltipText), { pointerType: 'mouse' })
    expect((await screen.findByRole('tooltip')).textContent).toBe(tooltipText)

    activateTab('B')
    expect(onContentChange).toHaveBeenCalledWith({ user_prompt_variant: 'b' })

    const bSegment = taskSegment()
    bSegment.content.user_prompt_b = 'Reverse-engineered prompt'
    bSegment.content.user_prompt_variant = 'b'
    rerender(
      <TooltipProvider delayDuration={0}>
        <TaskSegmentEditor segment={bSegment} onContentChange={onContentChange} />
      </TooltipProvider>,
    )

    const prompt = screen.getByRole('textbox', { name: 'Prompt' }) as HTMLElement
    expect(prompt.textContent).toBe('Reverse-engineered prompt')
    inputEditable(prompt, 'Updated B prompt')
    expect(onContentChange).toHaveBeenCalledWith({ user_prompt_b: 'Updated B prompt' })
    expect(bSegment.content.user_prompt).toBe('Initial prompt')
  })

  it('offers last-frame mode after edit', () => {
    const onContentChange = vi.fn()
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={onContentChange} />)

    fireEvent.click(screen.getByRole('combobox'))
    const options = screen.getAllByRole('option').map((option) => option.textContent)

    expect(options).toEqual([
      'FL2V',
      'R2V',
      'VI2V',
      'L2V',
    ])
    fireEvent.click(screen.getByRole('option', { name: 'L2V' }))
    expect(onContentChange).toHaveBeenCalledWith({ task_mode: 'l2v' })
  })

  it.each([
    [0, 'T2V'],
    [1, 'I2V'],
    [2, 'FL2V'],
    [3, 'FMLF2V'],
  ])('shows the default task type for %i task images', (imageCount, expectedLabel) => {
    const segment = taskSegment()
    segment.content.images = Array.from({ length: imageCount }, (_, index) => ({
      id: `image-${index}`,
      source_type: 'input' as const,
      file_path: `image-${index}.png`,
    }))

    render(<TaskSegmentEditor segment={segment} onContentChange={vi.fn()} />)

    expect(screen.getByRole('combobox', { name: 'Task mode' }).textContent).toContain(expectedLabel)
  })

  it('keeps task prompt wheel scrolling inside the editor', () => {
    const onCanvasWheel = vi.fn()
    render(
      <div
        onWheelCapture={(event) => {
          const target = event.target as HTMLElement
          const captureElement = target.closest('[data-capture-wheel="true"]')
          if (!captureElement?.contains(document.activeElement)) onCanvasWheel()
        }}
      >
        <TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />
      </div>,
    )

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    prompt.focus()
    const keepsNativeScroll = fireEvent.wheel(prompt, { deltaY: 100 })

    expect(keepsNativeScroll).toBe(true)
    expect(onCanvasWheel).not.toHaveBeenCalled()
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

    expect(screen.getByRole('option', { name: 'RV2V' })).not.toBeNull()
  })

  it('uses rv2v for reference mode with preset video', () => {
    render(
      <TaskSegmentEditor
        segment={taskSegment()}
        videoSegments={[videoSegment(2, 5, 'preset')]}
        onContentChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('combobox'))

    expect(screen.getByRole('option', { name: 'RV2V' })).not.toBeNull()
  })

  it.each([
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

    expect(screen.getByRole('option', { name: 'R2V' })).not.toBeNull()
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
    const onOpenImagePreview = vi.fn()
    render(
      <TaskSegmentEditor
        segment={taskSegment()}
        onContentChange={onContentChange}
        onOpenImagePreview={onOpenImagePreview}
      />,
    )

    expect(screen.getByTestId('task-image-a').className).toContain('bg-black')
    const firstImage = screen.getByAltText('a.png')
    expect(firstImage.className).toContain('absolute')
    expect(firstImage.className).toContain('inset-0')
    expect(firstImage.className).toContain('h-full')
    expect(firstImage.className).toContain('w-full')
    expect(firstImage.className).toContain('object-contain')
    expect(screen.getByTestId('task-image-actions-a').className).toContain('opacity-0')
    expect(screen.getByTestId('task-image-actions-a').className).toContain('right-1')
    expect(screen.getByTestId('task-image-actions-a').className).toContain('top-1')
    expect(screen.getByTestId('task-image-index-a').textContent).toBe('0')
    expect(screen.getByTestId('task-image-index-a').className).toContain('bottom-0')
    expect(screen.getByTestId('task-image-index-b').textContent).toBe('1')
    expect(screen.getAllByRole('button', { name: 'Preview image' })[0].className).toContain('cursor-pointer')
    expect(screen.getAllByRole('button', { name: 'Preview image' })).toHaveLength(2)
    expect(screen.queryByRole('button', { name: '720° panorama preview' })).toBeNull()
    expect(screen.getAllByRole('button', { name: 'Delete image' })[0].className).toContain('cursor-pointer')
    expect(screen.getAllByRole('button', { name: 'Delete image' })[0].className).toContain('text-destructive')
    const previewButton = screen.getAllByRole('button', { name: 'Preview image' })[0]
    expect(previewButton.querySelector('.lucide-eye')).not.toBeNull()

    fireEvent.click(previewButton)
    expect(onOpenImagePreview).toHaveBeenCalledWith('a')
    expect(screen.queryByTestId('media-selector-mock')).toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: 'Delete image' })[0])
    expect(onContentChange).toHaveBeenCalledWith({
      images: [expect.objectContaining({ id: 'b' })],
    })
    expect(screen.queryByTestId('media-selector-mock')).toBeNull()
  })

  it('labels slot image placeholders with their slot number', () => {
    const segment = taskSegment()
    segment.content.images = [{
      id: 'slot-image',
      source_type: 'slot',
      slot_name: 'image2',
      file_name: 'image2',
    }]

    render(<TaskSegmentEditor segment={segment} onContentChange={vi.fn()} />)

    expect(screen.getByTestId('task-image-slot-image').textContent).toContain('Image 2')
  })

  it('shares an image across tasks before each task local images', () => {
    const first = taskSegment()
    first.content.images = [
      { id: 'local-a', source_type: 'input', file_path: 'local-a.png' },
      { id: 'shared', source_type: 'input', file_path: 'shared.png' },
    ]
    const second = secondTaskSegment()
    second.content.images = [
      { id: 'same-path', source_type: 'input', file_path: 'shared.png' },
      { id: 'local-b', source_type: 'input', file_path: 'local-b.png' },
    ]
    const onTrackSegmentsContentChange = vi.fn()

    render(
      <TaskSegmentEditor
        segment={first}
        trackSegments={[first, second]}
        onContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
      />,
    )
    fireEvent.click(screen.getByTestId('task-image-shared-shared'))

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([
      {
        segmentId: first.id,
        patch: { images: [
          expect.objectContaining({ id: 'shared', shared_reference: true }),
          expect.objectContaining({ id: 'local-a' }),
        ] },
      },
      {
        segmentId: second.id,
        patch: { images: [
          expect.objectContaining({ id: 'same-path', shared_reference: true }),
          expect.objectContaining({ id: 'local-b' }),
        ] },
      },
    ])
  })

  it('reselects an image in place and opens the chosen image first without batch controls', () => {
    const onContentChange = vi.fn()
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={onContentChange} />)

    fireEvent.click(screen.getByTestId('task-image-a'))

    const selector = screen.getByTestId('media-selector-mock')
    expect(selector.dataset.value).toBe('a.png')
    expect(selector.dataset.multiple).toBe('false')
    expect(selector.dataset.limit).toBe('1')

    fireEvent.click(screen.getByRole('button', { name: 'choose replacement' }))
    expect(onContentChange).toHaveBeenLastCalledWith({
      images: [
        expect.objectContaining({ id: 'a', file_path: 'replacement.png' }),
        expect.objectContaining({ id: 'b', file_path: 'b.png' }),
      ],
    })
  })

  it('passes the remaining image capacity to the add-image selector', () => {
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />)

    fireEvent.click(screen.getByRole('button', { name: 'Select image' }))

    const selector = screen.getByTestId('media-selector-mock')
    expect(selector.dataset.value).toBe('')
    expect(selector.dataset.multiple).toBe('true')
    expect(selector.dataset.limit).toBe('7')
  })

  it('supports one-based image item numbering', () => {
    render(
      <TaskSegmentEditor
        segment={taskSegment()}
        imageIndexOffset={1}
        onContentChange={vi.fn()}
      />,
    )

    expect(screen.getByTestId('task-image-index-a').textContent).toBe('1')
    expect(screen.getByTestId('task-image-index-b').textContent).toBe('2')
  })

  it('preserves the default image grid while exposing a container for narrow layouts', () => {
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />)

    const dropZone = screen.getByTestId('task-image-drop-zone')
    const grid = screen.getByTestId('task-image-grid')
    expect(dropZone.closest('.task-segment-editor')).not.toBeNull()
    expect(dropZone.className).toContain('aspect-square')
    expect(grid.className).toContain('grid-cols-2')
    expect(screen.getByTestId('task-image-a').className).toContain('w-full')
    expect(screen.getByTestId('task-image-a').className).toContain('self-start')
    expect(screen.getByRole('button', { name: 'Select image' }).className).toContain('w-full')
    expect(screen.getByRole('button', { name: 'Select image' }).className).toContain('self-start')
    expect(screen.getByTestId('task-prompt-panel').className).toContain('min-w-0')
  })

  it('renders saved panorama framing without showing a panorama action icon', () => {
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

    expect(screen.queryByRole('button', { name: '720° panorama preview' })).toBeNull()
    expect(screen.getAllByRole('button', { name: 'Preview image' })).toHaveLength(2)
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

  it('keeps the image grid stable without opening a hover focus preview', () => {
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />)

    const grid = screen.getByTestId('task-image-grid')
    const firstImage = screen.getByTestId('task-image-a')
    const secondImage = screen.getByTestId('task-image-b')

    expect(grid.className).toContain('grid-cols-2')
    expect(screen.queryByTestId('task-image-focus-preview')).toBeNull()
    fireEvent.mouseEnter(secondImage)
    expect(grid.className).toContain('grid-cols-2')
    expect(screen.queryByTestId('task-image-focus-preview')).toBeNull()
    expect(secondImage.className).toContain('relative')
    expect(secondImage.className).toContain('cursor-pointer')
    expect(secondImage.className).not.toContain('opacity-0')
    expect(firstImage.className).not.toContain('opacity-0')

    fireEvent.mouseLeave(grid)
    expect(grid.className).toContain('grid-cols-2')
    expect(screen.queryByTestId('task-image-focus-preview')).toBeNull()
    expect(secondImage.className).toContain('relative')
    expect(secondImage.className).not.toContain('opacity-0')
  })

  it('uses a three-column image grid once four or more task images are present', () => {
    const segment = taskSegment()
    segment.content.images = [
      ...(segment.content.images ?? []),
      { id: 'c', source_type: 'input', file_path: 'c.png', file_name: 'c.png' },
      { id: 'd', source_type: 'input', file_path: 'd.png', file_name: 'd.png' },
    ]

    render(<TaskSegmentEditor segment={segment} onContentChange={vi.fn()} />)

    expect(screen.getByTestId('task-image-grid').className).toContain('grid-cols-3')
  })

  it('uses locale messages for task mode and prompt labels', () => {
    render(
      <LocaleContext.Provider value="zh">
        <TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />
      </LocaleContext.Provider>,
    )

    expect(screen.getByRole('combobox').textContent).toContain('首尾生视频')
    expect(screen.getByRole('textbox', { name: '提示词' })).not.toBeNull()
    expect(screen.getByLabelText('任务图片拖放区域')).not.toBeNull()
  })

  it('opens an @ resource menu with task images and populated media tracks', () => {
    const onContentChange = vi.fn()
    const audio = videoSegment(0, 3)
    audio.id = 'audio-segment'
    audio.content = { media_type: 'audio', file_path: 'voice.wav' }
    render(
      <TaskSegmentEditor
        segment={taskSegment()}
        mediaTracks={[
          mediaTrack('audio', [audio]),
          mediaTrack('video', [videoSegment(0, 3)]),
          { ...mediaTrack('video', []), id: 'empty-video-track', name: 'Empty video track' },
        ]}
        onContentChange={onContentChange}
      />,
    )

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    inputEditable(prompt, '@')

    const menu = screen.getByRole('listbox', { name: 'Reference resources' })
    expect(menu.parentElement?.classList.contains('comfyui-easy-media')).toBe(true)
    expect(menu.className).toContain('bg-popover')
    expect(menu.textContent).toContain('Picture 1')
    expect(menu.textContent).toContain('Picture 2')
    expect(menu.textContent).toContain('Audio 1')
    expect(menu.textContent).toContain('Audio 2')
    expect(menu.textContent).toContain('Video 1')
    expect(menu.textContent).not.toContain('Empty video track')
    expect(screen.getByRole('option', { name: /Audio 1 Source video track/ })).not.toBeNull()
    expect(screen.getByRole('option', { name: /Audio 2 Voice track/ })).not.toBeNull()

    expect(screen.getByRole('option', { name: /Picture 1/ }).getAttribute('aria-selected')).toBe('true')
    fireEvent.keyDown(prompt, { key: 'Enter' })

    expect(onContentChange).toHaveBeenLastCalledWith({ user_prompt: '@Picture 1' })
    expect(screen.queryByRole('listbox', { name: 'Reference resources' })).toBeNull()
    const chip = prompt.querySelector('[data-prompt-reference-token="@Picture 1"]')
    expect(chip?.querySelector('img')?.getAttribute('src')).toContain('a.png')
    expect((chip as HTMLElement | null)?.style.color).toBe('var(--multitrack-task-bg)')
  })

  it('shows the interactive MiniMax placeholder only for an empty prompt', () => {
    const onContentChange = vi.fn()
    const segment = taskSegment()
    segment.content.user_prompt = ''
    const { rerender } = render(
      <LocaleContext.Provider value="zh">
        <TaskSegmentEditor segment={segment} format="MiniMax" onContentChange={onContentChange} />
      </LocaleContext.Provider>,
    )

    const prompt = screen.getByRole('textbox', { name: '提示词' })
    expect(prompt.getAttribute('data-placeholder')).toBe('参考模式下最多支持9张图片，3个音频，3个视频， 输入文字或 @ 参考内容， 自由组合图、文、音，视频多元素，定义精彩互动。例如 @图片1 模仿 @视频1 的动作，音色参考 @音频1。')
    expect(prompt.textContent).toBe('')

    const placeholderTrigger = screen.getByRole('button', { name: '打开引用资源' })
    expect(placeholderTrigger.className).toContain('prompt-reference-chip')
    fireEvent.click(placeholderTrigger)
    fireEvent.click(screen.getByRole('option', { name: /图片1/ }))
    expect(onContentChange).toHaveBeenLastCalledWith({ user_prompt: '@图片1' })

    rerender(
      <LocaleContext.Provider value="zh">
        <TaskSegmentEditor segment={segment} format="Seedance" onContentChange={onContentChange} />
      </LocaleContext.Provider>,
    )
    expect(screen.getByRole('textbox', { name: '提示词' }).getAttribute('data-placeholder')).toBe('双击描述您想要生成的内容...')
    expect(screen.queryByRole('button', { name: '打开引用资源' })).toBeNull()
  })

  it('restores the MiniMax placeholder immediately after deleting back to empty', () => {
    const onContentChange = vi.fn()
    render(<TaskSegmentEditor segment={taskSegment()} format="MiniMax" onContentChange={onContentChange} />)

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    prompt.innerHTML = '<br>'
    fireEvent.input(prompt)

    expect(onContentChange).toHaveBeenLastCalledWith({ user_prompt: '' })
    expect(prompt.textContent).toBe('')
    expect(screen.getByRole('button', { name: 'Open reference resources' })).not.toBeNull()
  })

  it('does not open the resource menu when clicking prompt text that already contains @', () => {
    const segment = taskSegment()
    segment.content.user_prompt = 'Keep this @ character'
    render(<TaskSegmentEditor segment={segment} onContentChange={vi.fn()} />)

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    fireEvent.focus(prompt)
    fireEvent.click(prompt)

    expect(screen.queryByRole('listbox', { name: 'Reference resources' })).toBeNull()
  })

  it('centers the empty reference state below its header', () => {
    const segment = taskSegment()
    segment.content.images = []
    render(<TaskSegmentEditor segment={segment} mediaTracks={[]} onContentChange={vi.fn()} />)

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    inputEditable(prompt, '@')

    const menu = screen.getByRole('listbox', { name: 'Reference resources' })
    const emptyState = screen.getByTestId('reference-resources-empty')
    expect(menu.className).toContain('h-64')
    expect(emptyState.className).toContain('flex-1')
    expect(emptyState.className).toContain('items-center')
    expect(emptyState.className).toContain('justify-center')
    expect(emptyState.className).toContain('text-center')
    expect(emptyState.textContent).toContain('Add task images or media clips to a track first.')
  })

  it('highlights official media tags with thumbnails, Lucide icons, and track colors', () => {
    const segment = taskSegment()
    segment.content.user_prompt = '<Picture 1> <Audio 1> <Audio 2> <Video 1>'
    const audio = videoSegment(0, 3)
    audio.id = 'audio-segment'
    audio.content = { media_type: 'audio', file_path: 'voice.wav' }
    render(
      <TaskSegmentEditor
        segment={segment}
        mediaTracks={[
          mediaTrack('audio', [audio]),
          mediaTrack('video', [videoSegment(0, 3)]),
        ]}
        onContentChange={vi.fn()}
      />,
    )

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    const chips = prompt.querySelectorAll('[data-prompt-reference-token]')
    expect(chips).toHaveLength(4)
    expect(chips[0].querySelector('img')?.getAttribute('src')).toContain('a.png')
    expect(chips[1].querySelector('[data-reference-icon="audio"] svg.lucide')).not.toBeNull()
    expect(chips[2].querySelector('[data-reference-icon="audio"] svg.lucide')).not.toBeNull()
    expect(chips[3].querySelector('[data-reference-icon="video"] svg.lucide')).not.toBeNull()
    expect((chips[0] as HTMLElement).className).toContain('align-middle')
    expect((chips[0] as HTMLElement).className).toContain('items-center')
    expect((chips[0] as HTMLElement).className).toContain('leading-none')
    expect((chips[0] as HTMLElement).className).toContain('rounded-md')
    expect((chips[0] as HTMLElement).className).toContain('bg-background')
    expect((chips[0] as HTMLElement).className).toContain('border-border')
    expect(chips[0].querySelector('span:last-child')?.className).toContain('items-center')
    expect((chips[1] as HTMLElement).style.color).toBe('var(--multitrack-video-waveform)')
    expect((chips[2] as HTMLElement).style.color).toBe('var(--multitrack-audio-waveform)')
    expect((chips[3] as HTMLElement).style.color).toBe('var(--multitrack-video-waveform)')
  })

  it('uses orange for H3 language and speaker tags and theme color for dialogue text', () => {
    const segment = taskSegment()
    segment.content.user_prompt = '<Subject 12> (S1,S2) says <d>[Chinese] 你好</d> [English] [Shot 3] <scenetrans> <cutoff> [reference generation]'
    render(<TaskSegmentEditor segment={segment} onContentChange={vi.fn()} />)

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    const semantics = prompt.querySelectorAll('[data-prompt-semantic-token]')
    expect(Array.from(semantics, (item) => item.textContent)).toEqual([
      '<Subject 12>',
      '(S1,S2)',
      '<d>',
      '[Chinese]',
      '</d>',
      '[English]',
      '[Shot 3]',
      '<scenetrans>',
      '<cutoff>',
    ])
    expect(semantics[0].className).toContain('text-prompt-semantic')
    expect(semantics[1].className).toContain('text-prompt-semantic')
    expect(semantics[2].className).toContain('text-prompt-semantic')
    expect(semantics[3].className).toContain('text-prompt-semantic')
    expect(semantics[4].className).toContain('text-prompt-semantic')
    expect(semantics[6].className).toContain('text-highlight')
    expect(Array.from(semantics).filter((_, index) => index !== 6)
      .every((item) => item.className.includes('text-prompt-semantic'))).toBe(true)
    const dialogue = prompt.querySelector('[data-prompt-dialogue-content]')
    expect(dialogue?.className).toContain('text-highlight')
    expect(dialogue?.textContent).toBe(' 你好')
    expect(prompt.textContent).toContain('[reference generation]')
  })

  it('keeps reference tokens stable when typing adjacent text', () => {
    const onContentChange = vi.fn()
    const segment = taskSegment()
    segment.content.user_prompt = '@Picture 1'
    render(<TaskSegmentEditor segment={segment} onContentChange={onContentChange} />)

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    const chip = prompt.querySelector('[data-prompt-reference-token="@Picture 1"]')
    expect(chip).not.toBeNull()
    chip?.after(document.createTextNode(' moves forward'))
    fireEvent.input(prompt)

    expect(onContentChange).toHaveBeenLastCalledWith({ user_prompt: '@Picture 1 moves forward' })
    expect(prompt.querySelector('[data-prompt-reference-token="@Picture 1"]')).toBe(chip)
  })

  it('pastes plain text into the prompt without bubbling to the ComfyUI canvas', () => {
    const onContentChange = vi.fn()
    const onCanvasPaste = vi.fn()
    const segment = taskSegment()
    segment.content.user_prompt = ''
    render(
      <div onPaste={onCanvasPaste}>
        <TaskSegmentEditor segment={segment} onContentChange={onContentChange} />
      </div>,
    )

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    prompt.focus()
    const range = document.createRange()
    range.selectNodeContents(prompt)
    range.collapse(false)
    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)
    fireEvent.paste(prompt, {
      clipboardData: {
        getData: (type: string) => type === 'text/plain' ? '<Picture 1> moves' : '',
      },
    })

    expect(onCanvasPaste).not.toHaveBeenCalled()
    expect(onContentChange).toHaveBeenLastCalledWith({ user_prompt: '<Picture 1> moves' })
    expect(prompt.querySelector('[data-prompt-reference-token="<Picture 1>"]')).not.toBeNull()
  })

  it('copies newly inserted reference chips as their prompt tag text', () => {
    render(<TaskSegmentEditor segment={taskSegment()} onContentChange={vi.fn()} />)

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    inputEditable(prompt, '@')
    fireEvent.keyDown(prompt, { key: 'Enter' })
    expect(prompt.querySelector('[data-prompt-reference-token="@Picture 1"]')).not.toBeNull()

    const selection = window.getSelection()
    const selectAll = document.createRange()
    selectAll.selectNodeContents(prompt)
    selection?.removeAllRanges()
    selection?.addRange(selectAll)
    const setData = vi.fn()
    fireEvent.copy(prompt, {
      clipboardData: { setData },
    })

    expect(setData).toHaveBeenCalledWith('text/plain', '@Picture 1')
  })

  it('deletes a reference chip as one atomic value with Backspace', () => {
    const onContentChange = vi.fn()
    const segment = taskSegment()
    segment.content.user_prompt = 'Before <Picture 1> after'
    render(<TaskSegmentEditor segment={segment} onContentChange={onContentChange} />)

    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    const chip = prompt.querySelector('[data-prompt-reference-token="<Picture 1>"]')
    expect(chip).not.toBeNull()
    prompt.focus()
    const selection = window.getSelection()
    const range = document.createRange()
    range.setStartAfter(chip!)
    range.collapse(true)
    selection?.removeAllRanges()
    selection?.addRange(range)

    fireEvent.keyDown(prompt, { key: 'Backspace' })

    expect(onContentChange).toHaveBeenLastCalledWith({ user_prompt: 'Before  after' })
    expect(prompt.querySelector('[data-prompt-reference-token]')).toBeNull()
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
          { modes: ['default', 'l2v'], format: 'MiniMax', system_prompt: 'MiniMax base template' },
          { modes: ['ref', 'edit'], format: 'MiniMax', system_prompt: 'MiniMax reference template' },
        ],
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    const { rerender, unmount } = render(<TaskSegmentEditor segment={taskSegment()} onContentChange={onContentChange} />)

    fireEvent.click(screen.getByRole('button', { name: 'System Prompt' }))
    const systemPrompt = await screen.findByRole('textbox', { name: 'System Prompt' }) as HTMLElement
    await waitFor(() => {
      expect(systemPrompt.textContent).toBe('Default image-to-video template')
    })

    const resetSegment = {
      ...taskSegment(),
      content: {
        ...taskSegment().content,
        system_prompt: '',
      },
    }
    rerender(<TaskSegmentEditor segment={resetSegment} onContentChange={onContentChange} />)
    expect(systemPrompt.textContent).toBe('Default image-to-video template')

    const refSegment = {
      ...taskSegment(),
      content: {
        ...taskSegment().content,
        task_mode: 'ref' as const,
      },
    }
    rerender(<TaskSegmentEditor segment={refSegment} onContentChange={onContentChange} />)
    await waitFor(() => {
      expect(systemPrompt.textContent).toBe('Default reference-to-video template')
    })

    rerender(
      <TaskSegmentEditor
        segment={refSegment}
        videoSegments={[videoSegment(2, 5)]}
        onContentChange={onContentChange}
      />,
    )
    await waitFor(() => {
      expect(systemPrompt.textContent).toBe('Default reference-edit template')
    })

    rerender(<TaskSegmentEditor segment={resetSegment} format="MiniMax" onContentChange={onContentChange} />)
    await waitFor(() => {
      expect(systemPrompt.textContent).toBe('MiniMax base template')
    })

    rerender(<TaskSegmentEditor segment={refSegment} format="MiniMax" onContentChange={onContentChange} />)
    await waitFor(() => {
      expect(systemPrompt.textContent).toBe('MiniMax reference template')
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith('/easy-media/prompt/system-prompts')

    unmount()
    render(<TaskSegmentEditor segment={refSegment} onContentChange={onContentChange} />)
    fireEvent.click(screen.getByRole('button', { name: 'System Prompt' }))
    const remountedSystemPrompt = await screen.findByRole('textbox', { name: 'System Prompt' }) as HTMLElement
    expect(remountedSystemPrompt.textContent).toBe('Default reference-to-video template')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it('edits a customized system prompt and highlights complete placeholders', () => {
    const onContentChange = vi.fn()
    const customizedSegment = {
      ...taskSegment(),
      content: {
        ...taskSegment().content,
        system_prompt: 'Create {subject} in <style> with <Picture 1> and @图片1',
      },
    }
    const { container } = render(
      <TooltipProvider>
        <TaskSegmentEditor segment={customizedSegment} onContentChange={onContentChange} />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'System Prompt' }))
    const textarea = screen.getByRole('textbox', { name: 'System Prompt' }) as HTMLElement
    const highlightedVariables = container.querySelectorAll('[data-system-prompt-variable="true"]')

    expect(textarea.getAttribute('contenteditable')).toBe('true')
    expect(textarea.classList.contains('text-transparent')).toBe(false)
    expect(textarea.classList.contains('caret-foreground')).toBe(true)
    expect(Array.from(highlightedVariables, (item) => item.textContent)).toEqual(['{subject}', '<style>', '<Picture 1>'])
    expect(textarea.querySelectorAll('[data-prompt-reference-token]')).toHaveLength(0)
    expect(textarea.textContent).toBe('Create {subject} in <style> with <Picture 1> and @图片1')

    inputEditable(textarea, 'Create {character} in <location>')

    expect(onContentChange).toHaveBeenCalledWith({ system_prompt: 'Create {character} in <location>' })
  })

  it('keeps the caret position when completing a highlighted tag after a line break', () => {
    const onContentChange = vi.fn()
    const segment = taskSegment()
    segment.content.system_prompt = 'first\n<style'
    render(
      <TooltipProvider>
        <TaskSegmentEditor segment={segment} onContentChange={onContentChange} />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'System Prompt' }))
    const editor = screen.getByRole('textbox', { name: 'System Prompt' })
    const secondLine = editor.childNodes[2]
    expect(secondLine.nodeType).toBe(Node.TEXT_NODE)
    const textNode = secondLine as Text
    textNode.insertData(textNode.length, '>')
    const range = document.createRange()
    range.setStart(textNode, textNode.length)
    range.collapse(true)
    const selection = window.getSelection()
    selection?.removeAllRanges()
    selection?.addRange(range)
    fireEvent.input(editor)

    expect(onContentChange).toHaveBeenLastCalledWith({ system_prompt: 'first\n<style>' })
    const highlighted = editor.querySelector('[data-system-prompt-variable="true"]')
    expect(highlighted?.textContent).toBe('<style>')
    expect(selection?.anchorNode).toBe(highlighted?.firstChild)
    expect(selection?.anchorOffset).toBe('<style>'.length)
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
    inputEditable(screen.getByRole('textbox', { name: 'Prompt' }), 'First rewritten|Second rewritten')

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([
      { segmentId: 'task-segment', patch: { user_prompt: 'First rewritten' } },
      { segmentId: 'task-segment-2', patch: { user_prompt: 'Second rewritten' } },
    ])
  })

  it('reads and writes each task selected A/B prompt in combined mode', () => {
    const onTrackSegmentsContentChange = vi.fn()
    const first = taskSegment()
    const second = secondTaskSegment()
    second.content.user_prompt_b = 'Selected B prompt'
    second.content.user_prompt_variant = 'b'
    render(
      <TaskSegmentEditor
        segment={first}
        trackSegments={[first, second]}
        onContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
      />,
    )

    activateTab('Combined')
    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    expect(prompt.textContent).toBe('Initial prompt|Selected B prompt')

    inputEditable(prompt, 'Updated A|Updated B')

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([
      { segmentId: 'task-segment', patch: { user_prompt: 'Updated A' } },
      { segmentId: 'task-segment-2', patch: { user_prompt_b: 'Updated B' } },
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
    const prompt = screen.getByRole('textbox', { name: 'Prompt' }) as HTMLElement
    inputEditable(prompt, 'First pasted｜Second pasted|Third pasted')

    expect(prompt.textContent).toBe('First pasted|Second pasted|Third pasted')
    const highlight = screen.getByTestId('combined-prompt-highlight')
    expect(highlight.querySelectorAll('[data-pipe="true"]')).toHaveLength(2)
    expect(highlight.querySelector('[data-pipe="true"]')?.className).toContain('text-highlight')
    const updated = onTrackSegmentsChange.mock.lastCall?.[0] as MultiTrackSegment[]
    expect(updated).toHaveLength(3)
    expect(updated.map(({ start_frame, end_frame, content }) => ({
      start_frame,
      end_frame,
      user_prompt: content.user_prompt,
    }))).toEqual([
      { start_frame: 0, end_frame: 4, user_prompt: 'First pasted' },
      { start_frame: 4, end_frame: 7, user_prompt: 'Second pasted' },
      { start_frame: 7, end_frame: 10, user_prompt: 'Third pasted' },
    ])
    expect(updated[0].id).toBe('task-segment')
    expect(updated[1].id).toBe('task-segment-2')
    expect(updated[2]).toMatchObject({
      content: { media_type: 'none', task_mode: 'default', images: [] },
    })

    inputEditable(prompt, '')
    const cleared = onTrackSegmentsChange.mock.lastCall?.[0] as MultiTrackSegment[]
    expect(cleared).toHaveLength(1)
    expect(cleared[0]).toMatchObject({
      id: 'task-segment',
      start_frame: 0,
      end_frame: 10,
      content: { user_prompt: '' },
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

    expect(screen.getByText('Segment 2').className).toContain('text-[10px]')
    expect(screen.getByText('Segment 2').className).toContain('text-primary')
    expect(screen.getByText('00:00:03').className).toContain('text-[10px]')

    fireEvent.click(screen.getByRole('button', { name: 'Edit task duration' }))
    expect(screen.queryByText('Segment 2')).toBeNull()
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

  it('uses responsive preview editor sizing without rendering the empty image picker as a button element', () => {
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

  it('shows connected image inputs in the media selector slot list', () => {
    const emptyImageSegment = {
      ...taskSegment(),
      content: {
        ...taskSegment().content,
        images: [],
      },
    }
    const sourceNode = {
      outputs: [{ shape: 0 }],
      imgs: [{ currentSrc: '/view?filename=reference.png&type=input' }],
    }
    const node = {
      inputs: [{ name: 'image', type: 'IMAGE', link: 7 }],
    }
    const app = {
      graph: {
        links: { 7: { origin_id: 3, origin_slot: 0 } },
        getNodeById: (id: number) => id === 3 ? sourceNode : null,
      },
    }

    render(
      <TaskSegmentEditor
        segment={emptyImageSegment}
        node={node}
        app={app}
        onContentChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Task image drop zone' }))
    expect(screen.getByTestId('media-selector-mock').getAttribute('data-slot-items'))
      .toBe('__slot__:image')
  })
})
