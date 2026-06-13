import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: () => null,
}))

import { EditPanel } from './EditPanel'
import type { ImageItem, MaintainSegment, Segment } from '@/types/timeline'

function segment(
  id: string,
  text: string,
  start: number,
  end: number,
  images: ImageItem[] = [],
): MaintainSegment {
  return {
    id,
    start_frame: start,
    end_frame: end,
    content: { text, images, type: 'flf' },
    color: 'var(--secondary)',
  }
}

async function renderPanel(
  allSegments: MaintainSegment[],
  onAllSegmentsChange = vi.fn<(segments: Segment[]) => void>(),
) {
  const props = {
    segment: allSegments[0],
    allSegments,
    totalFrames: 120,
    frameRate: 24,
    displayFormat: 'frames' as const,
    areaWidth: 600,
    canvasScale: 1,
    trackColor: 'var(--secondary)',
    onContentChange: vi.fn(),
    onAllSegmentsChange,
  }

  const result = render(<EditPanel {...props} />)
  const combinedTab = screen.getByRole('tab', { name: 'Combined' })
  fireEvent.pointerDown(combinedTab, { button: 0, ctrlKey: false })
  fireEvent.mouseDown(combinedTab, { button: 0, ctrlKey: false })
  fireEvent.click(combinedTab)
  await waitFor(() => {
    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe(
      allSegments.map((s) => s.content.text).join('|'),
    )
  })
  return {
    ...result,
    onAllSegmentsChange,
    rerenderWithSegments(nextSegments: MaintainSegment[]) {
      result.rerender(<EditPanel {...props} segment={nextSegments[0]} allSegments={nextSegments} />)
    },
  }
}

describe('EditPanel combined prompt editor', () => {
  it('syncs displayed combined text when segment list changes outside the editor', async () => {
    const first = [segment('a', 'A', 0, 59), segment('b', 'B', 60, 119)]
    const next = [first[0], first[1], segment('c', 'C', 120, 179)]
    const { rerenderWithSegments } = await renderPanel(first)

    rerenderWithSegments(next)

    expect((screen.getByRole('textbox') as HTMLTextAreaElement).value).toBe('A|B|C')
  })

  it('commits empty parts so backspace and trailing separators can clear or add segments', async () => {
    const onAllSegmentsChange = vi.fn<(segments: Segment[]) => void>()
    await renderPanel([segment('a', 'A', 0, 59), segment('b', 'B', 60, 119)], onAllSegmentsChange)

    fireEvent.change(screen.getByRole('textbox'), { target: { value: 'A|' } })

    expect(onAllSegmentsChange).toHaveBeenLastCalledWith([
      expect.objectContaining({ content: expect.objectContaining({ text: 'A' }) }),
      expect.objectContaining({ content: expect.objectContaining({ text: '' }) }),
    ])

    fireEvent.change(screen.getByRole('textbox'), { target: { value: '' } })

    expect(onAllSegmentsChange).toHaveBeenLastCalledWith([])
  })

  it('keeps textarea editable during IME composition before committing on composition end', async () => {
    const onAllSegmentsChange = vi.fn<(segments: Segment[]) => void>()
    await renderPanel([segment('a', '', 0, 119)], onAllSegmentsChange)
    const textarea = screen.getByRole('textbox') as HTMLTextAreaElement

    fireEvent.compositionStart(textarea)
    fireEvent.change(textarea, { target: { value: '测试' } })

    expect(textarea.value).toBe('测试')
    expect(onAllSegmentsChange).not.toHaveBeenCalled()

    fireEvent.compositionEnd(textarea)

    expect(onAllSegmentsChange).toHaveBeenLastCalledWith([
      expect.objectContaining({ content: expect.objectContaining({ text: '测试' }) }),
    ])
  })

  it('commits pasted combined text as multiple segments', async () => {
    const onAllSegmentsChange = vi.fn<(segments: Segment[]) => void>()
    await renderPanel([segment('a', '', 0, 119)], onAllSegmentsChange)

    fireEvent.change(screen.getByRole('textbox'), { target: { value: '一|二|三' } })

    expect(onAllSegmentsChange).toHaveBeenLastCalledWith([
      expect.objectContaining({ content: expect.objectContaining({ text: '一' }) }),
      expect.objectContaining({ content: expect.objectContaining({ text: '二' }) }),
      expect.objectContaining({ content: expect.objectContaining({ text: '三' }) }),
    ])
  })

  it('highlights separators without changing the visual text layout', async () => {
    const { container } = await renderPanel([
      segment('a', 'A', 0, 59),
      segment('b', 'B', 60, 119),
    ])

    const pipe = container.querySelector('[data-pipe="1"]') as HTMLSpanElement | null

    expect(pipe).not.toBeNull()
    expect(pipe?.style.padding).toBe('')
    expect(pipe?.style.fontWeight).toBe('')
  })
})

describe('EditPanel image sub-block sync', () => {
  it('updates same-segment image ranges immediately when parent duration changes', () => {
    const first = segment('a', 'A', 0, 252, [
      { source_type: 'input', file_path: 'a.png', file_name: 'a.png', start_frame: 0, end_frame: 126 },
      { source_type: 'input', file_path: 'b.png', file_name: 'b.png', start_frame: 127, end_frame: 252 },
    ])
    const next = segment('a', 'A', 0, 120, [
      { source_type: 'input', file_path: 'a.png', file_name: 'a.png', start_frame: 0, end_frame: 60 },
      { source_type: 'input', file_path: 'b.png', file_name: 'b.png', start_frame: 61, end_frame: 120 },
    ])
    const props = {
      segment: first,
      allSegments: [first],
      totalFrames: 253,
      frameRate: 24,
      displayFormat: 'frames' as const,
      areaWidth: 600,
      canvasScale: 1,
      trackColor: 'var(--secondary)',
      onContentChange: vi.fn(),
      onAllSegmentsChange: vi.fn(),
    }

    const { rerender } = render(<EditPanel {...props} />)

    rerender(<EditPanel {...props} segment={next} allSegments={[next]} totalFrames={121} />)

    const block = screen.getByText('b.png').closest('[role="button"]') as HTMLElement
    expect(parseFloat(block.style.left)).toBeCloseTo(305, 0)
  })

  it('uploads dropped local image files into the sub-track at the drop frame', async () => {
    const onContentChange = vi.fn()
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ name: 'pic.png', subfolder: 'sub' }),
    })
    vi.stubGlobal('fetch', fetchMock)
    const seg = segment('a', 'A', 0, 119)

    render(
      <EditPanel
        segment={seg}
        allSegments={[seg]}
        totalFrames={120}
        frameRate={24}
        displayFormat="frames"
        areaWidth={600}
        canvasScale={1}
        trackColor="var(--secondary)"
        onContentChange={onContentChange}
        onAllSegmentsChange={vi.fn()}
      />,
    )

    const addButton = screen.getByText('Add image').closest('button') as HTMLButtonElement
    const subTrack = addButton.parentElement as HTMLElement
    vi.spyOn(subTrack, 'getBoundingClientRect').mockReturnValue({
      left: 0,
      top: 0,
      right: 600,
      bottom: 48,
      width: 600,
      height: 48,
      x: 0,
      y: 0,
      toJSON: () => ({}),
    })

    const file = new File(['image'], 'pic.png', { type: 'image/png' })
    fireEvent.drop(subTrack, {
      clientX: 300,
      dataTransfer: {
        files: [file],
        types: ['Files'],
      },
    })

    await waitFor(() => {
      expect(onContentChange).toHaveBeenCalledWith({
        images: [
          expect.objectContaining({
            file_path: 'sub/pic.png',
            file_name: 'pic.png',
            source_type: 'input',
            start_frame: 60,
          }),
        ],
      })
    })
  })
})
