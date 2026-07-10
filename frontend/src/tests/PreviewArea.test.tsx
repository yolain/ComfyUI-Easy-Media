import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PreviewArea } from '@/components/widgets/multitrack/PreviewArea'
import type { SelectedMultiTrackSegment } from '@/lib/multitrack-utils'
import type { TrackData } from '@/types/multitrack'

vi.mock('@/lib/video-utils', () => ({
  loadBrowserVideoMetadata: vi.fn(() => new Promise(() => {})),
}))

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: ({ onChange }: { onChange: (value: string, source?: 'input' | 'output' | 'local') => void }) => (
    <button type="button" onClick={() => onChange('picked.png', 'input')}>
      mock select image
    </button>
  ),
}))

function trackData(): { data: TrackData; selectedSegment: SelectedMultiTrackSegment } {
  const videoSegment = {
    id: 'selected-video',
    start_frame: 0,
    end_frame: 2,
    color: 'var(--primary)',
    content: {
      media_type: 'video' as const,
      source_type: 'input' as const,
      file_path: 'clips/shot.mp4',
      file_name: 'shot.mp4',
      duration: 2,
    },
  }
  const data: TrackData = {
    muted: false,
    frame_rate: 24,
    total_length: 5,
    tracks: [
      {
        id: 'video-track',
        name: 'Video 1',
        type: 'video',
        color: 'var(--primary)',
        muted: false,
        locked: false,
        segments: [videoSegment],
      },
    ],
  }
  return {
    data,
    selectedSegment: {
      trackId: 'video-track',
      trackType: 'video',
      segment: videoSegment,
    },
  }
}

function addActiveTaskTrack(data: TrackData): void {
  data.tracks.unshift({
    id: 'task-track',
    name: 'Task 1',
    type: 'task',
    color: 'var(--primary)',
    muted: false,
    locked: false,
    segments: [
      {
        id: 'task-before',
        start_frame: 0,
        end_frame: 24,
        color: 'var(--primary)',
        content: { media_type: 'none', images: [] },
      },
      {
        id: 'active-task',
        start_frame: 24,
        end_frame: 48,
        color: 'var(--primary)',
        content: {
          media_type: 'none',
          images: [
            {
              id: 'first',
              source_type: 'input',
              file_path: 'tasks/first.png',
              file_name: 'first.png',
              panorama_view: {
                version: 1,
                projection: 'equirectangular',
                yaw: 30,
                pitch: -10,
                hfov: 75,
                aspect_ratio: 16 / 9,
              },
            },
            { id: 'second', source_type: 'url', url: 'https://example.com/second.png', file_name: 'second.png' },
          ],
        },
      },
    ],
  })
}

describe('PreviewArea', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      unobserve() {}
      disconnect() {}
    })
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it('shows track guidance in the preview when the timeline has no segments', () => {
    const { data } = trackData()
    data.tracks[0].segments = []

    render(
      <PreviewArea
        data={data}
        currentTime={0}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    expect(screen.getByTestId('multitrack-empty-preview')).not.toBeNull()
    expect(screen.getByText('Add a task segment or video clip to the tracks below.')).not.toBeNull()
    expect(screen.getByText('Task track')).not.toBeNull()
    expect(screen.getByText('Video track')).not.toBeNull()
    expect(screen.getByText('Audio track')).not.toBeNull()
    expect(screen.getByText('Subtitle track')).not.toBeNull()
    expect(screen.getByText('Task track').closest('dt')?.querySelector('svg')).not.toBeNull()
    expect(screen.getByText('Video track').closest('dt')?.querySelector('svg')).not.toBeNull()
    expect(screen.getByText('Audio track').closest('dt')?.querySelector('svg')).not.toBeNull()
    expect(screen.getByText('Subtitle track').closest('dt')?.querySelector('svg')).not.toBeNull()
    expect(screen.getByTestId('multitrack-video-stage').closest('.hidden')).not.toBeNull()
  })

  it('does not bubble preview clicks to the widget clear-selection handler when a segment is selected', () => {
    const onParentClick = vi.fn()
    const { data, selectedSegment } = trackData()

    render(
      <div onClick={onParentClick}>
        <PreviewArea
          data={data}
          currentTime={3}
          selectedSegment={selectedSegment}
          isPlaying={false}
          node={{ widgets: [] }}
          onGlobalSettingsChange={vi.fn()}
          onSelectedSegmentContentChange={vi.fn()}
          onSelectedSegmentDurationChange={vi.fn()}
        />
      </div>,
    )

    fireEvent.click(screen.getByTestId('multitrack-black-frame'))

    expect(onParentClick).not.toHaveBeenCalled()
  })

  it('keeps subtitle speech settings in memory while the settings panel unmounts', () => {
    const { data } = trackData()
    const subtitleSegment = {
      id: 'subtitle-1',
      start_frame: 0,
      end_frame: 48,
      color: 'var(--primary)',
      content: {
        media_type: 'subtitle' as const,
        text: '字幕',
        subtitle_style: {
          font_size: 24,
          color: '#ffffff',
          outline_color: '#000000',
          background_color: 'transparent',
          background_opacity: 0.7,
          x: 0.15,
          y: 0.8,
          width: 0.7,
        },
      },
    }
    const taskSegment = {
      id: 'task-1',
      start_frame: 0,
      end_frame: 48,
      color: 'var(--primary)',
      content: { media_type: 'none' as const },
    }
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle',
      type: 'subtitle',
      color: 'var(--primary)',
      muted: false,
      locked: false,
      segments: [subtitleSegment],
    })
    data.tracks.push({
      id: 'task-track',
      name: 'Task',
      type: 'task',
      color: 'var(--primary)',
      muted: false,
      locked: false,
      segments: [taskSegment],
    })
    const subtitleSelection: SelectedMultiTrackSegment = {
      trackId: 'subtitle-track',
      trackType: 'subtitle',
      segment: subtitleSegment,
    }
    const taskSelection: SelectedMultiTrackSegment = {
      trackId: 'task-track',
      trackType: 'task',
      segment: taskSegment,
    }
    const props = {
      data,
      currentTime: 0,
      selectedSegment: subtitleSelection,
      isPlaying: false,
      node: { widgets: [] },
      onGlobalSettingsChange: vi.fn(),
      onSelectedSegmentContentChange: vi.fn(),
      onSelectedSegmentDurationChange: vi.fn(),
    }

    const { rerender } = render(<PreviewArea {...props} />)

    fireEvent.mouseDown(screen.getByRole('tab', { name: /Speech/ }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Control prompt' }), { target: { value: '四川话' } })
    fireEvent.change(screen.getByRole('spinbutton', { name: 'CFG' }), { target: { value: '3.4' } })
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Steps' }), { target: { value: '18' } })
    fireEvent.click(screen.getByRole('button', { name: 'Add reference audio' }))
    fireEvent.click(screen.getByRole('button', { name: 'mock select image' }))

    rerender(<PreviewArea {...props} selectedSegment={taskSelection} />)
    expect(screen.queryByTestId('subtitle-settings-panel')).toBeNull()

    rerender(<PreviewArea {...props} selectedSegment={subtitleSelection} />)
    fireEvent.mouseDown(screen.getByRole('tab', { name: /Speech/ }))

    expect((screen.getByRole('textbox', { name: 'Control prompt' }) as HTMLTextAreaElement).value).toBe('四川话')
    expect(screen.getByRole('spinbutton', { name: 'CFG' }).getAttribute('value')).toBe('3.4')
    expect(screen.getByRole('spinbutton', { name: 'Steps' }).getAttribute('value')).toBe('18')
    expect(screen.getByText('picked.png')).not.toBeNull()
  })

  it('opens and exits the enlarged image preview for a selected task segment', () => {
    const { data } = trackData()
    const taskSegment = {
      id: 'selected-task',
      start_frame: 0,
      end_frame: 24,
      color: 'var(--primary)',
      content: {
        media_type: 'none' as const,
        images: [
          { id: 'a', source_type: 'input' as const, file_path: 'a.png', file_name: 'a.png' },
          { id: 'b', source_type: 'input' as const, file_path: 'b.png', file_name: 'b.png' },
        ],
      },
    }
    data.tracks.unshift({
      id: 'task-track',
      name: 'Task',
      type: 'task',
      color: 'var(--primary)',
      muted: false,
      locked: false,
      segments: [taskSegment],
    })
    const selectedSegment: SelectedMultiTrackSegment = {
      trackId: 'task-track',
      trackType: 'task',
      segment: taskSegment,
    }
    const props = {
      data,
      currentTime: 0,
      selectedSegment,
      isPlaying: false,
      node: { widgets: [] },
      onGlobalSettingsChange: vi.fn(),
      onSelectedSegmentContentChange: vi.fn(),
      onSelectedSegmentDurationChange: vi.fn(),
    }
    const { rerender } = render(<PreviewArea {...props} />)

    expect(screen.getByTestId('task-image-drop-zone')).not.toBeNull()
    expect(screen.queryByRole('button', { name: '720° panorama preview' })).toBeNull()
    fireEvent.click(screen.getAllByRole('button', { name: 'Preview image' })[0])
    const expandedPreview = screen.getByTestId('task-image-expanded-preview')
    const expandedImage = screen.getByRole('img', { name: 'a.png' }) as HTMLImageElement
    expect(expandedPreview).not.toBeNull()
    expect(expandedImage.className).toContain('object-contain')
    expect(screen.queryByTestId('task-image-drop-zone')).toBeNull()
    expect(screen.getByTestId('task-image-expanded-preview').closest('[data-multitrack-preview-area]')).not.toBeNull()

    vi.spyOn(expandedImage, 'getBoundingClientRect').mockReturnValue({
      bottom: 250,
      height: 200,
      left: 100,
      right: 500,
      top: 50,
      width: 400,
      x: 100,
      y: 50,
      toJSON: () => ({}),
    })
    fireEvent.wheel(expandedPreview, { clientX: 200, clientY: 100, deltaY: -100 })
    expect(expandedPreview.getAttribute('data-zoom')).toBe('1.15')
    expect(expandedImage.style.transform).toBe('scale(1.15)')
    expect(expandedImage.style.transformOrigin).toBe('25% 25%')

    fireEvent.wheel(expandedPreview, { clientX: 200, clientY: 100, deltaY: 100 })
    expect(expandedPreview.getAttribute('data-zoom')).toBe('1.00')

    const backButton = screen.getByRole('button', { name: 'Back to preview' })
    expect(backButton.className).toContain('z-20')
    fireEvent.click(backButton)
    expect(screen.getByTestId('task-image-drop-zone')).not.toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: 'Preview image' })[0])
    rerender(<PreviewArea {...props} selectedSegment={{ ...selectedSegment, segment: { ...taskSegment, id: 'other-task' } }} />)
    expect(screen.queryByTestId('task-image-expanded-preview')).toBeNull()
  })

  it('updates preview aspect ratio when resolution child widgets change after render', () => {
    const { data, selectedSegment } = trackData()
    const widthWidget = { name: 'resolution.width', value: [1024] }
    const heightWidget = { name: 'resolution.height', value: [576] }
    const node = {
      widgets: [
        { name: 'resolution', value: ['width x height (custom)'] },
        { name: 'resolution.resize_method', value: ['crop'] },
        widthWidget,
        heightWidget,
      ],
    }

    render(
      <PreviewArea
        data={data}
        currentTime={1}
        selectedSegment={selectedSegment}
        isPlaying={false}
        node={node}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const previewFrame = screen.getByTestId('multitrack-video-preview').parentElement as HTMLElement
    expect(previewFrame.style.aspectRatio).toBe('1024 / 576')

    widthWidget.value = [576]
    heightWidget.value = [1024]

    act(() => {
      vi.advanceTimersByTime(300)
    })

    expect(previewFrame.style.aspectRatio).toBe('576 / 1024')
  })

  it('shows images from the task segment at the current time when no segment is selected', () => {
    const { data } = trackData()
    addActiveTaskTrack(data)

    const props = {
      data,
      selectedSegment: null,
      isPlaying: false,
      node: { widgets: [] },
      onGlobalSettingsChange: vi.fn(),
      onSelectedSegmentContentChange: vi.fn(),
      onSelectedSegmentDurationChange: vi.fn(),
    }
    const { rerender } = render(<PreviewArea {...props} currentTime={36} />)

    const imageArea = screen.getByTestId('task-preview-images')
    expect(imageArea.getAttribute('data-layout')).toBe('flow')
    expect(imageArea.className).toContain('flex-wrap')
    expect(imageArea.className).toContain('overflow-y-auto')
    expect(screen.queryByTestId('multitrack-video-preview')).toBeNull()
    expect(screen.getAllByRole('img').map((image) => image.getAttribute('src'))).toContain(
      '/view?filename=first.png&type=input&subfolder=tasks',
    )
    expect(screen.getAllByRole('img').map((image) => image.getAttribute('src'))).toContain('https://example.com/second.png')
    expect(screen.getByRole('img', { name: 'second.png' }).className).toContain('w-auto')
    expect(screen.getByTestId('task-preview-image-first').className).toContain('h-32')
    expect(screen.getByTestId('task-preview-image-second').className).toContain('h-32')
    expect(screen.getByTestId('panorama-image-preview-first').className).toContain('aspect-video')
    expect(screen.queryByLabelText('720° panorama preview')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'second.png' }))
    expect(screen.getByTestId('task-preview-image-second').className).toContain('border-primary')

    const addImage = screen.getByTestId('task-preview-add-image')
    expect(addImage.className).toContain('h-32')
    expect(addImage.className).toContain('w-32')
    expect(imageArea.lastElementChild).toBe(addImage)

    rerender(<PreviewArea {...props} currentTime={48} />)
    expect(screen.queryByTestId('task-preview-images')).toBeNull()
  })

  it('edits the active task user prompt inline when no segment is selected', () => {
    const { data } = trackData()
    data.tracks.unshift({
      id: 'task-track',
      name: 'Task 1',
      type: 'task',
      color: 'var(--primary)',
      muted: false,
      locked: false,
      segments: [{
        id: 'active-task',
        start_frame: 24,
        end_frame: 48,
        color: 'var(--primary)',
        content: {
          media_type: 'none',
          user_prompt: 'A long active task prompt that should stay on one preview line and truncate when needed',
        },
      }],
    })

    const onTrackSegmentsContentChange = vi.fn()
    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const overlay = screen.getByTestId('task-prompt-overlay')
    expect(overlay.className).toContain('bg-black/')
    expect(screen.getByTestId('task-prompt-text').className).toContain('truncate')
    expect(screen.getByTestId('task-prompt-text').textContent).toBe(
      'A long active task prompt that should stay on one preview line and truncate when needed',
    )

    fireEvent.doubleClick(screen.getByTestId('task-prompt-text'))
    const editor = screen.getByTestId('task-prompt-editor') as HTMLTextAreaElement
    expect(editor.value).toBe('A long active task prompt that should stay on one preview line and truncate when needed')
    fireEvent.change(editor, { target: { value: 'Updated preview prompt' } })

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'active-task',
      patch: { user_prompt: 'Updated preview prompt' },
    }])
    expect((screen.getByTestId('task-prompt-editor') as HTMLTextAreaElement).value).toBe('Updated preview prompt')
  })

  it('shows and updates the active task mode from the prompt bar', () => {
    const { data } = trackData()
    data.tracks.unshift({
      id: 'task-track',
      name: 'Task 1',
      type: 'task',
      color: 'var(--primary)',
      muted: false,
      locked: false,
      segments: [{
        id: 'active-task',
        start_frame: 24,
        end_frame: 48,
        color: 'var(--primary)',
        content: {
          media_type: 'none',
          task_mode: 'ref',
          user_prompt: 'Use this reference',
        },
      }],
    })

    const onTrackSegmentsContentChange = vi.fn()
    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const modeSelect = screen.getByTestId('task-mode-select') as HTMLSelectElement
    expect(modeSelect.value).toBe('ref')
    expect(modeSelect.selectedOptions[0].textContent).toBe('Reference')
    expect(screen.getByText('|').className).toContain('text-secondary')

    fireEvent.change(modeSelect, { target: { value: 'edit' } })

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'active-task',
      patch: { task_mode: 'edit' },
    }])
  })

  it('shows an editable active task prompt placeholder and toggles image/video layout', () => {
    const { data } = trackData()
    data.tracks[0].segments[0].end_frame = 48
    addActiveTaskTrack(data)
    data.tracks[0].segments[1].content.user_prompt = ''
    const onTrackSegmentsContentChange = vi.fn()

    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    expect(screen.getByTestId('task-prompt-text').textContent).toBe('Double-click to describe what you want to generate...')
    const imageArea = screen.getByTestId('task-preview-images')
    expect(imageArea.className).toContain('w-20')
    expect(imageArea.parentElement?.className).toContain('w-fit')
    expect(screen.getByTestId('multitrack-video-stage').parentElement?.className).toContain('h-full')

    fireEvent.click(screen.getByRole('button', { name: 'Enlarge task images' }))

    expect(screen.getByTestId('task-preview-images').className).toContain('w-32')
    expect(screen.getByTestId('multitrack-video-stage').parentElement?.className).toContain('h-full')
    expect(screen.getByRole('button', { name: 'Use balanced preview layout' })).not.toBeNull()

    fireEvent.doubleClick(screen.getByTestId('task-prompt-text'))
    fireEvent.change(screen.getByTestId('task-prompt-editor'), { target: { value: 'Prompt from preview' } })

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'active-task',
      patch: { user_prompt: 'Prompt from preview' },
    }])
  })

  it('does not show the active task prompt while another segment is selected', () => {
    const { data, selectedSegment } = trackData()
    data.tracks.unshift({
      id: 'task-track',
      name: 'Task 1',
      type: 'task',
      color: 'var(--primary)',
      muted: false,
      locked: false,
      segments: [{
        id: 'active-task',
        start_frame: 0,
        end_frame: 48,
        color: 'var(--primary)',
        content: { media_type: 'none', user_prompt: 'Hidden prompt' },
      }],
    })

    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={selectedSegment}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    expect(screen.queryByTestId('task-prompt-overlay')).toBeNull()
  })

  it('opens and exits the enlarged preview from an active task image', () => {
    const { data } = trackData()
    addActiveTaskTrack(data)
    const onTrackSegmentsContentChange = vi.fn()

    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const firstImage = screen.getByTestId('task-preview-image-first')
    const viewButton = firstImage.querySelector<HTMLButtonElement>('[aria-label="Preview image"]')
    expect(viewButton).not.toBeNull()
    fireEvent.click(viewButton!)
    const expandedPreview = screen.getByTestId('task-image-expanded-preview')
    expect(expandedPreview).not.toBeNull()
    expect(expandedPreview.querySelector('img[alt="first.png"]')).not.toBeNull()
    expect(screen.queryByLabelText('720° panorama preview')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Back to preview' }))
    expect(screen.getByTestId('task-preview-images')).not.toBeNull()
    expect(onTrackSegmentsContentChange).not.toHaveBeenCalled()
  })

  it('deletes an active task image from the global preview', () => {
    const { data } = trackData()
    addActiveTaskTrack(data)
    const onTrackSegmentsContentChange = vi.fn()

    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const firstImage = screen.getByTestId('task-preview-image-first')
    expect(firstImage.className).toContain('cursor-pointer')
    const deleteButton = screen.getByRole('button', { name: 'Delete image first.png' })
    expect(firstImage.contains(deleteButton)).toBe(true)
    expect(deleteButton.parentElement?.className).toContain('right-1')
    expect(deleteButton.className).toContain('h-5')
    expect(deleteButton.className).toContain('text-destructive')

    fireEvent.click(deleteButton)

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'active-task',
      patch: {
        images: [expect.objectContaining({ id: 'second' })],
      },
    }])
  })

  it('reorders active task images by dragging in the global preview', () => {
    const { data } = trackData()
    addActiveTaskTrack(data)
    const onTrackSegmentsContentChange = vi.fn()

    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    fireEvent.dragStart(screen.getByTestId('task-preview-image-first'))
    fireEvent.dragOver(screen.getByTestId('task-preview-image-second'))
    fireEvent.drop(screen.getByTestId('task-preview-image-second'))

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'active-task',
      patch: {
        images: [
          expect.objectContaining({ id: 'second' }),
          expect.objectContaining({ id: 'first' }),
        ],
      },
    }])
  })

  it('adds an image to the active task from the global preview media selector', async () => {
    vi.useRealTimers()
    const { data } = trackData()
    addActiveTaskTrack(data)
    const onTrackSegmentsContentChange = vi.fn()

    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByTestId('task-preview-add-image'))
    fireEvent.click(await screen.findByRole('button', { name: 'mock select image' }))

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'active-task',
      patch: {
        images: [
          expect.objectContaining({ id: 'first' }),
          expect.objectContaining({ id: 'second' }),
          expect.objectContaining({ source_type: 'input', file_path: 'picked.png', file_name: 'picked.png' }),
        ],
      },
    }])
  })

  it('opens the media selector from the empty image-only task preview', async () => {
    vi.useRealTimers()
    const { data } = trackData()
    data.tracks.unshift({
      id: 'task-track',
      name: 'Task 1',
      type: 'task',
      color: 'var(--primary)',
      muted: false,
      locked: false,
      segments: [{
        id: 'active-task',
        start_frame: 24,
        end_frame: 48,
        color: 'var(--primary)',
        content: { media_type: 'none', images: [] },
      }],
    })
    const onTrackSegmentsContentChange = vi.fn()

    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    expect(screen.getByTestId('task-preview-images').className).toContain('flex-wrap')
    fireEvent.click(screen.getByTestId('task-preview-empty-add-image'))
    fireEvent.click(await screen.findByRole('button', { name: 'mock select image' }))

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'active-task',
      patch: {
        images: [
          expect.objectContaining({ source_type: 'input', file_path: 'picked.png', file_name: 'picked.png' }),
        ],
      },
    }])
  })

  it('uploads a local image dropped on the active task preview add slot', async () => {
    vi.useRealTimers()
    const { data } = trackData()
    addActiveTaskTrack(data)
    const onTrackSegmentsContentChange = vi.fn()
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({ name: 'dropped.png', subfolder: 'tasks' }),
    } as Response)

    render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const file = new File(['image'], 'dropped.png', { type: 'image/png' })
    fireEvent.drop(screen.getByTestId('task-preview-add-image'), {
      dataTransfer: {
        files: [file],
        items: [{ kind: 'file', type: 'image/png' }],
      },
    })

    await waitFor(() => expect(onTrackSegmentsContentChange).toHaveBeenCalled())
    expect(fetchMock).toHaveBeenCalledWith('/upload/image', expect.objectContaining({ method: 'POST' }))
    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'active-task',
      patch: {
        images: [
          expect.objectContaining({ id: 'first' }),
          expect.objectContaining({ id: 'second' }),
          expect.objectContaining({ source_type: 'input', file_path: 'tasks/dropped.png', file_name: 'dropped.png' }),
        ],
      },
    }])
    fetchMock.mockRestore()
  })

  it('does not show task images while another segment is selected', () => {
    const { data, selectedSegment } = trackData()
    data.tracks.unshift({
      id: 'task-track',
      name: 'Task 1',
      type: 'task',
      color: 'var(--primary)',
      muted: false,
      locked: false,
      segments: [{
        id: 'active-task',
        start_frame: 0,
        end_frame: 48,
        color: 'var(--primary)',
        content: {
          media_type: 'none',
          images: [{ id: 'first', source_type: 'input', file_path: 'first.png' }],
        },
      }],
    })

    render(
      <PreviewArea
        data={data}
        currentTime={1}
        selectedSegment={selectedSegment}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    expect(screen.queryByTestId('task-preview-images')).toBeNull()
  })

  it('shows a constrained full-width waveform and media toolbar for a selected audio segment', () => {
    const { data } = trackData()
    const audioSegment = {
      id: 'selected-audio',
      start_frame: 0,
      end_frame: 48,
      color: 'var(--multitrack-audio-bg)',
      content: {
        media_type: 'audio' as const,
        source_type: 'input' as const,
        file_path: 'audio.wav',
        volume_db: 2,
      },
    }
    data.tracks.push({
      id: 'audio-track',
      name: 'Audio 1',
      type: 'audio',
      color: 'var(--multitrack-audio-bg)',
      muted: false,
      locked: false,
      segments: [audioSegment],
    })

    render(
      <PreviewArea
        data={data}
        currentTime={12}
        selectedSegment={{ trackId: 'audio-track', trackType: 'audio', segment: audioSegment }}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    expect(screen.queryByTestId('selected-audio-waveform')).toBeNull()
    expect(screen.getByRole('button', { name: 'Audio settings' })).not.toBeNull()
  })

  it('overlays active subtitles on the video preview frame', () => {
    const { data } = trackData()
    data.tracks[0].segments[0].end_frame = 96
    const subtitleSegment = {
      id: 'subtitle-active',
      start_frame: 24,
      end_frame: 48,
      color: '#9D4937',
      content: {
        media_type: 'subtitle' as const,
        text: 'Recognized line',
        subtitle_style: {
          font_size: 12,
          color: '#ffffff',
          outline_color: '#000000',
          background_color: 'rgba(0, 0, 0, 0.7)',
          background_opacity: 0.7,
          x: 0.15,
          y: 0.8,
          width: 0.7,
        },
      },
    }
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [subtitleSegment],
    })

    const { rerender } = render(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    expect(screen.getByTestId('subtitle-preview-overlay').textContent).toContain('Recognized line')
    expect(screen.getByTestId('subtitle-preview-overlay').closest('[data-testid="multitrack-video-stage"]')).not.toBeNull()
    expect(screen.getByTestId('subtitle-preview-overlay').style.top).toBe('80%')
    expect(screen.getByTestId('subtitle-preview-overlay').style.backgroundColor).toBe('')
    expect(screen.getByTestId('subtitle-preview-text').style.backgroundColor).toBe('rgba(0, 0, 0, 0.7)')

    rerender(
      <PreviewArea
        data={data}
        currentTime={36}
        selectedSegment={{ trackId: 'subtitle-track', trackType: 'subtitle', segment: subtitleSegment }}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    expect(screen.getByTestId('multitrack-video-preview')).not.toBeNull()
    expect(screen.getByTestId('multitrack-video-stage').parentElement?.className).not.toContain('transition')
    expect(screen.getByTestId('subtitle-settings-panel')).not.toBeNull()
    expect(screen.getByRole('tab', { name: /Text/ })).not.toBeNull()
  })

  it('renders active subtitles from multiple subtitle tracks at the same time', () => {
    const { data } = trackData()
    data.tracks[0].segments[0].end_frame = 96
    data.tracks.push(
      {
        id: 'subtitle-track-a',
        name: 'Subtitle 1',
        type: 'subtitle',
        color: '#9D4937',
        muted: false,
        locked: false,
        segments: [{
          id: 'subtitle-a',
          start_frame: 0,
          end_frame: 48,
          color: '#9D4937',
          content: { media_type: 'subtitle' as const, text: 'First subtitle' },
        }],
      },
      {
        id: 'subtitle-track-b',
        name: 'Subtitle 2',
        type: 'subtitle',
        color: '#9D4937',
        muted: false,
        locked: false,
        segments: [{
          id: 'subtitle-b',
          start_frame: 0,
          end_frame: 48,
          color: '#9D4937',
          content: { media_type: 'subtitle' as const, text: 'Second subtitle' },
        }],
      },
    )

    render(
      <PreviewArea
        data={data}
        currentTime={12}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const overlays = screen.getAllByTestId('subtitle-preview-overlay')
    expect(overlays).toHaveLength(2)
    expect(overlays.map((overlay) => overlay.textContent)).toEqual([
      expect.stringContaining('First subtitle'),
      expect.stringContaining('Second subtitle'),
    ])
    expect(overlays.every((overlay) => overlay.closest('[data-testid="multitrack-video-stage"]'))).toBe(true)
  })

  it('renders transparent subtitle backgrounds without filling the preview width', () => {
    const { data } = trackData()
    const subtitleSegment = {
      id: 'subtitle-transparent',
      start_frame: 0,
      end_frame: 48,
      color: '#9D4937',
      content: {
        media_type: 'subtitle' as const,
        text: 'Transparent line',
        subtitle_style: {
          font_size: 12,
          color: '#ffffff',
          outline_color: '#000000',
          background_color: 'transparent',
          background_opacity: 0.7,
          x: 0.15,
          y: 0.8,
          width: 0.7,
        },
      },
    }
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [subtitleSegment],
    })

    render(
      <PreviewArea
        data={data}
        currentTime={12}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    expect(screen.getByTestId('subtitle-preview-overlay').style.width).toBe('max-content')
    expect(screen.getByTestId('subtitle-preview-overlay').style.maxWidth).toBe('70%')
    expect(screen.getByTestId('subtitle-preview-overlay').style.backgroundColor).toBe('')
    expect(screen.getByTestId('subtitle-preview-text').className).toContain('inline-block')
    expect(screen.getByTestId('subtitle-preview-text').style.backgroundColor).toBe('transparent')
  })

  it('selects an active subtitle from the preview overlay', () => {
    const { data } = trackData()
    const onSelectSegment = vi.fn()
    const subtitleSegment = {
      id: 'subtitle-clickable',
      start_frame: 0,
      end_frame: 48,
      color: '#9D4937',
      content: {
        media_type: 'subtitle' as const,
        text: 'Clickable line',
      },
    }
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [subtitleSegment],
    })

    render(
      <PreviewArea
        data={data}
        currentTime={12}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onSelectSegment={onSelectSegment}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const overlay = screen.getByTestId('subtitle-preview-overlay')
    expect(overlay.className).toContain('cursor-pointer')
    expect(overlay.className).not.toContain('pointer-events-none')
    expect(overlay.style.left).toBe('50%')
    expect(overlay.style.maxWidth).toBe('75%')
    expect(overlay.style.transform).toBe('translateX(-50%)')

    fireEvent.mouseDown(overlay)

    expect(onSelectSegment).toHaveBeenCalledWith('subtitle-clickable')
  })

  it('edits an active subtitle inline by double clicking the preview overlay', () => {
    const { data } = trackData()
    const onSelectSegment = vi.fn()
    const onTrackSegmentsContentChange = vi.fn()
    const subtitleSegment = {
      id: 'subtitle-editable',
      start_frame: 0,
      end_frame: 48,
      color: '#9D4937',
      content: {
        media_type: 'subtitle' as const,
        text: 'Original line',
      },
    }
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [subtitleSegment],
    })

    render(
      <PreviewArea
        data={data}
        currentTime={12}
        selectedSegment={null}
        isPlaying={false}
        node={{ widgets: [] }}
        onSelectSegment={onSelectSegment}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const overlay = screen.getByTestId('subtitle-preview-overlay')
    fireEvent.doubleClick(overlay)
    const editor = screen.getByTestId('subtitle-preview-editor')
    expect(overlay.contains(editor)).toBe(true)
    expect(screen.queryByRole('dialog')).toBeNull()
    fireEvent.change(editor, { target: { value: 'Updated line' } })
    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSelectSegment).toHaveBeenCalledWith('subtitle-editable')
    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'subtitle-editable',
      patch: { text: 'Updated line' },
    }])
  })

  it('edits a selected subtitle inline when editing is requested externally', () => {
    const { data } = trackData()
    const onSelectedSegmentContentChange = vi.fn()
    const subtitleSegment = {
      id: 'subtitle-edit-request',
      start_frame: 0,
      end_frame: 48,
      color: '#9D4937',
      content: {
        media_type: 'subtitle' as const,
        text: 'Timeline line',
      },
    }
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [subtitleSegment],
    })

    render(
      <PreviewArea
        data={data}
        currentTime={12}
        selectedSegment={{ trackId: 'subtitle-track', trackType: 'subtitle', segment: subtitleSegment }}
        editingSubtitleSegmentId="subtitle-edit-request"
        isPlaying={false}
        node={{ widgets: [] }}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={onSelectedSegmentContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    const overlay = screen.getByTestId('subtitle-preview-overlay')
    const editor = screen.getByTestId('subtitle-preview-editor')
    expect(overlay.contains(editor)).toBe(true)
    expect((editor as HTMLInputElement).value).toBe('Timeline line')
    fireEvent.change(editor, { target: { value: 'Edited from timeline' } })
    fireEvent.keyDown(editor, { key: 'Enter' })

    expect(onSelectedSegmentContentChange).toHaveBeenCalledWith({ text: 'Edited from timeline' })
  })
})
