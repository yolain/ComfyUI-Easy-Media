import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PreviewArea } from '@/components/widgets/multitrack/PreviewArea'
import type { SelectedMultiTrackSegment } from '@/lib/multitrack-utils'
import type { TrackData } from '@/types/multitrack'

vi.mock('@/lib/video-utils', () => ({
  loadBrowserVideoMetadata: vi.fn(() => new Promise(() => {})),
}))

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: () => null,
}))

vi.mock('@/components/widgets/timeline/AudioWaveform', () => ({
  AudioWaveform: () => <canvas data-testid="audio-waveform" />,
}))

vi.mock('@/components/widgets/panorama/PanoramaViewerOverlay', () => ({
  PanoramaViewerOverlay: ({ onPanoramaViewChange, onExit }: {
    onPanoramaViewChange: (view: unknown) => void
    onExit: () => void
  }) => (
    <div data-testid="mock-panorama-overlay">
      <button
        type="button"
        onClick={() => onPanoramaViewChange({
          version: 1,
          projection: 'equirectangular',
          yaw: 30,
          pitch: -10,
          hfov: 75,
          aspect_ratio: 1.6,
        })}
      >
        mock apply panorama
      </button>
      <button type="button" onClick={() => onPanoramaViewChange(undefined)}>mock restore panorama</button>
      <button type="button" onClick={onExit}>mock exit panorama</button>
    </div>
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
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined)
  })

  afterEach(() => {
    vi.useRealTimers()
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

  it('opens panorama mode and updates only the selected task image metadata', () => {
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
    const onSelectedSegmentContentChange = vi.fn()
    const props = {
      data,
      currentTime: 0,
      selectedSegment,
      isPlaying: false,
      node: { widgets: [] },
      onGlobalSettingsChange: vi.fn(),
      onSelectedSegmentContentChange,
      onSelectedSegmentDurationChange: vi.fn(),
    }
    const { rerender } = render(<PreviewArea {...props} />)

    expect(screen.getByTestId('task-image-drop-zone')).not.toBeNull()
    fireEvent.click(screen.getAllByRole('button', { name: '720° panorama preview' })[0])
    expect(screen.getByTestId('mock-panorama-overlay')).not.toBeNull()
    expect(screen.queryByTestId('task-image-drop-zone')).toBeNull()
    expect(screen.getByTestId('mock-panorama-overlay').closest('[data-multitrack-preview-area]')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'mock apply panorama' }))
    expect(onSelectedSegmentContentChange).toHaveBeenLastCalledWith({
      images: [
        expect.objectContaining({ id: 'a', panorama_view: expect.objectContaining({ yaw: 30 }) }),
        expect.objectContaining({ id: 'b' }),
      ],
    })

    fireEvent.click(screen.getByRole('button', { name: 'mock restore panorama' }))
    const restoredImages = onSelectedSegmentContentChange.mock.lastCall?.[0].images
    expect(restoredImages[0]).toEqual(expect.objectContaining({ id: 'a' }))
    expect(restoredImages[0]).not.toHaveProperty('panorama_view')

    fireEvent.click(screen.getByRole('button', { name: 'mock exit panorama' }))
    expect(screen.getByTestId('task-image-drop-zone')).not.toBeNull()

    fireEvent.click(screen.getAllByRole('button', { name: '720° panorama preview' })[0])
    rerender(<PreviewArea {...props} selectedSegment={{ ...selectedSegment, segment: { ...taskSegment, id: 'other-task' } }} />)
    expect(screen.queryByTestId('mock-panorama-overlay')).toBeNull()
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
    expect(imageArea.className).toContain('flex-col')
    expect(imageArea.className).toContain('w-full')
    expect(screen.queryByTestId('multitrack-video-preview')).toBeNull()
    expect(screen.getAllByRole('img').map((image) => image.getAttribute('src'))).toContain(
      '/view?filename=first.png&type=input&subfolder=tasks',
    )
    expect(screen.getAllByRole('img').map((image) => image.getAttribute('src'))).toContain('https://example.com/second.png')
    expect(screen.getByRole('img', { name: 'second.png' }).className).toContain('object-contain')
    expect(screen.getAllByTestId('panorama-image-preview-first')[0].className).toContain('aspect-video')
    expect(screen.getAllByLabelText('720° panorama preview')[0].className).toContain('text-highlight')

    fireEvent.click(screen.getByRole('button', { name: 'second.png' }))
    expect(screen.getAllByRole('img', { name: 'second.png' })).toHaveLength(2)

    rerender(<PreviewArea {...props} currentTime={48} />)
    expect(screen.queryByTestId('task-preview-images')).toBeNull()
  })

  it('shows and collapses the active task user prompt when no segment is selected', () => {
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
          text: 'Fallback prompt',
        },
      }],
    })

    render(
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

    const overlay = screen.getByTestId('task-prompt-overlay')
    expect(overlay.className).toContain('bg-black/')
    expect(screen.getByTestId('task-prompt-text').className).toContain('truncate')
    expect(screen.getByTestId('task-prompt-text').textContent).toBe(
      'A long active task prompt that should stay on one preview line and truncate when needed',
    )

    fireEvent.click(screen.getByRole('button', { name: 'Hide task prompt' }))

    expect(screen.getByTestId('task-prompt-overlay').className).toContain('w-8')
    expect(screen.queryByTestId('task-prompt-text')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Show task prompt' }))

    expect(screen.getByTestId('task-prompt-text').textContent).toContain('A long active task prompt')
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

  it('opens and updates panorama mode from an unselected active task image', () => {
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

    fireEvent.click(screen.getAllByLabelText('720° panorama preview')[0])
    expect(screen.getByTestId('mock-panorama-overlay')).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'mock apply panorama' }))

    expect(onTrackSegmentsContentChange).toHaveBeenCalledWith([{
      segmentId: 'active-task',
      patch: {
        images: [
          expect.objectContaining({ id: 'first', panorama_view: expect.objectContaining({ yaw: 30 }) }),
          expect.objectContaining({ id: 'second' }),
        ],
      },
    }])
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
    expect(firstImage.querySelector('[aria-label="Delete image first.png"]')).toBeNull()
    const deleteButton = screen.getByRole('button', { name: 'Delete image first.png' })
    expect(deleteButton.parentElement?.className).toContain('right-2')
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

    const waveform = screen.getByTestId('selected-audio-waveform')
    expect(waveform.className).toContain('h-20')
    expect(waveform.className).toContain('w-full')
    expect(screen.getByRole('button', { name: 'Audio settings' })).not.toBeNull()
    expect(screen.queryByTestId('multitrack-video-preview')).toBeNull()
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
    expect(screen.getByRole('button', { name: 'Subtitle text settings' })).not.toBeNull()
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
