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
})
