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
              { id: 'first', source_type: 'input', file_path: 'tasks/first.png', file_name: 'first.png' },
              { id: 'second', source_type: 'url', url: 'https://example.com/second.png', file_name: 'second.png' },
            ],
          },
        },
      ],
    })

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
    expect(imageArea.className).toContain('w-20')
    expect(screen.getByText('Task1')).toBeTruthy()
    expect(screen.getAllByRole('img').map((image) => image.getAttribute('src'))).toEqual([
      '/view?filename=first.png&type=input&subfolder=tasks',
      'https://example.com/second.png',
    ])
    expect(screen.getAllByRole('img').every((image) => image.className.includes('object-contain'))).toBe(true)

    rerender(<PreviewArea {...props} currentTime={48} />)
    expect(screen.queryByTestId('task-preview-images')).toBeNull()
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
