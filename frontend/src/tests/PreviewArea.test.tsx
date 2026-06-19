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
    volume: 1,
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
})
