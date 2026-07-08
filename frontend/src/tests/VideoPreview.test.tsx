import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { VideoPreview } from '@/components/widgets/multitrack/VideoPreview'
import type { ActivePreviewVideoSegment, MultiTrackPreviewResolution } from '@/lib/multitrack-utils'

const resolution: MultiTrackPreviewResolution = {
  width: 16,
  height: 9,
  resizeMethod: 'resize',
  mode: 'fixed',
}

function activeVideo(localTime: number): ActivePreviewVideoSegment {
  return {
    trackId: 'video-track',
    localTime,
    segment: {
      id: 'video-segment',
      start_frame: 2,
      end_frame: 5,
      color: 'var(--primary)',
      content: {
        media_type: 'video',
        source_type: 'input',
        file_path: 'clips/shot.mp4',
        file_name: 'shot.mp4',
        duration: 3,
      },
    },
  }
}

describe('VideoPreview', () => {
  beforeEach(() => {
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined)
  })

  it('seeks the video element to the active local time', () => {
    const { rerender } = render(
      <VideoPreview
        activeVideo={activeVideo(1.25)}
        resolution={resolution}
        isPlaying={false}
        muted
        volume={0.5}
      />,
    )

    const video = screen.getByTestId('multitrack-video-preview') as HTMLVideoElement
    expect(video.src).toContain('/view?filename=shot.mp4&type=input&subfolder=clips')
    expect(video.currentTime).toBeCloseTo(1.25)
    expect(video.muted).toBe(true)
    expect(video.volume).toBe(0.5)

    rerender(
      <VideoPreview
        activeVideo={activeVideo(2)}
        resolution={resolution}
        isPlaying={false}
        muted={false}
        volume={1}
      />,
    )

    expect(video.currentTime).toBeCloseTo(2)
    expect(video.muted).toBe(false)
    expect(video.volume).toBe(1)
  })

  it('shows a black frame when no active video is available', () => {
    render(
      <VideoPreview
        activeVideo={null}
        resolution={resolution}
        isPlaying={false}
        muted
        volume={1}
      />,
    )

    expect(screen.queryByTestId('multitrack-video-preview')).toBeNull()
    expect(screen.getByTestId('multitrack-black-frame')).not.toBeNull()
  })

  it('does not repeatedly seek while the preview video is already playing', () => {
    const seekTimes: number[] = []
    const currentTimes = new WeakMap<HTMLMediaElement, number>()
    Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', {
      configurable: true,
      get() {
        return currentTimes.get(this) ?? 0
      },
      set(value: number) {
        currentTimes.set(this, value)
        seekTimes.push(value)
      },
    })

    const { rerender } = render(
      <VideoPreview
        activeVideo={activeVideo(0)}
        resolution={resolution}
        isPlaying
        playbackNonce={0}
        muted={false}
        volume={1}
      />,
    )

    seekTimes.length = 0

    rerender(
      <VideoPreview
        activeVideo={activeVideo(1 / 24)}
        resolution={resolution}
        isPlaying
        playbackNonce={0}
        muted={false}
        volume={1}
      />,
    )

    expect(seekTimes).toEqual([])
  })

  it('seeks while already playing when a new playback session starts', () => {
    const seekTimes: number[] = []
    const currentTimes = new WeakMap<HTMLMediaElement, number>()
    Object.defineProperty(HTMLMediaElement.prototype, 'currentTime', {
      configurable: true,
      get() {
        return currentTimes.get(this) ?? 0
      },
      set(value: number) {
        currentTimes.set(this, value)
        seekTimes.push(value)
      },
    })

    const { rerender } = render(
      <VideoPreview
        activeVideo={activeVideo(3)}
        resolution={resolution}
        isPlaying
        playbackNonce={0}
        muted={false}
        volume={1}
      />,
    )

    seekTimes.length = 0

    rerender(
      <VideoPreview
        activeVideo={activeVideo(0)}
        resolution={resolution}
        isPlaying
        playbackNonce={1}
        muted={false}
        volume={1}
      />,
    )

    expect(seekTimes).toEqual([0])
  })
})
