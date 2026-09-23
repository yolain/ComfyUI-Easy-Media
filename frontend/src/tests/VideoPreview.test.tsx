import { act, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
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
    vi.stubGlobal('fetch', vi.fn(() => new Promise(() => {})))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    Reflect.deleteProperty(URL, 'createObjectURL')
    Reflect.deleteProperty(URL, 'revokeObjectURL')
  })

  it('requests the output-sampled frame when paused and hides the native video', async () => {
    const fetchMock = vi.fn((url: string, _options?: RequestInit) => Promise.resolve(url.endsWith('/source-fps')
      ? { ok: true, json: async () => ({ fps: 30 }) }
      : { ok: true, blob: async () => new Blob(['frame'], { type: 'image/jpeg' }) }))
    vi.stubGlobal('fetch', fetchMock)
    Object.defineProperty(URL, 'createObjectURL', {
      configurable: true, value: vi.fn(() => 'blob:timeline-frame'),
    })
    Object.defineProperty(URL, 'revokeObjectURL', {
      configurable: true, value: vi.fn(),
    })

    render(
      <VideoPreview
        frameRate={24}
        activeVideo={activeVideo(204 / 24)}
        resolution={resolution}
        isPlaying={false}
        muted
        volume={1}
      />,
    )

    expect((screen.getByTestId('multitrack-video-preview') as HTMLVideoElement).style.visibility).toBe('hidden')
    await waitFor(() => expect(screen.getByTestId('multitrack-timeline-frame').getAttribute('src'))
      .toBe('blob:timeline-frame'))
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      '/easy-media/video/source-fps', '/easy-media/video/preview-frame',
    ])
    expect(JSON.parse(String(fetchMock.mock.calls[1][1]?.body))).toMatchObject({ frame: 204, fps: 24 })
  })

  it('uses the native seek when source and timeline frame rates match', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ fps: 24 }),
    })
    vi.stubGlobal('fetch', fetchMock)
    render(
      <VideoPreview
        frameRate={24}
        activeVideo={activeVideo(8.5)}
        resolution={resolution}
        isPlaying={false}
        muted
        volume={1}
      />,
    )

    const video = screen.getByTestId('multitrack-video-preview') as HTMLVideoElement
    await act(async () => { await Promise.resolve() })
    expect(video.style.visibility).toBe('hidden')
    fireEvent.seeked(video)
    expect(video.style.visibility).toBe('visible')
    expect(screen.queryByTestId('multitrack-timeline-frame')).toBeNull()
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock.mock.calls[0][0]).toBe('/easy-media/video/source-fps')
  })

  it('uses the native video for URL sources without probing or requesting a frame', () => {
    const urlVideo = activeVideo(8.5)
    urlVideo.segment.content = {
      media_type: 'video',
      source_type: 'url',
      url: 'https://example.com/shot.mp4',
    }
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    render(
      <VideoPreview
        frameRate={24}
        activeVideo={urlVideo}
        resolution={resolution}
        isPlaying={false}
        muted
        volume={1}
      />,
    )

    const video = screen.getByTestId('multitrack-video-preview') as HTMLVideoElement
    expect(video.src).toBe('https://example.com/shot.mp4')
    expect(video.currentTime).toBeCloseTo(8.5)
    expect(video.style.visibility).toBe('visible')
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('seeks the video element to the active local time', () => {
    const { rerender } = render(
      <VideoPreview
        frameRate={24}
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
        frameRate={24}
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
        frameRate={24}
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

  it('shows a labeled placeholder for a connected video slot', () => {
    const slotVideo = activeVideo(0)
    slotVideo.segment.content = {
      media_type: 'video',
      source_type: 'slot',
      slot_name: 'video2',
      file_name: 'video2',
    }

    render(
      <VideoPreview
        frameRate={24}
        activeVideo={slotVideo}
        resolution={resolution}
        isPlaying={false}
        muted
        volume={1}
      />,
    )

    expect(screen.getByTestId('multitrack-video-slot-placeholder').textContent).toBe('Video 2')
    expect(screen.queryByTestId('multitrack-black-frame')).toBeNull()
  })

  it('keeps the video out of layout flow so intrinsic dimensions cannot expand the stage', () => {
    render(
      <VideoPreview
        frameRate={24}
        activeVideo={activeVideo(0)}
        resolution={resolution}
        isPlaying={false}
        muted
        volume={1}
      />,
    )

    const video = screen.getByTestId('multitrack-video-preview')
    expect(video.className).toContain('absolute')
    expect(video.className).toContain('inset-0')
    expect(video.className).toContain('min-h-0')
    expect(video.className).toContain('min-w-0')
    expect(video.className).toContain('max-h-full')
    expect(video.className).toContain('max-w-full')
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
        frameRate={24}
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
        frameRate={24}
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
        frameRate={24}
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
        frameRate={24}
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
