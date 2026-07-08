import { render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { PreviewAudioPlayback } from '@/components/widgets/multitrack/PreviewAudioPlayback'
import type { ActivePreviewAudioSource } from '@/lib/multitrack-utils'

function audioSource(localTime: number): ActivePreviewAudioSource {
  return {
    trackId: 'audio-track',
    localTime,
    volumeDb: 0,
    segment: {
      id: 'audio-segment',
      start_frame: 0,
      end_frame: 120,
      color: 'var(--multitrack-audio-bg)',
      content: {
        media_type: 'audio',
        source_type: 'input',
        file_path: 'audio/voice.wav',
        file_name: 'voice.wav',
      },
    },
  }
}

describe('PreviewAudioPlayback', () => {
  beforeEach(() => {
    vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => undefined)
  })

  it('seeks all active audio while already playing when a new playback session starts', () => {
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
      <PreviewAudioPlayback
        sources={[audioSource(3)]}
        isPlaying
        playbackNonce={0}
      />,
    )

    expect(screen.getByTestId('preview-audio-audio-segment')).not.toBeNull()
    seekTimes.length = 0

    rerender(
      <PreviewAudioPlayback
        sources={[audioSource(0)]}
        isPlaying
        playbackNonce={1}
      />,
    )

    expect(seekTimes).toEqual([0])
  })
})
