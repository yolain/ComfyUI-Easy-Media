import { describe, expect, it } from 'vitest'
import { resolveBrowserAudioPreviewGain } from '@/lib/audio-utils'

describe('browser audio preview gain', () => {
  it('keeps positive gain for same-origin audio', () => {
    expect(resolveBrowserAudioPreviewGain(6, '/view?filename=audio.wav', 'https://comfy.test/app')).toBeCloseTo(2, 1)
  })

  it('disables positive gain for cross-origin audio', () => {
    expect(resolveBrowserAudioPreviewGain(6, 'https://media.test/audio.wav', 'https://comfy.test/app')).toBe(1)
  })

  it('keeps attenuation for cross-origin audio', () => {
    expect(resolveBrowserAudioPreviewGain(-6, 'https://media.test/audio.wav', 'https://comfy.test/app')).toBeCloseTo(0.5, 1)
  })
})
