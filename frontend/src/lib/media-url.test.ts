import { describe, expect, it } from 'vitest'
import { audioContentToViewUrl, mediaPathToViewUrl } from './media-url'
import type { AudioContent } from '@/types/timeline'

describe('media view urls', () => {
  it('splits input subfolder paths into filename and subfolder query params', () => {
    expect(mediaPathToViewUrl('samples/drums/kick 01.wav', 'input')).toBe(
      '/view?filename=kick%2001.wav&type=input&subfolder=samples%2Fdrums',
    )
  })

  it('builds audio content urls with output source type preserved', () => {
    const content: AudioContent = {
      source_type: 'output',
      file_path: 'renders/take 1.wav',
      file_name: 'take 1.wav',
    }

    expect(audioContentToViewUrl(content)).toBe(
      '/view?filename=take%201.wav&type=output&subfolder=renders',
    )
  })
})
