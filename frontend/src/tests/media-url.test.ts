import { describe, expect, it } from 'vitest'
import { addMediaRevision, audioContentToViewUrl, mediaContentToViewUrl, mediaPathToViewUrl } from '@/lib/media-url'
import type { AudioContent } from '@/types/timeline'

describe('media view urls', () => {
  it('adds a cache revision without replacing existing query parameters', () => {
    expect(addMediaRevision('/view?filename=clip.mp4&type=output', '123456789')).toBe(
      '/view?filename=clip.mp4&type=output&v=123456789',
    )
  })

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

  it('builds generic media urls for input video content', () => {
    expect(mediaContentToViewUrl({
      source_type: 'input',
      file_path: 'clips/shot 01.mp4',
    })).toBe('/view?filename=shot%2001.mp4&type=input&subfolder=clips')
  })

  it('encodes unicode video filename and subfolder as URL components', () => {
    expect(mediaPathToViewUrl(
      '中文 子目录/jimeng-2026-02-13-9387-基于经典“子弹时间”段落进行镜头语言级复刻，镜头角度、景别、节奏与原片保持一致，....mp4',
      'input',
    )).toContain(
      '/view?filename=jimeng-2026-02-13-9387-%E5%9F%BA%E4%BA%8E%E7%BB%8F%E5%85%B8%E2%80%9C%E5%AD%90%E5%BC%B9%E6%97%B6%E9%97%B4%E2%80%9D',
    )
  })
})
