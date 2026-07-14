import { describe, expect, it, vi } from 'vitest'
import { uploadInputMediaFile } from '@/lib/media-upload'

describe('uploadInputMediaFile', () => {
  it('sends the requested input subfolder to ComfyUI', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ name: 'voice.wav', subfolder: 'input/ref' }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await uploadInputMediaFile(new File(['audio'], 'voice.wav', { type: 'audio/wav' }), 'input/ref')

    const body = fetchMock.mock.calls[0]?.[1]?.body as FormData
    expect(body.get('subfolder')).toBe('input/ref')
  })
})
