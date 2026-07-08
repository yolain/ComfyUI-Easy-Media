import { beforeEach, describe, expect, it, vi } from 'vitest'
import {
  clearMediaListCache,
  getMediaList,
  invalidateMediaListCache,
} from '@/stores/media-list-store'

const request = {
  source: 'inputs' as const,
  mediaType: 'video' as const,
  localPath: '',
  subfolder: '',
}

describe('media list store', () => {
  beforeEach(() => {
    clearMediaListCache()
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ items: [{ type: 'file', name: 'clip.mp4', path: 'clip.mp4' }] }),
    }))
  })

  it('shares cached media lists across callers', async () => {
    const first = await getMediaList(request)
    const second = await getMediaList(request)

    expect(first).toEqual(second)
    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('deduplicates concurrent requests for the same list', async () => {
    await Promise.all([getMediaList(request), getMediaList(request)])

    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('refetches invalidated inputs without dropping output caches', async () => {
    const outputRequest = { ...request, source: 'outputs' as const }
    await getMediaList(request)
    await getMediaList(outputRequest)

    invalidateMediaListCache('inputs')
    await getMediaList(request)
    await getMediaList(outputRequest)

    expect(fetch).toHaveBeenCalledTimes(3)
  })

  it('does not cache failed requests', async () => {
    vi.mocked(fetch)
      .mockResolvedValueOnce({ ok: false, status: 500 } as Response)
      .mockResolvedValueOnce({ ok: true, json: async () => ({ items: [] }) } as Response)

    await expect(getMediaList(request)).rejects.toThrow('500')
    await expect(getMediaList(request)).resolves.toEqual([])
    expect(fetch).toHaveBeenCalledTimes(2)
  })
})
