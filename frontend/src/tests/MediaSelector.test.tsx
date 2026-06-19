import { render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'

vi.mock('@/lib/i18n', () => ({
  useT: () => (key: string) => key,
}))

vi.mock('@/lib/comfy-api', () => ({
  $error: vi.fn(),
}))

describe('MediaSelector', () => {
  beforeEach(() => {
    vi.stubGlobal('IntersectionObserver', class {
      private readonly callback: IntersectionObserverCallback

      constructor(callback: IntersectionObserverCallback) {
        this.callback = callback
      }

      observe(target: Element) {
        this.callback([{ isIntersecting: true, target } as IntersectionObserverEntry], this as unknown as IntersectionObserver)
      }

      disconnect() {}
      unobserve() {}
      takeRecords() { return [] }
      root = null
      rootMargin = '0px'
      thresholds = [0]
    })
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [{
          type: 'file',
          name: 'clip.mp4',
          path: 'clips/clip.mp4',
          url: '/view?filename=clip.mp4&type=input&subfolder=clips',
          size: 1024,
          mtime: 1,
        }],
      }),
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('renders a video thumbnail for video files', async () => {
    render(<MediaSelector value="" mediaType="video" onChange={vi.fn()} />)

    const fileName = await screen.findByTitle('clip.mp4')
    const fileButton = fileName.closest('button')

    await waitFor(() => {
      const video = fileButton?.querySelector('video')
      expect(video).not.toBeNull()
      expect(video?.getAttribute('src')).toContain('clip.mp4')
    })
  })
})
