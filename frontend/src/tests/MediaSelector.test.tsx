import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import { DEFAULT_LAZY_INTERSECTION_DELAY_MS } from '@/hooks/use-delayed-intersection'
import { clearMediaListCache } from '@/stores/media-list-store'

vi.mock('@/lib/i18n', () => ({
  useT: () => (key: string) => key,
}))

vi.mock('@/lib/comfy-api', () => ({
  $error: vi.fn(),
}))

describe('MediaSelector', () => {
  beforeEach(() => {
    clearMediaListCache()
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
    cleanup()
    vi.useRealTimers()
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
    expect(screen.getByText('1.0 KB')).toBeTruthy()
  })

  it('does not load thumbnails that only briefly enter the viewport', async () => {
    const callbacks: IntersectionObserverCallback[] = []
    vi.stubGlobal('IntersectionObserver', class {
      constructor(callback: IntersectionObserverCallback) {
        callbacks.push(callback)
      }

      observe() {}
      disconnect() {}
      unobserve() {}
      takeRecords() { return [] }
      root = null
      rootMargin = '0px'
      thresholds = [0]
    })

    render(<MediaSelector value="" mediaType="video" onChange={vi.fn()} />)

    const fileName = await screen.findByTitle('clip.mp4')
    const video = fileName.closest('button')?.querySelector('video')
    if (!video) throw new Error('Expected a video thumbnail')
    expect(video?.getAttribute('src')).toBeNull()

    vi.useFakeTimers()
    const callback = callbacks[0]
    const entry = (isIntersecting: boolean): IntersectionObserverEntry => ({
      isIntersecting,
      target: video,
    } as unknown as IntersectionObserverEntry)

    act(() => {
      callback([entry(true)], {} as IntersectionObserver)
      vi.advanceTimersByTime(DEFAULT_LAZY_INTERSECTION_DELAY_MS - 1)
    })
    expect(video?.getAttribute('src')).toBeNull()

    act(() => {
      callback([entry(false)], {} as IntersectionObserver)
      vi.advanceTimersByTime(DEFAULT_LAZY_INTERSECTION_DELAY_MS)
    })
    expect(video?.getAttribute('src')).toBeNull()

    act(() => {
      callback([entry(true)], {} as IntersectionObserver)
      vi.advanceTimersByTime(DEFAULT_LAZY_INTERSECTION_DELAY_MS)
    })
    expect(video?.getAttribute('src')).toContain('clip.mp4')
  })

  it('reuses the media list after the selector is reopened', async () => {
    const { unmount } = render(<MediaSelector value="" mediaType="video" onChange={vi.fn()} />)
    await screen.findByTitle('clip.mp4')
    unmount()

    render(<MediaSelector value="" mediaType="video" onChange={vi.fn()} />)
    await screen.findByTitle('clip.mp4')

    expect(fetch).toHaveBeenCalledTimes(1)
  })

  it('clears the search query when entering a subdirectory', async () => {
    vi.mocked(fetch).mockImplementation(async (input) => {
      const url = new URL(String(input), 'http://localhost')
      const subfolder = url.searchParams.get('subfolder')

      return {
        ok: true,
        json: async () => ({
          items: subfolder === 'clips'
            ? [{
              type: 'file',
              name: 'inside.mp4',
              path: 'clips/inside.mp4',
              url: '/view?filename=inside.mp4&type=input&subfolder=clips',
              size: 2048,
              mtime: 2,
            }]
            : [{
              type: 'dir',
              name: 'clips',
              path: 'clips',
            }],
        }),
      } as Response
    })

    render(<MediaSelector value="" mediaType="video" onChange={vi.fn()} />)

    await screen.findByTitle('clips')

    const searchInput = screen.getByPlaceholderText('mediaSelector.searchPlaceholder') as HTMLInputElement
    fireEvent.change(searchInput, { target: { value: 'cli' } })
    expect(searchInput.value).toBe('cli')

    fireEvent.click(screen.getByTitle('clips'))

    await screen.findByTitle('inside.mp4')
    expect(searchInput.value).toBe('')
    expect(fetch).toHaveBeenLastCalledWith('/easy-media/media/list?source=inputs&type=video&subfolder=clips')
  })

  it('keeps folders after files until folders-first sorting is selected', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [
          {
            type: 'dir',
            name: 'folder',
            path: 'folder',
          },
          {
            type: 'file',
            name: 'alpha.mp4',
            path: 'alpha.mp4',
            url: '/view?filename=alpha.mp4&type=input&subfolder=',
            size: 1024,
            mtime: 1,
          },
          {
            type: 'file',
            name: 'zulu.mp4',
            path: 'zulu.mp4',
            url: '/view?filename=zulu.mp4&type=input&subfolder=',
            size: 2048,
            mtime: 2,
          },
        ],
      }),
    } as Response)

    render(<MediaSelector value="zulu.mp4" mediaType="video" onChange={vi.fn()} />)

    await screen.findByTitle('zulu.mp4')
    const selectedFile = screen.getByTitle('zulu.mp4')
    const unselectedFile = screen.getByTitle('alpha.mp4')
    const folder = screen.getByTitle('folder')

    expect(selectedFile.compareDocumentPosition(unselectedFile) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    expect(unselectedFile.compareDocumentPosition(folder) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()

    const sortButton = screen.getByTitle('mediaSelector.sort')
    fireEvent.click(sortButton)
    fireEvent.click(sortButton)

    await waitFor(() => {
      expect(selectedFile.compareDocumentPosition(folder) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
      expect(folder.compareDocumentPosition(unselectedFile) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy()
    })
  })
})
