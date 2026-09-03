import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import { DEFAULT_LAZY_INTERSECTION_DELAY_MS } from '@/hooks/use-delayed-intersection'
import { clearMediaListCache } from '@/stores/media-list-store'
import { uploadInputMediaFile } from '@/lib/media-upload'

vi.mock('@/lib/media-upload', () => ({
  uploadInputMediaFile: vi.fn().mockResolvedValue('ref/uploaded.wav'),
}))

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
    const fileButton = fileName.closest('[role="button"]')

    await waitFor(() => {
      const video = fileButton?.querySelector('video')
      expect(video).not.toBeNull()
      expect(video?.getAttribute('src')).toContain('clip.mp4')
    })
    expect(screen.getByText('1.0 KB')).toBeTruthy()
  })

  it('offers connected video slots and labels them as video', () => {
    const onChange = vi.fn()
    render(
      <MediaSelector
        value=""
        mediaType="video"
        slotItems={[{ label: 'video1', value: '__slot__:video1' }]}
        onChange={onChange}
      />,
    )

    fireEvent.mouseDown(screen.getByRole('tab', { name: 'mediaSelector.tabSlot' }))
    fireEvent.click(screen.getByText('mediaSelector.slotVideo'))

    expect(onChange).toHaveBeenCalledWith('__slot__:video1')
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'mediaSelector.tabInputs' }))
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
    const video = fileName.closest('[role="button"]')?.querySelector('video')
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

  it('refreshes the current directory with a short cooldown', async () => {
    render(<MediaSelector value="" mediaType="video" onChange={vi.fn()} />)
    await screen.findByTitle('clip.mp4')

    const selector = document.querySelector('[data-media-selector]')
    expect(selector?.className).toContain('w-80')

    vi.useFakeTimers()
    const refreshButton = screen.getByRole('button', { name: 'mediaSelector.refresh' })
    fireEvent.click(refreshButton)
    await act(async () => Promise.resolve())

    expect(fetch).toHaveBeenCalledTimes(2)
    expect((refreshButton as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(refreshButton)
    expect(fetch).toHaveBeenCalledTimes(2)

    act(() => vi.advanceTimersByTime(2_000))
    expect((refreshButton as HTMLButtonElement).disabled).toBe(false)
    fireEvent.click(refreshButton)
    await act(async () => Promise.resolve())
    expect(fetch).toHaveBeenCalledTimes(3)
  })

  it('uses one inverse-icon button to toggle grid and list views', async () => {
    render(<MediaSelector value="" mediaType="video" onChange={vi.fn()} />)
    await screen.findByTitle('clip.mp4')

    const viewButton = screen.getByTitle('mediaSelector.viewList')
    expect(viewButton.querySelector('.lucide-layout-list')).not.toBeNull()
    expect(screen.queryByTitle('mediaSelector.viewGrid')).toBeNull()

    fireEvent.click(viewButton)

    const gridButton = screen.getByTitle('mediaSelector.viewGrid')
    expect(gridButton.querySelector('.lucide-layout-grid')).not.toBeNull()
    expect(screen.queryByTitle('mediaSelector.viewList')).toBeNull()
  })

  it('keeps image selection single by default', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [{
          type: 'file',
          name: 'a.png',
          path: 'a.png',
          url: '/view?filename=a.png&type=input&subfolder=',
          size: 10,
          mtime: 1,
        }],
      }),
    } as Response)
    const onChange = vi.fn()

    render(<MediaSelector value="" mediaType="image" onChange={onChange} />)

    fireEvent.click(await screen.findByTitle('a.png'))
    expect(screen.queryByText('mediaSelector.filter')).toBeNull()
    expect(onChange).toHaveBeenCalledWith('a.png', 'input')
  })

  it('selects image files in batches without exceeding the supplied limit', async () => {
    vi.mocked(fetch).mockResolvedValue({
      ok: true,
      json: async () => ({
        items: [
          { type: 'dir', name: 'folder', path: 'folder' },
          ...['a.png', 'b.png', 'c.png'].map((name, index) => ({
            type: 'file',
            name,
            path: name,
            url: `/view?filename=${name}&type=input&subfolder=`,
            size: 10,
            mtime: index,
          })),
        ],
      }),
    } as Response)
    const onChange = vi.fn()

    render(
      <MediaSelector
        value=""
        mediaType="image"
        allowMultipleSelection
        maxSelectionCount={2}
        onChange={onChange}
      />,
    )
    await screen.findByTitle('a.png')
    expect(screen.queryAllByRole('checkbox')).toHaveLength(0)

    fireEvent.click(screen.getByText('mediaSelector.filter'))

    expect(screen.queryByTitle('mediaSelector.viewList')).toBeNull()
    expect(screen.queryByText('mediaSelector.filter')).toBeNull()
    expect(screen.getAllByRole('checkbox')).toHaveLength(3)
    fireEvent.click(screen.getByText('mediaSelector.selectAll'))
    expect(screen.getAllByRole('checkbox', { checked: true })).toHaveLength(2)

    fireEvent.click(screen.getByText('mediaSelector.confirm'))
    expect(onChange).toHaveBeenCalledWith('a.png|MULTIPLE|b.png', 'input')
  })

  it('navigates each level of a nested breadcrumb', async () => {
    vi.mocked(fetch).mockImplementation(async (input) => {
      const subfolder = new URL(String(input), 'http://localhost').searchParams.get('subfolder') ?? ''
      const items = subfolder === 'parent/child'
        ? [{ type: 'file', name: 'inside.mp4', path: 'parent/child/inside.mp4', size: 10, mtime: 1 }]
        : subfolder === 'parent'
          ? [{ type: 'dir', name: 'child', path: 'parent/child' }]
          : [{ type: 'dir', name: 'parent', path: 'parent' }]
      return { ok: true, json: async () => ({ items }) } as Response
    })

    render(<MediaSelector value="" mediaType="all" onChange={vi.fn()} />)
    fireEvent.click(await screen.findByTitle('parent'))
    fireEvent.click(await screen.findByTitle('child'))
    await screen.findByTitle('inside.mp4')

    fireEvent.click(screen.getByRole('button', { name: 'parent' }))
    await waitFor(() => {
      expect(screen.queryByTitle('inside.mp4')).toBeNull()
    })
    expect(screen.getByTitle('child')).not.toBeNull()
  })

  it('splits Windows-style subfolders into separate breadcrumb levels', async () => {
    vi.mocked(fetch).mockImplementation(async (input) => {
      const subfolder = new URL(String(input), 'http://localhost').searchParams.get('subfolder') ?? ''
      const items = subfolder === 'parent\\child'
        ? [{ type: 'file', name: 'inside.mp4', path: 'parent\\child\\inside.mp4', size: 10, mtime: 1 }]
        : subfolder === 'parent'
          ? [{ type: 'dir', name: 'child', path: 'parent\\child' }]
          : [{ type: 'dir', name: 'parent', path: 'parent' }]
      return { ok: true, json: async () => ({ items }) } as Response
    })

    render(<MediaSelector value="clip.mp4" mediaType="all" onChange={vi.fn()} />)
    fireEvent.click(await screen.findByTitle('parent'))
    fireEvent.click(await screen.findByTitle('child'))
    await screen.findByTitle('inside.mp4')

    expect(screen.getByRole('button', { name: 'parent' })).not.toBeNull()
    expect(screen.getByRole('button', { name: 'child' })).not.toBeNull()
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

  it('reopens a selector with a selected value at its source tab and last subdirectory', async () => {
    vi.mocked(fetch).mockImplementation(async (input) => {
      const url = new URL(String(input), 'http://localhost')
      const source = url.searchParams.get('source')
      const subfolder = url.searchParams.get('subfolder')
      return {
        ok: true,
        json: async () => ({
          items: source === 'outputs' && subfolder === 'ref'
            ? [{ type: 'file', name: 'chosen.png', path: 'ref/chosen.png', size: 10, mtime: 1 }]
            : [{ type: 'dir', name: 'ref', path: 'ref' }],
        }),
      } as Response
    })

    const first = render(<MediaSelector value="" mediaType="image" onChange={vi.fn()} />)
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'mediaSelector.tabOutputs' }))
    await screen.findByTitle('ref')
    fireEvent.click(screen.getByTitle('ref'))
    await screen.findByTitle('chosen.png')
    fireEvent.click(screen.getByTitle('chosen.png'))
    first.unmount()

    render(<MediaSelector value="ref/chosen.png" mediaType="image" defaultTab="outputs" onChange={vi.fn()} />)

    expect(screen.getByRole('tab', { name: 'mediaSelector.tabOutputs' }).getAttribute('data-state')).toBe('active')
    await screen.findByTitle('chosen.png')
    expect(fetch).toHaveBeenLastCalledWith('/easy-media/media/list?source=outputs&type=image&subfolder=ref')
  })

  it('uploads local files into the current input subdirectory', async () => {
    vi.mocked(fetch).mockImplementation(async (input) => {
      const url = new URL(String(input), 'http://localhost')
      const subfolder = url.searchParams.get('subfolder')
      return {
        ok: true,
        json: async () => ({
          items: subfolder === 'ref'
            ? [{ type: 'file', name: 'existing.wav', path: 'ref/existing.wav', size: 10, mtime: 1 }]
            : [{ type: 'dir', name: 'ref', path: 'ref' }],
        }),
      } as Response
    })
    const originalCreateElement = document.createElement.bind(document)
    vi.spyOn(document, 'createElement').mockImplementation(((tagName: string, options?: ElementCreationOptions) => {
      const element = originalCreateElement(tagName, options)
      if (tagName.toLowerCase() === 'input') {
        const file = new File(['audio'], 'voice.wav', { type: 'audio/wav' })
        Object.defineProperty(element, 'files', { configurable: true, value: [file] })
        element.click = () => { void (element as HTMLInputElement).onchange?.(new Event('change')) }
      }
      return element
    }) as typeof document.createElement)

    render(<MediaSelector value="" mediaType="audio" onChange={vi.fn()} />)
    await screen.findByTitle('ref')
    fireEvent.click(screen.getByTitle('ref'))
    await screen.findByTitle('existing.wav')
    fireEvent.click(screen.getByText('mediaSelector.addLocal'))

    await waitFor(() => expect(uploadInputMediaFile).toHaveBeenCalledWith(expect.any(File), 'ref'))
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

  it('shows recent generated output media in the history tab', async () => {
    const onChange = vi.fn()
    vi.mocked(fetch).mockImplementation(async (input) => {
      const url = new URL(String(input), 'http://localhost')
      if (url.pathname === '/easy-media/media/recent') {
        return {
          ok: true,
          json: async () => ({
            items: [{
              type: 'file',
              name: 'recent.mp4',
              path: 'renders/recent.mp4',
              url: '/view?filename=recent.mp4&type=output&subfolder=renders',
              size: 10,
              mtime: 2,
            }],
          }),
        } as Response
      }
      return { ok: true, json: async () => ({ items: [] }) } as Response
    })

    render(<MediaSelector value="" mediaType="video" onChange={onChange} />)
    fireEvent.mouseDown(screen.getByRole('tab', { name: 'mediaSelector.tabHistory' }))

    fireEvent.click(await screen.findByTitle('recent.mp4'))
    expect(onChange).toHaveBeenCalledWith('renders/recent.mp4', 'output')
    expect(fetch).toHaveBeenCalledWith('/easy-media/media/recent?source=outputs&type=video&hours=48&limit=50')
  })
})
