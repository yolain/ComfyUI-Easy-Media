import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { MultiImagesLoaderWidget } from '@/components/widgets/MultiImagesLoaderWidget'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import type { ImageData } from '@/components/widgets/MultiImagesLoaderWidget'

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: ({ onChange }: { onChange: (value: string, source: 'input') => void }) => (
    <div><button type="button" onClick={() => onChange('new.png', 'input')}>Choose image</button></div>
  ),
}))

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

function mockImageUploads(failingName?: string) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (_input, init) => {
    const file = (init?.body as FormData).get('image') as File
    return {
      ok: file.name !== failingName,
      json: async () => ({ name: file.name }),
    } as Response
  })
}

function renderWidget(count: number, onChange = vi.fn()) {
  const value: ImageData = {
    images: Array.from({ length: count }, (_, index) => ({
      id: String(index),
      source_type: 'input',
      file_path: `image-${index}.png`,
    })),
  }
  const props = {
    value,
    onChange,
    app: { canvas: { ds: { scale: 1 } }, ui: { settings: { settingsValues: { 'Comfy.VueNodes.Enabled': true } } } },
    node: {},
    widget: {},
    inputName: 'image_data',
  } as unknown as ReactWidgetProps<ImageData>
  const view = render(<MultiImagesLoaderWidget {...props} />)
  return { ...view, onChange }
}

describe('MultiImagesLoaderWidget', () => {
  it('shows the multitrack image picker across the empty area', () => {
    const { container } = renderWidget(0)
    expect(container.querySelector('.task-image-picker-empty')).not.toBeNull()
    const uploadIcon = container.querySelector('.task-image-picker-empty svg')
    expect(uploadIcon?.getAttribute('class')).toContain('size-12')
    expect(uploadIcon?.closest('button')).toBeNull()
    expect(screen.getByText('Select image')).toBeTruthy()
    expect(screen.getByText('Click to choose, or drag files here.')).toBeTruthy()
    expect(container.querySelector('.task-image-grid-add')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Clear all' })).toBeNull()
  })

  it('opens the media selector from the empty area', () => {
    const onChange = vi.fn()
    renderWidget(0, onChange)
    fireEvent.click(screen.getByRole('button', { name: 'Select image' }))
    fireEvent.click(screen.getByRole('button', { name: 'Choose image' }))
    expect(onChange).toHaveBeenCalledWith({ images: [expect.objectContaining({
      source_type: 'input',
      file_path: 'new.png',
    })] })
  })

  it.each([[1, 2], [4, 3], [9, 4], [16, 5], [25, 5]])(
    'uses a grid with %i images and %i columns including the add tile', (count, columns) => {
      const { container } = renderWidget(count)
      const grid = container.querySelector('[style*="grid-template-columns"]')
      expect(grid?.getAttribute('style')).toContain(`repeat(${columns}, minmax(0, 1fr))`)
      expect(container.querySelector('.nodeNew')).not.toBeNull()
      expect(container.querySelectorAll('.task-image-grid-add')).toHaveLength(count === 25 ? 0 : 1)
      expect(screen.queryByText(`${count} / 25`)).toBeNull()
      expect(screen.getByText(`Images: ${count}`)).toBeTruthy()
      expect(screen.getByRole('button', { name: 'Clear all' })).toBeTruthy()
    },
  )

  it('keeps the chosen order when deleting an image', () => {
    const { onChange } = renderWidget(3)
    fireEvent.click(screen.getAllByRole('button', { name: 'Delete image' })[1])
    expect(onChange).toHaveBeenCalledWith({ images: [
      expect.objectContaining({ id: '0' }),
      expect.objectContaining({ id: '2' }),
    ] })
  })

  it('uploads files dropped on an existing image tile', async () => {
    const upload = mockImageUploads()
    const { container, onChange } = renderWidget(1)
    const tile = container.querySelector('.task-image-grid .group') as HTMLElement

    fireEvent.drop(tile, {
      dataTransfer: { files: [new File(['image'], 'dropped.png', { type: 'image/png' })] },
    })

    await waitFor(() => expect(onChange).toHaveBeenCalledWith({ images: [
      expect.objectContaining({ file_path: 'image-0.png' }),
      expect.objectContaining({ file_path: 'dropped.png' }),
    ] }))
    expect(upload).toHaveBeenCalledTimes(1)
  })

  it('still reorders images dragged between tiles', () => {
    const { container, onChange } = renderWidget(2)
    const tiles = container.querySelectorAll('.task-image-grid .group')

    fireEvent.dragStart(tiles[0])
    fireEvent.drop(tiles[1], { dataTransfer: { files: [] } })
    fireEvent.dragEnd(tiles[0])

    expect(onChange).toHaveBeenCalledWith({ images: [
      expect.objectContaining({ id: '1' }),
      expect.objectContaining({ id: '0' }),
    ] })
  })

  it('keeps successful images when another file in the drop fails to upload', async () => {
    mockImageUploads('broken.png')
    const logError = vi.spyOn(console, 'error').mockImplementation(() => {})
    const { container, onChange } = renderWidget(0)
    const files = [
      new File(['first'], 'first.png', { type: 'image/png' }),
      new File(['broken'], 'broken.png', { type: 'image/png' }),
      new File(['last'], 'last.png', { type: 'image/png' }),
    ]

    fireEvent.drop(container.querySelector('.task-image-picker-empty') as HTMLElement, {
      dataTransfer: { files },
    })

    await waitFor(() => expect(onChange).toHaveBeenCalledWith({ images: [
      expect.objectContaining({ file_path: 'first.png' }),
      expect.objectContaining({ file_path: 'last.png' }),
    ] }))
    expect(logError).toHaveBeenCalledTimes(1)
  })

  it('clears all images from the fixed footer', () => {
    const { onChange } = renderWidget(3)
    fireEvent.click(screen.getByRole('button', { name: 'Clear all' }))
    expect(onChange).toHaveBeenCalledWith({ images: [] })
  })

  it('previews an image inside the widget with zoom and a return control', () => {
    renderWidget(1)
    fireEvent.click(screen.getByRole('button', { name: 'Preview image' }))

    const preview = screen.getByTestId('multi-images-expanded-preview')
    expect(screen.queryByRole('dialog')).toBeNull()
    expect(preview.querySelector('img')?.getAttribute('src')).toContain('image-0.png')
    expect(preview.getAttribute('data-zoom')).toBe('1.00')

    fireEvent.wheel(preview, { deltaY: -100, clientX: 10, clientY: 10 })
    expect(preview.getAttribute('data-zoom')).toBe('1.15')

    fireEvent.click(screen.getByRole('button', { name: 'Back to preview' }))
    expect(screen.queryByTestId('multi-images-expanded-preview')).toBeNull()
  })
})
