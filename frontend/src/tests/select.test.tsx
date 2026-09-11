import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'

const originalScrollIntoView = HTMLElement.prototype.scrollIntoView

describe('Select', () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
      configurable: true,
      writable: true,
      value: vi.fn(),
    })
  })

  afterEach(() => {
    if (originalScrollIntoView) {
      HTMLElement.prototype.scrollIntoView = originalScrollIntoView
    } else {
      Reflect.deleteProperty(HTMLElement.prototype, 'scrollIntoView')
    }
  })

  it('closes through the Node 2.0-safe backdrop when clicking outside', async () => {
    render(
      <Select defaultValue="first">
        <SelectTrigger aria-label="Mode">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="first">First</SelectItem>
          <SelectItem value="second">Second</SelectItem>
        </SelectContent>
      </Select>,
    )

    expect(document.querySelector('[data-easy-media-select-backdrop]')).toBeNull()
    fireEvent.click(screen.getByRole('combobox', { name: 'Mode' }))

    expect(await screen.findByRole('listbox')).not.toBeNull()
    const backdrop = document.querySelector<HTMLElement>('[data-easy-media-select-backdrop]')
    expect(backdrop).not.toBeNull()
    fireEvent.pointerDown(backdrop!, { pointerType: 'mouse' })

    await waitFor(() => expect(screen.queryByRole('listbox')).toBeNull())
    expect(document.querySelector('[data-easy-media-select-backdrop]')).toBeNull()
  })
})
