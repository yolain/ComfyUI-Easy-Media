import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'

describe('ContextMenu', () => {
  it('closes through the Node 2.0-safe backdrop when clicking outside', async () => {
    render(
      <ContextMenu>
        <ContextMenuTrigger asChild>
          <div>Timeline segment</div>
        </ContextMenuTrigger>
        <ContextMenuContent>
          <ContextMenuItem>Delete segment</ContextMenuItem>
        </ContextMenuContent>
      </ContextMenu>,
    )

    fireEvent.contextMenu(screen.getByText('Timeline segment'), {
      clientX: 100,
      clientY: 80,
    })

    expect(await screen.findByRole('menu')).not.toBeNull()
    const backdrop = document.querySelector<HTMLElement>('[data-easy-media-context-menu-backdrop]')
    expect(backdrop).not.toBeNull()
    fireEvent.pointerDown(backdrop!, { pointerType: 'mouse' })

    await waitFor(() => expect(screen.queryByRole('menu')).toBeNull())
    expect(document.querySelector('[data-easy-media-context-menu-backdrop]')).toBeNull()
  })
})
