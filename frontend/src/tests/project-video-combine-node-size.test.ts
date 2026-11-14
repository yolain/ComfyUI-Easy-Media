import { describe, expect, it, vi } from 'vitest'
import { preserveVideoCombineNodeSize } from '@/lib/project-video-combine-node-size'

describe('project video combine node size', () => {
  it('starts at editor size and restores saved height after widget layout changes', () => {
    vi.useFakeTimers()
    try {
      class Node {
        size = [450, 100]
        properties: Record<string, number> = {}
        declare onNodeCreated?: () => void
        declare onConfigure?: (data: unknown) => void
        declare onWidgetChanged?: (name: string, value: unknown, oldValue: unknown, widget: unknown) => void
        declare onResize?: (size: number[]) => void
        setDirtyCanvas = vi.fn()
        setSize(size: number[]) { this.size = size }
      }
      preserveVideoCombineNodeSize(Node, { name: 'easy multitrackProjectVideoCombine' })
      const node = new Node()
      node.onNodeCreated?.()
      expect(node.size).toEqual([800, 700])
      node.onConfigure?.({ size: [800, 820] })
      vi.runAllTimers()
      expect(node.size).toEqual([800, 820])
      node.onWidgetChanged?.('project_data', 'next', 'previous', {})
      node.size = [800, 100]
      node.onResize?.(node.size)
      vi.runAllTimers()
      expect(node.size).toEqual([800, 820])
    } finally {
      vi.useRealTimers()
    }
  })
})
