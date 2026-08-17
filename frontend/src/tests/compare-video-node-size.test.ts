import { describe, expect, it, vi } from 'vitest'
import { preserveCompareVideoNodeSize } from '@/lib/compare-video-node-size'

describe('preserveCompareVideoNodeSize', () => {
  it('sets a stable initial width and height for compare-video nodes', () => {
    const originalOnNodeCreated = vi.fn()
    class NodeType {
      size = [420, 360]
      setDirtyCanvas = vi.fn()
      onNodeCreated() {}
    }
    NodeType.prototype.onNodeCreated = originalOnNodeCreated

    preserveCompareVideoNodeSize(NodeType, { name: 'easy compareVideos' })
    const node = new NodeType()

    node.onNodeCreated()

    expect(originalOnNodeCreated).toHaveBeenCalledOnce()
    expect(node.size).toEqual([720, 520])
    expect(node.setDirtyCanvas).toHaveBeenCalledWith(true, true)
  })

  it('does not resize other node types', () => {
    class NodeType {
      size = [420, 360]
      setDirtyCanvas = vi.fn()
      onNodeCreated = vi.fn()
    }

    preserveCompareVideoNodeSize(NodeType, { name: 'easy saveVideo' })
    const node = new NodeType()

    node.onNodeCreated()

    expect(node.size).toEqual([420, 360])
  })
})
