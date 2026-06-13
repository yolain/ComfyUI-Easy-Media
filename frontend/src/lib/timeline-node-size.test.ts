import { describe, expect, it, vi } from 'vitest'
import { preserveTimelineEditorNodeHeight } from './timeline-node-size'
import { scaleImageItemsToDuration } from './timeline-utils'
import type { ImageItem } from '@/types/timeline'

function installTimelineHeightHooks() {
  class NodeType {
    size = [480, 360]
    properties: Record<string, unknown> = {}
    setSize = vi.fn((size: [number, number]) => {
      this.size = size
      ;(this as unknown as { onResize?: (size: unknown) => void }).onResize?.(size)
    })
    setDirtyCanvas = vi.fn()
  }

  preserveTimelineEditorNodeHeight(NodeType, { name: 'easy timelineEditor' })
  return NodeType
}

describe('preserveTimelineEditorNodeHeight', () => {
  it('stores explicit node resize heights', () => {
    const NodeType = installTimelineHeightHooks()
    const node = new NodeType() as InstanceType<typeof NodeType> & {
      onResize?: (size: unknown) => void
    }

    node.onResize?.([480, 420])

    expect(node.properties.easyMediaTimelineHeight).toBe(420)
  })

  it('does not replace stored height when widget option changes trigger an automatic resize', () => {
    vi.useFakeTimers()
    const NodeType = installTimelineHeightHooks()
    const node = new NodeType() as InstanceType<typeof NodeType> & {
      onResize?: (size: unknown) => void
      onWidgetChanged?: (name: string, value: unknown, oldValue: unknown, widget: unknown) => void
    }
    node.properties.easyMediaTimelineHeight = 420
    node.size = [480, 420]

    node.onWidgetChanged?.('some_option', 'next', 'previous', {})
    node.onResize?.([480, 260])

    expect(node.properties.easyMediaTimelineHeight).toBe(420)

    vi.runAllTimers()

    expect(node.size).toEqual([480, 420])
    expect(node.properties.easyMediaTimelineHeight).toBe(420)
    vi.useRealTimers()
  })
})

describe('scaleImageItemsToDuration', () => {
  it('keeps child image ranges proportional when a segment is shortened', () => {
    const images: ImageItem[] = [
      { source_type: 'input', file_path: 'a.png', file_name: 'a.png', start_frame: 0, end_frame: 126 },
      { source_type: 'input', file_path: 'b.png', file_name: 'b.png', start_frame: 127, end_frame: 252 },
    ]

    expect(scaleImageItemsToDuration(images, 253, 121)).toEqual([
      expect.objectContaining({ file_name: 'a.png', start_frame: 0, end_frame: 60 }),
      expect.objectContaining({ file_name: 'b.png', start_frame: 61, end_frame: 120 }),
    ])
  })
})
