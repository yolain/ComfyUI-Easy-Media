import { describe, expect, it, vi } from 'vitest'
import { adjustMultiTrackEditorNodeHeight, preserveTimelineEditorNodeHeight } from '@/lib/timeline-node-size'
import { scaleImageItemsToDuration } from '@/lib/timeline-utils'
import type { ImageItem } from '@/types/timeline'

function installTimelineHeightHooks(nodeName = 'easy timelineEditor') {
  class NodeType {
    size = [480, 360]
    properties: Record<string, unknown> = {}
    setSize = vi.fn((size: [number, number]) => {
      this.size = size
      ;(this as unknown as { onResize?: (size: unknown) => void }).onResize?.(size)
    })
    setDirtyCanvas = vi.fn()
  }

  preserveTimelineEditorNodeHeight(NodeType, { name: nodeName })
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

  it('restores easyMediaTimelineHeight after refresh-time default resizing', () => {
    vi.useFakeTimers()
    const NodeType = installTimelineHeightHooks('easy multiTrackEditor')
    const node = new NodeType() as InstanceType<typeof NodeType> & {
      onConfigure?: (serialisedNode: unknown) => void
      onResize?: (size: unknown) => void
    }
    node.size = [800, 700]

    node.onConfigure?.({
      size: [800, 812],
      properties: { easyMediaTimelineHeight: 812 },
    })
    node.onResize?.([800, 700])
    vi.runAllTimers()

    expect(node.size).toEqual([800, 812])
    expect(node.properties.easyMediaTimelineHeight).toBe(812)
    vi.useRealTimers()
  })

  it('keeps an intentional multitrack height adjustment over a pending widget resize restore', () => {
    vi.useFakeTimers()
    const NodeType = installTimelineHeightHooks('easy multiTrackEditor')
    const node = new NodeType() as InstanceType<typeof NodeType> & {
      onWidgetChanged?: (name: string, value: unknown, oldValue: unknown, widget: unknown) => void
    }
    node.size = [800, 700]
    node.properties.easyMediaTimelineHeight = 700

    node.onWidgetChanged?.('tracks', 'next', 'previous', {})
    adjustMultiTrackEditorNodeHeight(node, 64)
    vi.runAllTimers()

    expect(node.size).toEqual([800, 764])
    expect(node.properties.easyMediaTimelineHeight).toBe(764)
    vi.useRealTimers()
  })

  it('does not rewrite timeline duration when the format changes to MiniMax', () => {
    vi.useFakeTimers()
    const NodeType = installTimelineHeightHooks()
    const timelineWidget = {
      name: 'timeline_data',
      value: JSON.stringify({ total_length: 121, frame_rate: 24, tracks: [] }),
    }
    const node = new NodeType() as InstanceType<typeof NodeType> & {
      widgets: Array<{ name: string, value: string }>
      onWidgetChanged?: (name: string, value: unknown, oldValue: unknown, widget: unknown) => void
    }
    node.widgets = [timelineWidget]

    node.onWidgetChanged?.('format', 'MiniMax', 'Wan', {})

    expect(JSON.parse(timelineWidget.value).total_length).toBe(121)
    vi.runAllTimers()
    vi.useRealTimers()
  })

  it('does not rewrite multitrack duration when the format changes to MiniMax', () => {
    vi.useFakeTimers()
    const NodeType = installTimelineHeightHooks('easy multiTrackEditor')
    const trackDataWidget = {
      name: 'track_data',
      value: JSON.stringify({ total_length: 120, frame_rate: 24, tracks: [] }),
    }
    const node = new NodeType() as InstanceType<typeof NodeType> & {
      widgets: Array<{ name: string, value: string }>
      onWidgetChanged?: (name: string, value: unknown, oldValue: unknown, widget: unknown) => void
    }
    node.widgets = [trackDataWidget]

    node.onWidgetChanged?.('format', 'MiniMax', 'Wan', {})

    expect(JSON.parse(trackDataWidget.value).total_length).toBe(120)
    vi.runAllTimers()
    vi.useRealTimers()
  })

  it('does not rewrite timeline duration when loading a MiniMax workflow', () => {
    vi.useFakeTimers()
    const NodeType = installTimelineHeightHooks()
    const timelineWidget = {
      name: 'timeline_data',
      value: JSON.stringify({ total_length: 121, frame_rate: 24, tracks: [] }),
    }
    const node = new NodeType() as InstanceType<typeof NodeType> & {
      widgets: Array<{ name: string, value: string }>
      onConfigure?: (serialisedNode: unknown) => void
    }
    node.widgets = [
      { name: 'format', value: 'MiniMax' },
      timelineWidget,
    ]

    node.onConfigure?.({ size: [520, 430], properties: {} })

    expect(JSON.parse(timelineWidget.value).total_length).toBe(121)
    vi.runAllTimers()
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
