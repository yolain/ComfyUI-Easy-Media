import { act } from '@testing-library/react'
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { installProjectSamplingPreview } from '@/lib/project-sampling-preview-node'

const previewState = vi.hoisted(() => ({
  props: null as null | {
    interactive?: boolean
    onVisibilityChange?: (visible: boolean) => void
  },
}))

vi.mock('@/components/widgets/SamplingPreviewWidget', () => ({
  SamplingPreviewWidget: (props: typeof previewState.props) => {
    previewState.props = props
    return <div data-testid="sampling-preview" />
  },
}))

interface TestWidget {
  computedHeight?: number
  hidden?: boolean
  options: {
    getMinHeight: () => number
    getMaxHeight: () => number
  }
}

describe('installProjectSamplingPreview', () => {
  const app = {} as ComfyApp

  beforeEach(() => {
    previewState.props = null
    vi.stubGlobal('requestAnimationFrame', vi.fn((callback: FrameRequestCallback) => {
      callback(0)
      return 1
    }))
  })

  function createNodeType() {
    const host = document.createElement('div')
    document.body.append(host)
    let widget: TestWidget | undefined
    const node = {
      id: 42,
      size: [640, 400] as [number, number],
      graph: { setDirtyCanvas: vi.fn() },
      onResize: vi.fn(),
      setSize: vi.fn(function setSize(this: { size: [number, number] }, size: [number, number]) {
        this.size = size
      }),
      addDOMWidget: vi.fn((
        _name: string,
        _type: string,
        container: HTMLDivElement,
        options: TestWidget['options'],
      ) => {
        host.append(container)
        widget = { options }
        return widget
      }),
    }
    class NodeType {}
    Object.assign(NodeType.prototype, node)
    return { NodeType, host, node, getWidget: () => widget }
  }

  it('only installs on the multitrack project node', () => {
    const { NodeType, node } = createNodeType()
    installProjectSamplingPreview(NodeType, { name: 'easy multitrackEditor' }, app)

    const instance = new NodeType() as typeof node & { onNodeCreated?: () => void }
    act(() => instance.onNodeCreated?.())

    expect(node.addDOMWidget).not.toHaveBeenCalled()
  })

  it('stays inside the node and changes from a tiny slot to a bounded preview', async () => {
    const { NodeType, host, node, getWidget } = createNodeType()
    installProjectSamplingPreview(NodeType, { name: 'easy multitrackProject' }, app)

    const instance = new NodeType() as typeof node & { onNodeCreated?: () => void }
    await act(async () => instance.onNodeCreated?.())

    const widget = getWidget()
    const container = host.firstElementChild as HTMLDivElement
    expect(widget?.hidden).toBe(false)
    expect(widget?.options.getMinHeight()).toBe(8)
    expect(widget?.options.getMaxHeight()).toBe(8)
    expect(container.hidden).toBe(true)
    expect(container.style.position).toBe('relative')
    expect(container.style.height).toBe('280px')
    expect(host.style.maxHeight).toBe('8px')
    expect(host.style.pointerEvents).toBe('none')
    expect(previewState.props?.interactive).toBeUndefined()

    act(() => previewState.props?.onVisibilityChange?.(true))

    expect(widget?.options.getMinHeight()).toBe(280)
    expect(widget?.options.getMaxHeight()).toBe(280)
    expect(container.hidden).toBe(false)
    expect(host.style.maxHeight).toBe('280px')
    expect(host.style.overflow).toBe('hidden')
    expect(host.style.pointerEvents).toBe('auto')
    expect(node.setSize).toHaveBeenCalledWith([640, 672])
    expect(node.onResize).toHaveBeenCalledWith([640, 672])
    expect(node.graph.setDirtyCanvas).toHaveBeenCalledWith(true, true)

    act(() => previewState.props?.onVisibilityChange?.(false))
    expect(node.setSize).toHaveBeenLastCalledWith([640, 400])
  })
})
