import { createRoot, type Root } from 'react-dom/client'
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import { SamplingPreviewWidget } from '@/components/widgets/SamplingPreviewWidget'
import { CUSTOM_NODE_CLASS } from '@/lib/constants'

const NODE_NAME = 'easy multitrackProject'
const WIDGET_NAME = 'easy-media-sampling-preview'
const COLLAPSED_HEIGHT = 8
const PREVIEW_HEIGHT = 280
const PREVIEW_HEIGHT_DELTA = PREVIEW_HEIGHT - COLLAPSED_HEIGHT

interface PreviewWidget {
  computedHeight?: number
  hidden?: boolean
}

interface PreviewNode {
  id: string | number
  size?: [number, number]
  graph?: { setDirtyCanvas?: (foreground: boolean, background: boolean) => void }
  addDOMWidget: (
    name: string,
    type: string,
    container: HTMLDivElement,
    options: Record<string, unknown>,
  ) => PreviewWidget
  setSize?: (size: [number, number]) => void
  onResize?: (size: [number, number] | undefined) => void
  onNodeCreated?: (...args: unknown[]) => void
  onExecutionStart?: (...args: unknown[]) => void
  onRemoved?: (...args: unknown[]) => void
  easyMediaSamplingPreviewCleanup?: () => void
  easyMediaSamplingPreviewReset?: () => void
}

interface PreviewNodeType {
  prototype: object
}

function installWidget(node: PreviewNode, app: ComfyApp) {
  let visible = false
  let previewHeightApplied = false
  let executionRevision = 0
  let root: Root | null = null
  let hostObserver: MutationObserver | null = null
  let hostStyleObserver: MutationObserver | null = null
  let trackedHost: HTMLElement | null = null
  const container = document.createElement('div')
  container.classList.add('comfyui-react-widget', CUSTOM_NODE_CLASS)
  container.hidden = true
  Object.assign(container.style, {
    width: '100%',
    height: `${PREVIEW_HEIGHT}px`,
    maxHeight: `${PREVIEW_HEIGHT}px`,
    overflow: 'hidden',
    position: 'relative',
    pointerEvents: 'none',
  })

  const widget = node.addDOMWidget(WIDGET_NAME, WIDGET_NAME, container, {
    getMinHeight: () => visible ? PREVIEW_HEIGHT : COLLAPSED_HEIGHT,
    getMaxHeight: () => visible ? PREVIEW_HEIGHT : COLLAPSED_HEIGHT,
    hideOnZoom: false,
    margin: 0,
    serialize: false,
  })
  widget.hidden = false

  const syncHostBounds = () => {
    const host = container.parentElement
    if (!host) return
    if (host !== trackedHost) {
      hostStyleObserver?.disconnect()
      trackedHost = host
      hostStyleObserver = new MutationObserver(syncHostBounds)
      hostStyleObserver.observe(host, { attributes: true, attributeFilter: ['style'] })
      hostObserver?.disconnect()
      hostObserver = null
    }
    const height = visible ? PREVIEW_HEIGHT : COLLAPSED_HEIGHT
    const pointerEvents = visible ? 'auto' : 'none'
    if (host.style.maxHeight !== `${height}px`) host.style.maxHeight = `${height}px`
    if (host.style.overflow !== 'hidden') host.style.overflow = 'hidden'
    if (host.style.pointerEvents !== pointerEvents) host.style.pointerEvents = pointerEvents
  }
  hostObserver = new MutationObserver(syncHostBounds)
  if (document.body) hostObserver.observe(document.body, { childList: true, subtree: true })
  globalThis.requestAnimationFrame?.(syncHostBounds)

  const setVisible = (nextVisible: boolean) => {
    if (visible === nextVisible) return
    visible = nextVisible
    container.hidden = !visible
    container.style.pointerEvents = visible ? 'auto' : 'none'
    delete widget.computedHeight
    if (node.size && node.setSize) {
      if (visible && !previewHeightApplied) {
        node.setSize([node.size[0], node.size[1] + PREVIEW_HEIGHT_DELTA])
        previewHeightApplied = true
      } else if (!visible && previewHeightApplied) {
        node.setSize([node.size[0], Math.max(COLLAPSED_HEIGHT, node.size[1] - PREVIEW_HEIGHT_DELTA)])
        previewHeightApplied = false
      }
    }
    node.onResize?.(node.size)
    node.graph?.setDirtyCanvas?.(true, true)
    globalThis.requestAnimationFrame?.(syncHostBounds)
  }

  const render = () => root?.render(
    <SamplingPreviewWidget
      app={app}
      nodeId={() => node.id}
      executionRevision={executionRevision}
      onVisibilityChange={setVisible}
    />,
  )

  root = createRoot(container)
  render()

  return {
    reset: () => {
      executionRevision += 1
      render()
    },
    cleanup: () => {
      hostObserver?.disconnect()
      hostStyleObserver?.disconnect()
      root?.unmount()
      container.remove()
      root = null
    },
  }
}

export function installProjectSamplingPreview(
  nodeType: PreviewNodeType,
  nodeData: { name?: string },
  app: ComfyApp,
) {
  if (nodeData.name !== NODE_NAME) return
  const prototype = nodeType.prototype as PreviewNode
  const originalCreated = prototype.onNodeCreated
  const originalExecutionStart = prototype.onExecutionStart
  const originalRemoved = prototype.onRemoved

  prototype.onNodeCreated = function onProjectNodeCreated(this: PreviewNode, ...args: unknown[]) {
    originalCreated?.apply(this, args)
    const preview = installWidget(this, app)
    this.easyMediaSamplingPreviewCleanup = preview.cleanup
    this.easyMediaSamplingPreviewReset = preview.reset
  }
  prototype.onExecutionStart = function onProjectExecutionStart(this: PreviewNode, ...args: unknown[]) {
    this.easyMediaSamplingPreviewReset?.()
    originalExecutionStart?.apply(this, args)
  }
  prototype.onRemoved = function onProjectNodeRemoved(this: PreviewNode, ...args: unknown[]) {
    this.easyMediaSamplingPreviewCleanup?.()
    delete this.easyMediaSamplingPreviewCleanup
    delete this.easyMediaSamplingPreviewReset
    originalRemoved?.apply(this, args)
  }
}
