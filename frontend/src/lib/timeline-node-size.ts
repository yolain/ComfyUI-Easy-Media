const TIMELINE_NODE_NAME = 'easy timelineEditor'
const TIMELINE_HEIGHT_PROPERTY = 'easyMediaTimelineHeight'

type NodeSize = [number, number]

function readSize(size: unknown): NodeSize | null {
  if (Array.isArray(size) && size.length >= 2) {
    const width = Number(size[0])
    const height = Number(size[1])
    return Number.isFinite(width) && Number.isFinite(height) ? [width, height] : null
  }
  if (ArrayBuffer.isView(size) && 'length' in size && Number(size.length) >= 2) {
    const arrayLikeSize = size as ArrayLike<number>
    const width = Number(arrayLikeSize[0])
    const height = Number(arrayLikeSize[1])
    return Number.isFinite(width) && Number.isFinite(height) ? [width, height] : null
  }
  if (!size || typeof size !== 'object') return null
  const sizeObject = size as Record<string, unknown>
  const width = Number(sizeObject.width ?? sizeObject[0])
  const height = Number(sizeObject.height ?? sizeObject[1])
  return Number.isFinite(width) && Number.isFinite(height) ? [width, height] : null
}

function readStoredHeight(node: any, serialisedNode?: any): number | null {
  const propertyHeight = Number(
    node?.properties?.[TIMELINE_HEIGHT_PROPERTY]
      ?? serialisedNode?.properties?.[TIMELINE_HEIGHT_PROPERTY],
  )
  if (Number.isFinite(propertyHeight) && propertyHeight > 0) return propertyHeight

  return readSize(serialisedNode?.size)?.[1] ?? null
}

function preserveHeight(node: any, height: unknown) {
  const numericHeight = Number(height)
  if (!Number.isFinite(numericHeight) || numericHeight <= 0) return
  node.properties ??= {}
  node.properties[TIMELINE_HEIGHT_PROPERTY] = numericHeight
}

function applyHeight(node: any, height: number, fallbackWidth?: number) {
  const currentSize = readSize(node.size)
  if (!currentSize || Math.abs(currentSize[1] - height) < 1) return

  const width = currentSize[0] || fallbackWidth || 480
  if (typeof node.setSize === 'function') {
    node.setSize([width, height])
  } else {
    node.size = [width, height]
  }
  node.setDirtyCanvas?.(true, true)
}

function restoreHeight(node: any, height: number | null, fallbackWidth?: number) {
  if (!Number.isFinite(height) || height === null || height <= 0) return
  window.setTimeout(() => applyHeight(node, height, fallbackWidth), 100)
}

export function preserveTimelineEditorNodeHeight(nodeType: any, nodeData: { name?: string }) {
  if (nodeData.name !== TIMELINE_NODE_NAME) return

  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  const originalOnConfigure = nodeType.prototype.onConfigure
  const originalOnResize = nodeType.prototype.onResize
  const originalOnSerialize = nodeType.prototype.onSerialize

  nodeType.prototype.onNodeCreated = function () {
    originalOnNodeCreated?.call(this)
    restoreHeight(this, readStoredHeight(this))
  }

  nodeType.prototype.onConfigure = function (serialisedNode: any) {
    originalOnConfigure?.call(this, serialisedNode)
    const savedHeight = readStoredHeight(this, serialisedNode)
    if (savedHeight !== null) preserveHeight(this, savedHeight)
    restoreHeight(this, savedHeight, readSize(serialisedNode?.size)?.[0])
  }

  nodeType.prototype.onResize = function (size: unknown) {
    originalOnResize?.call(this, size)
    const nextSize = readSize(size)
    if (nextSize) preserveHeight(this, nextSize[1])
  }

  nodeType.prototype.onSerialize = function (serialisedNode: any) {
    originalOnSerialize?.call(this, serialisedNode)
    const currentSize = readSize(this.size)
    if (currentSize) preserveHeight(this, currentSize[1])
    serialisedNode.properties ??= {}
    serialisedNode.properties[TIMELINE_HEIGHT_PROPERTY] = this.properties?.[TIMELINE_HEIGHT_PROPERTY]
  }
}
