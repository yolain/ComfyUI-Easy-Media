const NODE_NAME = 'easy multiImagesLoader'

export function preserveMultiImagesLoaderNodeSize(nodeType: any, nodeData: { name?: string }) {
  if (nodeData.name !== NODE_NAME) return
  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function () {
    originalOnNodeCreated?.call(this)
    const size: [number, number] = [600, 700]
    if (typeof this.setSize === 'function') this.setSize(size)
    else this.size = size
    this.setDirtyCanvas?.(true, true)
  }
}
