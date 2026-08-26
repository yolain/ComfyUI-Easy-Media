const NODE_NAME = 'easy multitrackProjectVideoCombine'
const DEFAULT_SIZE: [number, number] = [450, 100]

function applyNodeSize(node: any, width: number, height: number) {
  if (typeof node.setSize === 'function') {
    node.setSize([width, height])
  } else {
    node.size = [width, height]
  }
  node.setDirtyCanvas?.(true, true)
}

export function preserveVideoCombineNodeSize(nodeType: any, nodeData: { name?: string }) {
  if (nodeData.name !== NODE_NAME) return

  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function onVideoCombineNodeCreated() {
    originalOnNodeCreated?.call(this)
    const [width, height] = DEFAULT_SIZE
    applyNodeSize(this, width, height)
  }
}
