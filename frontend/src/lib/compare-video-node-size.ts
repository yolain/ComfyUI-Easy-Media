const COMPARE_VIDEO_NODE_NAME = 'easy compareVideos'
const COMPARE_VIDEO_DEFAULT_SIZE: [number, number] = [720, 520]

function applyNodeSize(node: any, width: number, height: number) {
  if (typeof node.setSize === 'function') {
    node.setSize([width, height])
  } else {
    node.size = [width, height]
  }
  node.setDirtyCanvas?.(true, true)
}

export function preserveCompareVideoNodeSize(nodeType: any, nodeData: { name?: string }) {
  if (nodeData.name !== COMPARE_VIDEO_NODE_NAME) return

  const originalOnNodeCreated = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function onCompareVideoNodeCreated() {
    originalOnNodeCreated?.call(this)
    const [width, height] = COMPARE_VIDEO_DEFAULT_SIZE
    applyNodeSize(this, width, height)
  }
}
