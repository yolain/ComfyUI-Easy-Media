interface BeautySizeNode {
  size: [number, number]
  setSize?: (size: [number, number]) => void
  setDirtyCanvas?: (foreground: boolean, background: boolean) => void
  onNodeCreated?: (...args: unknown[]) => unknown
}
export function preserveFaceBeautyNodeSize(nodeType: { prototype: BeautySizeNode }, nodeData: { name?: string }) {
  if (nodeData.name !== 'easy faceBeautify') return
  const original = nodeType.prototype.onNodeCreated
  nodeType.prototype.onNodeCreated = function (...args: unknown[]) {
    const result = original?.apply(this, args)
    if (this.setSize) this.setSize([960, 640])
    else this.size = [960, 640]
    this.setDirtyCanvas?.(true, true)
    return result
  }
}
