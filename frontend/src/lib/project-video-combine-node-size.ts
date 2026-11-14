import { preserveTimelineEditorNodeSize } from '@/lib/timeline-node-size'

export function preserveVideoCombineNodeSize(
  nodeType: Parameters<typeof preserveTimelineEditorNodeSize>[0],
  nodeData: { name?: string },
) {
  if (nodeData.name !== 'easy multitrackProjectVideoCombine') return
  preserveTimelineEditorNodeSize(nodeType, nodeData, [800, 700])
}
