/**
 * ComfyUI graph utility functions for tracing and navigating node connections.
 */

export interface NodeInfo {
  nodeId: number
  nodeName: string
  nodeType: string
  outputSlotIndex: number
  isList: boolean
}

/**
 * Get output slot information from a node.
 */
export function getOutputSlotInfo(
  nodeId: number,
  slotIndex: number,
  graph: any,
): { isList: boolean; linkId: number | null } | null {
  const node: any = graph.getNodeById(nodeId)
  if (!node) return null

  const outputSlot: any = node.outputs?.[slotIndex]
  if (!outputSlot) return null

  return {
    isList: outputSlot.shape == 6,
    linkId: outputSlot.link ?? null,
  }
}

/**
 * Get link information by link ID.
 */
export function getLinkInfo(linkId: number, graph: any): {
  originId: number
  originSlot: number
  targetId: number
  targetSlot: number
} | null {
  const link: any = graph.links[linkId]
  if (!link) return null

  return {
    originId: link.origin_id,
    originSlot: link.origin_slot,
    targetId: link.target_id,
    targetSlot: link.target_slot,
  }
}

/**
 * Recursively trace back through links to find the original source node.
 * Returns the root node ID, or null if not found.
 */
export function traceToRootSource(
  nodeId: number,
  slotIndex: number,
  graph: any,
  visited: Set<string> = new Set(),
): number | null {
  let currentNodeId: number | null = nodeId
  let currentSlotIdx: number | null = slotIndex

  while (currentNodeId !== null && currentSlotIdx !== null) {
    const key = `${currentNodeId}:${currentSlotIdx}`
    if (visited.has(key)) {
      console.debug(`[trace] Cycle detected at node ${currentNodeId}, slot ${currentSlotIdx}`)
      break
    }
    visited.add(key)

    const node: any = graph.getNodeById(currentNodeId)
    if (!node) break

    const outputSlot: any = node.outputs?.[currentSlotIdx]
    if (!outputSlot) break

    console.debug(
      `[trace] Node #${currentNodeId} "${node.name || node.type || 'unknown'}" slot ${currentSlotIdx} isList=${outputSlot.shape == 6}`,
    )

    const linkId: number | null = outputSlot.link
    if (linkId === null || linkId === undefined) {
      console.debug(`[trace] Root source found: #${currentNodeId}`)
      return currentNodeId
    }

    const link: any = graph.links[linkId]
    if (!link) break

    currentNodeId = link.origin_id
    currentSlotIdx = link.origin_slot
  }

  console.debug('[trace] Root source: none')
  return null
}

/**
 * Trace from a link ID to find the root source node.
 * The link points from a source output to a target input.
 */
export function traceToRootSourceViaLink(linkId: number, graph: any): number | null {
  if (!linkId) {
    console.debug('[traceViaLink] No link ID')
    return null
  }
  const link: any = graph.links[linkId]
  if (!link) {
    console.debug('[traceViaLink] Link not found:', linkId)
    return null
  }

  console.debug('[traceViaLink] Found link, tracing from:', {
    originId: link.origin_id,
    originSlot: link.origin_slot,
  })

  return traceToRootSource(link.origin_id, link.origin_slot, graph)
}