import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'

export const EASY_MEDIA_SYNC_PLAY_NODE_TYPES = new Set([
  'easy saveVideo',
  'easy compareVideos',
  'easy multiTrackEditor',
])

export const EASY_MEDIA_SYNC_PLAY_MENU_LABEL = 'Sync Play'

export interface EasyMediaSyncPlayNode {
  id?: string | number
  comfyClass?: string
  type?: string
  widgets?: unknown[]
  imgs?: unknown[]
  selected?: boolean
  __easyMediaSyncPlay?: (startAt: number) => void | Promise<void>
}

export interface EasyMediaSyncPlayCanvas {
  selected_nodes?: Record<string, EasyMediaSyncPlayNode>
  graph?: {
    _nodes?: EasyMediaSyncPlayNode[]
  }
}

export interface EasyMediaMenuOption {
  content?: string
  disabled?: boolean
  callback?: () => void | Promise<void>
}

export type EasyMediaMenuEntry = EasyMediaMenuOption | null

interface NodeConstructor {
  prototype: EasyMediaSyncPlayNode & {
    getExtraMenuOptions?: unknown
  }
}

type EasyMediaMenuGetter = (
  this: EasyMediaSyncPlayNode,
  canvas: EasyMediaSyncPlayCanvas,
  options: EasyMediaMenuEntry[],
) => EasyMediaMenuEntry[]

function isTargetNode(node: EasyMediaSyncPlayNode | null | undefined): node is EasyMediaSyncPlayNode {
  return EASY_MEDIA_SYNC_PLAY_NODE_TYPES.has(node?.comfyClass ?? node?.type ?? '')
}

function uniqueNodes(nodes: EasyMediaSyncPlayNode[]): EasyMediaSyncPlayNode[] {
  const seen = new Set<EasyMediaSyncPlayNode>()
  return nodes.filter((node) => {
    if (seen.has(node)) return false
    seen.add(node)
    return true
  })
}

export function getSyncPlayTargetNodes(
  currentNode: EasyMediaSyncPlayNode,
  canvas: EasyMediaSyncPlayCanvas | undefined,
): EasyMediaSyncPlayNode[] {
  const selectedNodes = Object.values(canvas?.selected_nodes ?? {}).filter(isTargetNode)
  if (selectedNodes.length > 0) return uniqueNodes(selectedNodes)
  if (isTargetNode(currentNode)) return [currentNode]
  return []
}

function isHTMLElement(value: unknown): value is HTMLElement {
  return value instanceof HTMLElement
}

function collectVideosFromValue(value: unknown, videos: Set<HTMLVideoElement>) {
  if (value instanceof HTMLVideoElement) {
    videos.add(value)
    return
  }
  if (isHTMLElement(value)) {
    value.querySelectorAll('video').forEach((video) => videos.add(video))
  }
}

function collectNativeVideos(node: EasyMediaSyncPlayNode): HTMLVideoElement[] {
  const videos = new Set<HTMLVideoElement>()
  for (const widget of node.widgets ?? []) {
    if (!widget || typeof widget !== 'object') continue
    for (const key of ['element', 'inputEl', 'video', 'container']) {
      collectVideosFromValue((widget as Record<string, unknown>)[key], videos)
    }
  }
  for (const preview of node.imgs ?? []) collectVideosFromValue(preview, videos)
  return [...videos]
}

export function playNativeNodeVideosFromStart(node: EasyMediaSyncPlayNode): boolean {
  const videos = collectNativeVideos(node)
  for (const video of videos) {
    try {
      video.pause()
      video.currentTime = 0
      const playResult = video.play()
      if (playResult) {
        playResult.catch((error) => {
          console.error('[EasyMedia Sync Play] failed to play native video:', error)
        })
      }
    } catch (error) {
      console.error('[EasyMedia Sync Play] failed to reset native video:', error)
    }
  }
  return videos.length > 0
}

export async function syncPlayNodes(nodes: EasyMediaSyncPlayNode[]) {
  const startAt = performance.now()
  await Promise.all(nodes.map(async (node) => {
    try {
      if (node.__easyMediaSyncPlay) {
        await node.__easyMediaSyncPlay(startAt)
        return
      }
      if (!playNativeNodeVideosFromStart(node)) {
        console.warn('[EasyMedia Sync Play] no playable video found for node:', node)
      }
    } catch (error) {
      console.error('[EasyMedia Sync Play] failed to sync play node:', error)
    }
  }))
}

export function installEasyMediaSyncPlay(nodeType: NodeConstructor, nodeData: { name?: string }) {
  if (!EASY_MEDIA_SYNC_PLAY_NODE_TYPES.has(nodeData.name ?? '')) return

  const originalGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions as EasyMediaMenuGetter | undefined
  nodeType.prototype.getExtraMenuOptions = function getExtraMenuOptions(
    this: EasyMediaSyncPlayNode,
    canvas: EasyMediaSyncPlayCanvas,
    options: EasyMediaMenuEntry[],
  ) {
    const nextOptions = originalGetExtraMenuOptions?.call(this, canvas, options) ?? options
    const targets = getSyncPlayTargetNodes(this, canvas)
    if (targets.length === 0) return nextOptions

    return [
      ...nextOptions,
      null,
      {
        content: EASY_MEDIA_SYNC_PLAY_MENU_LABEL,
        disabled: targets.length === 0,
        callback: () => syncPlayNodes(getSyncPlayTargetNodes(this, canvas)),
      },
    ]
  }

  nodeType.prototype.__easyMediaSyncPlay ??= function syncPlayNativeVideo() {
    playNativeNodeVideosFromStart(this)
  }
}

export function installEasyMediaSyncPlayForNode(app: ComfyApp, nodeType: NodeConstructor, nodeData: { name?: string }) {
  void app
  installEasyMediaSyncPlay(nodeType, nodeData)
}
