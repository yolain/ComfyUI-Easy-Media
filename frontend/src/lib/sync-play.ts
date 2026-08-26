import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import { translate } from './i18n'

export const EASY_MEDIA_SYNC_PLAY_NODE_TYPES = new Set([
  'easy saveVideo',
  'easy compareVideos',
  'easy multiTrackEditor',
  'easy multitrackProjectVideoCombine',
])

export const EASY_MEDIA_SYNC_PLAY_MENU_LABEL_KEY = 'common.syncPlay'

export interface EasyMediaSyncPlayNode {
  id?: string | number
  comfyClass?: string
  type?: string
  widgets?: unknown[]
  imgs?: unknown[]
  selected?: boolean
  __easyMediaSyncPlay?: (startAt: number, muted?: boolean) => void | Promise<void>
}

export interface EasyMediaSyncPlayCanvas {
  selected_nodes?: Record<string, EasyMediaSyncPlayNode>
  graph?: {
    _nodes?: EasyMediaSyncPlayNode[]
  } | null
}

export interface EasyMediaMenuOption {
  content: string
  disabled?: boolean
  callback?: () => void | Promise<void>
}

export type EasyMediaMenuEntry = EasyMediaMenuOption | null

interface NodeConstructor {
  prototype: EasyMediaSyncPlayNode
}

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

const SYNC_PLAY_AUDIO_PRIORITY: Readonly<Record<string, number>> = {
  'easy saveVideo': 0,
  'easy multitrackProjectVideoCombine': 1,
  'easy multiTrackEditor': 2,
}

function nodeType(node: EasyMediaSyncPlayNode): string {
  return node.comfyClass ?? node.type ?? ''
}

export function getSyncPlayNodeMuted(
  node: EasyMediaSyncPlayNode,
  nodes: EasyMediaSyncPlayNode[],
): boolean | undefined {
  const priority = SYNC_PLAY_AUDIO_PRIORITY[nodeType(node)]
  if (priority === undefined) return undefined
  const activePriority = nodes.reduce((best, candidate) => {
    const candidatePriority = SYNC_PLAY_AUDIO_PRIORITY[nodeType(candidate)]
    return candidatePriority === undefined ? best : Math.min(best, candidatePriority)
  }, Number.POSITIVE_INFINITY)
  return Number.isFinite(activePriority) ? priority > activePriority : undefined
}

export function playNativeNodeVideosFromStart(node: EasyMediaSyncPlayNode, muted?: boolean): boolean {
  const videos = collectNativeVideos(node)
  for (const video of videos) {
    try {
      video.pause()
      video.currentTime = 0
      if (typeof muted === 'boolean') video.muted = muted
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
      const muted = getSyncPlayNodeMuted(node, nodes)
      if (node.__easyMediaSyncPlay) {
        await node.__easyMediaSyncPlay(startAt, muted)
        return
      }
      if (!playNativeNodeVideosFromStart(node, muted)) {
        console.warn('[EasyMedia Sync Play] no playable video found for node:', node)
      }
    } catch (error) {
      console.error('[EasyMedia Sync Play] failed to sync play node:', error)
    }
  }))
}

export function getEasyMediaSyncPlayMenuItems(
  currentNode: EasyMediaSyncPlayNode,
  canvas: EasyMediaSyncPlayCanvas | undefined,
  locale?: string,
): EasyMediaMenuEntry[] {
  if (!isTargetNode(currentNode)) return []
  const targets = getSyncPlayTargetNodes(currentNode, canvas)
  if (targets.length === 0) return []
  return [
    null,
    {
      content: translate(locale, EASY_MEDIA_SYNC_PLAY_MENU_LABEL_KEY),
      callback: () => syncPlayNodes(getSyncPlayTargetNodes(currentNode, canvas)),
    },
  ]
}

export function installEasyMediaSyncPlay(nodeType: NodeConstructor, nodeData: { name?: string }) {
  if (!EASY_MEDIA_SYNC_PLAY_NODE_TYPES.has(nodeData.name ?? '')) return

  nodeType.prototype.__easyMediaSyncPlay ??= function syncPlayNativeVideo(_startAt: number, muted?: boolean) {
    playNativeNodeVideosFromStart(this, muted)
  }
}

export function installEasyMediaSyncPlayForNode(app: ComfyApp, nodeType: NodeConstructor, nodeData: { name?: string }) {
  void app
  installEasyMediaSyncPlay(nodeType, nodeData)
}
