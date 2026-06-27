import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  EASY_MEDIA_SYNC_PLAY_MENU_LABEL,
  getSyncPlayTargetNodes,
  installEasyMediaSyncPlay,
  playNativeNodeVideosFromStart,
  syncPlayNodes,
  type EasyMediaMenuEntry,
  type EasyMediaSyncPlayNode,
} from '@/lib/sync-play'

afterEach(() => {
  vi.restoreAllMocks()
})

describe('sync play node targeting', () => {
  it('uses selected easy media nodes when multiple target nodes are selected', () => {
    const current = { comfyClass: 'easy saveVideo' }
    const saveVideo = { comfyClass: 'easy saveVideo' }
    const editor = { comfyClass: 'easy multiTrackEditor' }
    const other = { comfyClass: 'KSampler' }

    expect(getSyncPlayTargetNodes(current, {
      selected_nodes: { saveVideo, editor, other },
    })).toEqual([saveVideo, editor])
  })

  it('falls back to the current node when there is no selected target node', () => {
    const current = { comfyClass: 'easy saveVideo' }

    expect(getSyncPlayTargetNodes(current, {
      selected_nodes: { other: { comfyClass: 'KSampler' } },
    })).toEqual([current])
  })
})

describe('sync play menu installation', () => {
  it('adds a right click menu action that plays selected target nodes', async () => {
    const saveVideo = { comfyClass: 'easy saveVideo', __easyMediaSyncPlay: vi.fn() }
    const editor = { comfyClass: 'easy multiTrackEditor', __easyMediaSyncPlay: vi.fn() }
    const nodeType = function NodeType() {} as unknown as {
      prototype: EasyMediaSyncPlayNode & { getExtraMenuOptions?: unknown }
    }
    nodeType.prototype = { comfyClass: 'easy saveVideo' }

    installEasyMediaSyncPlay(nodeType, { name: 'easy saveVideo' })

    const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions as (
      canvas: { selected_nodes: Record<string, EasyMediaSyncPlayNode> },
      options: EasyMediaMenuEntry[],
    ) => EasyMediaMenuEntry[]
    const options = getExtraMenuOptions({
      selected_nodes: { saveVideo, editor },
    }, [])
    const syncPlayOption = options.find((option) => option?.content === EASY_MEDIA_SYNC_PLAY_MENU_LABEL)
    await syncPlayOption?.callback?.()

    expect(syncPlayOption).toBeDefined()
    expect(saveVideo.__easyMediaSyncPlay).toHaveBeenCalledOnce()
    expect(editor.__easyMediaSyncPlay).toHaveBeenCalledOnce()
  })
})

describe('native video sync play', () => {
  it('resets and plays native video elements attached to node widgets', () => {
    const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    const container = document.createElement('div')
    const video = document.createElement('video')
    container.append(video)
    video.currentTime = 12

    const played = playNativeNodeVideosFromStart({
      comfyClass: 'easy saveVideo',
      widgets: [{ element: container }],
    })

    expect(played).toBe(true)
    expect(pause).toHaveBeenCalledOnce()
    expect(play).toHaveBeenCalledOnce()
    expect(video.currentTime).toBe(0)
  })

  it('uses the custom node play method before native video fallback', async () => {
    const customPlay = vi.fn()
    const nativePlay = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    const video = document.createElement('video')

    await syncPlayNodes([{
      comfyClass: 'easy multiTrackEditor',
      widgets: [{ element: video }],
      __easyMediaSyncPlay: customPlay,
    }])

    expect(customPlay).toHaveBeenCalledOnce()
    expect(nativePlay).not.toHaveBeenCalled()
  })
})
