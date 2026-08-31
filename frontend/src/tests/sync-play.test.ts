import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  getEasyMediaSyncPlayMenuItems,
  getSyncPlayNodeMuted,
  getSyncPlayTargetNodes,
  installEasyMediaSyncPlay,
  playNativeNodeVideosFromStart,
  syncPlayNodes,
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
    const projectCombine = { comfyClass: 'easy multitrackProjectVideoCombine' }
    const other = { comfyClass: 'KSampler' }

    expect(getSyncPlayTargetNodes(current, {
      selected_nodes: { saveVideo, editor, projectCombine, other },
    })).toEqual([saveVideo, editor, projectCombine])
  })

  it('falls back to the current node when there is no selected target node', () => {
    const current = { comfyClass: 'easy saveVideo' }

    expect(getSyncPlayTargetNodes(current, {
      selected_nodes: { other: { comfyClass: 'KSampler' } },
    })).toEqual([current])
  })
})

describe('sync play menu installation', () => {
  it('returns a right click menu action that plays selected target nodes', async () => {
    const saveVideo = { comfyClass: 'easy saveVideo', __easyMediaSyncPlay: vi.fn() }
    const editor = { comfyClass: 'easy multiTrackEditor', __easyMediaSyncPlay: vi.fn() }
    const projectCombine = { comfyClass: 'easy multitrackProjectVideoCombine', __easyMediaSyncPlay: vi.fn() }
    const options = getEasyMediaSyncPlayMenuItems(saveVideo, {
      selected_nodes: { saveVideo, editor, projectCombine },
    })
    const syncPlayOption = options.find((option) => option?.content === 'Sync Play')
    await syncPlayOption?.callback?.()

    expect(syncPlayOption).toBeDefined()
    expect(saveVideo.__easyMediaSyncPlay).toHaveBeenCalledWith(expect.any(Number), false)
    expect(editor.__easyMediaSyncPlay).toHaveBeenCalledWith(expect.any(Number), true)
    expect(projectCombine.__easyMediaSyncPlay).toHaveBeenCalledWith(expect.any(Number), true)
  })

  it('does not add the menu action to unrelated nodes', () => {
    expect(getEasyMediaSyncPlayMenuItems({ comfyClass: 'KSampler' }, {
      selected_nodes: {},
    })).toEqual([])
  })

  it('installs native playback support without replacing existing custom playback', () => {
    const existingPlayback = vi.fn()
    const nodeType = function NodeType() {} as unknown as {
      prototype: EasyMediaSyncPlayNode
    }
    nodeType.prototype = { __easyMediaSyncPlay: existingPlayback }

    installEasyMediaSyncPlay(nodeType, { name: 'easy saveVideo' })

    expect(nodeType.prototype.__easyMediaSyncPlay).toBe(existingPlayback)
  })
})

describe('sync play audio priority', () => {
  it('uses save video, project combine, then multitrack editor priority', () => {
    const saveVideo = { comfyClass: 'easy saveVideo' }
    const projectCombine = { comfyClass: 'easy multitrackProjectVideoCombine' }
    const editor = { comfyClass: 'easy multiTrackEditor' }

    expect(getSyncPlayNodeMuted(saveVideo, [saveVideo, projectCombine, editor])).toBe(false)
    expect(getSyncPlayNodeMuted(projectCombine, [saveVideo, projectCombine, editor])).toBe(true)
    expect(getSyncPlayNodeMuted(editor, [saveVideo, projectCombine, editor])).toBe(true)
    expect(getSyncPlayNodeMuted(projectCombine, [projectCombine, editor])).toBe(false)
    expect(getSyncPlayNodeMuted(editor, [projectCombine, editor])).toBe(true)
    expect(getSyncPlayNodeMuted(editor, [editor])).toBe(false)
  })

  it('does not change compare video audio state', () => {
    expect(getSyncPlayNodeMuted(
      { comfyClass: 'easy compareVideos' },
      [{ comfyClass: 'easy saveVideo' }, { comfyClass: 'easy compareVideos' }],
    )).toBeUndefined()
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
