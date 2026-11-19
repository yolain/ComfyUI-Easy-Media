import { uuid } from '@/lib/uuid'
import type { MultiTrack, MultiTrackSegment, MultiTrackTaskMod } from '@/types/multitrack'

export const MAX_TASK_MODS = 8

export function taskMods(segment: MultiTrackSegment): MultiTrackTaskMod[] {
  return Array.isArray(segment.content.mods) ? segment.content.mods : []
}

export function appendTaskMods(mods: MultiTrackTaskMod[], paths: string[]): MultiTrackTaskMod[] {
  const known = new Set(mods.map((mod) => mod.file_path))
  const selected = [...new Set(paths)]
    .filter((path) => !known.has(path))
    .slice(0, Math.max(0, MAX_TASK_MODS - mods.length))
    .map((file_path) => ({ id: uuid(), file_path }))
  return selected.length ? [...mods, ...selected] : mods
}

export function taskModName(path: string): string {
  return (path.split('/').pop() ?? path).replace(/\.safetensors$/, '')
}

export function taskModPreviewUrl(path: string): string {
  return `/easy-media/refmods/preview?${new URLSearchParams({ path })}`
}

export function canShareTaskMod(segments: MultiTrackSegment[], mod: MultiTrackTaskMod): boolean {
  return segments.every((segment) => {
    const mods = taskMods(segment)
    return mods.length < MAX_TASK_MODS || mods.some((item) => item.file_path === mod.file_path)
  })
}

function withSharedMods(mods: MultiTrackTaskMod[], shared: Map<string, MultiTrackTaskMod>): MultiTrackTaskMod[] {
  const existing = new Map(mods.map((item) => [item.file_path, item]))
  const common = Array.from(shared, ([path, mod]) => {
    const local = existing.get(path)
    return local
      ? { ...local, shared_reference: true }
      : { ...mod, id: uuid(), muted: undefined, shared_reference: true, shared_reference_copy: true }
  })
  return [...common, ...mods.filter((item) => !shared.has(item.file_path))].slice(0, MAX_TASK_MODS)
}

export function sharedTaskModUpdates(
  segments: MultiTrackSegment[],
  mod: MultiTrackTaskMod,
  enabled: boolean,
  removeFromSegmentId?: string,
): Array<{ segmentId: string; mods: MultiTrackTaskMod[] }> {
  return segments.map((segment) => {
    const mods = taskMods(segment)
    if (enabled) {
      const shared = new Map(mods.filter((item) => item.shared_reference).map((item) => [item.file_path, item]))
      shared.set(mod.file_path, mod)
      return { segmentId: segment.id, mods: withSharedMods(mods, shared) }
    }
    return {
      segmentId: segment.id,
      mods: mods.flatMap((item) => {
        if (item.file_path !== mod.file_path) return [item]
        if (item.shared_reference_copy || segment.id === removeFromSegmentId) return []
        return [{ ...item, shared_reference: false, shared_reference_copy: undefined }]
      }),
    }
  })
}

export function synchronizeSharedTaskMods(tracks: MultiTrack[]): MultiTrack[] {
  const shared = new Map<string, MultiTrackTaskMod>()
  for (const track of tracks) {
    if (track.type !== 'task') continue
    for (const segment of track.segments) {
      for (const mod of taskMods(segment)) {
        if (mod.shared_reference && !shared.has(mod.file_path)) shared.set(mod.file_path, mod)
      }
    }
  }
  if (!shared.size) return tracks
  return tracks.map((track) => track.type === 'task' ? {
    ...track,
    segments: track.segments.map((segment) => ({
      ...segment,
      content: { ...segment.content, mods: withSharedMods(taskMods(segment), shared) },
    })),
  } : track)
}
