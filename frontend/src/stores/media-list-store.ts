export type MediaListSource = 'inputs' | 'outputs' | 'local'
export type MediaListMediaType = 'all' | 'image' | 'audio' | 'video'

export interface MediaDirEntry {
  type: 'dir'
  name: string
  path: string
}

export interface MediaFileEntry {
  type: 'file'
  name: string
  path: string
  url: string
  size: number
  mtime: number
  width?: number
  height?: number
}

export type MediaItem = MediaDirEntry | MediaFileEntry

export interface MediaListRequest {
  source: MediaListSource
  mediaType: MediaListMediaType
  localPath: string
  subfolder: string
}

interface CacheEntry {
  request: MediaListRequest
  items: MediaItem[]
}

interface PendingEntry {
  request: MediaListRequest
  promise: Promise<MediaItem[]>
}

const cache = new Map<string, CacheEntry>()
const pending = new Map<string, PendingEntry>()
const listeners = new Set<() => void>()
let revision = 0
const generations: Record<MediaListSource, number> = {
  inputs: 0,
  outputs: 0,
  local: 0,
}

function requestKey(request: MediaListRequest): string {
  return JSON.stringify(request)
}

function normalizeMediaItems(data: unknown): MediaItem[] {
  const raw = data && typeof data === 'object' ? data as Record<string, unknown> : {}
  const rawList = Array.isArray(raw.items) ? raw.items : Array.isArray(raw.files) ? raw.files : []
  return rawList.reduce<MediaItem[]>((items, entry) => {
    if (!entry || typeof entry !== 'object') return items
    const item = entry as Record<string, unknown>
    items.push(item.type === 'dir'
      ? item as unknown as MediaDirEntry
      : { ...item, type: 'file' } as unknown as MediaFileEntry)
    return items
  }, [])
}

function mediaListUrl(request: MediaListRequest): string {
  const params = new URLSearchParams({ source: request.source, type: request.mediaType })
  if (request.source === 'local') params.set('path', request.localPath)
  if (request.subfolder) params.set('subfolder', request.subfolder)
  return `/easy-media/media/list?${params}`
}

export function getMediaList(request: MediaListRequest): Promise<MediaItem[]> {
  const key = requestKey(request)
  const cached = cache.get(key)
  if (cached) return Promise.resolve(cached.items)

  const activeRequest = pending.get(key)
  if (activeRequest) return activeRequest.promise

  const requestGeneration = generations[request.source]
  const promise = fetch(mediaListUrl(request))
    .then((response) => {
      if (!response.ok) throw new Error(`${response.status}`)
      return response.json() as Promise<unknown>
    })
    .then((data) => {
      const items = normalizeMediaItems(data)
      if (requestGeneration === generations[request.source]) cache.set(key, { request, items })
      return items
    })
    .finally(() => {
      if (pending.get(key)?.promise === promise) pending.delete(key)
    })

  pending.set(key, { request, promise })
  return promise
}

export function invalidateMediaListCache(source?: MediaListSource): void {
  if (source) {
    generations[source] += 1
  } else {
    generations.inputs += 1
    generations.outputs += 1
    generations.local += 1
  }
  for (const [key, entry] of cache) {
    if (!source || entry.request.source === source) cache.delete(key)
  }
  for (const [key, entry] of pending) {
    if (!source || entry.request.source === source) pending.delete(key)
  }
  revision += 1
  listeners.forEach((listener) => listener())
}

export function clearMediaListCache(): void {
  invalidateMediaListCache()
}

export function subscribeMediaListStore(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function getMediaListStoreRevision(): number {
  return revision
}
