import { useState, useEffect, useRef, useSyncExternalStore } from 'react'
import {
  Search,
  ArrowUpDown,
  LayoutList,
  LayoutGrid,
  CheckCircle2,
  FileAudio,
  FileVideo,
  Image as ImageIcon,
  File,
  Folder,
  ChevronRight,
  Link2,
  Plus,
} from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { useT } from '@/lib/i18n'
import { $error } from '@/lib/comfy-api'
import { uploadInputMediaFile } from '@/lib/media-upload'
import { useDelayedIntersection } from '@/hooks/use-delayed-intersection'
import type { SlotItem } from '@/lib/timeline-utils'
import {
  getMediaList,
  getMediaListStoreRevision,
  invalidateMediaListCache,
  subscribeMediaListStore,
  type MediaDirEntry,
  type MediaFileEntry,
  type MediaItem,
  type MediaListMediaType,
} from '@/stores/media-list-store'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type MediaType = MediaListMediaType
export type MediaTab = 'inputs' | 'outputs' | 'local' | 'url' | 'slot'
type ViewMode = 'grid' | 'list'
type SortBy = 'name' | 'date' | 'folders'

const MULTIPLE_MEDIA_SEPARATOR = '|MULTIPLE|'

interface MediaSelectorChangeEvent {
  filePath: string
  sourceType: 'input' | 'output' | 'local'
}

export interface MediaSelectorProps {
  value: string
  onChange: (value: string, source?: 'input' | 'output' | 'local') => void
  onSourceChange?: (event: MediaSelectorChangeEvent) => void
  mediaType?: MediaType
  /** Which tab to show initially */
  defaultTab?: MediaTab
  /** Slot items computed from the connected node graph (only for image/audio media types) */
  slotItems?: SlotItem[]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function getFileIcon(name: string, mediaType: MediaType) {
  if (mediaType === 'audio') return FileAudio
  if (mediaType === 'image') return ImageIcon
  if (mediaType === 'video') return FileVideo
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  if (['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac', 'opus', 'wma'].includes(ext)) return FileAudio
  if (['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tiff', 'tif'].includes(ext)) return ImageIcon
  if (['mp4', 'webm', 'mov', 'mkv', 'avi', 'm4v'].includes(ext)) return FileVideo
  return File
}

function isImageFile(name: string, mediaType: MediaType): boolean {
  if (mediaType === 'image') return true
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  return ['png', 'jpg', 'jpeg', 'webp', 'gif', 'bmp', 'tiff', 'tif'].includes(ext)
}

function isAudioFile(name: string, mediaType: MediaType): boolean {
  if (mediaType === 'audio') return true
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  return ['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac', 'opus', 'wma'].includes(ext)
}

function isVideoFile(name: string, mediaType: MediaType): boolean {
  if (mediaType === 'video') return true
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  return ['mp4', 'webm', 'mov', 'mkv', 'avi', 'm4v'].includes(ext)
}

function getSelectedMediaValues(value: string): Set<string> {
  if (!value) return new Set()
  return new Set(value.split(MULTIPLE_MEDIA_SEPARATOR).filter((item) => item.length > 0))
}

function formatMediaInfo(file: MediaFileEntry, mediaType: MediaType): string {
  if (isImageFile(file.name, mediaType) && file.width && file.height) {
    return `${file.width}×${file.height}`
  }
  return formatSize(file.size)
}

function sortFiles(files: MediaFileEntry[], sortBy: SortBy): MediaFileEntry[] {
  return [...files].sort((a, b) => {
    if (sortBy === 'date') return b.mtime - a.mtime
    return a.name.localeCompare(b.name)
  })
}

function sortDirs(dirs: MediaDirEntry[]): MediaDirEntry[] {
  return [...dirs].sort((a, b) => a.name.localeCompare(b.name))
}

function sortByLabelKey(sortBy: SortBy): string {
  if (sortBy === 'name') return 'mediaSelector.sortName'
  if (sortBy === 'date') return 'mediaSelector.sortDate'
  return 'mediaSelector.sortFolders'
}

// ---------------------------------------------------------------------------
// LazyImage — only loads src once it enters the viewport/scroll-container
// ---------------------------------------------------------------------------

function LazyImage({
  src,
  alt,
  className,
}: Readonly<{ src: string; alt: string; className?: string }>) {
  const ref = useRef<HTMLImageElement>(null)
  const visible = useDelayedIntersection(ref)

  return (
    <img
      ref={ref}
      src={visible ? src : undefined}
      alt={alt}
      className={className}
      onError={(e) => {
        ;(e.target as HTMLImageElement).style.display = 'none'
      }}
    />
  )
}

function LazyVideo({
  src,
  className,
}: Readonly<{ src: string; className?: string }>) {
  const ref = useRef<HTMLVideoElement>(null)
  const visible = useDelayedIntersection(ref)

  return (
    <video
      ref={ref}
      src={visible ? src : undefined}
      className={className}
      muted
      playsInline
      preload="metadata"
      onLoadedMetadata={(event) => {
        const video = event.currentTarget
        if (Number.isFinite(video.duration) && video.duration > 0.1) video.currentTime = 0.1
      }}
      onError={(event) => {
        event.currentTarget.style.display = 'none'
      }}
    />
  )
}

// ---------------------------------------------------------------------------
// FileThumbnail (grid)
// ---------------------------------------------------------------------------

function FileThumbnail({
  file,
  mediaType,
  isSelected,
}: Readonly<{ file: MediaFileEntry; mediaType: MediaType; isSelected: boolean }>) {
  const Icon = getFileIcon(file.name, mediaType)
  const showImage = isImageFile(file.name, mediaType) && !!file.url
  const showVideo = !showImage && isVideoFile(file.name, mediaType) && !!file.url
  const isAudio = !showImage && !showVideo && isAudioFile(file.name, mediaType)

  return (
    <div
      className={cn(
        'relative w-full aspect-square rounded overflow-hidden flex items-center justify-center bg-muted',
        isSelected && 'ring-2 ring-primary',
      )}
    >
      {showImage ? (
        <LazyImage src={file.url} alt={file.name} className="w-full h-full object-cover" />
      ) : showVideo ? (
        <LazyVideo src={file.url} className="w-full h-full object-cover" />
      ) : (
        <Icon className={`w-6 h-6 ${isAudio ? 'text-highlight' : 'text-muted-foreground'}`} />
      )}
      {isSelected && (
        <div className="absolute top-1 right-1">
          <CheckCircle2 className="w-3.5 h-3.5 text-primary fill-primary-foreground" />
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Skeleton rows
// ---------------------------------------------------------------------------

function GridSkeleton() {
  return (
    <div className="grid grid-cols-4 gap-2 p-2">
      {Array.from({ length: 8 }, (_, i) => `sk-grid-${i}`).map((key) => (
        <div key={key} className="space-y-1">
          <Skeleton className="w-full aspect-square rounded" />
          <Skeleton className="h-3 w-3/4" />
          <Skeleton className="h-2.5 w-1/2" />
        </div>
      ))}
    </div>
  )
}

function ListSkeleton() {
  return (
    <div className="flex flex-col gap-1 p-2">
      {Array.from({ length: 6 }, (_, i) => `sk-list-${i}`).map((key) => (
        <div key={key} className="flex items-center gap-2 h-8">
          <Skeleton className="w-6 h-6 rounded shrink-0" />
          <Skeleton className="h-3 flex-1" />
          <Skeleton className="h-3 w-14 shrink-0" />
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Breadcrumb
// ---------------------------------------------------------------------------

function Breadcrumb({
  subfolder,
  onNavigate,
}: Readonly<{ subfolder: string; onNavigate: (path: string) => void }>) {
  if (!subfolder) return null
  const parts = subfolder.split('/').filter(Boolean)
  return (
    <div className="flex items-center gap-0.5 px-2 py-0.5 border-b border-border text-[11px] text-muted-foreground flex-wrap">
      <button
        type="button"
        className="hover:text-foreground transition-colors"
        onClick={() => onNavigate('')}
      >
        …
      </button>
      {parts.map((part, i) => {
        const path = parts.slice(0, i + 1).join('/')
        return (
          <span key={path} className="flex items-center gap-0.5">
            <ChevronRight className="w-3 h-3 shrink-0" />
            <button
              type="button"
              className="hover:text-foreground transition-colors truncate max-w-20"
              onClick={() => onNavigate(path)}
            >
              {part}
            </button>
          </span>
        )
      })}
    </div>
  )
}

// ---------------------------------------------------------------------------
// RemoteFileList — tree-aware, lazy images
// ---------------------------------------------------------------------------

function RemoteFileList({
  source,
  mediaType,
  localPath,
  viewMode,
  sortBy,
  searchQuery,
  value,
  onChange,
  onNavigateSubfolder,
  onAddLocalFile,
}: Readonly<{
  source: 'inputs' | 'outputs' | 'local'
  mediaType: MediaType
  localPath: string
  viewMode: ViewMode
  sortBy: SortBy
  searchQuery: string
  value: string
  onChange: (v: string, source: 'input' | 'output' | 'local') => void
  onNavigateSubfolder?: () => void
  onAddLocalFile?: () => void
}>) {
  const t = useT()
  const [items, setItems] = useState<MediaItem[]>([])
  const [subfolder, setSubfolder] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const cacheRevision = useSyncExternalStore(
    subscribeMediaListStore,
    getMediaListStoreRevision,
    getMediaListStoreRevision,
  )
  const selectedValues = getSelectedMediaValues(value)

  // Reset to root when the source or local path changes
  const rootKey = `${source}|${localPath}`
  const prevRootKeyRef = useRef(rootKey)

  useEffect(() => {
    const prevKey = prevRootKeyRef.current
    prevRootKeyRef.current = rootKey
    if (prevKey !== rootKey) {
      setSubfolder('')
      setItems([])
    }
  }, [rootKey])

  useEffect(() => {
    if (source === 'local' && !localPath) {
      setLoading(false)
      return
    }

    setLoading(true)
    setError(null)

    let cancelled = false
    getMediaList({ source, mediaType, localPath, subfolder })
      .then((list) => {
        if (!cancelled) setItems(list)
      })
      .catch((e: unknown) => {
        if (!cancelled) setError(e instanceof Error ? e.message : JSON.stringify(e))
      })
      .finally(() => { if (!cancelled) setLoading(false) })

    return () => { cancelled = true }
  }, [source, mediaType, localPath, subfolder, cacheRevision])

  const dirs = sortDirs(items.filter((i): i is MediaDirEntry => i.type === 'dir'))
  const files = sortFiles(
    (items.filter((i): i is MediaFileEntry => i.type === 'file') as MediaFileEntry[])
      .filter((f) => f.name.toLowerCase().includes(searchQuery.toLowerCase())),
    sortBy,
  )

  const filteredDirs = dirs.filter((d) =>
    searchQuery ? d.name.toLowerCase().includes(searchQuery.toLowerCase()) : true,
  )

  if (loading) return viewMode === 'grid' ? <GridSkeleton /> : <ListSkeleton />

  if (error) {
    return (
      <div className="flex items-center justify-center h-24 text-destructive text-xs">
        {t('mediaSelector.error', { msg: error })}
      </div>
    )
  }

  const isEmpty = filteredDirs.length === 0 && files.length === 0
  const selectedFiles = files.filter((file) => selectedValues.has(file.path))
  const sortedItems: MediaItem[] = sortBy === 'folders'
    ? [...filteredDirs, ...files]
    : [...files, ...filteredDirs]
  const unselectedItems = sortedItems.filter((item) => item.type === 'dir' || !selectedValues.has(item.path))

  function navigateSubfolder(path: string) {
    if (path === subfolder) return
    setSubfolder(path)
    onNavigateSubfolder?.()
  }

  function renderGridFile(file: MediaFileEntry, selected: boolean) {
    return (
      <button
        key={file.path}
        type="button"
        className="flex flex-col gap-1 text-left hover:opacity-80 transition-opacity"
        onClick={() => onChange(file.path, source === 'outputs' ? 'output' : 'input')}
      >
        <FileThumbnail file={file} mediaType={mediaType} isSelected={selected} />
        <span className="text-[10px] truncate leading-tight max-w-full" title={file.name}>
          {file.name}
        </span>
        <span className="text-[10px] text-muted-foreground truncate leading-tight max-w-full">
          {formatMediaInfo(file, mediaType)}
        </span>
      </button>
    )
  }

  function renderListFile(file: MediaFileEntry, selected: boolean) {
    const Icon = getFileIcon(file.name, mediaType)
    const showThumb = isImageFile(file.name, mediaType) && !!file.url
    const showVideoThumb = !showThumb && isVideoFile(file.name, mediaType) && !!file.url
    const showAudioIcon = !showThumb && !showVideoThumb && isAudioFile(file.name, mediaType)

    return (
      <button
        key={file.path}
        type="button"
        className={cn(
          'flex items-center gap-2 px-2 py-1 text-left hover:bg-accent transition-colors',
          selected && 'bg-accent',
        )}
        onClick={() => onChange(file.path, source === 'outputs' ? 'output' : 'input')}
      >
        {showThumb && (
          <div className="w-4 h-4 rounded overflow-hidden shrink-0 bg-muted">
            <LazyImage src={file.url} alt={file.name} className="w-full h-full object-cover" />
          </div>
        )}
        {showVideoThumb && (
          <div className="w-4 h-4 rounded overflow-hidden shrink-0 bg-muted">
            <LazyVideo src={file.url} className="w-full h-full object-cover" />
          </div>
        )}
        {showAudioIcon && (
          <div className="w-4 h-4 rounded flex items-center justify-center bg-[#34d399] shrink-0">
            <Icon className="w-3 h-3 text-white" />
          </div>
        )}
        {!showThumb && !showVideoThumb && !showAudioIcon && (
          <Icon className="w-4 h-4 text-muted-foreground shrink-0" />
        )}
        <span className="flex-1 text-xs truncate min-w-0" title={file.name}>
          {file.name}
        </span>
        <span className="text-[10px] text-muted-foreground shrink-0">
          {formatMediaInfo(file, mediaType)}
        </span>
        {selected && <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0" />}
      </button>
    )
  }

  function renderGridDir(dir: MediaDirEntry) {
    return (
      <button
        key={dir.path}
        type="button"
        className="flex flex-col gap-1 text-left hover:opacity-80 transition-opacity"
        onClick={() => navigateSubfolder(dir.path)}
      >
        <div className="relative w-full aspect-square rounded overflow-hidden bg-muted flex items-center justify-center">
          <Folder className="w-6 h-6 text-warning" />
        </div>
        <span className="text-[10px] truncate leading-tight max-w-full" title={dir.name}>
          {dir.name}
        </span>
      </button>
    )
  }

  function renderListDir(dir: MediaDirEntry) {
    return (
      <button
        key={dir.path}
        type="button"
        className="flex items-center gap-2 px-2 py-1 text-left hover:bg-accent transition-colors"
        onClick={() => navigateSubfolder(dir.path)}
      >
        <div className="w-4 h-4 rounded flex items-center justify-center bg-muted shrink-0">
          <Folder className="w-3 h-3 text-warning" />
        </div>
        <span className="flex-1 text-xs truncate min-w-0" title={dir.name}>
          {dir.name}
        </span>
      </button>
    )
  }

  function renderGridItem(item: MediaItem) {
    return item.type === 'dir'
      ? renderGridDir(item)
      : renderGridFile(item, selectedValues.has(item.path))
  }

  function renderListItem(item: MediaItem) {
    return item.type === 'dir'
      ? renderListDir(item)
      : renderListFile(item, selectedValues.has(item.path))
  }

  function renderGrid() {
    return (
      <div className="grid grid-cols-4 gap-2 p-2 overflow-y-auto flex-1">
        {/* Local file button */}
        {onAddLocalFile && (
          <button
            key="__add_local__"
            type="button"
            className="flex flex-col gap-1 text-left hover:opacity-80 transition-opacity"
            onClick={onAddLocalFile}
          >
            <div className="relative w-full aspect-square rounded overflow-hidden bg-muted flex items-center justify-center border-2 border-dashed border-muted-foreground/50">
              <Plus className="w-6 h-6 text-muted-foreground" />
            </div>
            <span className="text-[10px] truncate leading-tight max-w-full text-muted-foreground">
              {t('mediaSelector.addLocal')}
            </span>
          </button>
        )}
        {selectedFiles.map((file) => renderGridFile(file, true))}
        {unselectedItems.map(renderGridItem)}
      </div>
    )
  }

  function renderList() {
    return (
      <div className="flex flex-col overflow-y-auto flex-1">
        {/* Local file button */}
        {onAddLocalFile && (
          <button
            key="__add_local__"
            type="button"
            className="flex items-center gap-2 px-2 py-1 text-left hover:bg-accent transition-colors text-muted-foreground"
            onClick={onAddLocalFile}
          >
            <div className="w-4 h-4 rounded flex items-center justify-center bg-muted shrink-0">
              <Plus className="w-3 h-3" />
            </div>
            <span className="flex-1 text-xs truncate min-w-0">
              {t('mediaSelector.addLocal')}
            </span>
          </button>
        )}
        {selectedFiles.map((file) => renderListFile(file, true))}
        {unselectedItems.map(renderListItem)}
      </div>
    )
  }

  return (
    <div className="flex flex-col flex-1 overflow-hidden">
      <Breadcrumb subfolder={subfolder} onNavigate={navigateSubfolder} />
      {isEmpty && (
        <div className="flex items-center justify-center h-24 text-muted-foreground text-xs">
          {t('mediaSelector.empty')}
        </div>
      )}
      {!isEmpty && viewMode === 'grid' && renderGrid()}
      {!isEmpty && viewMode !== 'grid' && renderList()}
    </div>
  )
}

// ---------------------------------------------------------------------------
// MediaSelector
// ---------------------------------------------------------------------------

export function MediaSelector({
  value,
  onChange,
  onSourceChange,
  mediaType = 'all',
  defaultTab = 'inputs',
  slotItems = [],
}: Readonly<MediaSelectorProps>) {
  const t = useT()
  const showSlotTab = mediaType === 'image' || mediaType === 'audio'
  const [activeTab, setActiveTab] = useState<MediaTab>(defaultTab)
  const [viewMode, setViewMode] = useState<ViewMode>('grid')
  const [sortBy, setSortBy] = useState<SortBy>('name')
  const [searchQuery, setSearchQuery] = useState('')
  const [localPath, setLocalPath] = useState('')
  const [urlInput, setUrlInput] = useState(activeTab === 'url' ? value : '')
  const [urlChecking, setUrlChecking] = useState(false)
  const selectedValues = getSelectedMediaValues(value)

  // Sync defaultTab when it changes (e.g. popover re-opens for a different segment)
  useEffect(() => {
    setActiveTab(defaultTab)
    setSearchQuery('')
  }, [defaultTab])

  function cycleSortBy() {
    setSortBy((prev) => {
      if (prev === 'name') return 'date'
      if (prev === 'date') return 'folders'
      return 'name'
    })
  }

  function handleFileChange(filePath: string, source: 'input' | 'output' | 'local') {
    onChange(filePath, source)
    onSourceChange?.({ filePath, sourceType: source })
  }

  async function handleUrlConfirm() {
    if (urlChecking) return
    setUrlChecking(true)
    try {
      const img = new globalThis.Image()
      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = () => reject(new Error('Image load failed'))
        img.src = urlInput
      })
      onChange(urlInput)
    } catch {
      await downloadViaBackend()
    } finally {
      setUrlChecking(false)
    }
  }

  async function downloadViaBackend() {
    try {
      const res = await fetch('/easy-media/download-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: urlInput }),
      })
      if (!res.ok) {
        $error(t('mediaSelector.urlErrorTitle'), t('mediaSelector.urlErrorNotFound'))
        return
      }
      const data = await res.json() as { source_type: 'url' | 'input'; url?: string; file_name?: string }
      if (data.source_type === 'url') {
        onChange(data.url!)
      } else {
        invalidateMediaListCache('inputs')
        onChange(data.file_name!)
      }
    } catch {
      onChange(urlInput)
    }
  }

  function handleAddLocalFile() {
    const input = document.createElement('input')
    input.type = 'file'
    const accept = mediaType === 'image'
      ? 'image/*'
      : mediaType === 'audio'
        ? 'audio/*'
        : mediaType === 'video'
          ? 'video/*'
          : '*/*'
    input.accept = accept
    input.multiple = true
    input.onchange = async () => {
      if (!input.files || input.files.length === 0) return

      // For single file, select it directly
      if (input.files.length === 1) {
        const file = input.files[0]
        try {
          const uploaded = await uploadInputMediaFile(file)
          invalidateMediaListCache('inputs')
          onChange(uploaded)
        } catch (err) {
          console.error('[MediaSelector] upload failed:', err)
        }
        return
      }

      // For multiple files, select all of them (caller handles distribution)
      const paths: string[] = []
      for (const file of input.files) {
        try {
          const uploaded = await uploadInputMediaFile(file)
          paths.push(uploaded)
        } catch (err) {
          console.error('[MediaSelector] upload failed:', err)
        }
      }
      if (paths.length > 0) {
        invalidateMediaListCache('inputs')
        // Select first file for single selection, but indicate multiple were uploaded
        onChange(paths.join(MULTIPLE_MEDIA_SEPARATOR))
      }
    }
    input.click()
  }

  return (
    <div data-media-selector="" className="flex flex-col w-72 h-80 text-xs select-none">
      <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as MediaTab)} className="flex flex-col flex-1 overflow-hidden">
        {/* Tab header */}
        <TabsList className="w-full rounded-none rounded-t-md h-7 p-0.5 gap-0.5 shrink-0">
          <TabsTrigger value="inputs" className="flex-1 h-full text-[11px] px-1">
            {t('mediaSelector.tabInputs')}
          </TabsTrigger>
          <TabsTrigger value="outputs" className="flex-1 h-full text-[11px] px-1">
            {t('mediaSelector.tabOutputs')}
          </TabsTrigger>
          {/* <TabsTrigger value="local" className="flex-1 h-full text-[11px] px-1">
            {t('mediaSelector.tabLocal')}
          </TabsTrigger> */}
          <TabsTrigger value="url" className="flex-1 h-full text-[11px] px-1">
            {t('mediaSelector.tabUrl')}
          </TabsTrigger>
          {showSlotTab && (
            <TabsTrigger value="slot" className="flex-1 h-full text-[11px] px-1">
              {t('mediaSelector.tabSlot')}
            </TabsTrigger>
          )}
        </TabsList>

        {/* Search + view controls (shared across inputs/outputs/local) */}
        {activeTab !== 'url' && activeTab !== 'slot' && (
          <div className="flex items-center gap-1 px-2 py-1 border-b border-border shrink-0">
            <div className="relative flex-1">
              <Search className="absolute left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 text-muted-foreground pointer-events-none" />
              <Input
                className="h-6 pl-6 text-[11px]"
                placeholder={t('mediaSelector.searchPlaceholder')}
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              title={t('mediaSelector.sort', { by: t(sortByLabelKey(sortBy)) })}
              onClick={cycleSortBy}
            >
              <ArrowUpDown className="w-3 h-3" />
            </Button>
            <Button
              variant={viewMode === 'list' ? 'secondary' : 'ghost'}
              size="icon"
              className="h-6 w-6"
              title={t('mediaSelector.viewList')}
              onClick={() => setViewMode('list')}
            >
              <LayoutList className="w-3 h-3" />
            </Button>
            <Button
              variant={viewMode === 'grid' ? 'secondary' : 'ghost'}
              size="icon"
              className="h-6 w-6"
              title={t('mediaSelector.viewGrid')}
              onClick={() => setViewMode('grid')}
            >
              <LayoutGrid className="w-3 h-3" />
            </Button>
          </div>
        )}

        {/* Tab panels */}
        <TabsContent value="inputs" className="mt-0 flex-1 overflow-hidden flex flex-col">
          <RemoteFileList
            source="inputs"
            mediaType={mediaType}
            localPath=""
            viewMode={viewMode}
            sortBy={sortBy}
            searchQuery={searchQuery}
            value={value}
            onChange={(path) => handleFileChange(path, 'input')}
            onNavigateSubfolder={() => setSearchQuery('')}
            onAddLocalFile={handleAddLocalFile}
          />
        </TabsContent>

        <TabsContent value="outputs" className="mt-0 flex-1 overflow-hidden flex flex-col">
          <RemoteFileList
            source="outputs"
            mediaType={mediaType}
            localPath=""
            viewMode={viewMode}
            sortBy={sortBy}
            searchQuery={searchQuery}
            value={value}
            onChange={(path) => handleFileChange(path, 'output')}
            onNavigateSubfolder={() => setSearchQuery('')}
          />
        </TabsContent>

        <TabsContent value="local" className="mt-0 flex-1 overflow-hidden flex flex-col">
          {/* Local path input */}
          <div className="flex gap-1 px-2 pt-1 pb-1 border-b border-border">
            <Input
              className="h-6 text-[11px] flex-1"
              placeholder={t('mediaSelector.localPathPlaceholder')}
              value={localPath}
              onChange={(e) => setLocalPath(e.target.value)}
            />
          </div>
          <RemoteFileList
            source="local"
            mediaType={mediaType}
            localPath={localPath}
            viewMode={viewMode}
            sortBy={sortBy}
            searchQuery={searchQuery}
            value={value}
            onChange={(path) => handleFileChange(path, 'local')}
            onNavigateSubfolder={() => setSearchQuery('')}
          />
        </TabsContent>

        <TabsContent value="url" className="mt-0 flex-1 p-2">
          <div className="space-y-2">
            <p className="text-muted-foreground text-[11px]">{t('mediaSelector.urlHint')}</p>
            <Input
              className="h-7 text-[11px]"
              placeholder="https://..."
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') handleUrlConfirm() }}
            />
            <Button
              size="sm"
              className="h-6 text-[11px] w-full"
              disabled={!urlInput.startsWith('http') || urlChecking}
              onClick={handleUrlConfirm}
            >
              {urlChecking ? t('mediaSelector.urlChecking') : t('mediaSelector.urlConfirm')}
            </Button>
          </div>
        </TabsContent>

        {showSlotTab && (
          <TabsContent value="slot" className="mt-0 flex-1 overflow-hidden flex flex-col">
            {slotItems.length === 0 ? (
              <div className="flex-1 flex items-center justify-center text-muted-foreground text-[11px]">
                {t('mediaSelector.slotEmpty')}
              </div>
            ) : (
              <div className="flex flex-col overflow-y-auto flex-1">
                {slotItems
                  .map((item, index) => ({ item, index }))
                  .sort((a, b) => Number(selectedValues.has(b.item.value)) - Number(selectedValues.has(a.item.value)))
                  .map(({ item, index }) => {
                  const selected = selectedValues.has(item.value)
                  const isImage = item.value.startsWith('__slot__:image')
                  const isAudio = item.value.startsWith('__slot__:audio')
                  const displayLabel = isImage
                    ? t('mediaSelector.slotImage', { n: index + 1 })
                    : t('mediaSelector.slotAudio', { n: index + 1 })
                  return (
                    <button
                      key={item.value}
                      type="button"
                      className={cn(
                        'flex items-center gap-2 px-2 py-1 text-left hover:bg-accent transition-colors',
                        selected && 'bg-accent',
                      )}
                      onClick={() => onChange(item.value)}
                    >
                      {item.img ? (
                        <img
                          src={item.img}
                          alt={displayLabel}
                          className="w-8 h-8 object-cover rounded shrink-0 bg-muted"
                        />
                      ) : isAudio ? (
                        <div className="w-8 h-8 rounded flex items-center justify-center bg-[#34d399] shrink-0">
                          <FileAudio className="w-4 h-4 text-white" />
                        </div>
                      ) : (
                        <Link2 className="w-3.5 h-3.5 text-muted-foreground shrink-0" />
                      )}
                      <span className="flex-1 text-xs truncate min-w-0">{displayLabel}</span>
                      {isAudio && item.audio_name && (
                        <span className="text-[10px] text-muted-foreground truncate max-w-24" title={item.audio_name}>
                          {item.audio_name}
                        </span>
                      )}
                      {selected && <CheckCircle2 className="w-3.5 h-3.5 text-primary shrink-0" />}
                    </button>
                  )
                })}
              </div>
            )}
          </TabsContent>
        )}
      </Tabs>
    </div>
  )
}
