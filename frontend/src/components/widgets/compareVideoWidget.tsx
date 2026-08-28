import { memo, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Check,
  Columns2,
  Download,
  FilePlay,
  FolderOpen,
  History,
  Pause,
  Play,
  Settings2,
  Volume2,
  VolumeX,
  X,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useCanvasScale } from '@/hooks/use-canvas-scale'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import { LocaleContext, useT } from '@/lib/i18n'
import { mediaPathToViewUrl } from '@/lib/media-url'
import { cn } from '@/lib/utils'
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import type { RecentMediaHistoryEntry } from '@/stores/media-list-store'

type CompareMode = 'source' | 'compare' | 'side-by-side' | 'output'

export interface CompareVideoSettings {
  save_output: boolean
  filename_prefix: string
  watch_output_history: boolean
  compare_mode?: CompareMode
  source?: CompareVideoMediaSelection | null
  output?: CompareVideoMediaSelection | null
}

type CompareVideoMediaSourceType = 'input' | 'output' | 'temp' | 'url'

export interface CompareVideoMediaSelection {
  source_type: CompareVideoMediaSourceType
  file_path?: string
  url?: string
}

const DEFAULT_COMPARE_VIDEO_SETTINGS: CompareVideoSettings = {
  save_output: false,
  filename_prefix: 'ComfyUI',
  watch_output_history: false,
  compare_mode: 'compare',
  source: null,
  output: null,
}

const WATCH_OUTPUT_HISTORY_LIMIT = 50
const WATCH_VIDEO_EXTENSIONS: ReadonlySet<string> = new Set([
  '.mp4',
  '.webm',
  '.mov',
  '.avi',
  '.mkv',
  '.flv',
  '.wmv',
  '.m4v',
])

export interface CompareVideoResult {
  filename: string
  subfolder?: string
  type?: 'input' | 'output' | 'temp'
}

export interface CompareVideoPayload {
  source?: CompareVideoResult | null
  output?: CompareVideoResult | null
  fps?: number | null
  frame_count?: number | null
  duration?: number | null
}

interface CompareVideoInnerProps {
  app: ComfyApp
  node: CompareVideoNode
  settings: CompareVideoSettings
  onSettingsChange: (settings: CompareVideoSettings) => void
  allowMediaSelection?: boolean
  bindNodeEvents?: boolean
  pausePlaybackNonce?: number
}

export interface CompareVideoProps {
  app: ComfyApp
  sourceUrl: string
  outputUrl: string
  pausePlaybackNonce?: number
}

type ExecutedHandler = (output: unknown) => void
type CompareVideoEventCallback = (event: CustomEvent<unknown>) => void

interface CompareVideoEventApi {
  addEventListener?: (type: string, callback: CompareVideoEventCallback) => void
  removeEventListener?: (type: string, callback: CompareVideoEventCallback) => void
  addCustomEventListener?: (type: string, callback: CompareVideoEventCallback) => void
  removeCustomEventListener?: (type: string, callback: CompareVideoEventCallback) => void
}

interface CompareVideoNode {
  id?: string | number
  onExecuted?: ExecutedHandler
  __easyMediaCompareVideos?: CompareVideoPayload | null
  __easyMediaSyncPlay?: () => void
}

function resultToUrl(result: CompareVideoResult | null | undefined): string | null {
  if (!result?.filename) return null
  const type = result.type ?? 'temp'
  const subfolder = result.subfolder ?? ''
  return `/view?filename=${encodeURIComponent(result.filename)}&type=${encodeURIComponent(type)}&subfolder=${encodeURIComponent(subfolder)}`
}

function normalizeCompareVideoMediaSelection(value: unknown): CompareVideoMediaSelection | null {
  if (!value || typeof value !== 'object') return null
  const record = value as Record<string, unknown>
  const rawType = record.source_type ?? record.type
  const sourceType: CompareVideoMediaSourceType | undefined =
    rawType === 'input' || rawType === 'output' || rawType === 'temp' || rawType === 'url'
      ? rawType
      : undefined
  const filePath =
    typeof record.file_path === 'string'
      ? record.file_path
      : typeof record.path === 'string'
        ? record.path
        : undefined
  const url = typeof record.url === 'string' ? record.url : undefined
  const rawValue = url ?? filePath ?? ''

  if (!rawValue.trim()) return null
  if (sourceType === 'url' || /^https?:\/\//i.test(rawValue)) {
    return { source_type: 'url', url: rawValue }
  }
  if (!sourceType || !filePath) return null
  return { source_type: sourceType, file_path: filePath }
}

function mediaSelectionToUrl(selection: CompareVideoMediaSelection | null | undefined): string | null {
  if (!selection) return null
  const sourceType = selection.source_type
  const filePath = selection.file_path
  const url = selection.url

  if (sourceType === 'url' || /^https?:\/\//i.test(url ?? filePath ?? '')) {
    return url ?? filePath ?? null
  }
  if ((sourceType === 'input' || sourceType === 'output' || sourceType === 'temp') && filePath) {
    return mediaPathToViewUrl(filePath, sourceType)
  }
  return null
}

function mediaSelectionValue(selection: CompareVideoMediaSelection | null | undefined): string {
  if (!selection) return ''
  return selection.file_path ?? selection.url ?? ''
}

function mediaSelectionTab(
  selection: CompareVideoMediaSelection | null | undefined,
  slot: 'source' | 'output',
): 'inputs' | 'outputs' | 'url' {
  if (selection?.source_type === 'url') return 'url'
  if (selection?.source_type === 'output') return 'outputs'
  return slot === 'output' ? 'outputs' : 'inputs'
}

function mediaSelectionName(selection: CompareVideoMediaSelection | null | undefined): string {
  const value = mediaSelectionValue(selection)
  return value.split(/[\\/]/).pop() || value
}

function recentMediaHistoryKey(entry: RecentMediaHistoryEntry): string | null {
  return entry.path ? `${entry.source_type}:${entry.path}` : null
}

function executedOutputToHistory(value: unknown, mtime: number): RecentMediaHistoryEntry | null {
  if (!value || typeof value !== 'object') return null
  const record = value as Record<string, unknown>
  const filename = typeof record.filename === 'string' ? record.filename : ''
  const sourceType = record.type === 'output' || record.type === 'temp' ? record.type : undefined
  if (!filename || !sourceType) return null
  if (filename.toLowerCase().includes('easy_compare')) return null

  const extensionIndex = filename.lastIndexOf('.')
  if (extensionIndex <= 0) return null
  const extension = filename.slice(extensionIndex).toLowerCase()
  if (!WATCH_VIDEO_EXTENSIONS.has(extension)) return null

  const subfolder = typeof record.subfolder === 'string'
    ? record.subfolder.replaceAll('\\', '/')
    : ''
  const relativePath = subfolder ? `${subfolder}/${filename}` : filename
  return {
    type: 'file',
    name: filename,
    path: relativePath,
    url: mediaPathToViewUrl(relativePath, sourceType),
    size: 0,
    mtime,
    source_type: sourceType,
  }
}

function collectExecutedWatchOutputs(detail: unknown): RecentMediaHistoryEntry[] {
  if (!detail || typeof detail !== 'object') return []
  const output = (detail as Record<string, unknown>).output
  if (!output || typeof output !== 'object') return []

  const outputRecord = output as Record<string, unknown>
  const entries: RecentMediaHistoryEntry[] = []
  const seen = new Set<string>()
  let mtime = Date.now()

  for (const key of ['video', 'images'] as const) {
    const values = outputRecord[key]
    if (!Array.isArray(values)) continue
    for (const value of values) {
      const entry = executedOutputToHistory(value, mtime)
      if (!entry) continue
      const entryKey = recentMediaHistoryKey(entry)
      if (!entryKey || seen.has(entryKey)) continue
      seen.add(entryKey)
      entries.push(entry)
      mtime += 1
    }
  }

  return entries
}

function outputMediaSelection(filePath: string): CompareVideoMediaSelection {
  return { source_type: 'output', file_path: filePath }
}

function tempMediaSelection(filePath: string): CompareVideoMediaSelection {
  return { source_type: 'temp', file_path: filePath }
}

function compareMediaSelectionsEqual(
  left: CompareVideoMediaSelection | null | undefined,
  right: CompareVideoMediaSelection | null | undefined,
): boolean {
  if (!left && !right) return true
  if (!left || !right) return false
  return left.source_type === right.source_type
    && left.file_path === right.file_path
    && left.url === right.url
}

function compareMediaSelectionKey(selection: CompareVideoMediaSelection | null | undefined): string | null {
  if (!selection) return null
  const value = selection.file_path ?? selection.url ?? ''
  return value ? `${selection.source_type}:${value}` : null
}

function historyVideoToSelection(video: RecentMediaHistoryEntry): CompareVideoMediaSelection {
  return video.source_type === 'temp'
    ? tempMediaSelection(video.path)
    : outputMediaSelection(video.path)
}

export function mergeWatchOutputHistorySelections(
  current: Pick<CompareVideoSettings, 'source' | 'output'>,
  videos: readonly RecentMediaHistoryEntry[],
  manualSlots: ReadonlySet<'source' | 'output'> = new Set(),
): Pick<CompareVideoSettings, 'source' | 'output'> {
  const uniqueSelections: CompareVideoMediaSelection[] = []
  const seenKeys = new Set<string>()
  for (const video of videos) {
    const selection = historyVideoToSelection(video)
    const key = compareMediaSelectionKey(selection)
    if (!key || seenKeys.has(key)) continue
    seenKeys.add(key)
    uniqueSelections.push(selection)
  }

  const sourceIsManual = manualSlots.has('source') && current.source != null
  const outputIsManual = manualSlots.has('output') && current.output != null

  const source = sourceIsManual
    ? current.source ?? null
    : outputIsManual
      ? uniqueSelections.find((selection) => (
          compareMediaSelectionKey(selection) !== compareMediaSelectionKey(current.output)
        )) ?? null
      : uniqueSelections.length >= 2
        ? uniqueSelections[1]
        : null

  const output = outputIsManual
    ? current.output ?? null
    : sourceIsManual
      ? uniqueSelections.find((selection) => (
          compareMediaSelectionKey(selection) !== compareMediaSelectionKey(source)
        )) ?? null
      : uniqueSelections.length >= 1
        ? uniqueSelections[0]
        : null

  return { source, output }
}

function normalizeCompareVideoPayload(value: unknown): CompareVideoPayload | null {
  if (Array.isArray(value)) {
    for (const item of value) {
      const payload = normalizeCompareVideoPayload(item)
      if (payload) return payload
    }
    return null
  }
  if (!value || typeof value !== 'object') return null
  return value as CompareVideoPayload
}

function normalizeCompareVideoSettings(value: unknown): CompareVideoSettings {
  if (!value || typeof value !== 'object') return { ...DEFAULT_COMPARE_VIDEO_SETTINGS }
  const record = value as Record<string, unknown>
  const rawMode = record.compare_mode
  const compareMode: CompareMode = (rawMode === 'source' || rawMode === 'compare' || rawMode === 'side-by-side' || rawMode === 'output')
    ? rawMode
    : 'compare'
  return {
    save_output: record.save_output === true,
    filename_prefix: typeof record.filename_prefix === 'string'
      ? record.filename_prefix
      : DEFAULT_COMPARE_VIDEO_SETTINGS.filename_prefix,
    watch_output_history: record.watch_output_history === true,
    compare_mode: compareMode,
    source: normalizeCompareVideoMediaSelection(record.source),
    output: normalizeCompareVideoMediaSelection(record.output),
  }
}

function parseCompareVideoPayload(output: unknown): CompareVideoPayload | null {
  if (!output || typeof output !== 'object') return null
  const record = output as Record<string, unknown>
  const direct = normalizeCompareVideoPayload(record.compare_videos)
  if (direct) return direct
  const nestedOutput = record.output
  if (nestedOutput && typeof nestedOutput === 'object') {
    const nestedDirect = normalizeCompareVideoPayload((nestedOutput as Record<string, unknown>).compare_videos)
    if (nestedDirect) return nestedDirect
  }
  const ui = record.ui
  if (ui && typeof ui === 'object') {
    const nested = normalizeCompareVideoPayload((ui as Record<string, unknown>).compare_videos)
    if (nested) return nested
  }
  return null
}

function executedEventBelongsToNode(detail: unknown, node: CompareVideoNode): boolean {
  if (!detail || typeof detail !== 'object') return true
  const eventNode = (detail as Record<string, unknown>).node
  if (eventNode === undefined || eventNode === null || node.id === undefined || node.id === null) return true
  return String(eventNode) === String(node.id)
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '0:00'
  const minutes = Math.floor(seconds / 60)
  const wholeSeconds = Math.floor(seconds % 60)
  return `${minutes}:${wholeSeconds.toString().padStart(2, '0')}`
}

function seek(video: HTMLVideoElement | null, time: number) {
  if (!video || !Number.isFinite(time)) return
  try {
    video.currentTime = time
  } catch (error) {
    console.error('[CompareVideoWidget] failed to seek video:', error)
  }
}

export function CompareVideoWidget({ app, node, value, onChange }: Readonly<ReactWidgetProps<CompareVideoSettings>>) {
  const locale = app?.ui?.settings?.settingsValues?.['Comfy.Locale']
  const compareNode = node as CompareVideoNode
  const settings = normalizeCompareVideoSettings(value)

  return (
    <LocaleContext.Provider value={locale}>
      <CompareVideoWidgetInner app={app} node={compareNode} settings={settings} onSettingsChange={onChange} />
    </LocaleContext.Provider>
  )
}

/** Compare two externally selected videos without exposing the media pickers. */
export const CompareVideo = memo(function CompareVideo({
  app,
  sourceUrl,
  outputUrl,
  pausePlaybackNonce,
}: Readonly<CompareVideoProps>) {
  const locale = app?.ui?.settings?.settingsValues?.['Comfy.Locale']
  const [compareMode, setCompareMode] = useState<CompareMode>('side-by-side')
  const settings = useMemo<CompareVideoSettings>(() => ({
    ...DEFAULT_COMPARE_VIDEO_SETTINGS,
    compare_mode: compareMode,
    source: { source_type: 'url', url: sourceUrl },
    output: { source_type: 'url', url: outputUrl },
  }), [compareMode, outputUrl, sourceUrl])

  return (
    <LocaleContext.Provider value={locale}>
      <CompareVideoWidgetInner
        app={app}
        node={{}}
        settings={settings}
        onSettingsChange={(nextSettings) => setCompareMode(nextSettings.compare_mode ?? 'side-by-side')}
        allowMediaSelection={false}
        bindNodeEvents={false}
        pausePlaybackNonce={pausePlaybackNonce}
      />
    </LocaleContext.Provider>
  )
})

function CompareVideoSettingsFields({
  settings,
  onSettingsChange,
}: Readonly<Pick<CompareVideoInnerProps, 'settings' | 'onSettingsChange'>>) {
  const t = useT()
  return (
    <div className="flex w-64 flex-col gap-3">
      <div className="flex items-center gap-2">
        <Button
          type="button"
          variant="ghost"
          role="checkbox"
          aria-checked={settings.save_output}
          disabled={settings.watch_output_history}
          className={cn(
            'h-auto flex-1 cursor-pointer justify-start gap-2 px-0 py-1 hover:bg-transparent',
            settings.watch_output_history && 'cursor-not-allowed opacity-50',
          )}
          onClick={() => onSettingsChange({ ...settings, save_output: !settings.save_output })}
        >
          <span
            className={cn(
              'flex h-4 w-4 shrink-0 items-center justify-center rounded-sm border border-border',
              settings.save_output && 'border-primary bg-primary text-primary-foreground',
            )}
          >
            {settings.save_output ? <Check className="h-3 w-3" /> : null}
          </span>
          <span className="text-left">{t('compareVideo.saveOutput')}</span>
        </Button>
        <Button
          type="button"
          variant="ghost"
          role="checkbox"
          aria-checked={settings.watch_output_history}
          className="h-auto flex-1 cursor-pointer justify-start gap-2 px-0 py-1 hover:bg-transparent"
          onClick={() => {
            const nextWatchOutputHistory = !settings.watch_output_history
            onSettingsChange({
              ...settings,
              watch_output_history: nextWatchOutputHistory,
              source: nextWatchOutputHistory ? null : settings.source,
              output: nextWatchOutputHistory ? null : settings.output,
              save_output: nextWatchOutputHistory ? false : settings.save_output,
            })
          }}
        >
          <span
            className={cn(
              'flex h-4 w-4 shrink-0 items-center justify-center rounded-sm border border-border',
              settings.watch_output_history && 'border-primary bg-primary text-primary-foreground',
            )}
          >
            {settings.watch_output_history ? <Check className="h-3 w-3" /> : null}
          </span>
          <History className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-left">{t('compareVideo.watchOutputHistory')}</span>
        </Button>
      </div>
      <label className="flex flex-col gap-1.5 text-xs text-muted-foreground">
        <span>{t('compareVideo.filenamePrefix')}</span>
        <Input
          value={settings.filename_prefix}
          disabled={settings.watch_output_history}
          placeholder="ComfyUI"
          aria-label={t('compareVideo.filenamePrefix')}
          onChange={(event) => onSettingsChange({ ...settings, filename_prefix: event.currentTarget.value })}
        />
      </label>
    </div>
  )
}

function CompareVideoMediaPicker({
  slot,
  selection,
  disabled = false,
  watchHistory = false,
  onSelect,
  onClear,
}: Readonly<{
  slot: 'source' | 'output'
  selection: CompareVideoMediaSelection | null | undefined
  disabled?: boolean
  watchHistory?: boolean
  onSelect: (filePath: string, source?: 'input' | 'output' | 'temp' | 'local') => void
  onClear: () => void
}>) {
  const t = useT()
  const [open, setOpen] = useState(false)
  const selectedName = mediaSelectionName(selection)
  const label = slot === 'source' ? t('compareVideo.source') : t('compareVideo.output')
  const selectLabel = slot === 'source' ? t('compareVideo.selectSource') : t('compareVideo.selectOutput')
  const clearLabel = slot === 'source' ? t('compareVideo.clearSource') : t('compareVideo.clearOutput')

  return (
    <div className="flex items-center gap-1">
      <Popover open={open} onOpenChange={(nextOpen) => {
        if (disabled) return
        setOpen(nextOpen)
      }}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            disabled={disabled}
            className={cn(
              'h-7 gap-1.5 border-border/50 bg-muted/60 px-2 text-foreground shadow',
              disabled ? 'cursor-not-allowed opacity-50' : 'cursor-pointer hover:bg-muted/90',
            )}
            aria-label={selectedName ? `${label}: ${selectedName}` : selectLabel}
          >
            <FolderOpen className="h-3.5 w-3.5" />
            <span className="max-w-36 truncate text-[11px]">{selectedName || selectLabel}</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent
          className="w-auto p-0"
          align={slot === 'source' ? 'start' : 'end'}
        >
          <MediaSelector
            key={`${slot}-${selectedName || 'empty'}`}
            value={mediaSelectionValue(selection)}
            mediaType="video"
            defaultTab={watchHistory ? 'history' : mediaSelectionTab(selection, slot)}
            historySource="combined"
            historyOnly={watchHistory}
            onChange={(filePath, source) => {
              if (source === undefined) {
                onSelect(filePath, source)
                setOpen(false)
              }
            }}
            onSourceChange={(event) => {
              onSelect(event.filePath, event.sourceType)
              setOpen(false)
            }}
          />
        </PopoverContent>
      </Popover>
      {selection ? (
        <Button
          type="button"
          variant="ghost"
          size="icon"
          className="h-7 w-7 cursor-pointer bg-muted/60 text-foreground shadow hover:bg-muted/90"
          aria-label={clearLabel}
          disabled={disabled}
          onClick={onClear}
        >
          <X className="h-3.5 w-3.5" />
        </Button>
      ) : null}
    </div>
  )
}

function CompareVideoWidgetInner({
  app,
  node,
  settings,
  onSettingsChange,
  allowMediaSelection = true,
  bindNodeEvents = true,
  pausePlaybackNonce,
}: Readonly<CompareVideoInnerProps>) {
  const t = useT()
  const canvasScale = useCanvasScale(app)
  const sourceRef = useRef<HTMLVideoElement>(null)
  const outputRef = useRef<HTMLVideoElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const [payload, setPayload] = useState<CompareVideoPayload | null>(() => node.__easyMediaCompareVideos ?? null)
  const [mode, setMode] = useState<CompareMode>(settings.compare_mode ?? 'compare')
  const [split, setSplit] = useState(50)
  const [isPointerInside, setIsPointerInside] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(0.75)
  const [currentTime, setCurrentTime] = useState(0)
  const [metadataDuration, setMetadataDuration] = useState(0)
  const settingsRef = useRef(settings)
  const onSettingsChangeRef = useRef(onSettingsChange)
  const manualSlotsRef = useRef<Set<'source' | 'output'>>(new Set())
  const previousWatchRef = useRef(settings.watch_output_history)
  settingsRef.current = settings
  onSettingsChangeRef.current = onSettingsChange

  const sourceUrl = useMemo(
    () => mediaSelectionToUrl(settings.source)
      ?? (settings.watch_output_history ? null : resultToUrl(payload?.source)),
    [payload, settings.source, settings.watch_output_history],
  )
  const outputUrl = useMemo(
    () => mediaSelectionToUrl(settings.output)
      ?? (settings.watch_output_history ? null : resultToUrl(payload?.output)),
    [payload, settings.output, settings.watch_output_history],
  )
  const hasSource = Boolean(sourceUrl)
  const hasOutput = Boolean(outputUrl)
  const hasAnyVideo = hasSource || hasOutput
  const canCompare = hasSource && hasOutput
  const duration = hasOutput ? metadataDuration : Math.max(payload?.duration ?? 0, metadataDuration, 0)

  const visibleMode: CompareMode = canCompare ? mode : hasSource ? 'source' : 'output'
  const isSideBySide = canCompare && visibleMode === 'side-by-side'
  const displaySplit = visibleMode === 'source' ? 100 : visibleMode === 'output' ? 0 : isSideBySide ? 50 : split
  const animateComparison = visibleMode !== 'compare' || !isPointerInside

  useEffect(() => {
    if (!bindNodeEvents) return

    function applyExecutedOutput(output: unknown) {
      const nextPayload = parseCompareVideoPayload(output)
      if (!nextPayload) return
      node.__easyMediaCompareVideos = nextPayload
      setPayload(nextPayload)
      setCurrentTime(0)
      setIsPlaying(false)
      setMetadataDuration(0)
    }

    const original = node.onExecuted as ExecutedHandler | undefined
    const wrapped = function onCompareVideosExecuted(this: unknown, output: unknown) {
      original?.call(this, output)
      applyExecutedOutput(output)
    }

    node.onExecuted = wrapped
    const api = app.api as CompareVideoEventApi | undefined
    const handleExecuted: CompareVideoEventCallback = (event) => {
      if (!executedEventBelongsToNode(event.detail, node)) return
      applyExecutedOutput(event.detail)
    }
    api?.addEventListener?.('executed', handleExecuted)
    api?.addCustomEventListener?.('executed', handleExecuted)
    return () => {
      if (node.onExecuted === wrapped) node.onExecuted = original
      api?.removeEventListener?.('executed', handleExecuted)
      api?.removeCustomEventListener?.('executed', handleExecuted)
    }
  }, [app.api, bindNodeEvents, node])

  useEffect(() => {
    if (sourceRef.current) {
      sourceRef.current.muted = muted || hasOutput
      sourceRef.current.volume = volume
    }
    if (outputRef.current) {
      outputRef.current.muted = muted
      outputRef.current.volume = volume
    }
  }, [hasOutput, muted, volume])

  useEffect(() => {
    if (canCompare) {
      setMode(settings.compare_mode ?? 'compare')
    } else if (hasSource) {
      setMode('source')
    } else if (hasOutput) {
      setMode('output')
    }
  }, [canCompare, hasOutput, hasSource, settings.compare_mode])

  useEffect(() => {
    const previousWatch = previousWatchRef.current
    previousWatchRef.current = settings.watch_output_history
    if (settings.watch_output_history && !previousWatch) {
      manualSlotsRef.current.clear()
    } else if (!settings.watch_output_history && previousWatch) {
      node.__easyMediaCompareVideos = null
      setPayload(null)
      setCurrentTime(0)
      setMetadataDuration(0)
    }
  }, [node, settings.watch_output_history])

  useEffect(() => {
    if (!settings.watch_output_history) return

    const watchHistory: RecentMediaHistoryEntry[] = []
    let runOutputs: RecentMediaHistoryEntry[] = []
    let runOutputKeys = new Set<string>()

    function applyWatchHistory() {
      const currentSettings = settingsRef.current
      if (!currentSettings.watch_output_history) return

      const { source: nextSource, output: nextOutput } = mergeWatchOutputHistorySelections(
        currentSettings,
        watchHistory,
        manualSlotsRef.current,
      )

      if (
        compareMediaSelectionsEqual(currentSettings.source, nextSource)
        && compareMediaSelectionsEqual(currentSettings.output, nextOutput)
      ) {
        return
      }

      onSettingsChangeRef.current({ ...currentSettings, source: nextSource, output: nextOutput })
      setCurrentTime(0)
      setMetadataDuration(0)
      setIsPlaying(false)
    }

    function handleExecuted(event: CustomEvent<unknown>) {
      const entries = collectExecutedWatchOutputs(event.detail)
      if (entries.length === 0) return

      for (const entry of entries) {
        const entryKey = recentMediaHistoryKey(entry)
        if (!entryKey || runOutputKeys.has(entryKey)) continue
        runOutputKeys.add(entryKey)
        runOutputs.push(entry)
      }
    }

    function handleExecutionStart() {
      runOutputs = []
      runOutputKeys = new Set()
    }

    function handleExecutionSuccess() {
      const newestFirstRunOutputs = [...runOutputs].reverse()
      const mergedHistory: RecentMediaHistoryEntry[] = []
      const historyKeys = new Set<string>()

      for (const entry of [...newestFirstRunOutputs, ...watchHistory]) {
        const entryKey = recentMediaHistoryKey(entry)
        if (!entryKey || historyKeys.has(entryKey)) continue
        historyKeys.add(entryKey)
        mergedHistory.push(entry)
      }

      watchHistory.splice(0, watchHistory.length, ...mergedHistory.slice(0, WATCH_OUTPUT_HISTORY_LIMIT))

      runOutputs = []
      runOutputKeys = new Set()
      applyWatchHistory()
    }

    const api = app.api as CompareVideoEventApi | undefined
    api?.addEventListener?.('execution_start', handleExecutionStart)
    api?.addEventListener?.('executed', handleExecuted)
    api?.addEventListener?.('execution_success', handleExecutionSuccess)
    api?.addCustomEventListener?.('execution_start', handleExecutionStart)
    api?.addCustomEventListener?.('executed', handleExecuted)
    api?.addCustomEventListener?.('execution_success', handleExecutionSuccess)

    return () => {
      api?.removeEventListener?.('execution_start', handleExecutionStart)
      api?.removeEventListener?.('executed', handleExecuted)
      api?.removeEventListener?.('execution_success', handleExecutionSuccess)
      api?.removeCustomEventListener?.('execution_start', handleExecutionStart)
      api?.removeCustomEventListener?.('executed', handleExecuted)
      api?.removeCustomEventListener?.('execution_success', handleExecutionSuccess)
    }
  }, [app.api, settings.watch_output_history])

  const syncVideos = useCallback((time: number) => {
    seek(sourceRef.current, time)
    seek(outputRef.current, time)
  }, [])

  const playVideos = useCallback(() => {
    setIsPlaying(true)
    for (const video of [sourceRef.current, outputRef.current]) {
      if (!video) continue
      const playResult = video.play()
      if (playResult) {
        playResult.catch((error: unknown) => {
          console.error('[CompareVideoWidget] failed to play video:', error)
          setIsPlaying(false)
        })
      }
    }
  }, [])

  const pauseVideos = useCallback(() => {
    setIsPlaying(false)
    sourceRef.current?.pause()
    outputRef.current?.pause()
  }, [])

  useEffect(() => {
    if (pausePlaybackNonce === undefined) return
    pauseVideos()
  }, [pausePlaybackNonce, pauseVideos])

  useEffect(() => () => {
    sourceRef.current?.pause()
    outputRef.current?.pause()
  }, [])

  useEffect(() => {
    if (!bindNodeEvents) return
    node.__easyMediaSyncPlay = () => {
      syncVideos(0)
      setCurrentTime(0)
      playVideos()
    }
    return () => {
      if (node.__easyMediaSyncPlay) delete node.__easyMediaSyncPlay
    }
  }, [bindNodeEvents, node, playVideos, syncVideos])

  function updateSplitFromPointer(event: React.PointerEvent<HTMLDivElement>) {
    if (!canCompare || visibleMode !== 'compare') return
    const rect = stageRef.current?.getBoundingClientRect()
    if (!rect) return
    const scale = canvasScale > 0 ? canvasScale : 1
    const localX = (event.clientX - rect.left) / scale
    const width = rect.width / scale
    const percent = width > 0 ? (localX / width) * 100 : 50
    setSplit(Math.max(0, Math.min(100, percent)))
  }

  function handleTimeUpdate(event: React.SyntheticEvent<HTMLVideoElement>) {
    const video = event.currentTarget
    const master = outputRef.current ?? sourceRef.current
    if (video !== master) return
    setCurrentTime(video.currentTime)
    const other = video === sourceRef.current ? outputRef.current : sourceRef.current
    if (other && Math.abs(other.currentTime - video.currentTime) > 0.08) {
      seek(other, video.currentTime)
    }
  }

  function handleEnded() {
    syncVideos(0)
    setCurrentTime(0)
    if (isPlaying) playVideos()
  }

  function handleSeek(value: number[]) {
    const nextTime = value[0] ?? 0
    setCurrentTime(nextTime)
    syncVideos(nextTime)
  }

  function cycleMode() {
    setMode((current) => {
      let next: CompareMode = 'compare'
      if (current === 'compare') next = 'side-by-side'
      else if (current === 'side-by-side') next = 'source'
      else if (current === 'source') next = 'output'
      onSettingsChange({ ...settings, compare_mode: next })
      return next
    })
  }

  function handleMediaSelect(
    slot: 'source' | 'output',
    filePath: string,
    source?: 'input' | 'output' | 'temp' | 'local',
  ) {
    const isUrl = /^https?:\/\//i.test(filePath)
    const sourceType: CompareVideoMediaSourceType = source === 'output'
      ? 'output'
      : source === 'temp'
        ? 'temp'
      : source === 'input'
        ? 'input'
        : isUrl
          ? 'url'
          : 'input'
    const selection: CompareVideoMediaSelection = sourceType === 'url'
      ? { source_type: 'url', url: filePath }
      : { source_type: sourceType, file_path: filePath }
    const nextSettings = { ...settings, [slot]: selection }
    const otherSelection = slot === 'source' ? nextSettings.output : nextSettings.source

    if (settings.watch_output_history) {
      manualSlotsRef.current.add(slot)
    }

    onSettingsChange(nextSettings)
    setCurrentTime(0)
    setMetadataDuration(0)
    setIsPlaying(false)
    if (mediaSelectionToUrl(selection) && mediaSelectionToUrl(otherSelection)) {
      setMode(nextSettings.compare_mode ?? 'compare')
    }
  }

  function handleMediaClear(slot: 'source' | 'output') {
    if (settings.watch_output_history) {
      manualSlotsRef.current.delete(slot)
    }
    onSettingsChange({ ...settings, [slot]: null })
    setCurrentTime(0)
    setMetadataDuration(0)
    setIsPlaying(false)
  }

  function downloadOutputVideo() {
    const downloadName = settings.watch_output_history
      ? mediaSelectionName(settings.output)
      : payload?.output?.filename ?? mediaSelectionName(settings.output)
    if (!outputUrl || !downloadName) return
    try {
      const anchor = document.createElement('a')
      anchor.href = outputUrl
      anchor.download = downloadName
      anchor.rel = 'noopener'
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
    } catch (error) {
      console.error('[CompareVideoWidget] failed to download output video:', error)
    }
  }

  return (
    <TooltipProvider>
      <div className="group relative h-full min-h-[320px] w-full overflow-hidden rounded border border-border bg-background text-foreground">
        <div
          ref={stageRef}
          className="relative h-full min-h-[320px] overflow-hidden bg-black"
          onPointerEnter={() => setIsPointerInside(true)}
          onPointerMove={(event) => {
            updateSplitFromPointer(event)
          }}
          onPointerLeave={() => {
            setIsPointerInside(false)
            if (visibleMode === 'compare') setSplit(50)
          }}
          onPointerCancel={() => {
            setIsPointerInside(false)
            if (visibleMode === 'compare') setSplit(50)
          }}
        >
          {sourceUrl ? (
            <video
              ref={sourceRef}
              className={cn(
                'absolute inset-0 h-full w-full object-contain',
                !hasSource && 'invisible',
                isSideBySide && 'right-auto w-1/2',
              )}
              data-compare-video-panel={isSideBySide ? 'source' : undefined}
              src={sourceUrl}
              loop
              muted={muted || hasOutput}
              playsInline
              preload="auto"
              onLoadedMetadata={(event) => {
                const nextDuration = event.currentTarget.duration || 0
                if (!hasOutput) setMetadataDuration(nextDuration)
              }}
              onTimeUpdate={handleTimeUpdate}
              onEnded={handleEnded}
            />
          ) : null}
          {outputUrl ? (
            <video
              ref={outputRef}
              className={cn(
                'absolute inset-0 h-full w-full object-contain',
                !hasOutput && 'invisible',
                isSideBySide && 'left-1/2 w-1/2',
              )}
              style={{
                clipPath: isSideBySide ? undefined : `inset(0 0 0 ${displaySplit}%)`,
                transition: !isSideBySide && animateComparison ? 'clip-path 260ms ease' : undefined,
              }}
              data-compare-video-panel={isSideBySide ? 'output' : undefined}
              src={outputUrl}
              loop
              muted={muted}
              playsInline
              preload="auto"
              onLoadedMetadata={(event) => {
                const nextDuration = event.currentTarget.duration || 0
                setMetadataDuration(nextDuration)
              }}
              onTimeUpdate={handleTimeUpdate}
              onEnded={handleEnded}
            />
          ) : null}

          {!hasAnyVideo ? (
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-4 bg-background p-4 pt-12">
              <CompareVideoSettingsFields settings={settings} onSettingsChange={onSettingsChange} />
              <p className="text-xs text-muted-foreground mt-2">{t('compareVideo.empty')}</p>
            </div>
          ) : null}

          {allowMediaSelection ? (
            <>
              <div className="absolute left-3 top-2 z-30">
                <CompareVideoMediaPicker
                  slot="source"
                  selection={settings.source}
                  watchHistory={settings.watch_output_history}
                  onSelect={(filePath, source) => handleMediaSelect('source', filePath, source)}
                  onClear={() => handleMediaClear('source')}
                />
              </div>
              <div className="absolute right-3 top-2 z-30">
                <CompareVideoMediaPicker
                  slot="output"
                  selection={settings.output}
                  watchHistory={settings.watch_output_history}
                  onSelect={(filePath, source) => handleMediaSelect('output', filePath, source)}
                  onClear={() => handleMediaClear('output')}
                />
              </div>
            </>
          ) : null}

          {canCompare && (visibleMode === 'compare' || isSideBySide) ? (
            <div
              className="pointer-events-none absolute inset-y-0 w-px bg-white shadow"
              style={{
                left: `${displaySplit}%`,
                transition: animateComparison ? 'left 260ms ease' : undefined,
              }}
            />
          ) : null}

          {canCompare ? (
            <Button
              type="button"
              className="absolute left-1/2 top-2 z-30 -translate-x-1/2 cursor-pointer bg-muted/60 text-foreground shadow hover:bg-muted/90"
              onClick={cycleMode}
              onPointerDown={(event) => event.stopPropagation()}
              size="sm"
              variant="secondary"
            >
              {visibleMode === 'source' || visibleMode === 'output' ? <FilePlay className="h-4 w-4" /> : <Columns2 className="h-4 w-4" />}
              {visibleMode === 'source'
                ? t('compareVideo.sourceOnly')
                : visibleMode === 'output'
                  ? t('compareVideo.outputOnly')
                  : visibleMode === 'side-by-side'
                    ? t('compareVideo.sideBySide')
                    : t('compareVideo.compare')}
            </Button>
          ) : null}
        </div>

        <div
          data-compare-video-toolbar
          className="@container/compare-video-controls pointer-events-none absolute inset-x-0 bottom-0 z-10 grid translate-y-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 border-t border-border bg-background/90 px-1 opacity-0 shadow-sm backdrop-blur-sm transition-[opacity,transform] duration-300 ease-out @max-[420px]/compare-video-controls:grid-cols-[minmax(0,1fr)_auto] @max-[420px]/compare-video-controls:gap-x-1 @max-[420px]/compare-video-controls:gap-y-0 @max-[420px]/compare-video-controls:px-2 @max-[420px]/compare-video-controls:py-1 group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:translate-y-0 group-focus-within:opacity-100"
        >
          <div
            data-compare-video-playback
            className="@max-[420px]/compare-video-controls:col-start-1 @max-[420px]/compare-video-controls:row-start-2"
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="cursor-pointer"
                  onClick={isPlaying ? pauseVideos : playVideos}
                  aria-label={isPlaying ? t('compareVideo.pause') : t('compareVideo.play')}
                >
                  {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{isPlaying ? t('compareVideo.pause') : t('compareVideo.play')}</TooltipContent>
            </Tooltip>
          </div>

          <div
            data-compare-video-seek
            className="grid min-w-0 grid-cols-[auto_minmax(4rem,1fr)_auto] items-center gap-2 @max-[420px]/compare-video-controls:col-span-2 @max-[420px]/compare-video-controls:col-start-1 @max-[420px]/compare-video-controls:row-start-1 @max-[420px]/compare-video-controls:w-full @max-[420px]/compare-video-controls:px-1 @max-[420px]/compare-video-controls:py-1"
          >
            <span className="w-10 text-right text-[11px] tabular-nums text-muted-foreground">{formatTime(currentTime)}</span>
            <div className="min-w-0">
              <Slider
                className="w-full"
                value={[Math.min(currentTime, duration || currentTime)]}
                min={0}
                max={Math.max(duration, currentTime, 0.01)}
                step={0.01}
                onValueChange={handleSeek}
                aria-label={t('compareVideo.seek')}
              />
            </div>
            <span className="w-10 text-[11px] tabular-nums text-muted-foreground">{formatTime(duration)}</span>
          </div>

          <div
            data-compare-video-actions
            className="flex items-center gap-2 @max-[420px]/compare-video-controls:col-start-2 @max-[420px]/compare-video-controls:row-start-2 @max-[420px]/compare-video-controls:gap-1"
          >
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="cursor-pointer"
                  onClick={() => setMuted((current) => !current)}
                  aria-label={muted ? t('compareVideo.unmute') : t('compareVideo.mute')}
                >
                  {muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>{muted ? t('compareVideo.unmute') : t('compareVideo.mute')}</TooltipContent>
            </Tooltip>
            <div data-compare-video-volume>
              <Slider
                className="w-20"
                value={[volume]}
                min={0}
                max={1}
                step={0.01}
                onValueChange={(value) => setVolume(value[0] ?? 0)}
                aria-label={t('compareVideo.volume')}
              />
            </div>

            {allowMediaSelection ? (
              <Popover>
                <PopoverTrigger asChild>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="cursor-pointer"
                    aria-label={t('compareVideo.settings')}
                  >
                    <Settings2 className="h-4 w-4" />
                  </Button>
                </PopoverTrigger>
                <PopoverContent align="end" className="w-auto">
                  <CompareVideoSettingsFields settings={settings} onSettingsChange={onSettingsChange} />
                </PopoverContent>
              </Popover>
            ) : null}
            {allowMediaSelection && hasOutput ? (
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    className="cursor-pointer"
                    aria-label={t('compareVideo.downloadOutput')}
                    onClick={downloadOutputVideo}
                  >
                    <Download className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{t('compareVideo.downloadOutput')}</TooltipContent>
              </Tooltip>
            ) : null}
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}

export { parseCompareVideoPayload }
