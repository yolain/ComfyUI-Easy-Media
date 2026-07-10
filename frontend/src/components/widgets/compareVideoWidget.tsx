import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Check, Columns2, Download, FilePlay, Pause, Play, Settings2, Volume2, VolumeX } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useCanvasScale } from '@/hooks/use-canvas-scale'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import { LocaleContext, useT } from '@/lib/i18n'
import { cn } from '@/lib/utils'
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'

type CompareMode = 'source' | 'compare' | 'output'

export interface CompareVideoSettings {
  save_output: boolean
  filename_prefix: string
}

const DEFAULT_COMPARE_VIDEO_SETTINGS: CompareVideoSettings = {
  save_output: false,
  filename_prefix: 'ComfyUI',
}

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
  if (!value || typeof value !== 'object') return DEFAULT_COMPARE_VIDEO_SETTINGS
  const record = value as Record<string, unknown>
  return {
    save_output: record.save_output === true,
    filename_prefix: typeof record.filename_prefix === 'string'
      ? record.filename_prefix
      : DEFAULT_COMPARE_VIDEO_SETTINGS.filename_prefix,
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

function CompareVideoSettingsFields({
  settings,
  onSettingsChange,
}: Readonly<Pick<CompareVideoInnerProps, 'settings' | 'onSettingsChange'>>) {
  const t = useT()
  return (
    <div className="flex w-64 flex-col gap-3">
      <Button
        type="button"
        variant="ghost"
        role="checkbox"
        aria-checked={settings.save_output}
        className="h-auto cursor-pointer justify-start gap-2 px-0 py-1 hover:bg-transparent"
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
        <span>{t('compareVideo.saveOutput')}</span>
      </Button>
      <label className="flex flex-col gap-1.5 text-xs text-muted-foreground">
        <span>{t('compareVideo.filenamePrefix')}</span>
        <Input
          value={settings.filename_prefix}
          placeholder="ComfyUI"
          aria-label={t('compareVideo.filenamePrefix')}
          onChange={(event) => onSettingsChange({ ...settings, filename_prefix: event.currentTarget.value })}
        />
      </label>
    </div>
  )
}

function CompareVideoWidgetInner({ app, node, settings, onSettingsChange }: Readonly<CompareVideoInnerProps>) {
  const t = useT()
  const canvasScale = useCanvasScale(app)
  const sourceRef = useRef<HTMLVideoElement>(null)
  const outputRef = useRef<HTMLVideoElement>(null)
  const stageRef = useRef<HTMLDivElement>(null)
  const [payload, setPayload] = useState<CompareVideoPayload | null>(() => node.__easyMediaCompareVideos ?? null)
  const [mode, setMode] = useState<CompareMode>('compare')
  const [split, setSplit] = useState(50)
  const [isPointerInside, setIsPointerInside] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(0.75)
  const [currentTime, setCurrentTime] = useState(0)
  const [metadataDuration, setMetadataDuration] = useState(0)

  const sourceUrl = useMemo(() => resultToUrl(payload?.source), [payload])
  const outputUrl = useMemo(() => resultToUrl(payload?.output), [payload])
  const hasSource = Boolean(sourceUrl)
  const hasOutput = Boolean(outputUrl)
  const canCompare = hasSource && hasOutput
  const duration = Math.max(payload?.duration ?? 0, metadataDuration, 0)

  const visibleMode: CompareMode = canCompare ? mode : hasSource ? 'source' : 'output'
  const displaySplit = canCompare
    ? visibleMode === 'source'
      ? 100
      : visibleMode === 'output'
        ? 0
        : split
    : hasSource ? 100 : 0
  const animateComparison = visibleMode !== 'compare' || !isPointerInside

  useEffect(() => {
    function applyExecutedOutput(output: unknown) {
      const nextPayload = parseCompareVideoPayload(output)
      if (!nextPayload) return
      node.__easyMediaCompareVideos = nextPayload
      setPayload(nextPayload)
      setCurrentTime(0)
      setIsPlaying(false)
      setMetadataDuration(0)
      setMode(nextPayload.source && nextPayload.output ? 'compare' : nextPayload.source ? 'source' : 'output')
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
  }, [app.api, node])

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
    if (!canCompare && hasSource) setMode('source')
    if (!canCompare && hasOutput) setMode('output')
  }, [canCompare, hasOutput, hasSource])

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
    node.__easyMediaSyncPlay = () => {
      syncVideos(0)
      setCurrentTime(0)
      playVideos()
    }
    return () => {
      if (node.__easyMediaSyncPlay) delete node.__easyMediaSyncPlay
    }
  }, [node, playVideos, syncVideos])

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
    setMode((current) => current === 'compare' ? 'source' : current === 'source' ? 'output' : 'compare')
  }

  function downloadOutputVideo() {
    if (!outputUrl || !payload?.output?.filename) return
    try {
      const anchor = document.createElement('a')
      anchor.href = outputUrl
      anchor.download = payload.output.filename
      anchor.rel = 'noopener'
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
    } catch (error) {
      console.error('[CompareVideoWidget] failed to download output video:', error)
    }
  }

  if (!payload || (!hasSource && !hasOutput)) {
    return (
      <div className="flex h-full min-h-[260px] w-full flex-col items-center justify-center gap-4 rounded border border-border bg-background p-4 text-xs text-muted-foreground">
        <p>{t('compareVideo.empty')}</p>
        <CompareVideoSettingsFields settings={settings} onSettingsChange={onSettingsChange} />
      </div>
    )
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
              )}
              src={sourceUrl}
              loop
              muted={muted || hasOutput}
              playsInline
              preload="auto"
              onLoadedMetadata={(event) => {
                const nextDuration = event.currentTarget.duration || 0
                setMetadataDuration((prev) => Math.max(prev, nextDuration))
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
              )}
              style={{
                clipPath: `inset(0 0 0 ${displaySplit}%)`,
                transition: animateComparison ? 'clip-path 260ms ease' : undefined,
              }}
              src={outputUrl}
              loop
              muted={muted}
              playsInline
              preload="auto"
              onLoadedMetadata={(event) => {
                const nextDuration = event.currentTarget.duration || 0
                setMetadataDuration((prev) => Math.max(prev, nextDuration))
              }}
              onTimeUpdate={handleTimeUpdate}
              onEnded={handleEnded}
            />
          ) : null}

          {hasSource && visibleMode == 'compare' ? (
            <Badge className="absolute left-3 top-2 bg-muted/60 text-foreground shadow-sm hover:bg-muted/90">
              {t('compareVideo.source')}
            </Badge>
          ) : null}
          {hasOutput && visibleMode =='compare' ? (
            <Badge className="absolute right-3 top-2 bg-muted/60 text-foreground shadow-sm hover:bg-muted/90">
              {t('compareVideo.output')}
            </Badge>
          ) : null}

          {canCompare ? (
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
              className="absolute left-1/2 top-2 -translate-x-1/2 cursor-pointer bg-muted/60 text-foreground shadow hover:bg-muted/90"
              onClick={cycleMode}
              onPointerDown={(event) => event.stopPropagation()}
              size="sm"
              variant="secondary"
            >
              {visibleMode === 'source' ? <FilePlay className="h-4 w-4" /> : visibleMode === 'output' ? <FilePlay className="h-4 w-4" /> : <Columns2 className="h-4 w-4" />}
              {visibleMode === 'source' ? t('compareVideo.sourceOnly') : visibleMode === 'output' ? t('compareVideo.outputOnly') : t('compareVideo.compare')}
            </Button>
          ) : null}
        </div>

        <div
          data-compare-video-toolbar
          className="pointer-events-none absolute inset-x-0 bottom-0 z-10 grid translate-y-full grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2 border-t border-border bg-background/90 px-1 opacity-0 shadow-sm backdrop-blur-sm transition-[opacity,transform] duration-300 ease-out group-hover:pointer-events-auto group-hover:translate-y-0 group-hover:opacity-100 group-focus-within:pointer-events-auto group-focus-within:translate-y-0 group-focus-within:opacity-100"
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

          <div className="grid min-w-0 grid-cols-[auto_minmax(0,1fr)_auto] items-center gap-2">
            <span className="w-10 text-right text-[11px] tabular-nums text-muted-foreground">{formatTime(currentTime)}</span>
            <Slider
              value={[Math.min(currentTime, duration || currentTime)]}
              min={0}
              max={Math.max(duration, currentTime, 0.01)}
              step={0.01}
              onValueChange={handleSeek}
              aria-label={t('compareVideo.seek')}
            />
            <span className="w-10 text-[11px] tabular-nums text-muted-foreground">{formatTime(duration)}</span>
          </div>

          <div className="flex items-center gap-2">
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
            <Slider
              className="w-20"
              value={[volume]}
              min={0}
              max={1}
              step={0.01}
              onValueChange={(value) => setVolume(value[0] ?? 0)}
              aria-label={t('compareVideo.volume')}
            />
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
            {hasOutput ? (
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
