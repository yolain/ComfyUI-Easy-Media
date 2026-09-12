import { useEffect, useMemo, useRef, useState } from 'react'
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import { Eye, EyeOff } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  useSamplingPreviewFrames,
  useSamplingPreviewPlayback,
  type SamplingPreviewFrame,
} from '@/hooks/use-sampling-preview-progress'
import { LocaleContext, useT } from '@/lib/i18n'

export const SAMPLING_PREVIEW_EVENT = 'easy_media.sampling_preview'
const EXECUTION_START_EVENT = 'execution_start'

export interface SamplingPreviewPayload {
  node_id: string
  display_node_id?: string
  prompt_id: string | null
  image: string
  images?: string[]
  mime: string
  step: number
  total: number
  fps: number | null
  frame_count: number
  segment_index: number
  sampling_pass: string
}

interface SamplingPreviewWidgetProps {
  app: ComfyApp
  nodeId: string | number | (() => string | number)
  executionRevision?: number
  onVisibilityChange?: (visible: boolean) => void
  interactive?: boolean
  visibilityControlOnly?: boolean
}

function terminalNodeId(value: string | number | undefined) {
  return value === undefined ? '' : String(value).split(':').at(-1)?.split('.').at(-1)
}

function eventTargetsNode(
  payload: SamplingPreviewPayload,
  nodeId: string | number | (() => string | number),
) {
  const currentNodeId = typeof nodeId === 'function' ? nodeId() : nodeId
  return [payload.display_node_id, payload.node_id]
    .some((candidate) => terminalNodeId(candidate) === String(currentNodeId))
}

function previewBlob(image: string, mime: string) {
  const binary = atob(image)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return new Blob([bytes], { type: mime || 'image/jpeg' })
}

function PreviewCanvas({ frame, label }: { frame: SamplingPreviewFrame, label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    canvas.width = frame.bitmap.width
    canvas.height = frame.bitmap.height
    const context = canvas.getContext('2d')
    if (!context) {
      console.error('[SamplingPreviewWidget] 2D canvas context is unavailable')
      return
    }
    context.drawImage(frame.bitmap, 0, 0)
  }, [frame])

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={label}
      className="absolute inset-0 h-full w-full object-contain"
    />
  )
}

function SamplingPreviewWidgetInner({
  app,
  nodeId,
  executionRevision = 0,
  onVisibilityChange,
  interactive = true,
  visibilityControlOnly = false,
}: SamplingPreviewWidgetProps) {
  const t = useT()
  const [preview, setPreview] = useState<SamplingPreviewPayload | null>(null)
  const [pausedPreview, setPausedPreview] = useState<SamplingPreviewPayload | null>(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [legacyPreviewUrls, setLegacyPreviewUrls] = useState<string[]>([])
  const [isPlaying, setIsPlaying] = useState(true)
  const [isPreviewVisible, setIsPreviewVisible] = useState(true)
  const [frozenFallbackUrl, setFrozenFallbackUrl] = useState('')
  const fallbackImageRef = useRef<HTMLImageElement>(null)
  const frozenFallbackCanvasRef = useRef<HTMLCanvasElement>(null)
  const activePromptIdRef = useRef<string | null>(null)
  const playbackInteractive = interactive && !visibilityControlOnly
  const visibilityInteractive = interactive || visibilityControlOnly
  const displayedPreview = isPlaying ? preview : (pausedPreview ?? preview)

  useEffect(() => {
    const handleExecutionStart = (event: CustomEvent<unknown>) => {
      const detail = event.detail
      activePromptIdRef.current = detail && typeof detail === 'object'
        && 'prompt_id' in detail && typeof detail.prompt_id === 'string'
        ? detail.prompt_id
        : null
    }
    const handlePreview = (event: CustomEvent<unknown>) => {
      const payload = event.detail as SamplingPreviewPayload | undefined
      if (!payload?.image || !eventTargetsNode(payload, nodeId)) return
      if (
        activePromptIdRef.current !== null
        && payload.prompt_id !== activePromptIdRef.current
      ) return
      setPreview(payload)
    }
    app.api.addCustomEventListener(EXECUTION_START_EVENT, handleExecutionStart)
    app.api.addCustomEventListener(SAMPLING_PREVIEW_EVENT, handlePreview)
    return () => {
      app.api.removeCustomEventListener(EXECUTION_START_EVENT, handleExecutionStart)
      app.api.removeCustomEventListener(SAMPLING_PREVIEW_EVENT, handlePreview)
    }
  }, [app.api, nodeId])

  useEffect(() => {
    setPreview(null)
    setPausedPreview(null)
    setIsPlaying(true)
    setFrozenFallbackUrl('')
  }, [executionRevision])

  useEffect(() => {
    onVisibilityChange?.(preview !== null)
  }, [onVisibilityChange, preview])

  const previewData = useMemo(
    () => displayedPreview ? previewBlob(displayedPreview.image, displayedPreview.mime) : null,
    [displayedPreview],
  )

  useEffect(() => {
    if (!previewData) {
      setPreviewUrl('')
      return
    }
    const nextUrl = URL.createObjectURL(previewData)
    setPreviewUrl(nextUrl)
    return () => URL.revokeObjectURL(nextUrl)
  }, [previewData])

  useEffect(() => {
    const legacyImages = displayedPreview?.images ?? []
    if (legacyImages.length === 0) {
      setLegacyPreviewUrls([])
      return
    }
    const nextUrls = legacyImages.map((image) => URL.createObjectURL(
      previewBlob(image, 'image/jpeg'),
    ))
    setLegacyPreviewUrls(nextUrls)
    return () => nextUrls.forEach((url) => URL.revokeObjectURL(url))
  }, [displayedPreview])

  const decodedFrames = useSamplingPreviewFrames(
    previewData,
    displayedPreview?.image,
    playbackInteractive,
  )
  const playbackFrameCount = legacyPreviewUrls.length
    || displayedPreview?.frame_count
    || decodedFrames.length
  const playbackRevision = displayedPreview
    ? `${displayedPreview.segment_index}:${displayedPreview.sampling_pass}:${displayedPreview.frame_count}:${displayedPreview.fps}`
    : undefined

  const { frameIndex, progress: playbackProgress } = useSamplingPreviewPlayback(
    playbackRevision,
    playbackFrameCount,
    displayedPreview?.fps ?? null,
    isPlaying,
  )

  if (!displayedPreview) return null
  const passLabel = t(`samplingPreview.${displayedPreview.sampling_pass}`)
  const decodedFrameIndex = decodedFrames.length <= 1 || playbackFrameCount <= 1
    ? 0
    : Math.min(
      decodedFrames.length - 1,
      Math.floor(frameIndex / playbackFrameCount * decodedFrames.length),
    )
  const decodedFrame = decodedFrames[decodedFrameIndex]
  const legacyPreviewUrl = legacyPreviewUrls[
    Math.min(frameIndex, legacyPreviewUrls.length - 1)
  ]
  const usesAnimatedImageFallback = (
    displayedPreview.mime === 'image/webp'
    && legacyPreviewUrls.length === 0
    && decodedFrame === undefined
  )
  const freezeAnimatedFallback = () => {
    if (!usesAnimatedImageFallback) return
    const image = fallbackImageRef.current
    const canvas = frozenFallbackCanvasRef.current
    if (!image?.complete || image.naturalWidth <= 0 || image.naturalHeight <= 0 || !canvas) return
    canvas.width = image.naturalWidth
    canvas.height = image.naturalHeight
    const context = canvas.getContext('2d')
    if (!context) {
      console.error('[SamplingPreviewWidget] 2D canvas context is unavailable')
      return
    }
    context.drawImage(image, 0, 0)
    setFrozenFallbackUrl(previewUrl)
  }
  const togglePlayback = () => {
    if (isPlaying) {
      freezeAnimatedFallback()
      setPausedPreview(preview)
      setIsPlaying(false)
    } else {
      setPausedPreview(null)
      setIsPlaying(true)
    }
  }
  const playbackLabel = t(isPlaying ? 'samplingPreview.pausePreview' : 'samplingPreview.resumePreview')
  const metaLabel = t('samplingPreview.meta', {
    segment: displayedPreview.segment_index + 1,
    pass: passLabel,
    step: displayedPreview.step,
    total: displayedPreview.total,
  })
  const progressBar = (
    <div
      role="progressbar"
      aria-label={t('samplingPreview.playbackProgress')}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(playbackProgress)}
      className="h-1.5 w-full overflow-hidden rounded-full bg-muted"
    >
      <div
        className="h-full rounded-full bg-primary"
        style={{ width: `${playbackProgress}%` }}
      />
    </div>
  )

  return (
    <div className="relative flex h-full w-full flex-col overflow-hidden rounded-lg border border-border bg-black">
      {isPreviewVisible ? (
        playbackInteractive ? (
          <Button
            type="button"
            variant="ghost"
            aria-label={playbackLabel}
            onClick={togglePlayback}
            className="relative min-h-0 flex-1 rounded-none p-0 hover:bg-black"
          >
            {legacyPreviewUrl ? (
              <img
                src={legacyPreviewUrl}
                alt={t('samplingPreview.title')}
                draggable={false}
                className="absolute inset-0 h-full w-full object-contain"
              />
            ) : decodedFrame ? (
              <PreviewCanvas
                frame={decodedFrame}
                label={t('samplingPreview.title')}
              />
            ) : previewUrl ? (
              <>
                <img
                  ref={fallbackImageRef}
                  src={previewUrl}
                  alt={t('samplingPreview.title')}
                  draggable={false}
                  onLoad={() => {
                    if (!isPlaying && !frozenFallbackUrl) freezeAnimatedFallback()
                  }}
                  className="absolute inset-0 h-full w-full object-contain"
                />
                <canvas
                  ref={frozenFallbackCanvasRef}
                  data-testid="sampling-preview-frozen-frame"
                  aria-hidden="true"
                  hidden={isPlaying || !frozenFallbackUrl}
                  className="absolute inset-0 h-full w-full object-contain"
                />
              </>
            ) : null}
          </Button>
        ) : (
          <div className="relative min-h-0 flex-1">
            {legacyPreviewUrl ? (
              <img
                src={legacyPreviewUrl}
                alt={t('samplingPreview.title')}
                draggable={false}
                className="absolute inset-0 h-full w-full object-contain"
              />
            ) : previewUrl ? (
              <img
                src={previewUrl}
                alt={t('samplingPreview.title')}
                draggable={false}
                className="absolute inset-0 h-full w-full object-contain"
              />
            ) : null}
          </div>
        )
      ) : (
        <div
          data-testid="sampling-preview-placeholder"
          aria-hidden="true"
          className="min-h-0 flex-1 bg-black"
        />
      )}
      <div className="shrink-0 border-t border-border bg-background px-2 py-1 gap-0">
        <div className="flex items-center justify-between text-[10px] text-muted-foreground">
          <span>{metaLabel}</span>
          {visibilityInteractive ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              aria-label={t(
                isPreviewVisible
                  ? 'samplingPreview.hidePreview'
                  : 'samplingPreview.showPreview',
              )}
              onClick={() => setIsPreviewVisible((visible) => !visible)}
              className="pointer-events-auto h-6 w-6"
            >
              {isPreviewVisible ? <EyeOff /> : <Eye />}
            </Button>
          ) : null}
        </div>
        {playbackInteractive ? (
          <Button
            type="button"
            variant="ghost"
            aria-label={playbackLabel}
            onClick={togglePlayback}
            className="h-3 w-full rounded-full p-0 hover:bg-muted"
          >
            {progressBar}
          </Button>
        ) : (
          <div className="h-3 w-full rounded-full p-0">{progressBar}</div>
        )}
      </div>
    </div>
  )
}

export function SamplingPreviewWidget(props: SamplingPreviewWidgetProps) {
  const locale = props.app?.ui?.settings?.settingsValues?.['Comfy.Locale']
  return (
    <LocaleContext.Provider value={locale}>
      <SamplingPreviewWidgetInner {...props} />
    </LocaleContext.Provider>
  )
}
