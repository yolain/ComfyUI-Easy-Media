import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, Check, RotateCcw, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useT } from '@/lib/i18n'
import { isValidPanoramaView, normalizePanoramaView } from '@/lib/panorama-camera'
import type { MultiTrackPanoramaView } from '@/types/multitrack'
import { PanoramaCanvas } from './PanoramaCanvas'

interface PanoramaViewerOverlayProps {
  imageUrl: string
  savedView?: unknown
  onPanoramaViewChange: (view: MultiTrackPanoramaView | undefined) => void
  onExit: () => void
}

interface AxisVector {
  x: number
  y: number
  z: number
}

function projectAxisToHud(
  axis: AxisVector,
  yaw: number,
  pitch: number,
  origin: number,
): { x: number; y: number } {
  const yawRadians = yaw * Math.PI / 180
  const pitchRadians = pitch * Math.PI / 180
  const sinYaw = Math.sin(yawRadians)
  const cosYaw = Math.cos(yawRadians)
  const sinPitch = Math.sin(pitchRadians)
  const cosPitch = Math.cos(pitchRadians)
  const right = { x: -sinYaw, y: 0, z: cosYaw }
  const up = {
    x: -sinPitch * cosYaw,
    y: cosPitch,
    z: -sinPitch * sinYaw,
  }
  const screenX = axis.x * right.x + axis.y * right.y + axis.z * right.z
  const screenY = axis.x * up.x + axis.y * up.y + axis.z * up.z

  return {
    x: origin + screenX * 25,
    y: origin - screenY * 25,
  }
}

function axisLabelPosition(axisEnd: { x: number; y: number }, origin: number): { x: number; y: number } {
  const deltaX = axisEnd.x - origin
  const deltaY = axisEnd.y - origin
  const length = Math.hypot(deltaX, deltaY)
  if (length < 0.5) return { x: axisEnd.x + 5, y: axisEnd.y - 5 }
  return {
    x: axisEnd.x + deltaX / length * 6,
    y: axisEnd.y + deltaY / length * 6,
  }
}

export function PanoramaViewerOverlay({
  imageUrl,
  savedView,
  onPanoramaViewChange,
  onExit,
}: Readonly<PanoramaViewerOverlayProps>) {
  const t = useT()
  const savedViewIsValid = savedView === undefined || isValidPanoramaView(savedView)
  const [draft, setDraft] = useState(() => normalizePanoramaView(savedView))
  const [aspectRatio, setAspectRatio] = useState(() => draft.aspect_ratio)
  const [loading, setLoading] = useState(true)
  const [renderError, setRenderError] = useState<string | null>(null)

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onExit()
    }
    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target
      if (target instanceof Element && target.closest('[data-multitrack-preview-area]')) return
      onExit()
    }
    document.addEventListener('keydown', handleKeyDown)
    document.addEventListener('pointerdown', handlePointerDown)
    return () => {
      document.removeEventListener('keydown', handleKeyDown)
      document.removeEventListener('pointerdown', handlePointerDown)
    }
  }, [onExit])

  const handleLoad = useCallback(() => {
    setLoading(false)
  }, [])

  const handleRenderError = useCallback((error: Error) => {
    setLoading(false)
    setRenderError(error.message.includes('WebGL')
      ? t('panorama.webglUnavailable')
      : t('panorama.textureLoadFailed'))
  }, [t])

  const invalidViewMessage = savedViewIsValid ? null : t('panorama.invalidView')
  const axisOrigin = 38
  const xAxis = projectAxisToHud({ x: 1, y: 0, z: 0 }, draft.yaw, draft.pitch, axisOrigin)
  const yAxis = projectAxisToHud({ x: 0, y: 1, z: 0 }, draft.yaw, draft.pitch, axisOrigin)
  const zAxis = projectAxisToHud({ x: 0, y: 0, z: 1 }, draft.yaw, draft.pitch, axisOrigin)
  const xLabel = axisLabelPosition(xAxis, axisOrigin)
  const yLabel = axisLabelPosition(yAxis, axisOrigin)
  const zLabel = axisLabelPosition(zAxis, axisOrigin)

  const applyView = () => {
    const applied = normalizePanoramaView({ ...draft, aspect_ratio: aspectRatio })
    onPanoramaViewChange(applied)
    onExit()
  }

  const cancelPanorama = () => {
    onPanoramaViewChange(undefined)
    onExit()
  }

  return (
    <div className="absolute inset-0 z-30 overflow-hidden bg-black" data-testid="panorama-viewer-overlay">
      <PanoramaCanvas
        imageUrl={imageUrl}
        view={draft}
        onViewChange={setDraft}
        onAspectRatioChange={setAspectRatio}
        onLoad={handleLoad}
        onError={handleRenderError}
      />
      <div className="absolute left-2 top-2 z-20 flex gap-1 rounded-md border border-border bg-background/70 p-1 shadow-lg backdrop-blur-sm">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button type="button" size="icon" variant="ghost" className="h-8 w-8" aria-label={t('panorama.exit')} onClick={onExit}>
              <X />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('panorama.exit')}</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button type="button" size="icon" variant="ghost" className="h-8 w-8" aria-label={t('panorama.applyView')} disabled={renderError !== null} onClick={applyView}>
              <Check />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('panorama.applyView')}</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button type="button" size="icon" variant="ghost" className="h-8 w-8" aria-label={t('panorama.cancel')} onClick={cancelPanorama}>
              <RotateCcw />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('panorama.cancelHint')}</TooltipContent>
        </Tooltip>
      </div>
      {(invalidViewMessage || renderError) && (
        <div className="absolute left-1/2 top-3 z-10 flex -translate-x-1/2 items-center gap-2 rounded-md border border-border bg-background/70 px-3 py-2 text-xs text-destructive shadow backdrop-blur-sm">
          <AlertCircle className="h-4 w-4" />
          <span>{renderError ?? invalidViewMessage}</span>
        </div>
      )}
      {loading && !renderError && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-xs text-muted-foreground">
          {t('panorama.loading')}
        </div>
      )}
      <div
        data-testid="panorama-orientation-hud"
        className="pointer-events-none absolute bottom-2 right-2 z-20 w-32 rounded-xl border border-border bg-background/70 p-2 text-foreground shadow-lg backdrop-blur-sm"
      >
        <div data-testid="panorama-axis-stage" className="aspect-square rounded-lg bg-black/60 p-1">
          <svg viewBox="0 0 76 76" className="h-full w-full" aria-hidden="true">
            <line data-testid="panorama-axis-x" x1={axisOrigin} y1={axisOrigin} x2={xAxis.x} y2={xAxis.y} style={{ stroke: 'var(--destructive)' }} strokeWidth="2" strokeLinecap="round" />
            <line data-testid="panorama-axis-y" x1={axisOrigin} y1={axisOrigin} x2={yAxis.x} y2={yAxis.y} style={{ stroke: 'var(--highlight)' }} strokeWidth="2" strokeLinecap="round" />
            <line data-testid="panorama-axis-z" x1={axisOrigin} y1={axisOrigin} x2={zAxis.x} y2={zAxis.y} style={{ stroke: 'var(--panorama-axis-z)' }} strokeWidth="2" strokeLinecap="round" />
            <text x={xLabel.x} y={xLabel.y} dominantBaseline="middle" textAnchor="middle" style={{ fill: 'var(--destructive)' }} className="text-[9px] font-semibold">X</text>
            <text x={yLabel.x} y={yLabel.y} dominantBaseline="middle" textAnchor="middle" style={{ fill: 'var(--highlight)' }} className="text-[9px] font-semibold">Y</text>
            <text x={zLabel.x} y={zLabel.y} dominantBaseline="middle" textAnchor="middle" style={{ fill: 'var(--panorama-axis-z)' }} className="text-[9px] font-semibold">Z</text>
          </svg>
        </div>
        <div className="mt-1 space-y-0.5 text-center text-xs leading-tight tabular-nums">
          <div>{t('panorama.orientation', { yaw: Math.round(draft.yaw), pitch: Math.round(draft.pitch) })}</div>
          <div>{t('panorama.zoom', { fov: Math.round(draft.hfov), ratio: (60 / draft.hfov).toFixed(2) })}</div>
        </div>
      </div>
    </div>
  )
}
