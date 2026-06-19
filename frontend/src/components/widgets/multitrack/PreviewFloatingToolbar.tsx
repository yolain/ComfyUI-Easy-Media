import { Gauge, Volume2, VolumeX } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import { useT } from '@/lib/i18n'
import { MULTITRACK_DEFAULT_FRAME_RATE, MULTITRACK_FRAME_RATE_OPTIONS } from '@/lib/multitrack-utils'
import type { MultiTrackSegmentContent, TrackData } from '@/types/multitrack'

interface PreviewFloatingToolbarProps {
  globalMuted: boolean
  globalVolume: number
  frameRate: number
  selectedVideoVolume: number | null
  selectedVideoDuration: number | null
  onGlobalSettingsChange: (patch: Partial<Pick<TrackData, 'muted' | 'volume' | 'frame_rate'>>) => void
  onSelectedSegmentContentChange: (patch: Partial<MultiTrackSegmentContent>) => void
  onSelectedSegmentDurationChange: (duration: number) => void
}

function clampVolume(value: number): number {
  return Math.max(0, Math.min(1, value))
}

function parsePositiveNumber(value: string): number | null {
  const parsed = Number.parseFloat(value)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null
}

function nearestFrameRateIndex(frameRate: number): number {
  const exactIndex = MULTITRACK_FRAME_RATE_OPTIONS.findIndex((option) => option === frameRate)
  if (exactIndex >= 0) return exactIndex
  const fallbackIndex = MULTITRACK_FRAME_RATE_OPTIONS.findIndex((option) => option === MULTITRACK_DEFAULT_FRAME_RATE)
  if (!Number.isFinite(frameRate) || frameRate <= 0) return Math.max(0, fallbackIndex)
  return MULTITRACK_FRAME_RATE_OPTIONS.reduce((bestIndex, option, index) => {
    const best = MULTITRACK_FRAME_RATE_OPTIONS[bestIndex] ?? MULTITRACK_DEFAULT_FRAME_RATE
    return Math.abs(option - frameRate) < Math.abs(best - frameRate) ? index : bestIndex
  }, Math.max(0, fallbackIndex))
}

export function PreviewFloatingToolbar({
  globalMuted,
  globalVolume,
  frameRate,
  selectedVideoVolume,
  selectedVideoDuration,
  onGlobalSettingsChange,
  onSelectedSegmentContentChange,
  onSelectedSegmentDurationChange,
}: Readonly<PreviewFloatingToolbarProps>) {
  const t = useT()
  const hasSelectedVideo = selectedVideoVolume !== null
  const effectiveVolume = hasSelectedVideo ? selectedVideoVolume : globalVolume
  const muted = hasSelectedVideo ? effectiveVolume <= 0 : globalMuted

  function updateVolume(value: number) {
    const nextVolume = clampVolume(value)
    if (hasSelectedVideo) {
      onSelectedSegmentContentChange({ volume: nextVolume })
      return
    }
    onGlobalSettingsChange({ volume: nextVolume, muted: nextVolume <= 0 })
  }

  function toggleMute() {
    if (hasSelectedVideo) {
      onSelectedSegmentContentChange({ volume: muted ? 1 : 0 })
      return
    }
    onGlobalSettingsChange({ muted: !globalMuted })
  }

  function updateFrameRateIndex(index: number) {
    const nextIndex = Math.max(0, Math.min(MULTITRACK_FRAME_RATE_OPTIONS.length - 1, Math.round(index)))
    const nextFrameRate = MULTITRACK_FRAME_RATE_OPTIONS[nextIndex] ?? MULTITRACK_DEFAULT_FRAME_RATE
    if (nextFrameRate !== frameRate) onGlobalSettingsChange({ frame_rate: nextFrameRate })
  }

  const frameRateIndex = nearestFrameRateIndex(frameRate)

  return (
    <div className="absolute right-3 top-1/2 z-20 flex -translate-y-1/2 flex-col overflow-hidden rounded-lg border border-border bg-popover/90 text-popover-foreground shadow-lg backdrop-blur">
      <Popover>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            className="flex h-12 w-10 flex-col gap-1 rounded-none px-1 py-1 cursor-pointer"
            aria-label={t('multitrack.audioSettings')}
          >
            {muted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
            <span className="text-[8px]">{t('multitrack.audio')}</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-56 space-y-3" align="end" side="left">
          <div className="flex items-center justify-between gap-3 cursor-pointer">
            <span className="text-sm font-medium">{t('multitrack.audio')}</span>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              aria-label={muted ? t('multitrack.unmutePreviewAudio') : t('multitrack.mutePreviewAudio')}
              onClick={toggleMute}
            >
              {muted ? t('multitrack.unmute') : t('multitrack.mute')}
            </Button>
          </div>
          <label className="grid gap-2 text-xs text-muted-foreground">
            {t('multitrack.volume')}
            <Input
              aria-label={t('multitrack.volume')}
              type="number"
              min={0}
              max={1}
              step={0.01}
              value={effectiveVolume}
              onChange={(event) => {
                const parsed = Number.parseFloat(event.currentTarget.value)
                if (!Number.isNaN(parsed)) updateVolume(parsed)
              }}
            />
          </label>
          <Slider
            value={[clampVolume(effectiveVolume)]}
            min={0}
            max={1}
            step={0.01}
            onValueChange={(value) => updateVolume(value[0] ?? 0)}
          />
        </PopoverContent>
      </Popover>
      <div className="mx-3 h-px bg-border" />
      <Popover>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            className="flex h-12 w-10 flex-col gap-1 rounded-none px-1 py-1 cursor-pointer"
            aria-label={t('multitrack.speedSettings')}
          >
            <Gauge className="h-4 w-4" />
            <span className="text-[8px]">{t('multitrack.speed')}</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-72 space-y-4" align="end" side="left">
          <div className="grid gap-3 text-xs text-muted-foreground">
            <div className="flex items-center justify-between">
              <span>{t('multitrack.frameRate')}</span>
              <span className="tabular-nums text-foreground">
                {MULTITRACK_FRAME_RATE_OPTIONS[frameRateIndex] ?? MULTITRACK_DEFAULT_FRAME_RATE} {t('multitrack.fps')}
              </span>
            </div>
            <Slider
              aria-label={t('multitrack.frameRate')}
              value={[frameRateIndex]}
              min={0}
              max={MULTITRACK_FRAME_RATE_OPTIONS.length - 1}
              step={1}
              ticks={MULTITRACK_FRAME_RATE_OPTIONS.map((fps, index) => ({
                value: index,
                label: String(fps),
              }))}
              onValueChange={(value) => updateFrameRateIndex(value[0] ?? frameRateIndex)}
            />
            {selectedVideoDuration !== null && (
              <label className="grid gap-2 pt-2">
                {t('multitrack.duration')}
                <Input
                  aria-label={t('multitrack.duration')}
                  type="number"
                  min={0.01}
                  step={0.01}
                  value={selectedVideoDuration}
                  onChange={(event) => {
                    const duration = parsePositiveNumber(event.currentTarget.value)
                    if (duration !== null) onSelectedSegmentDurationChange(duration)
                  }}
                />
              </label>
            )}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}
