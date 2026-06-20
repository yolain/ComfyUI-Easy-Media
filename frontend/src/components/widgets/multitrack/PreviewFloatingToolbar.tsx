import { useEffect, useState } from 'react'
import { Gauge, Volume2, VolumeX } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { NumberInput } from '@/components/ui/number-input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Slider } from '@/components/ui/slider'
import { useT } from '@/lib/i18n'
import {
  formatMultiTrackDurationTimecode,
  MULTITRACK_DEFAULT_FRAME_RATE,
  MULTITRACK_FRAME_RATE_OPTIONS,
  MULTITRACK_MAX_VOLUME_DB,
  MULTITRACK_MIN_VOLUME_DB,
  clampMultiTrackVolumeDb,
  parseMultiTrackDurationTimecode,
} from '@/lib/multitrack-utils'
import type { MultiTrackSegmentContent, TrackData } from '@/types/multitrack'

interface PreviewFloatingToolbarProps {
  globalMuted: boolean
  globalVolumeDb: number
  frameRate: number
  selectedMediaVolumeDb: number | null
  selectedMediaMuted: boolean
  selectedMediaDuration: number | null
  onGlobalSettingsChange: (patch: Partial<Pick<TrackData, 'muted' | 'volume_db' | 'frame_rate'>>) => void
  onSelectedSegmentContentChange: (patch: Partial<MultiTrackSegmentContent>) => void
  onSelectedSegmentDurationChange: (duration: number) => void
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
  globalVolumeDb,
  frameRate,
  selectedMediaVolumeDb,
  selectedMediaMuted,
  selectedMediaDuration,
  onGlobalSettingsChange,
  onSelectedSegmentContentChange,
  onSelectedSegmentDurationChange,
}: Readonly<PreviewFloatingToolbarProps>) {
  const t = useT()
  const hasSelectedMedia = selectedMediaVolumeDb !== null
  const audioLabel = t(hasSelectedMedia ? 'multitrack.audio' : 'multitrack.globalAudio')
  const effectiveVolumeDb = hasSelectedMedia ? selectedMediaVolumeDb : globalVolumeDb
  const muted = hasSelectedMedia ? selectedMediaMuted : globalMuted
  const formattedDuration = selectedMediaDuration === null
    ? ''
    : formatMultiTrackDurationTimecode(selectedMediaDuration, frameRate)
  const [durationInput, setDurationInput] = useState(formattedDuration)

  useEffect(() => {
    setDurationInput(formattedDuration)
  }, [formattedDuration])

  function updateVolumeDb(value: number) {
    const nextVolumeDb = clampMultiTrackVolumeDb(value)
    if (hasSelectedMedia) {
      onSelectedSegmentContentChange({ volume_db: nextVolumeDb })
      return
    }
    onGlobalSettingsChange({ volume_db: nextVolumeDb })
  }

  function toggleMute() {
    if (hasSelectedMedia) {
      onSelectedSegmentContentChange({ muted: !muted })
      return
    }
    onGlobalSettingsChange({ muted: !globalMuted })
  }

  function updateFrameRateIndex(index: number) {
    const nextIndex = Math.max(0, Math.min(MULTITRACK_FRAME_RATE_OPTIONS.length - 1, Math.round(index)))
    const nextFrameRate = MULTITRACK_FRAME_RATE_OPTIONS[nextIndex] ?? MULTITRACK_DEFAULT_FRAME_RATE
    if (nextFrameRate !== frameRate) onGlobalSettingsChange({ frame_rate: nextFrameRate })
  }

  function commitDuration() {
    const duration = parseMultiTrackDurationTimecode(durationInput, frameRate)
    if (duration === null) {
      setDurationInput(formattedDuration)
      return
    }
    setDurationInput(formatMultiTrackDurationTimecode(duration, frameRate))
    if (duration !== selectedMediaDuration) onSelectedSegmentDurationChange(duration)
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
            <span className="text-[8px]">{audioLabel}</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-56 space-y-3" align="end" side="left">
          <div className="flex items-center justify-between gap-3 cursor-pointer">
            <span className="text-sm font-medium">{audioLabel}</span>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              className="cursor-pointer"
              aria-label={muted ? t('multitrack.unmutePreviewAudio') : t('multitrack.mutePreviewAudio')}
              onClick={toggleMute}
            >
              {muted ? t('multitrack.unmute') : t('multitrack.mute')}
            </Button>
          </div>
          <div className="grid gap-2 text-xs text-muted-foreground">
            <span className="flex items-center justify-between">
              <span>{t('multitrack.volume')}</span>
              <span className="tabular-nums text-foreground">{effectiveVolumeDb.toFixed(1)} dB</span>
            </span>
            <NumberInput
              aria-label={t('multitrack.volume')}
              min={MULTITRACK_MIN_VOLUME_DB}
              max={MULTITRACK_MAX_VOLUME_DB}
              step={0.1}
              className="cursor-pointer"
              value={effectiveVolumeDb}
              onChange={updateVolumeDb}
            />
          </div>
          <Slider
            aria-label={t('multitrack.volume')}
            className="cursor-pointer"
            value={[clampMultiTrackVolumeDb(effectiveVolumeDb)]}
            min={MULTITRACK_MIN_VOLUME_DB}
            max={MULTITRACK_MAX_VOLUME_DB}
            step={0.1}
            onValueChange={(value) => updateVolumeDb(value[0] ?? 0)}
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
            {selectedMediaDuration !== null && (
              <label className="grid gap-2 pt-2">
                {t('multitrack.duration')}
                <Input
                  aria-label={t('multitrack.duration')}
                  type="text"
                  inputMode="numeric"
                  placeholder="00:00:00"
                  className="tabular-nums"
                  value={durationInput}
                  onChange={(event) => setDurationInput(event.currentTarget.value)}
                  onBlur={commitDuration}
                  onKeyDown={(event) => {
                    if (event.key !== 'Enter') return
                    event.preventDefault()
                    commitDuration()
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
