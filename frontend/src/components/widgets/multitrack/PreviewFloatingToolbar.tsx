import { useEffect, useState, type ReactNode } from 'react'
import { Gauge, Volume2, VolumeX } from 'lucide-react'
import { Switch } from '@/components/ui/switch'
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
  selectedTrackAudioLocked?: boolean
  onSelectedTrackAudioSettingsChange?: (patch: { audio_locked?: boolean }) => void
  rightInset?: number | string
  onGlobalSettingsChange: (patch: Partial<Pick<TrackData, 'muted' | 'volume_db' | 'frame_rate'>>) => void
  onSelectedSegmentContentChange: (patch: Partial<MultiTrackSegmentContent>) => void
  onSelectedSegmentDurationChange: (duration: number) => void
}

function SettingsRow({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  return (
    <div className="flex min-h-7 items-start gap-3">
      <span className="w-20 shrink-0 text-xs text-muted-foreground" title={title}>{label}</span>
      <div className="flex min-w-0 flex-1 justify-end">{children}</div>
    </div>
  )
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
  selectedTrackAudioLocked = false,
  onSelectedTrackAudioSettingsChange,
  rightInset = 12,
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

  function updateMuted(nextMuted: boolean) {
    if (hasSelectedMedia) {
      onSelectedSegmentContentChange({ muted: nextMuted })
      return
    }
    onGlobalSettingsChange({ muted: nextMuted })
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
    <div
      className="absolute top-1/2 z-20 flex -translate-y-1/2 flex-col overflow-hidden rounded-lg border border-border bg-popover/90 text-popover-foreground shadow-lg backdrop-blur"
      style={{ right: rightInset }}
    >
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
        <PopoverContent className="w-80 space-y-3" align="end" side="left">
          <div className="text-sm font-medium">{audioLabel}</div>
          <div className="flex flex-col gap-3">
            <SettingsRow label={t('multitrack.audioMuted')}>
              <Switch
                aria-label={t('multitrack.audioMuted')}
                checked={muted}
                onCheckedChange={updateMuted}
              />
            </SettingsRow>
            <SettingsRow label={t('multitrack.volumeDb')}>
              <div className="flex w-full min-w-0 items-center gap-2">
                <div className="min-w-0 flex-1">
                  <Slider
                    aria-label={t('multitrack.volumeDb')}
                    className="cursor-pointer"
                    value={[clampMultiTrackVolumeDb(effectiveVolumeDb)]}
                    min={MULTITRACK_MIN_VOLUME_DB}
                    max={MULTITRACK_MAX_VOLUME_DB}
                    step={0.1}
                    onValueChange={(value) => updateVolumeDb(value[0] ?? 0)}
                  />
                </div>
                <NumberInput
                  aria-label={t('multitrack.volumeDb')}
                  min={MULTITRACK_MIN_VOLUME_DB}
                  max={MULTITRACK_MAX_VOLUME_DB}
                  step={0.1}
                  className="w-20 shrink-0 cursor-pointer"
                  value={effectiveVolumeDb}
                  onChange={updateVolumeDb}
                />
              </div>
            </SettingsRow>
            {hasSelectedMedia && <>
              <SettingsRow label={t('multitrack.lockAudio')} title={t('multitrack.audioLockTooltip')}>
                <Switch
                  aria-label={t('multitrack.lockAudio')}
                  checked={selectedTrackAudioLocked}
                  disabled={!onSelectedTrackAudioSettingsChange}
                  onCheckedChange={(checked) => onSelectedTrackAudioSettingsChange?.({ audio_locked: checked })}
                />
              </SettingsRow>
            </>}
          </div>
        </PopoverContent>
      </Popover>
      <div className="mx-3 h-px bg-border" />
      <Popover>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            className="flex h-12 w-10 flex-col gap-1 rounded-none px-1 py-1 cursor-pointer"
            aria-label={t('multitrack.fpsSettings')}
          >
            <Gauge className="h-4 w-4" />
            <span className="text-[8px]">{t('multitrack.frameRate')}</span>
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-80 space-y-3" align="end" side="left">
          <div className="flex flex-col gap-3">
            <SettingsRow label={t('multitrack.frameRate')}>
              <div className="w-full min-w-0 space-y-2">
                <div className="text-right text-xs tabular-nums">
                  {MULTITRACK_FRAME_RATE_OPTIONS[frameRateIndex] ?? MULTITRACK_DEFAULT_FRAME_RATE} {t('multitrack.fps')}
                </div>
                <Slider
                  aria-label={t('multitrack.frameRate')}
                  className="cursor-pointer"
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
              </div>
            </SettingsRow>
            {selectedMediaDuration !== null && (
              <SettingsRow label={t('multitrack.duration')}>
                <Input
                  aria-label={t('multitrack.duration')}
                  type="text"
                  inputMode="numeric"
                  placeholder="00:00:00"
                  className="h-7 w-28 text-right text-xs tabular-nums"
                  value={durationInput}
                  onChange={(event) => setDurationInput(event.currentTarget.value)}
                  onBlur={commitDuration}
                  onKeyDown={(event) => {
                    if (event.key !== 'Enter') return
                    event.preventDefault()
                    commitDuration()
                  }}
                />
              </SettingsRow>
            )}
          </div>
        </PopoverContent>
      </Popover>
    </div>
  )
}
