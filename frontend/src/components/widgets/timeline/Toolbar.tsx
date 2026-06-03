import { useEffect } from 'react'
import { Select, SelectContent, SelectItem, SelectTrigger } from '@/components/ui/select'
import { NumberInput } from '@/components/ui/number-input'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { Play, Pause, Trash2, ZoomOut } from 'lucide-react'
import type { TimeDisplayFormat, MaintainSegment } from '@/types/timeline'
import { framesToSeconds, secondsToFrames } from '@/lib/timeline-utils'
import { useT } from '@/lib/i18n'

interface ToolbarProps {
  totalLength: number
  frameRate: number
  displayFormat: TimeDisplayFormat
  onTotalLengthChange: (frames: number) => void
  onFrameRateChange: (fps: number) => void
  onDisplayFormatChange: (format: TimeDisplayFormat) => void
  /** Whether playback is currently active */
  isPlaying: boolean
  onPlayPause: () => void
  /** Whether a segment is currently selected (enables delete) */
  hasSelection: boolean
  /** Whether the delete action is allowed (e.g., false when only one segment remains) */
  canDelete?: boolean
  onDeleteSelected: () => void
  /** Zoom level, 1 = fit to container */
  zoom: number
  onZoomChange: (zoom: number) => void
  /** Currently selected segment (for editing its duration) */
  selectedSegment: MaintainSegment | null
  /** Called when selected segment duration changes */
  onSelectedSegmentDurationChange: (segmentId: string, newDuration: number) => void
}

function getZoomConfig(totalLength: number, frameRate: number): { max: number; step: number } {
  const seconds = framesToSeconds(Math.max(totalLength, 1), Math.max(frameRate, 1))

  if (seconds <= 10) return { max: 2, step: 0.05 }
  if (seconds <= 30) return { max: 3, step: 0.05 }
  if (seconds <= 60) return { max: 4, step: 0.1 }
  if (seconds <= 120) return { max: 6, step: 0.1 }
  if (seconds <= 300) return { max: 8, step: 0.2 }
  return { max: 10, step: 0.25 }
}

export function Toolbar({
  totalLength,
  frameRate,
  displayFormat,
  onTotalLengthChange,
  onFrameRateChange,
  onDisplayFormatChange,
  isPlaying,
  onPlayPause,
  hasSelection,
  canDelete = hasSelection,
  onDeleteSelected,
  zoom,
  onZoomChange,
  selectedSegment,
  onSelectedSegmentDurationChange,
}: Readonly<ToolbarProps>) {
  const t = useT()
  const zoomConfig = getZoomConfig(totalLength, frameRate)

  useEffect(() => {
    if (zoom > zoomConfig.max) {
      onZoomChange(zoomConfig.max)
    }
  }, [zoom, zoomConfig.max, onZoomChange])

  // Duration for selected segment or total length
  const segmentDuration = selectedSegment
    ? selectedSegment.end_frame - selectedSegment.start_frame + 1
    : 0

  let durationValue: number
  if (selectedSegment) {
    durationValue = displayFormat === 'seconds'
      ? Math.round(framesToSeconds(segmentDuration, frameRate) * 100) / 100
      : segmentDuration
  } else {
    durationValue = displayFormat === 'seconds'
      ? Math.round(framesToSeconds(totalLength, frameRate) * 100) / 100
      : totalLength
  }

  function handleDurationChange(val: number) {
    if (val <= 0) return
    let frames =
      displayFormat === 'seconds' ? secondsToFrames(val, frameRate) : Math.round(val)

    if (displayFormat === 'seconds') {
      const currentFrames = selectedSegment
        ? selectedSegment.end_frame - selectedSegment.start_frame + 1
        : totalLength
      if (frames === currentFrames) {
        // val changed but didn't cross a frame boundary — step in the intended direction
        const currentSec = framesToSeconds(currentFrames, frameRate)
        if (val < currentSec) {
          frames = Math.max(1, currentFrames - 4)
        } else if (val > currentSec) {
          frames = currentFrames + 4
        } else {
          return
        }
      }
    }

    if (selectedSegment) {
      // Edit selected segment duration - other segments shift accordingly
      const oldDuration = selectedSegment.end_frame - selectedSegment.start_frame + 1
      const delta = frames - oldDuration
      if (delta === 0) return

      // Call parent to handle segment duration change with all segment positions shifting
      onSelectedSegmentDurationChange(selectedSegment.id, frames)
    } else {
      onTotalLengthChange(frames)
    }
  }

  function handleFpsChange(val: number) {
    if (val > 0) onFrameRateChange(Math.round(val))
  }

  return (
    <div className="flex justify-between items-center gap-2 py-1 border-b border-border text-xs shrink-0">
      {/* Left — playback controls */}
      <div className="flex items-center gap-0.5">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={onPlayPause}
              aria-label={isPlaying ? t('toolbar.pause') : t('toolbar.play')}
            >
              {isPlaying ? <Pause className="w-2 h-2" /> : <Play className="w-2 h-2" />}
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            {isPlaying ? t('toolbar.pause') : t('toolbar.play')}
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={onDeleteSelected}
              disabled={!canDelete}
              aria-label={t('toolbar.deleteSegment')}
            >
              <Trash2 className="w-2 h-2" />
            </Button>
          </TooltipTrigger>
          <TooltipContent side="bottom">{t('toolbar.deleteSegment')}</TooltipContent>
        </Tooltip>
      </div>

      {/* Center — duration / fps / zoom */}
      <div className="flex items-center gap-2" data-time-toolbar>
        <div className="flex items-center gap-1">
          <NumberInput
            value={durationValue}
            onChange={handleDurationChange}
            min={1}
            max={2000}
            step={displayFormat === 'seconds' ? 1 : 4}
            commitOnBlur={true}
            className="h-6 w-18"
          />
          <Select
            value={displayFormat}
            onValueChange={(v) => onDisplayFormatChange(v as TimeDisplayFormat)}
          >
            <SelectTrigger className="h-6 w-10 text-xs">
              {(displayFormat === 'frames' ? t('toolbar.frames') : t('toolbar.seconds')).substring(0, 1)}
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="frames">{t('toolbar.frames')}</SelectItem>
              <SelectItem value="seconds">{t('toolbar.seconds')}</SelectItem>
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center gap-1">
          <NumberInput
            value={frameRate}
            onChange={handleFpsChange}
            min={1}
            max={60}
            step={1}
            commitOnBlur={true}
            className="h-6 w-14"
          />
          <span className="text-muted-foreground">{t('toolbar.fps')}</span>
        </div>
      </div>
      {/* Right — zoom */}
      <div className="flex items-center gap-1 pr-1">
        {/* Zoom */}
        <div className="flex items-center gap-1">
          <Tooltip>
            <TooltipTrigger asChild>
              <ZoomOut className="w-4 h-4 text-muted-foreground shrink-0" />
            </TooltipTrigger>
            <TooltipContent side="bottom">{t('toolbar.zoom')}</TooltipContent>
          </Tooltip>
          <Slider
            min={1}
            max={zoomConfig.max}
            step={zoomConfig.step}
            value={[Math.min(zoom, zoomConfig.max)]}
            onValueChange={([v]) => onZoomChange(v)}
            className="w-12 h-3"
            aria-label={t('toolbar.zoom')}
          />
        </div>
      </div>
    </div>
  )
}
