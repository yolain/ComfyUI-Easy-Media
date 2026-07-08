import type { ReactNode } from 'react'
import { Magnet, Maximize2, Minimize2, Redo2, Trash2, Undo2, ZoomOut } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { SplitCenterIcon, SplitLeftDotsIcon, SplitRightDotsIcon } from '@/components/ui/custom-lucide-icon'
import { Slider } from '@/components/ui/slider'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useT } from '@/lib/i18n'
import { formatMultiTrackTime } from '@/lib/multitrack-utils'

const TOOLBAR_ICON_BUTTON_CLASS = 'h-6 w-6 shrink-0 [&_svg]:size-3.5'
const TOOLBAR_ICON_CLASS = 'size-3.5'

interface MultiTrackToolbarProps {
  currentTime: number
  totalLength: number
  frameRate: number
  isPlaying: boolean
  zoom: number
  snapEnabled: boolean
  timelineCollapsed: boolean
  onPlayPause: () => void
  onZoomChange: (zoom: number) => void
  onSnapEnabledChange: (enabled: boolean) => void
  onToggleTimeline: () => void
  canDelete: boolean
  onDeleteSelected: () => void
  onCutAtCurrentTime: () => void
  canTrimCenter: boolean
  canTrimLeft: boolean
  canTrimRight: boolean
  onTrimLeftAtCurrentTime: () => void
  onTrimRightAtCurrentTime: () => void
  canUndo: boolean
  canRedo: boolean
  onUndo: () => void
  onRedo: () => void
}

export function MultiTrackToolbar({
  currentTime,
  totalLength,
  frameRate,
  isPlaying,
  zoom,
  snapEnabled,
  timelineCollapsed,
  onPlayPause,
  onZoomChange,
  onSnapEnabledChange,
  onToggleTimeline,
  canDelete,
  onDeleteSelected,
  onCutAtCurrentTime,
  canTrimCenter,
  canTrimLeft,
  canTrimRight,
  onTrimLeftAtCurrentTime,
  onTrimRightAtCurrentTime,
  canUndo,
  canRedo,
  onUndo,
  onRedo,
}: Readonly<MultiTrackToolbarProps>) {
  const t = useT()

  function renderTooltip(label: string, children: ReactNode) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="inline-flex shrink-0">{children}</span>
        </TooltipTrigger>
        <TooltipContent>{label}</TooltipContent>
      </Tooltip>
    )
  }

  return (
    <div className="grid h-9 min-h-9 shrink-0 grid-cols-[1fr_auto_1fr] items-center overflow-hidden border-b border-border text-[10px] leading-none">
      <div className="flex items-center">
        {renderTooltip(t('multitrack.undo'), (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className={`${TOOLBAR_ICON_BUTTON_CLASS} text-muted-foreground cursor-pointer`}
            disabled={!canUndo}
            aria-label={t('multitrack.undo')}
            onClick={(event) => {
              event.stopPropagation()
              onUndo()
            }}
          >
            <Undo2 className={TOOLBAR_ICON_CLASS} />
          </Button>
        ))}
        {renderTooltip(t('multitrack.redo'), (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className={`${TOOLBAR_ICON_BUTTON_CLASS} text-muted-foreground cursor-pointer`}
            disabled={!canRedo}
            aria-label={t('multitrack.redo')}
            onClick={(event) => {
              event.stopPropagation()
              onRedo()
            }}
          >
            <Redo2 className={TOOLBAR_ICON_CLASS} />
          </Button>
        ))}
        {renderTooltip(t('multitrack.cutMode'), (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className={`${TOOLBAR_ICON_BUTTON_CLASS} text-muted-foreground cursor-pointer`}
            disabled={!canTrimCenter}
            aria-label={t('multitrack.cutMode')}
            onClick={(event) => {
              event.stopPropagation()
              onCutAtCurrentTime()
            }}
          >
            <SplitCenterIcon className={TOOLBAR_ICON_CLASS} />
          </Button>
        ))}
        
        {renderTooltip(t('multitrack.trimLeftAtPlayhead'), (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className={`${TOOLBAR_ICON_BUTTON_CLASS} text-muted-foreground cursor-pointer`}
            disabled={!canTrimLeft}
            aria-label={t('multitrack.trimLeftAtPlayhead')}
            onClick={(event) => {
              event.stopPropagation()
              onTrimLeftAtCurrentTime()
            }}
          >
            <SplitLeftDotsIcon className={TOOLBAR_ICON_CLASS} />
          </Button>
        ))}
        
        {renderTooltip(t('multitrack.trimRightAtPlayhead'), (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className={`${TOOLBAR_ICON_BUTTON_CLASS} text-muted-foreground cursor-pointer`}
            disabled={!canTrimRight}
            aria-label={t('multitrack.trimRightAtPlayhead')}
            onClick={(event) => {
              event.stopPropagation()
              onTrimRightAtCurrentTime()
            }}
          >
            <SplitRightDotsIcon className={TOOLBAR_ICON_CLASS} />
          </Button>
        ))}

        {renderTooltip(t('multitrack.deleteSelectedSegment'), (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className={`${TOOLBAR_ICON_BUTTON_CLASS} text-muted-foreground cursor-pointer`}
            disabled={!canDelete}
            aria-label={t('multitrack.deleteSelectedSegment')}
            onClick={(event) => {
              event.stopPropagation()
              onDeleteSelected()
            }}
          >
            <Trash2 className={TOOLBAR_ICON_CLASS} />
          </Button>
        ))}
      </div>

      <div className="flex items-center gap-2 text-muted-foreground">
        <span className="w-16 text-right tabular-nums text-gradient">{formatMultiTrackTime(currentTime, { frameRate, showFrames: true })}</span>
        {renderTooltip(isPlaying ? t('multitrack.pause') : t('multitrack.play'), (
          <Button
            type="button"
            variant="secondary"
            size="icon"
            className={`${TOOLBAR_ICON_BUTTON_CLASS} text-foreground rounded-full cursor-pointer`}
            aria-label={isPlaying ? t('multitrack.pause') : t('multitrack.play')}
            onClick={onPlayPause}
          >
            {isPlaying ? (
              <svg className={TOOLBAR_ICON_CLASS} viewBox="0 0 24 24" fill="none">
                <defs>
                  <linearGradient id="icon-gradient-pause" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="var(--primary)" />
                    <stop offset="100%" stopColor="var(--highlight)" />
                  </linearGradient>
                </defs>
                <rect x="6" y="5" width="4" height="14" rx="1" fill="url(#icon-gradient-pause)" />
                <rect x="14" y="5" width="4" height="14" rx="1" fill="url(#icon-gradient-pause)" />
              </svg>
            ) : (
              <svg className={TOOLBAR_ICON_CLASS} viewBox="0 0 24 24" fill="none">
                <defs>
                  <linearGradient id="icon-gradient-play" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="var(--primary)" />
                    <stop offset="100%" stopColor="var(--highlight)" />
                  </linearGradient>
                </defs>
                <path d="M8 5.5v13l11-6.5L8 5.5z" fill="url(#icon-gradient-play)" />
              </svg>
            )}
          </Button>
        ))}
        <span className="w-16 tabular-nums">{formatMultiTrackTime(totalLength, { frameRate, showFrames: true })}</span>
      </div>

      <div className="ml-auto flex h-6 max-h-6 min-h-6 items-center gap-0.5 overflow-hidden">
        {renderTooltip(t('multitrack.timelineSnap'), (
          <Button
            type="button"
            variant={snapEnabled ? "outline" : "ghost"}
            size="icon"
            className={`${TOOLBAR_ICON_BUTTON_CLASS} cursor-pointer ${snapEnabled ? 'text-highlight' : 'text-muted-foreground'}`}
            aria-label={t('multitrack.timelineSnap')}
            aria-pressed={snapEnabled}
            onClick={(event) => {
              event.stopPropagation()
              onSnapEnabledChange(!snapEnabled)
            }}
          >
            <Magnet className={TOOLBAR_ICON_CLASS} />
          </Button>
        ))}
        {renderTooltip(timelineCollapsed ? t('multitrack.showTimeline') : t('multitrack.hideTimeline'), (
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className={`${TOOLBAR_ICON_BUTTON_CLASS} text-muted-foreground cursor-pointer`}
            aria-label={timelineCollapsed ? t('multitrack.showTimeline') : t('multitrack.hideTimeline')}
            onClick={(event) => {
              event.stopPropagation()
              onToggleTimeline()
            }}
          >
            {timelineCollapsed ? (
              <Maximize2 className={TOOLBAR_ICON_CLASS} />
            ) : (
              <Minimize2 className={TOOLBAR_ICON_CLASS} />
            )}
          </Button>
        ))}
        {renderTooltip(t('multitrack.timelineZoom'), (
          <span className="inline-flex shrink-0" aria-label={t('multitrack.timelineZoom')}>
            <ZoomOut className={`${TOOLBAR_ICON_CLASS} text-muted-foreground`} />
          </span>
        ))}
        <div className="relative h-3 max-h-3 min-h-3 w-12 overflow-visible">
          <Slider
            min={1}
            max={6}
            step={0.25}
            value={[zoom]}
            onValueChange={([value]) => onZoomChange(value)}
            className="absolute inset-0 h-3 max-h-3 min-h-3 w-12 [&_[role=slider]]:h-2.5 [&_[role=slider]]:w-2.5 [&_[data-orientation=horizontal]]:h-1"
            aria-label={t('multitrack.timelineZoom')}
          />
        </div>
      </div>
    </div>
  )
}
