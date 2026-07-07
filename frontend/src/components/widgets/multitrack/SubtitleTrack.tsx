import { Captions, Eye, EyeOff, Plus, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useT } from '@/lib/i18n'
import type { MultiTrack } from '@/types/multitrack'
import { MULTITRACK_LEFT_GUTTER } from './MultiTrackRuler'
import { MultiTrackSegmentBlock } from './MultiTrackSegmentBlock'

interface SubtitleTrackProps {
  track: MultiTrack
  totalLength: number
  frameRate: number
  width: number
  canvasScale: number
  selectedSegmentIds: Set<string>
  onSelectSegment: (segmentId: string, mode?: 'replace' | 'toggle' | 'add') => void
  onAddSubtitleSegment: (trackId: string) => void
  onDeleteSegment: (segmentId: string) => void
  onDeleteTrack: (trackId: string) => void
  onTrackVisibilityChange?: (trackId: string, visible: boolean) => void
  onEditSubtitleSegment: (segmentId: string) => void
  onResizeSegment: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
  onResizeSegmentPreview: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
  onMoveSegment: (segmentId: string, nextStartTime: number, clientY: number) => void
  onDragPreviewChange: (segmentId: string, nextStartTime: number, clientY: number) => void
  getDragPreviewStart?: (segmentId: string, nextStartTime: number, clientY: number) => number
  onDragPreviewEnd: () => void
}

export function SubtitleTrack({
  track,
  totalLength,
  frameRate,
  width,
  canvasScale,
  selectedSegmentIds,
  onSelectSegment,
  onAddSubtitleSegment,
  onDeleteSegment,
  onDeleteTrack,
  onTrackVisibilityChange = () => {},
  onEditSubtitleSegment,
  onResizeSegment,
  onResizeSegmentPreview,
  onMoveSegment,
  onDragPreviewChange,
  getDragPreviewStart,
  onDragPreviewEnd,
}: Readonly<SubtitleTrackProps>) {
  const t = useT()
  const lastEnd = track.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0)
  const actionLeft = track.segments.length === 0 ? 6 : (lastEnd / Math.max(totalLength, 1)) * width + 6
  const visible = track.visible !== false

  return (
    <div className="relative flex h-[30px] border-b border-border">
      <div
        className="flex shrink-0 items-center justify-center border-r border-border"
        style={{ width: MULTITRACK_LEFT_GUTTER }}
      >
        <Captions className="h-3.5 w-3.5 text-muted-foreground" />
      </div>
      <div className="relative min-w-0 flex-1">
        {track.segments.map((segment, index) => (
          <MultiTrackSegmentBlock
            key={segment.id}
            trackType="subtitle"
            segmentIndex={index}
            segment={segment}
            totalLength={totalLength}
            frameRate={frameRate}
            areaWidth={width}
            canvasScale={canvasScale}
            selected={selectedSegmentIds.has(segment.id)}
            onSelect={onSelectSegment}
            onDelete={onDeleteSegment}
            onResize={onResizeSegment}
            onResizePreview={onResizeSegmentPreview}
            onMove={onMoveSegment}
            onDragPreviewChange={onDragPreviewChange}
            getDragPreviewStart={getDragPreviewStart}
            onDragPreviewEnd={onDragPreviewEnd}
            dimmed={!visible}
            onDoubleClick={() => {
              onSelectSegment(segment.id)
              onEditSubtitleSegment(segment.id)
            }}
          />
        ))}
        <div className="absolute top-1/2 flex -translate-y-1/2 flex-row gap-1" style={{ left: actionLeft }}>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant="secondary"
                size="icon"
                className="h-5 w-5 cursor-pointer"
                aria-label={t('multitrack.addSubtitleSegment')}
                onClick={(event) => {
                  event.stopPropagation()
                  onAddSubtitleSegment(track.id)
                }}
              >
                <Plus className="h-2.5 w-2.5" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>{t('multitrack.addSubtitleSegment')}</TooltipContent>
          </Tooltip>
          {track.segments.length === 0 ? (
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-5 w-5 cursor-pointer text-destructive"
                  aria-label={t('multitrack.deleteTrack', { name: track.name })}
                  onClick={() => onDeleteTrack(track.id)}
                >
                  <Trash2 className="h-2.5 w-2.5" />
                </Button>
              </TooltipTrigger>
              <TooltipContent>{t('multitrack.deleteTrack', { name: track.name })}</TooltipContent>
            </Tooltip>
          ) : null}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant={visible ? 'ghost' : 'outline'}
                size="icon"
                className={`h-5 w-5 cursor-pointer ${visible ? 'text-muted-foreground' : 'text-foreground'}`}
                aria-label={visible ? t('multitrack.hideSubtitleTrack', { name: track.name }) : t('multitrack.showSubtitleTrack', { name: track.name })}
                aria-pressed={!visible}
                onClick={(event) => {
                  event.stopPropagation()
                  onTrackVisibilityChange(track.id, !visible)
                }}
              >
                {visible ? <EyeOff className="h-2.5 w-2.5" /> : <Eye className="h-2.5 w-2.5" />}
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              {visible ? t('multitrack.hideSubtitleTrack', { name: track.name }) : t('multitrack.showSubtitleTrack', { name: track.name })}
            </TooltipContent>
          </Tooltip>
        </div>
      </div>
    </div>
  )
}
