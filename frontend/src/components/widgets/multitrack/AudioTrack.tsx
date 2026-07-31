import { useMemo, useRef, useState } from 'react'
import { Music2, Plus, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverAnchor, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import type { MediaTab } from '@/components/widgets/mediaSelector/MediaSelector'
import { useT } from '@/lib/i18n'
import { mediaPathToViewUrl } from '@/lib/media-url'
import type { SubtitleRecognitionMethod } from '@/lib/subtitle-recognition'
import { computeSlotItems } from '@/lib/timeline-utils'
import type { MultiTrack, MultiTrackSourceType } from '@/types/multitrack'
import { MULTITRACK_LEFT_GUTTER } from './MultiTrackRuler'
import { MultiTrackSegmentBlock } from './MultiTrackSegmentBlock'
import { TrackAudioControls } from './TrackAudioControls'
import { getTrackSegmentGaps, TrackGapAddButton } from './TrackGapAddButton'
import { TrackReorderHandle } from './TrackReorderHandle'

interface AudioTrackProps {
  track: MultiTrack
  totalLength: number
  frameRate: number
  width: number
  canvasScale: number
  selectedSegmentIds: Set<string>
  node: unknown
  app: unknown
  onAddAudio: (
    trackId: string,
    filePath: string,
    sourceType: MultiTrackSourceType,
    previewUrl?: string,
    startFrame?: number,
    endFrame?: number,
  ) => void
  onReplaceAudio: (
    trackId: string,
    segmentId: string,
    filePath: string,
    sourceType: MultiTrackSourceType,
    previewUrl?: string,
  ) => void
  onSelectSegment: (segmentId: string, mode?: 'replace' | 'toggle' | 'add') => void
  onDeleteSegment: (segmentId: string) => void
  onDeleteTrack: (trackId: string) => void
  onTrackAudioSettingsChange: (trackId: string, patch: Partial<Pick<MultiTrack, 'muted' | 'solo'>>) => void
  onResizeSegment: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
  onResizeSegmentPreview: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
  onMoveSegment: (segmentId: string, nextStartTime: number, clientY: number) => void
  onDragPreviewChange: (segmentId: string, nextStartTime: number, clientY: number) => void
  getDragPreviewStart?: (segmentId: string, nextStartTime: number, clientY: number) => number
  onDragPreviewEnd: () => void
  onRecognizeSubtitles?: (segmentId: string, method: SubtitleRecognitionMethod) => void
  cutMode: boolean
  onCutSegment: (segmentId: string, splitFrame: number) => void
}

function sourceTypeToTab(sourceType: MultiTrackSourceType | undefined): MediaTab {
  if (sourceType === 'output') return 'outputs'
  if (sourceType === 'local') return 'local'
  if (sourceType === 'url') return 'url'
  if (sourceType === 'slot') return 'slot'
  return 'inputs'
}

export function AudioTrack({
  track,
  totalLength,
  frameRate,
  width,
  canvasScale,
  selectedSegmentIds,
  node,
  app,
  onAddAudio,
  onReplaceAudio,
  onSelectSegment,
  onDeleteSegment,
  onDeleteTrack,
  onTrackAudioSettingsChange,
  onResizeSegment,
  onResizeSegmentPreview,
  onMoveSegment,
  onDragPreviewChange,
  getDragPreviewStart,
  onDragPreviewEnd,
  onRecognizeSubtitles = () => {},
  cutMode,
  onCutSegment,
}: Readonly<AudioTrackProps>) {
  const t = useT()
  const contentRef = useRef<HTMLDivElement>(null)
  const [mediaSelectorOpen, setMediaSelectorOpen] = useState(false)
  const [reselectAnchor, setReselectAnchor] = useState<{ segmentId: string; x: number; y: number } | null>(null)
  const slotItems = useMemo(
    () => computeSlotItems(node, app, 'audio'),
    [node, app, mediaSelectorOpen, reselectAnchor],
  )
  const lastEnd = track.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0)
  const actionLeft = track.segments.length === 0 ? 6 : (lastEnd / Math.max(totalLength, 1)) * width + 6
  const gaps = getTrackSegmentGaps(track.segments)
  const reselectSegment = reselectAnchor
    ? track.segments.find((segment) => segment.id === reselectAnchor.segmentId)
    : undefined
  const reselectValue = reselectSegment?.content.file_path
    ?? reselectSegment?.content.local_path
    ?? reselectSegment?.content.url
    ?? (reselectSegment?.content.slot_name ? `__slot__:${reselectSegment.content.slot_name}` : '')

  return (
    <div
      className="relative flex h-16 border-b border-border"
      data-multitrack-track-id={track.id}
    >
      <div className="shrink-0 border-r border-border" style={{ width: MULTITRACK_LEFT_GUTTER }}>
        <TrackAudioControls
          track={track}
          icon={(
            <TrackReorderHandle track={track}>
              <Music2 className="h-3.5 w-3.5 text-muted-foreground" />
            </TrackReorderHandle>
          )}
          preserveSelection={track.segments.some((segment) => selectedSegmentIds.has(segment.id))}
          onChange={(patch) => onTrackAudioSettingsChange(track.id, patch)}
        />
      </div>
      <div ref={contentRef} className="relative min-w-0 flex-1">
        {track.segments.map((segment, index) => (
          <MultiTrackSegmentBlock
            key={segment.id}
            trackType="audio"
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
            onRecognizeSubtitles={onRecognizeSubtitles}
            cutMode={cutMode}
            onCut={onCutSegment}
            onDoubleClick={(segmentId, event) => {
              const rect = contentRef.current?.getBoundingClientRect()
              if (!rect) return
              setReselectAnchor({
                segmentId,
                x: (event.clientX - rect.left) / Math.max(canvasScale, 0.01),
                y: (event.clientY - rect.top) / Math.max(canvasScale, 0.01),
              })
            }}
          />
        ))}
        {gaps.map((gap) => (
          <Popover key={`${gap.startFrame}-${gap.endFrame}`}>
            <PopoverTrigger asChild>
              <TrackGapAddButton
                gap={gap}
                totalLength={totalLength}
                width={width}
                ariaLabel={t('multitrack.addAudioSegment')}
              />
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="center">
              <MediaSelector
                value=""
                mediaType="audio"
                defaultTab="inputs"
                slotItems={slotItems}
                onChange={(filePath, sourceType = 'input') => {
                  const slotAudioName = slotItems.find((item) => item.value === filePath)?.audio_name
                  const previewUrl = slotAudioName ? mediaPathToViewUrl(slotAudioName, 'input') : undefined
                  onAddAudio(
                    track.id,
                    filePath,
                    sourceType,
                    previewUrl,
                    gap.startFrame,
                    gap.endFrame,
                  )
                }}
              />
            </PopoverContent>
          </Popover>
        ))}
        <div className="absolute top-1/2 flex -translate-y-1/2 gap-1" style={{ left: actionLeft }}>
          <Popover open={mediaSelectorOpen} onOpenChange={setMediaSelectorOpen}>
            <PopoverTrigger asChild>
              <Button
                type="button"
                variant="secondary"
                size="icon"
                className="h-5 w-5 cursor-pointer"
                aria-label={t('multitrack.addAudioSegment')}
              >
                <Plus className="h-2.5 w-2.5" />
              </Button>
            </PopoverTrigger>
            <PopoverContent className="w-auto p-0" align="start">
              <MediaSelector
                value=""
                mediaType="audio"
                defaultTab="inputs"
                slotItems={slotItems}
                onChange={(filePath, sourceType = 'input') => {
                  const slotAudioName = slotItems.find((item) => item.value === filePath)?.audio_name
                  const previewUrl = slotAudioName ? mediaPathToViewUrl(slotAudioName, 'input') : undefined
                  onAddAudio(track.id, filePath, sourceType, previewUrl)
                  setMediaSelectorOpen(false)
                }}
              />
            </PopoverContent>
          </Popover>
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
        </div>
        <Popover open={reselectAnchor !== null} onOpenChange={(open) => {
          if (!open) setReselectAnchor(null)
        }}>
          {reselectAnchor ? (
            <PopoverAnchor asChild>
              <div
                className="absolute h-px w-px"
                style={{ left: reselectAnchor.x, top: reselectAnchor.y }}
              />
            </PopoverAnchor>
          ) : null}
          <PopoverContent className="w-auto p-0" align="start">
            <MediaSelector
              value={reselectValue}
              mediaType="audio"
              defaultTab={sourceTypeToTab(reselectSegment?.content.source_type)}
              slotItems={slotItems}
              onChange={(filePath, sourceType = 'input') => {
                if (!reselectAnchor) return
                const slotAudioName = slotItems.find((item) => item.value === filePath)?.audio_name
                const previewUrl = slotAudioName ? mediaPathToViewUrl(slotAudioName, 'input') : undefined
                onReplaceAudio(track.id, reselectAnchor.segmentId, filePath, sourceType, previewUrl)
                setReselectAnchor(null)
              }}
            />
          </PopoverContent>
        </Popover>
      </div>
    </div>
  )
}
