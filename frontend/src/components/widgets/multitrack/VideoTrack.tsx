import { useRef, useState } from 'react'
import { Clapperboard, Plus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverAnchor, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import type { MediaTab } from '@/components/widgets/mediaSelector/MediaSelector'
import { useT } from '@/lib/i18n'
import type { MultiTrack, MultiTrackSourceType } from '@/types/multitrack'
import { MULTITRACK_LEFT_GUTTER } from './MultiTrackRuler'
import { MultiTrackSegmentBlock } from './MultiTrackSegmentBlock'

interface VideoTrackProps {
  track: MultiTrack
  totalLength: number
  frameRate: number
  width: number
  canvasScale: number
  selectedSegmentId: string | null
  onAddVideo: (trackId: string, filePath: string, sourceType: MultiTrackSourceType) => void
  onSelectSegment: (segmentId: string) => void
  onDeleteSegment: (segmentId: string) => void
  onResizeSegment: (segmentId: string, edge: 'start' | 'end', nextTime: number) => void
  onMoveSegment: (segmentId: string, nextStartTime: number, clientY: number) => void
  onDragPreviewChange: (segmentId: string, nextStartTime: number, clientY: number) => void
  onDragPreviewEnd: () => void
  onReplaceVideo: (trackId: string, segmentId: string, filePath: string, sourceType: MultiTrackSourceType) => void
}

function sourceTypeToTab(sourceType: MultiTrackSourceType | undefined): MediaTab {
  if (sourceType === 'output') return 'outputs'
  if (sourceType === 'local') return 'local'
  if (sourceType === 'url') return 'url'
  return 'inputs'
}

export function VideoTrack({
  track,
  totalLength,
  frameRate,
  width,
  canvasScale,
  selectedSegmentId,
  onAddVideo,
  onSelectSegment,
  onDeleteSegment,
  onResizeSegment,
  onMoveSegment,
  onDragPreviewChange,
  onDragPreviewEnd,
  onReplaceVideo,
}: Readonly<VideoTrackProps>) {
  const t = useT()
  const contentRef = useRef<HTMLDivElement>(null)
  const [reselectAnchor, setReselectAnchor] = useState<{ segmentId: string, x: number, y: number } | null>(null)
  const reselectSegment = reselectAnchor
    ? track.segments.find((segment) => segment.id === reselectAnchor.segmentId)
    : undefined
  const reselectValue = reselectSegment?.content.file_path
    ?? reselectSegment?.content.local_path
    ?? reselectSegment?.content.url
    ?? ''
  const lastEnd = track.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0)
  const addLeft = (lastEnd / Math.max(totalLength, 1)) * width

  return (
    <div className="relative flex h-16 border-b border-border">
      <div
        className="flex shrink-0 items-center justify-center border-r border-border"
        style={{ width: MULTITRACK_LEFT_GUTTER }}
      >
        <Clapperboard className="h-3.5 w-3.5 text-muted-foreground" />
      </div>
      <div ref={contentRef} className="relative min-w-0 flex-1">
        {track.segments.map((segment, index) => (
          <MultiTrackSegmentBlock
            key={segment.id}
            trackType={track.type}
            segmentIndex={index}
            segment={segment}
            totalLength={totalLength}
            frameRate={frameRate}
            areaWidth={width}
            canvasScale={canvasScale}
            selected={selectedSegmentId === segment.id}
            onSelect={onSelectSegment}
            onDelete={onDeleteSegment}
            onResize={onResizeSegment}
            onMove={onMoveSegment}
            onDragPreviewChange={onDragPreviewChange}
            onDragPreviewEnd={onDragPreviewEnd}
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

        <Popover>
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="secondary"
              size="icon"
              className="absolute top-1/2 h-5 w-5 cursor-pointer"
              style={{
                left: track.segments.length === 0 ? 6 : addLeft + 6,
                transform: 'translateY(-50%)',
              }}
              aria-label={t('multitrack.addVideo')}
            >
              <Plus className="h-2.5 w-2.5" />
            </Button>
          </PopoverTrigger>
          <PopoverContent className="w-auto p-0" align="start">
            <MediaSelector
              value=""
              mediaType="video"
              defaultTab="inputs"
              onChange={(filePath, sourceType = 'input') => {
                onAddVideo(track.id, filePath, sourceType)
              }}
            />
          </PopoverContent>
        </Popover>
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
              mediaType="video"
              defaultTab={sourceTypeToTab(reselectSegment?.content.source_type)}
              onChange={(filePath, sourceType = 'input') => {
                if (!reselectAnchor) return
                onReplaceVideo(track.id, reselectAnchor.segmentId, filePath, sourceType)
                setReselectAnchor(null)
              }}
            />
          </PopoverContent>
        </Popover>
      </div>
    </div>
  )
}
