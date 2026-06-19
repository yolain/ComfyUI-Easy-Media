import { useRef, useState } from 'react'
import { Captions, ListTree, Clapperboard, Layers2, Plus, Volume2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useT } from '@/lib/i18n'
import {
  getSegmentDragPlaceholder,
  getSegmentDragPreviewSegments,
  type SegmentDragPlaceholder,
} from '@/lib/multitrack-utils'
import type { MultiTrackSegment } from '@/types/multitrack'
import type { MultiTrackSourceType, TrackData } from '@/types/multitrack'
import { MULTITRACK_LEFT_GUTTER, MULTITRACK_RIGHT_RESERVE } from './MultiTrackRuler'
import { MultiTrackSegmentBlock } from './MultiTrackSegmentBlock'
import { VideoTrack } from './VideoTrack'

interface TrackAreaProps {
  data: TrackData
  width: number
  currentTime: number
  canvasScale: number
  selectedSegmentId: string | null
  onAddVideo: (trackId: string, filePath: string, sourceType: MultiTrackSourceType) => void
  onReplaceVideo: (trackId: string, segmentId: string, filePath: string, sourceType: MultiTrackSourceType) => void
  onAddTaskSegment: (trackId: string) => void
  onSelectSegment: (segmentId: string) => void
  onDeleteSegment: (segmentId: string) => void
  onResizeSegment: (segmentId: string, edge: 'start' | 'end', nextTime: number) => void
  onMoveSegment: (segmentId: string, targetTrackId: string, nextStartTime: number) => void
}

function trackHeight(type: string): number {
  if (type === 'task' || type === 'subtitle') return 30
  return 64
}

function TrackTypeIcon({ type }: Readonly<{ type: string }>) {
  const Icon = type === 'video'
    ? Clapperboard
    : type === 'audio'
      ? Volume2
      : type === 'subtitle'
        ? Captions
        : type === 'task'
          ? ListTree
          : Layers2
  return <Icon className="h-3.5 w-3.5 text-muted-foreground" />
}

function samePlaceholder(left: SegmentDragPlaceholder | null, right: SegmentDragPlaceholder | null): boolean {
  if (left === right) return true
  if (!left || !right) return false
  return left.segmentId === right.segmentId &&
    left.targetTrackId === right.targetTrackId &&
    left.insertIndex === right.insertIndex &&
    left.start_frame === right.start_frame &&
    left.end_frame === right.end_frame
}

export function TrackArea({
  data,
  width,
  currentTime,
  canvasScale,
  selectedSegmentId,
  onAddVideo,
  onReplaceVideo,
  onAddTaskSegment,
  onSelectSegment,
  onDeleteSegment,
  onResizeSegment,
  onMoveSegment,
}: Readonly<TrackAreaProps>) {
  const t = useT()
  const trackAreaRef = useRef<HTMLDivElement>(null)
  const [dragPlaceholder, setDragPlaceholder] = useState<SegmentDragPlaceholder | null>(null)
  const safeLength = Math.max(data.total_length, 1)
  const timelineWidth = Math.max(1, width - MULTITRACK_LEFT_GUTTER)
  const playableWidth = Math.max(1, timelineWidth - MULTITRACK_RIGHT_RESERVE)
  const playheadLeft = MULTITRACK_LEFT_GUTTER + (currentTime / safeLength) * playableWidth
  const reserveLeft = MULTITRACK_LEFT_GUTTER + playableWidth

  const trackAreaHeight = data.tracks.reduce((height, track) => height + trackHeight(track.type), 0)
  const trackBounds = data.tracks.reduce<Array<{ id: string, top: number, bottom: number }>>((bounds, track) => {
    const top = bounds.at(-1)?.bottom ?? 0
    bounds.push({ id: track.id, top, bottom: top + trackHeight(track.type) })
    return bounds
  }, [])

  function targetTrackIdFromClientY(clientY: number): string | null {
    const rect = trackAreaRef.current?.getBoundingClientRect()
    if (!rect) return null
    const y = (clientY - rect.top) / Math.max(canvasScale, 0.01)
    return trackBounds.find((track) => y >= track.top && y < track.bottom)?.id ?? null
  }

  function placeholderRect(placeholder: SegmentDragPlaceholder) {
    const bounds = trackBounds.find((track) => track.id === placeholder.targetTrackId)
    if (!bounds) return null
    const left = MULTITRACK_LEFT_GUTTER + (placeholder.start_frame / safeLength) * playableWidth
    const right = MULTITRACK_LEFT_GUTTER + (placeholder.end_frame / safeLength) * playableWidth
    return {
      left,
      top: bounds.top + 4,
      width: Math.max(2, right - left),
      height: Math.max(2, bounds.bottom - bounds.top - 8),
    }
  }

  function updateDragPlaceholder(segmentId: string, nextStartTime: number, clientY: number) {
    const targetTrackId = targetTrackIdFromClientY(clientY)
    if (!targetTrackId) {
      setDragPlaceholder((current) => samePlaceholder(current, null) ? current : null)
      return
    }
    const nextPlaceholder = getSegmentDragPlaceholder(data.tracks, segmentId, targetTrackId, nextStartTime, data.frame_rate)
    setDragPlaceholder((current) => samePlaceholder(current, nextPlaceholder) ? current : nextPlaceholder)
  }

  function handleMoveSegment(segmentId: string, fallbackTrackId: string, nextStartTime: number, clientY: number) {
    const targetTrackId = targetTrackIdFromClientY(clientY)
    setDragPlaceholder(null)
    if (!targetTrackId) return
    onMoveSegment(segmentId, targetTrackId ?? fallbackTrackId, nextStartTime)
  }

  const dragPlaceholderRect = dragPlaceholder ? placeholderRect(dragPlaceholder) : null
  const dragPreviewSegments = dragPlaceholder
    ? getSegmentDragPreviewSegments(data.tracks, dragPlaceholder, data.frame_rate)
    : null

  function previewSegment(trackId: string, segment: MultiTrackSegment): MultiTrackSegment {
    if (!dragPlaceholder || dragPlaceholder.targetTrackId !== trackId || segment.id === dragPlaceholder.segmentId) {
      return segment
    }
    return dragPreviewSegments?.find((item) => item.id === segment.id) ?? segment
  }

  return (
    <div ref={trackAreaRef} className="relative shrink-0 overflow-hidden" style={{ width, height: trackAreaHeight }}>
      <div
        className="pointer-events-none absolute top-0 z-10 h-full bg-black/30"
        style={{ left: reserveLeft, width: MULTITRACK_RIGHT_RESERVE }}
      />
      <div className="absolute top-0 z-20 h-full w-px bg-destructive" style={{ left: playheadLeft }} />
      {data.tracks.map((track) => {
        if (track.type === 'video') {
          return (
            <VideoTrack
              key={track.id}
              track={{
                ...track,
                segments: track.segments.map((segment) => previewSegment(track.id, segment)),
              }}
              totalLength={data.total_length}
              frameRate={data.frame_rate}
              width={playableWidth}
              canvasScale={canvasScale}
              selectedSegmentId={selectedSegmentId}
              onAddVideo={onAddVideo}
              onReplaceVideo={onReplaceVideo}
              onSelectSegment={onSelectSegment}
              onDeleteSegment={onDeleteSegment}
              onResizeSegment={onResizeSegment}
              onMoveSegment={(segmentId, nextStartTime, clientY) => {
                handleMoveSegment(segmentId, track.id, nextStartTime, clientY)
              }}
              onDragPreviewChange={updateDragPlaceholder}
              onDragPreviewEnd={() => setDragPlaceholder(null)}
            />
          )
        }

        return (
          <div
            key={track.id}
            className="relative flex border-b border-border"
            style={{ height: trackHeight(track.type) }}
          >
            <div
              className="flex shrink-0 items-center justify-center border-r border-border"
              style={{ width: MULTITRACK_LEFT_GUTTER }}
            >
              <TrackTypeIcon type={track.type} />
            </div>
            <div className="relative min-w-0 flex-1">
              {track.segments.map((segment, index) => (
                <MultiTrackSegmentBlock
                  key={segment.id}
                  trackType={track.type}
                  segmentIndex={index}
                  segment={previewSegment(track.id, segment)}
                  totalLength={data.total_length}
                  frameRate={data.frame_rate}
                  areaWidth={playableWidth}
                  canvasScale={canvasScale}
                  selected={selectedSegmentId === segment.id}
                  onSelect={onSelectSegment}
                  onDelete={onDeleteSegment}
                  onResize={onResizeSegment}
                  onMove={(segmentId, nextStartTime, clientY) => {
                    handleMoveSegment(segmentId, track.id, nextStartTime, clientY)
                  }}
                  onDragPreviewChange={updateDragPlaceholder}
                  onDragPreviewEnd={() => setDragPlaceholder(null)}
                />
              ))}
              {track.type === 'task' ? (
                <Button
                  type="button"
                  variant="secondary"
                  size="icon"
                  className="absolute top-1/2 h-5 w-5 cursor-pointer"
                  style={{
                    left: track.segments.length === 0
                      ? 6
                      : (track.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0) / safeLength) * playableWidth + 6,
                    transform: 'translateY(-50%)',
                  }}
                  aria-label={t('multitrack.addTaskSegment')}
                  onClick={(event) => {
                    event.stopPropagation()
                    onAddTaskSegment(track.id)
                  }}
                >
                  <Plus className="h-2.5 w-2.5" />
                </Button>
              ) : null}
            </div>
          </div>
        )
      })}
      {dragPlaceholderRect ? (
        <div
          className="pointer-events-none absolute z-10 rounded border border-border bg-muted/60"
          style={dragPlaceholderRect}
        />
      ) : null}
    </div>
  )
}
