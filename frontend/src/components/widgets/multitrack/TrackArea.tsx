import { useEffect, useRef, useState } from 'react'
import { Captions, Film, ListTree, Clapperboard, Layers2, Music2, Plus, Trash2, Volume2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useT } from '@/lib/i18n'
import { uploadInputMediaFile } from '@/lib/media-upload'
import { invalidateMediaListCache } from '@/stores/media-list-store'
import {
  getSegmentDragPlaceholder,
  getSegmentDragPreviewSegments,
  snapMultiTrackMoveStartTime,
  snapTimeToFrame,
  type SegmentDragPlaceholder,
} from '@/lib/multitrack-utils'
import type { MultiTrack, MultiTrackSegment, MultiTrackType } from '@/types/multitrack'
import type { MultiTrackSourceType, TrackData } from '@/types/multitrack'
import { MULTITRACK_LEFT_GUTTER, MULTITRACK_RIGHT_RESERVE } from './MultiTrackRuler'
import { MultiTrackSegmentBlock } from './MultiTrackSegmentBlock'
import { AudioTrack } from './AudioTrack'
import { VideoTrack } from './VideoTrack'
import { SubtitleTrack } from './SubtitleTrack'

interface TrackAreaProps {
  data: TrackData
  node: unknown
  app: unknown
  width: number
  currentTime: number
  snapEnabled: boolean
  canvasScale: number
  selectedSegmentIds: Set<string>
  onAddVideo: (trackId: string, filePath: string, sourceType: MultiTrackSourceType, startFrame?: number) => void
  onAddAudio: (
    trackId: string,
    filePath: string,
    sourceType: MultiTrackSourceType,
    previewUrl?: string,
    startFrame?: number,
  ) => void
  onAddTrack: (type: MultiTrackType) => void
  onAddSubtitleSegment: (trackId: string) => void
  onReplaceVideo: (trackId: string, segmentId: string, filePath: string, sourceType: MultiTrackSourceType) => void
  onAddTaskSegment: (trackId: string) => void
  onSelectSegment: (segmentId: string, mode?: 'replace' | 'toggle' | 'add') => void
  onSelectSegments: (segmentIds: string[]) => void
  onClearSelection: () => void
  onDeleteSegment: (segmentId: string) => void
  onDeleteTrack: (trackId: string) => void
  onTrackAudioSettingsChange: (trackId: string, patch: Partial<Pick<MultiTrack, 'muted' | 'solo'>>) => void
  onDistributeTaskSegments: (trackId: string) => void
  onCloneTaskSegment: (trackId: string, segmentId: string) => void
  onSplitTaskSegment?: (segmentId: string) => void
  onResizeSegment: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
  onResizeSegmentPreview: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
  onMoveSegment: (segmentId: string, targetTrackId: string, nextStartTime: number) => void
  onSmartSplit: (segmentId: string) => void
  onSmartSplitTasks: (segmentId: string) => void
  onRecognizeSubtitles?: (segmentId: string) => void
  onEditSubtitleSegment?: (segmentId: string) => void
  cutMode: boolean
  onCutSegment: (segmentId: string, splitFrame: number) => void
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

const AUDIO_EXTENSIONS = new Set(['mp3', 'wav', 'flac', 'ogg', 'm4a', 'aac', 'opus', 'wma'])
const VIDEO_EXTENSIONS = new Set(['mp4', 'webm', 'mov', 'mkv', 'avi', 'm4v'])

function externalMediaType(file: File): 'audio' | 'video' | null {
  if (file.type.startsWith('audio/')) return 'audio'
  if (file.type.startsWith('video/')) return 'video'
  const extension = file.name.split('.').pop()?.toLowerCase() ?? ''
  if (AUDIO_EXTENSIONS.has(extension)) return 'audio'
  if (VIDEO_EXTENSIONS.has(extension)) return 'video'
  return null
}

function draggedMediaType(dataTransfer: DataTransfer): 'audio' | 'video' | null {
  const file = firstDraggedFile(dataTransfer)
  if (file) return externalMediaType(file)
  for (const item of Array.from(dataTransfer.items)) {
    if (item.kind !== 'file') continue
    if (item.type.startsWith('audio/')) return 'audio'
    if (item.type.startsWith('video/')) return 'video'
  }
  return null
}

function hasExternalFiles(dataTransfer: DataTransfer): boolean {
  return Array.from(dataTransfer.types).includes('Files') ||
    dataTransfer.files.length > 0 ||
    Array.from(dataTransfer.items).some((item) => item.kind === 'file')
}

function firstDraggedFile(dataTransfer: DataTransfer): File | null {
  const file = Array.from(dataTransfer.files)[0]
  if (file) return file
  for (const item of Array.from(dataTransfer.items)) {
    if (item.kind !== 'file') continue
    const itemFile = item.getAsFile()
    if (itemFile) return itemFile
  }
  return null
}

interface ExternalDropSlot {
  trackId: string
  frame: number
}

export function TrackArea({
  data,
  node,
  app,
  width,
  currentTime,
  snapEnabled,
  canvasScale,
  selectedSegmentIds,
  onAddVideo,
  onAddAudio,
  onAddTrack,
  onAddSubtitleSegment,
  onReplaceVideo,
  onAddTaskSegment,
  onSelectSegment,
  onSelectSegments,
  onClearSelection,
  onDeleteSegment,
  onDeleteTrack,
  onTrackAudioSettingsChange,
  onDistributeTaskSegments,
  onCloneTaskSegment,
  onSplitTaskSegment = () => {},
  onResizeSegment,
  onResizeSegmentPreview,
  onMoveSegment,
  onSmartSplit,
  onSmartSplitTasks,
  onRecognizeSubtitles = () => {},
  onEditSubtitleSegment = () => {},
  cutMode,
  onCutSegment,
}: Readonly<TrackAreaProps>) {
  const t = useT()
  const trackAreaRef = useRef<HTMLDivElement>(null)
  const marqueeStartRef = useRef<{ x: number, y: number } | null>(null)
  const marqueeMovedRef = useRef(false)
  const suppressNextClickRef = useRef(false)
  const suppressGlobalClickCleanupRef = useRef<(() => void) | null>(null)
  const [dragPlaceholder, setDragPlaceholder] = useState<SegmentDragPlaceholder | null>(null)
  const [externalDropSlot, setExternalDropSlot] = useState<ExternalDropSlot | null>(null)
  const [marqueeRect, setMarqueeRect] = useState<{ left: number, top: number, width: number, height: number } | null>(null)
  const safeLength = Math.max(data.total_length, 1)
  const timelineWidth = Math.max(1, width - MULTITRACK_LEFT_GUTTER)
  const playableWidth = Math.max(1, timelineWidth - MULTITRACK_RIGHT_RESERVE)
  const playheadLeft = MULTITRACK_LEFT_GUTTER + (currentTime / safeLength) * playableWidth
  const reserveLeft = MULTITRACK_LEFT_GUTTER + playableWidth

  const addTrackHeight = 24
  const tracksHeight = data.tracks.reduce((height, track) => height + trackHeight(track.type), 0)
  const trackAreaHeight = tracksHeight + addTrackHeight
  const firstVideoTrackId = data.tracks.find((track) => track.type === 'video')?.id
  const audioTrackLimitReached = data.tracks.filter((track) => track.type === 'audio').length >= 2
  const subtitleTrackLimitReached = data.tracks.filter((track) => track.type === 'subtitle').length >= 2
  const trackBounds = data.tracks.reduce<Array<{ id: string, top: number, bottom: number }>>((bounds, track) => {
    const top = bounds.at(-1)?.bottom ?? 0
    bounds.push({ id: track.id, top, bottom: top + trackHeight(track.type) })
    return bounds
  }, [])

  useEffect(() => {
    return () => {
      suppressGlobalClickCleanupRef.current?.()
    }
  }, [])

  function targetTrackIdFromClientY(clientY: number): string | null {
    const rect = trackAreaRef.current?.getBoundingClientRect()
    if (!rect) return null
    const y = (clientY - rect.top) / Math.max(canvasScale, 0.01)
    return trackBounds.find((track) => y >= track.top && y < track.bottom)?.id ?? null
  }

  function localPoint(clientX: number, clientY: number): { x: number, y: number } | null {
    const rect = trackAreaRef.current?.getBoundingClientRect()
    if (!rect) return null
    const scale = Math.max(canvasScale, 0.01)
    return {
      x: (clientX - rect.left) / scale,
      y: (clientY - rect.top) / scale,
    }
  }

  function clampedLocalPoint(clientX: number, clientY: number): { x: number, y: number } | null {
    const point = localPoint(clientX, clientY)
    if (!point) return null
    return {
      x: Math.max(0, Math.min(width, point.x)),
      y: Math.max(0, Math.min(trackAreaHeight, point.y)),
    }
  }

  function normalizedRect(start: { x: number, y: number }, end: { x: number, y: number }) {
    return {
      left: Math.min(start.x, end.x),
      top: Math.min(start.y, end.y),
      width: Math.abs(end.x - start.x),
      height: Math.abs(end.y - start.y),
    }
  }

  function segmentIntersectsRect(trackId: string, segment: MultiTrackSegment, rect: { left: number, top: number, width: number, height: number }): boolean {
    const bounds = trackBounds.find((track) => track.id === trackId)
    if (!bounds) return false
    const left = MULTITRACK_LEFT_GUTTER + (segment.start_frame / safeLength) * playableWidth
    const right = MULTITRACK_LEFT_GUTTER + (segment.end_frame / safeLength) * playableWidth
    const top = bounds.top + 4
    const bottom = bounds.bottom - 4
    const rectRight = rect.left + rect.width
    const rectBottom = rect.top + rect.height
    return left < rectRight && right > rect.left && top < rectBottom && bottom > rect.top
  }

  function selectedIdsForRect(rect: { left: number, top: number, width: number, height: number }): string[] {
    return data.tracks.flatMap((track) => (
      track.segments
        .filter((segment) => segmentIntersectsRect(track.id, segment, rect))
        .map((segment) => segment.id)
    ))
  }

  function canStartMarquee(event: React.MouseEvent<HTMLDivElement>): boolean {
    if (event.button !== 0 || cutMode) return false
    const target = event.target as Element
    return !target.closest('[data-multitrack-segment]') &&
      !target.closest('button') &&
      !target.closest('[role="button"]') &&
      !target.closest('[data-radix-popper-content-wrapper]')
  }

  function suppressNextGlobalClick() {
    suppressGlobalClickCleanupRef.current?.()
    let timeoutId: ReturnType<typeof globalThis.setTimeout> | null = null
    const handleClick = (event: MouseEvent) => {
      event.preventDefault()
      event.stopPropagation()
      event.stopImmediatePropagation()
      cleanup()
    }
    function cleanup() {
      globalThis.removeEventListener('click', handleClick, true)
      if (timeoutId !== null) globalThis.clearTimeout(timeoutId)
      if (suppressGlobalClickCleanupRef.current === cleanup) suppressGlobalClickCleanupRef.current = null
    }
    suppressGlobalClickCleanupRef.current = cleanup
    globalThis.addEventListener('click', handleClick, true)
    timeoutId = globalThis.setTimeout(cleanup, 0)
  }

  function handleMarqueeMouseDown(event: React.MouseEvent<HTMLDivElement>) {
    if (!canStartMarquee(event)) return
    const point = localPoint(event.clientX, event.clientY)
    if (!point || point.y > trackAreaHeight) return
    event.preventDefault()
    event.stopPropagation()
    marqueeStartRef.current = point
    marqueeMovedRef.current = false

    function handleMove(moveEvent: MouseEvent) {
      const start = marqueeStartRef.current
      const current = clampedLocalPoint(moveEvent.clientX, moveEvent.clientY)
      if (!start || !current) return
      const rect = normalizedRect(start, current)
      if (rect.width > 3 || rect.height > 3) marqueeMovedRef.current = true
      setMarqueeRect(rect)
    }

    function handleUp(upEvent: MouseEvent) {
      const start = marqueeStartRef.current
      const current = clampedLocalPoint(upEvent.clientX, upEvent.clientY)
      marqueeStartRef.current = null
      setMarqueeRect(null)
      globalThis.removeEventListener('mousemove', handleMove)
      globalThis.removeEventListener('mouseup', handleUp)
      if (!start || !current) return
      if (!marqueeMovedRef.current) {
        onClearSelection()
        return
      }
      suppressNextClickRef.current = true
      suppressNextGlobalClick()
      const ids = selectedIdsForRect(normalizedRect(start, current))
      onSelectSegments(ids)
    }

    globalThis.addEventListener('mousemove', handleMove)
    globalThis.addEventListener('mouseup', handleUp)
  }

  function externalDropTarget(
    clientX: number,
    clientY: number,
    mediaType: 'audio' | 'video' | null,
  ): ExternalDropSlot | null {
    const rect = trackAreaRef.current?.getBoundingClientRect()
    const trackId = targetTrackIdFromClientY(clientY)
    const track = data.tracks.find((item) => item.id === trackId)
    if (!rect || !track || mediaType !== track.type) return null
    const x = (clientX - rect.left) / Math.max(canvasScale, 0.01) - MULTITRACK_LEFT_GUTTER
    const requestedFrame = Math.round((Math.max(0, Math.min(playableWidth, x)) / playableWidth) * safeLength)
    const sortedSegments = [...track.segments].sort((left, right) => left.start_frame - right.start_frame)
    const insertIndex = sortedSegments.filter((segment) => (
      segment.start_frame + (segment.end_frame - segment.start_frame) / 2 < requestedFrame
    )).length
    const previousEnd = sortedSegments[insertIndex - 1]?.end_frame ?? 0
    return { trackId: track.id, frame: Math.max(requestedFrame, previousEnd) }
  }

  function handleExternalDragOver(event: React.DragEvent<HTMLDivElement>) {
    if (!hasExternalFiles(event.dataTransfer)) return
    event.preventDefault()
    event.stopPropagation()
    event.dataTransfer.dropEffect = 'copy'
    const target = externalDropTarget(event.clientX, event.clientY, draggedMediaType(event.dataTransfer))
    setExternalDropSlot(target)
  }

  async function handleExternalDrop(event: React.DragEvent<HTMLDivElement>) {
    if (!hasExternalFiles(event.dataTransfer)) return
    event.preventDefault()
    event.stopPropagation()
    const file = firstDraggedFile(event.dataTransfer)
    const mediaType = file ? externalMediaType(file) : null
    const target = externalDropTarget(event.clientX, event.clientY, mediaType)
    setExternalDropSlot(null)
    if (!file || !target) return
    try {
      const filePath = await uploadInputMediaFile(file)
      invalidateMediaListCache('inputs')
      if (mediaType === 'video') {
        onAddVideo(target.trackId, filePath, 'input', target.frame)
      } else {
        onAddAudio(target.trackId, filePath, 'input', undefined, target.frame)
      }
    } catch (error) {
      console.error('[TrackArea] failed to upload dropped media:', error)
    }
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

  function dragSnapBrakeDistanceFrames(): number {
    return Math.max(1, Math.round((8 / playableWidth) * safeLength))
  }

  function snappedMoveStartTime(segmentId: string, nextStartTime: number): number {
    return snapEnabled
      ? snapMultiTrackMoveStartTime(data, segmentId, nextStartTime, dragSnapBrakeDistanceFrames(), currentTime)
      : snapTimeToFrame(nextStartTime, data.frame_rate)
  }

  function updateDragPlaceholder(segmentId: string, nextStartTime: number, clientY: number) {
    const targetTrackId = targetTrackIdFromClientY(clientY)
    if (!targetTrackId) {
      setDragPlaceholder((current) => samePlaceholder(current, null) ? current : null)
      return
    }
    const nextPlaceholder = getSegmentDragPlaceholder(
      data.tracks,
      segmentId,
      targetTrackId,
      snappedMoveStartTime(segmentId, nextStartTime),
      data.frame_rate,
    )
    setDragPlaceholder((current) => samePlaceholder(current, nextPlaceholder) ? current : nextPlaceholder)
  }

  function handleMoveSegment(segmentId: string, fallbackTrackId: string, nextStartTime: number, clientY: number) {
    const targetTrackId = targetTrackIdFromClientY(clientY)
    setDragPlaceholder(null)
    if (!targetTrackId) return
    onMoveSegment(segmentId, targetTrackId ?? fallbackTrackId, snappedMoveStartTime(segmentId, nextStartTime))
  }

  const dragPlaceholderRect = dragPlaceholder ? placeholderRect(dragPlaceholder) : null
  const dragPreviewSegments = dragPlaceholder
    ? getSegmentDragPreviewSegments(data.tracks, dragPlaceholder, data.frame_rate)
    : null

  function previewSegment(segment: MultiTrackSegment): MultiTrackSegment {
    if (!dragPlaceholder || segment.id === dragPlaceholder.segmentId) {
      return segment
    }
    return dragPreviewSegments?.find((item) => item.id === segment.id) ?? segment
  }

  return (
    <div
      ref={trackAreaRef}
      className="relative shrink-0"
      style={{ width, height: trackAreaHeight }}
      data-multitrack-track-area
      onDragOverCapture={handleExternalDragOver}
      onDragLeave={(event) => {
        const nextTarget = event.relatedTarget
        if (!(nextTarget instanceof Node) || !event.currentTarget.contains(nextTarget)) setExternalDropSlot(null)
      }}
      onDropCapture={handleExternalDrop}
      onMouseDown={handleMarqueeMouseDown}
      onClick={(event) => {
        event.stopPropagation()
        if (suppressNextClickRef.current) {
          suppressNextClickRef.current = false
          return
        }
        const target = event.target as Element
        if (!target.closest('[data-multitrack-segment]') && !target.closest('button') && !target.closest('[role="button"]')) {
          onClearSelection()
        }
      }}
    >
      <div
        className="pointer-events-none absolute top-0 z-10 bg-black/30"
        style={{ left: reserveLeft, width: MULTITRACK_RIGHT_RESERVE, height: tracksHeight }}
      />
      <div className="absolute top-0 z-20 w-px bg-destructive" style={{ left: playheadLeft, height: tracksHeight }} />
      {data.tracks.map((track) => {
        if (track.type === 'video') {
          return (
            <VideoTrack
              key={track.id}
              track={{
                ...track,
                segments: track.segments.map((segment) => previewSegment(segment)),
              }}
              totalLength={data.total_length}
              frameRate={data.frame_rate}
              width={playableWidth}
              canvasScale={canvasScale}
              selectedSegmentIds={selectedSegmentIds}
              onAddVideo={onAddVideo}
              onReplaceVideo={onReplaceVideo}
              onSelectSegment={onSelectSegment}
              onDeleteSegment={onDeleteSegment}
              onSmartSplit={onSmartSplit}
              onSmartSplitTasks={onSmartSplitTasks}
              onRecognizeSubtitles={onRecognizeSubtitles}
              cutMode={cutMode}
              onCutSegment={onCutSegment}
              canDeleteTrack={track.id !== firstVideoTrackId}
              onDeleteTrack={onDeleteTrack}
              onTrackAudioSettingsChange={onTrackAudioSettingsChange}
              onResizeSegment={onResizeSegment}
              onResizeSegmentPreview={onResizeSegmentPreview}
              onMoveSegment={(segmentId, nextStartTime, clientY) => {
                handleMoveSegment(segmentId, track.id, nextStartTime, clientY)
              }}
              onDragPreviewChange={updateDragPlaceholder}
              getDragPreviewStart={snappedMoveStartTime}
              onDragPreviewEnd={() => setDragPlaceholder(null)}
            />
          )
        }

        if (track.type === 'audio') {
          return (
            <AudioTrack
              key={track.id}
              track={{ ...track, segments: track.segments.map((segment) => previewSegment(segment)) }}
              totalLength={data.total_length}
              frameRate={data.frame_rate}
              width={playableWidth}
              canvasScale={canvasScale}
              selectedSegmentIds={selectedSegmentIds}
              node={node}
              app={app}
              onAddAudio={onAddAudio}
              onSelectSegment={onSelectSegment}
              onDeleteSegment={onDeleteSegment}
              onDeleteTrack={onDeleteTrack}
              onTrackAudioSettingsChange={onTrackAudioSettingsChange}
              onResizeSegment={onResizeSegment}
              onResizeSegmentPreview={onResizeSegmentPreview}
              onMoveSegment={(segmentId, nextStartTime, clientY) => handleMoveSegment(segmentId, track.id, nextStartTime, clientY)}
              onDragPreviewChange={updateDragPlaceholder}
              getDragPreviewStart={snappedMoveStartTime}
              onDragPreviewEnd={() => setDragPlaceholder(null)}
              onRecognizeSubtitles={onRecognizeSubtitles}
              cutMode={cutMode}
              onCutSegment={onCutSegment}
            />
          )
        }

        if (track.type === 'subtitle') {
          return (
            <SubtitleTrack
              key={track.id}
              track={{ ...track, segments: track.segments.map((segment) => previewSegment(segment)) }}
              totalLength={data.total_length}
              frameRate={data.frame_rate}
              width={playableWidth}
              canvasScale={canvasScale}
              selectedSegmentIds={selectedSegmentIds}
              onSelectSegment={onSelectSegment}
              onAddSubtitleSegment={onAddSubtitleSegment}
              onDeleteSegment={onDeleteSegment}
              onDeleteTrack={onDeleteTrack}
              onEditSubtitleSegment={onEditSubtitleSegment}
              onResizeSegment={onResizeSegment}
              onResizeSegmentPreview={onResizeSegmentPreview}
              onMoveSegment={(segmentId, nextStartTime, clientY) => handleMoveSegment(segmentId, track.id, nextStartTime, clientY)}
              onDragPreviewChange={updateDragPlaceholder}
              getDragPreviewStart={snappedMoveStartTime}
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
                  segment={previewSegment(segment)}
                  totalLength={data.total_length}
                  frameRate={data.frame_rate}
                  areaWidth={playableWidth}
                  canvasScale={canvasScale}
                  selected={selectedSegmentIds.has(segment.id)}
                  onSelect={onSelectSegment}
                  onDelete={onDeleteSegment}
                  onDistribute={track.type === 'task' ? () => onDistributeTaskSegments(track.id) : undefined}
                  onClone={track.type === 'task' ? (segmentId) => onCloneTaskSegment(track.id, segmentId) : undefined}
                  onSplitTask={track.type === 'task' ? onSplitTaskSegment : undefined}
                  cutMode={cutMode}
                  onCut={onCutSegment}
                  onResize={onResizeSegment}
                  onResizePreview={onResizeSegmentPreview}
                  onMove={(segmentId, nextStartTime, clientY) => {
                    handleMoveSegment(segmentId, track.id, nextStartTime, clientY)
                  }}
                  onDragPreviewChange={updateDragPlaceholder}
                  getDragPreviewStart={snappedMoveStartTime}
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
              ) : (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon"
                      className="absolute top-1/2 h-5 w-5 -translate-y-1/2 cursor-pointer text-destructive"
                      style={{
                        left: track.segments.length === 0
                          ? 6
                          : (track.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0) / safeLength) * playableWidth + 6,
                      }}
                      aria-label={t('multitrack.deleteTrack', { name: track.name })}
                      onClick={() => onDeleteTrack(track.id)}
                    >
                      <Trash2 className="h-2.5 w-2.5" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>{t('multitrack.deleteTrack', { name: track.name })}</TooltipContent>
                </Tooltip>
              )}
            </div>
          </div>
        )
      })}
      <div
        className="flex items-center justify-center gap-1 border-b border-border text-[10px] text-muted-foreground"
        style={{ height: addTrackHeight }}
      >
        <span>{t('multitrack.addTrack')}</span>
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="inline-flex">
              <Button type="button" variant="ghost" size="icon" className="h-6 w-6" disabled aria-label={t('multitrack.addVideoTrack')}>
                <Film className="h-3.5 w-3.5" />
              </Button>
            </span>
          </TooltipTrigger>
          <TooltipContent>{t('multitrack.notSupportedYet')}</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-6 w-6 cursor-pointer disabled:cursor-not-allowed"
              disabled={audioTrackLimitReached}
              aria-label={t('multitrack.addAudioTrack')}
              onClick={() => onAddTrack('audio')}
            >
              <Music2 className="h-3.5 w-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('multitrack.addAudioTrack')}</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="h-6 w-6 cursor-pointer disabled:cursor-not-allowed"
              disabled={subtitleTrackLimitReached}
              aria-label={t('multitrack.addSubtitleTrack')}
              onClick={() => onAddTrack('subtitle')}
            >
              <Captions className="h-3.5 w-3.5" />
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('multitrack.addSubtitleTrack')}</TooltipContent>
        </Tooltip>
      </div>
      {dragPlaceholderRect ? (
        <div
          className="pointer-events-none absolute z-10 rounded border border-border bg-muted/60"
          style={dragPlaceholderRect}
        />
      ) : null}
      {marqueeRect ? (
        <div
          className="pointer-events-none absolute z-40 rounded border border-primary bg-primary/10"
          style={marqueeRect}
        />
      ) : null}
      {externalDropSlot ? (() => {
        const bounds = trackBounds.find((track) => track.id === externalDropSlot.trackId)
        if (!bounds) return null
        return (
          <div
            data-testid="external-media-drop-slot"
            className="pointer-events-none absolute z-30 w-1 rounded bg-primary"
            style={{
              left: MULTITRACK_LEFT_GUTTER + (externalDropSlot.frame / safeLength) * playableWidth,
              top: bounds.top + 4,
              height: bounds.bottom - bounds.top - 8,
            }}
          />
        )
      })() : null}
    </div>
  )
}
