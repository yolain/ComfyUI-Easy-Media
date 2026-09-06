import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { Lock, Share2, Unlock } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSub,
  ContextMenuSubContent,
  ContextMenuSubTrigger,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { AudioWaveform } from '@/components/widgets/timeline/AudioWaveform'
import { mediaContentToViewUrl } from '@/lib/media-url'
import { useT } from '@/lib/i18n'
import { formatMultiTrackTime, getMultiTrackTaskModeLabel } from '@/lib/multitrack-utils'
import { getSegmentTrackPresentation } from '@/lib/multitrack-segment-style'
import { captureVideoPosterFrame } from '@/lib/video-utils'
import type { SubtitleRecognitionMethod } from '@/lib/subtitle-recognition'
import type { MultiTrackSegment, MultiTrackType } from '@/types/multitrack'

const RESIZE_ZONE_DEFAULT = 8
const RESIZE_ZONE_SELECTED = 12
const RESIZE_SNAP_DISTANCE = 8
const DRAG_START_DISTANCE = 4

type SegmentBlockStyle = CSSProperties & {
  '--multitrack-waveform'?: string
}

interface MultiTrackSegmentBlockProps {
  trackType: MultiTrackType
  segmentIndex: number
  segment: MultiTrackSegment
  totalLength: number
  frameRate: number
  areaWidth: number
  canvasScale: number
  selected: boolean
  onSelect: (segmentId: string, mode?: 'replace' | 'toggle' | 'add') => void
  onDelete: (segmentId: string) => void
  onDistribute?: () => void
  onClone?: (segmentId: string) => void
  onSplitTask?: (segmentId: string) => void
  onSmartSplit?: (segmentId: string) => void
  onSmartSplitTasks?: (segmentId: string) => void
  onRecognizeSubtitles?: (segmentId: string, method: SubtitleRecognitionMethod) => void
  cutMode?: boolean
  onCut?: (segmentId: string, splitFrame: number) => void
  onResize: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
  onResizePreview: (segmentId: string, edge: 'start' | 'end', nextTime: number, brakeDistanceFrames?: number) => void
  onMove: (segmentId: string, nextStartTime: number, clientY: number) => void
  onDragPreviewChange?: (segmentId: string, nextStartTime: number, clientY: number) => void
  getDragPreviewStart?: (segmentId: string, nextStartTime: number, clientY: number) => number
  onDragPreviewEnd?: () => void
  onDoubleClick?: (segmentId: string, event: React.MouseEvent) => void
  dimmed?: boolean
  showTaskIndex?: boolean
  taskOverview?: boolean
  audioLocked?: boolean
  audioLockEnabled?: boolean
  onAudioLockToggle?: (locked: boolean) => void
  sharedReference?: boolean
  onSharedReferenceToggle?: (enabled: boolean) => void
}

function segmentRect(segment: MultiTrackSegment, totalLength: number, areaWidth: number) {
  const safeLength = Math.max(totalLength, 1)
  const left = (segment.start_frame / safeLength) * areaWidth
  const right = (segment.end_frame / safeLength) * areaWidth
  return { left, width: Math.max(2, right - left) }
}

export function MultiTrackSegmentBlock({
  trackType,
  segmentIndex,
  segment,
  totalLength,
  frameRate,
  areaWidth,
  canvasScale,
  selected,
  onSelect,
  onDelete,
  onDistribute,
  onClone,
  onSplitTask,
  onSmartSplit,
  onSmartSplitTasks,
  onRecognizeSubtitles,
  cutMode = false,
  onCut,
  onResize,
  onResizePreview,
  onMove,
  onDragPreviewChange,
  getDragPreviewStart,
  onDragPreviewEnd,
  onDoubleClick,
  dimmed = false,
  showTaskIndex = true,
  taskOverview = false,
  audioLocked = false,
  audioLockEnabled = false,
  onAudioLockToggle,
  sharedReference = false,
  onSharedReferenceToggle,
}: Readonly<MultiTrackSegmentBlockProps>) {
  const t = useT()
  const didDragRef = useRef(false)
  const dragPreviewRef = useRef<{
    x: number
    y: number
    width: number
    height: number
    nextStartTime: number
    clientY: number
  } | null>(null)
  const dragFrameRef = useRef<number | null>(null)
  const isDraggingRef = useRef(false)
  const [isResizing, setIsResizing] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const [dragPreview, setDragPreview] = useState<{
    x: number
    y: number
    width: number
    height: number
    nextStartTime: number
    clientY: number
  } | null>(null)
  const [posterUrl, setPosterUrl] = useState<string | null>(null)
  const [cursorStyle, setCursorStyle] = useState<'ew-resize' | 'grab' | null>(null)
  const { left, width } = segmentRect(segment, totalLength, areaWidth)
  const mediaUrl = mediaContentToViewUrl({
    source_type: segment.content.source_type ?? 'input',
    file_path: segment.content.file_path,
    local_path: segment.content.local_path,
    url: segment.content.url,
  })
  const segmentDuration = Math.max(0, segment.end_frame - segment.start_frame)
  const sourceDurationFrames = segment.content.duration && segment.content.duration > 0
    ? segment.content.duration * Math.max(frameRate, 1)
    : null
  const sourceOriginFrame = segment.origin_start_frame ?? segment.start_frame
  const waveformStartRatio = sourceDurationFrames
    ? Math.max(0, Math.min(1, (segment.start_frame - sourceOriginFrame) / sourceDurationFrames))
    : 0
  const waveformEndRatio = sourceDurationFrames
    ? Math.max(waveformStartRatio, Math.min(1, (segment.end_frame - sourceOriginFrame) / sourceDurationFrames))
    : 1
  const sourceStartTime = Math.max(
    0,
    (segment.start_frame - sourceOriginFrame) / Math.max(frameRate, 1),
  )
  const taskModeLabel = getMultiTrackTaskModeLabel(segment.content.task_mode ?? 'default', t)
  const label = trackType === 'task'
    ? showTaskIndex
      ? t('multitrackSegment.taskLabel', {
          n: segmentIndex + 1,
          mode: taskModeLabel,
        })
      : taskModeLabel
    : trackType === 'subtitle'
      ? segment.content.text ?? t('multitrack.subtitle')
      : segment.content.file_name ?? segment.id
  const durationLabel = formatMultiTrackTime(segmentDuration, { frameRate, showFrames: true })
  const taskPrompt = segment.content.user_prompt?.trim() ?? ''
  const taskImageUrls = (segment.content.images ?? []).flatMap((image) => {
    const url = mediaContentToViewUrl({
      source_type: image.source_type ?? 'input',
      file_path: image.file_path,
      local_path: image.local_path,
      url: image.url,
      slot_name: image.slot_name,
    })
    return url ? [url] : []
  })
  const tiledTaskImages = Array.from({ length: 24 }, (_, repeatIndex) => (
    taskImageUrls.map((url, imageIndex) => ({
      key: `${repeatIndex}-${imageIndex}`,
      url,
    }))
  )).flat()
  const presentation = getSegmentTrackPresentation(trackType)
  const cloneLabel = trackType === 'video'
    ? t('multitrack.cloneVideoSegment')
    : trackType === 'audio'
      ? t('multitrack.cloneAudioSegment')
      : t('multitrack.cloneTaskSegment')
  const borderColor = isResizing
    ? 'var(--warning)'
    : selected
      ? presentation.borderColor
      : presentation.backgroundColorStrong
  const blockStyle: SegmentBlockStyle = {
    left: dragPreview ? 0 : left,
    top: dragPreview ? dragPreview.y : undefined,
    bottom: dragPreview ? undefined : undefined,
    width: dragPreview ? dragPreview.width : width,
    height: dragPreview ? dragPreview.height : undefined,
    position: 'absolute',
    transform: dragPreview ? `translate3d(${dragPreview.x}px, 0, 0)` : undefined,
    backgroundColor: presentation.backgroundColor,
    color: presentation.textColor,
    border: `1px solid ${borderColor}`,
    '--multitrack-waveform': presentation.waveformColor ?? undefined,
    cursor: cutMode ? 'text' : cursorStyle ?? 'grab',
    zIndex: isDragging ? 9999 : selected ? 30 : 1,
    boxShadow: isDragging ? '0 8px 24px rgb(0 0 0 / 0.35)' : undefined,
    pointerEvents: dragPreview ? 'none' : undefined,
  }

  function updateDragPreview(nextPreview: typeof dragPreviewRef.current) {
    dragPreviewRef.current = nextPreview
    if (dragFrameRef.current !== null) return
    dragFrameRef.current = requestAnimationFrame(() => {
      dragFrameRef.current = null
      setDragPreview(dragPreviewRef.current)
      if (dragPreviewRef.current) {
        onDragPreviewChange?.(segment.id, dragPreviewRef.current.nextStartTime, dragPreviewRef.current.clientY)
      }
    })
  }

  useEffect(() => {
    if (!presentation.showThumbnail || !mediaUrl) {
      setPosterUrl(null)
      return
    }

    let cancelled = false
    let objectUrl: string | null = null

    captureVideoPosterFrame(mediaUrl, sourceStartTime)
      .then((nextPosterUrl) => {
        if (cancelled) {
          URL.revokeObjectURL(nextPosterUrl)
          return
        }
        objectUrl = nextPosterUrl
        setPosterUrl(nextPosterUrl)
      })
      .catch((error: unknown) => {
        console.error('[MultiTrackSegmentBlock] failed to capture video poster:', error)
        if (!cancelled) setPosterUrl(null)
      })

    return () => {
      cancelled = true
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [mediaUrl, presentation.showThumbnail, sourceStartTime])

  useEffect(() => {
    return () => {
      if (dragFrameRef.current !== null) {
        cancelAnimationFrame(dragFrameRef.current)
      }
    }
  }, [])

  function resizeZone(): number {
    return selected ? RESIZE_ZONE_SELECTED : RESIZE_ZONE_DEFAULT
  }

  function renderedCoordinateScale(rectWidth: number): number {
    const measuredScale = rectWidth / Math.max(width, 1)
    return Number.isFinite(measuredScale) && measuredScale > 0.01
      ? measuredScale
      : Math.max(canvasScale, 0.01)
  }

  function resizeDeltaFromClientX(clientX: number, startX: number, coordinateScale: number): number {
    const adjustedDelta = (clientX - startX) / coordinateScale
    return (adjustedDelta / Math.max(areaWidth, 1)) * Math.max(totalLength, 1)
  }

  function resizeSnapBrakeDistanceFrames(): number {
    return Math.max(1, Math.round((RESIZE_SNAP_DISTANCE / Math.max(areaWidth, 1)) * Math.max(totalLength, 1)))
  }

  function handleMouseMove(event: React.MouseEvent) {
    if (cutMode) return
    const rect = event.currentTarget.getBoundingClientRect()
    const coordinateScale = renderedCoordinateScale(rect.width)
    const relX = (event.clientX - rect.left) / coordinateScale
    const zone = resizeZone()
    setCursorStyle(relX <= zone || relX >= rect.width / coordinateScale - zone ? 'ew-resize' : 'grab')
  }

  function handleMouseDown(event: React.MouseEvent) {
    if (event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    if (cutMode) return
    didDragRef.current = false
    isDraggingRef.current = false
    onSelect(segment.id, event.metaKey || event.ctrlKey ? 'toggle' : event.shiftKey || selected ? 'add' : 'replace')

    const rect = event.currentTarget.getBoundingClientRect()
    const coordinateScale = renderedCoordinateScale(rect.width)
    const relX = (event.clientX - rect.left) / coordinateScale
    const visualWidth = rect.width / coordinateScale
    const zone = resizeZone()
    const edge = relX <= zone ? 'start' : relX >= visualWidth - zone ? 'end' : null

    const startX = event.clientX
    const startY = event.clientY
    if (!edge) {
      const originalStart = segment.start_frame
      const startRect = event.currentTarget.getBoundingClientRect()
      const currentTarget = event.currentTarget as HTMLElement
      const containerRect = currentTarget.offsetParent?.getBoundingClientRect()
      const scale = coordinateScale
      const pointerOffsetY = (event.clientY - startRect.top) / scale
      const previewWidth = startRect.width / scale
      const previewHeight = startRect.height / scale

      function handleMove(moveEvent: MouseEvent) {
        const deltaFrames = resizeDeltaFromClientX(moveEvent.clientX, startX, coordinateScale)
        const distance = Math.hypot(moveEvent.clientX - startX, moveEvent.clientY - startY)
        if (distance < DRAG_START_DISTANCE) return
        if (!containerRect) return
        didDragRef.current = true
        if (!isDraggingRef.current) {
          isDraggingRef.current = true
          setIsDragging(true)
        }
        const requestedStartTime = originalStart + deltaFrames
        const previewStartTime = getDragPreviewStart?.(segment.id, requestedStartTime, moveEvent.clientY) ?? requestedStartTime
        const x = (previewStartTime / Math.max(totalLength, 1)) * areaWidth
        const y = (moveEvent.clientY - containerRect.top) / scale - pointerOffsetY
        updateDragPreview({
          x,
          y,
          width: previewWidth,
          height: previewHeight,
          nextStartTime: previewStartTime,
          clientY: moveEvent.clientY,
        })
      }

      function handleUp() {
        if (dragFrameRef.current !== null) {
          cancelAnimationFrame(dragFrameRef.current)
          dragFrameRef.current = null
        }
        isDraggingRef.current = false
        setIsDragging(false)
        const finalPreview = dragPreviewRef.current
        if (didDragRef.current && finalPreview) onMove(segment.id, finalPreview.nextStartTime, finalPreview.clientY)
        dragPreviewRef.current = null
        setDragPreview(null)
        onDragPreviewEnd?.()
        globalThis.removeEventListener('mousemove', handleMove)
        globalThis.removeEventListener('mouseup', handleUp)
      }

      globalThis.addEventListener('mousemove', handleMove)
      globalThis.addEventListener('mouseup', handleUp)
      return
    }

    const resizeEdge = edge
    const originalTime = resizeEdge === 'start' ? segment.start_frame : segment.end_frame
    const resizeAnchorX = resizeEdge === 'start' ? rect.left : rect.right
    const brakeDistanceFrames = resizeSnapBrakeDistanceFrames()
    setIsResizing(true)

    function handleMove(moveEvent: MouseEvent) {
      const deltaFrames = resizeDeltaFromClientX(moveEvent.clientX, resizeAnchorX, coordinateScale)
      if (Math.abs(deltaFrames) > 0) didDragRef.current = true
      onResizePreview(segment.id, resizeEdge, originalTime + deltaFrames, brakeDistanceFrames)
    }

    function handleUp(upEvent: MouseEvent) {
      const deltaFrames = resizeDeltaFromClientX(upEvent.clientX, resizeAnchorX, coordinateScale)
      if (didDragRef.current) onResize(segment.id, resizeEdge, originalTime + deltaFrames, brakeDistanceFrames)
      setIsResizing(false)
      globalThis.removeEventListener('mousemove', handleMove)
      globalThis.removeEventListener('mouseup', handleUp)
    }

    globalThis.addEventListener('mousemove', handleMove)
    globalThis.addEventListener('mouseup', handleUp)
  }

  return (
    <>
      <ContextMenu>
        <ContextMenuTrigger asChild>
        <div
          role="button"
          data-multitrack-segment=""
          tabIndex={0}
          className={`absolute top-1 bottom-1 flex items-center rounded select-none active:opacity-70 ${trackType === 'audio' ? 'overflow-visible' : 'overflow-hidden'} ${dimmed ? 'opacity-50' : ''}`}
          style={blockStyle}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setCursorStyle(null)}
          onClick={(event) => {
            event.stopPropagation()
            if (cutMode && onCut) {
              const rect = event.currentTarget.getBoundingClientRect()
              const visualWidth = rect.width / Math.max(canvasScale, 0.01)
              const x = (event.clientX - rect.left) / Math.max(canvasScale, 0.01)
              const ratio = Math.max(0, Math.min(1, x / Math.max(visualWidth, 1)))
              const splitFrame = Math.round(
                segment.start_frame + ratio * (segment.end_frame - segment.start_frame),
              )
              onCut(segment.id, splitFrame)
              return
            }
            if (!didDragRef.current) event.preventDefault()
          }}
          onContextMenu={(event) => {
            event.stopPropagation()
            onSelect(segment.id, selected ? 'add' : 'replace')
          }}
          onDoubleClick={(event) => {
            event.preventDefault()
            event.stopPropagation()
            if (cutMode) return
            onDoubleClick?.(segment.id, event)
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter') onSelect(segment.id, event.metaKey || event.ctrlKey ? 'toggle' : event.shiftKey ? 'add' : 'replace')
            if (event.key === 'Delete' || event.key === 'Backspace') onDelete(segment.id)
          }}
        >
          <div className="pointer-events-none relative flex h-full min-w-0 flex-1 flex-col gap-0.5 overflow-hidden rounded-sm">
            {trackType === 'task' && taskOverview ? (
              <>
                {tiledTaskImages.length > 0 ? (
                  <div className="absolute inset-0 overflow-hidden bg-black">
                    <div className="flex h-full w-max">
                      {tiledTaskImages.map(({ key, url }) => (
                        <img
                          key={key}
                          src={url}
                          alt=""
                          className="h-full w-auto max-w-none shrink-0 object-contain"
                          draggable={false}
                          onError={(event) => {
                            event.currentTarget.style.display = 'none'
                          }}
                        />
                      ))}
                    </div>
                  </div>
                ) : null}
                <div className={tiledTaskImages.length > 0
                  ? 'absolute inset-x-0 bottom-0 z-10 flex items-center bg-black px-1 py-0.5 text-white'
                  : 'absolute inset-0 flex items-center justify-center px-1 text-center'}
                >
                  <span className={`min-w-0 text-[9px] leading-tight ${tiledTaskImages.length > 0 ? 'truncate' : 'line-clamp-3'}`}>
                    {taskPrompt || label}
                  </span>
                </div>
              </>
            ) : (
              <>
                <div
                  className={`flex h-3.5 min-w-0 items-center gap-1 leading-none ${presentation.textClassName}`}
                  style={{ backgroundColor: presentation.titleBackgroundColor ?? undefined }}
                >
                  <span className="truncate rounded-sm px-1 font-medium">
                    {label}
                  </span>
                  <span className="shrink-0 rounded-sm px-1 tabular-nums">
                    {durationLabel}
                  </span>
                </div>
                {presentation.showThumbnail ? (
                  <div
                    className="h-8 overflow-hidden bg-black"
                    style={{
                      backgroundImage: posterUrl ? `url(${JSON.stringify(posterUrl)})` : undefined,
                      backgroundPosition: 'left center',
                      backgroundRepeat: 'repeat-x',
                      backgroundSize: 'auto 32px',
                    }}
                  />
                ) : null}
                {presentation.showWaveform ? (
                  <div
                    className={trackType === 'audio'
                      ? 'relative min-h-0 flex-1 overflow-hidden rounded-sm'
                      : 'relative h-2.5 overflow-hidden rounded-sm'}
                    style={{ backgroundColor: presentation.backgroundColorStrong }}
                  >
                    <div className="absolute inset-x-0 top-0 z-10 h-0.5 bg-warning" />
                    <AudioWaveform
                      content={{
                        source_type: segment.content.source_type ?? 'input',
                        file_path: segment.content.file_path,
                        local_path: segment.content.local_path,
                        url: segment.content.url,
                        slot_name: segment.content.slot_name,
                      }}
                      startRatio={waveformStartRatio}
                      endRatio={waveformEndRatio}
                      className="h-full w-full"
                      color={presentation.waveformColor ?? undefined}
                    />
                  </div>
                ) : trackType !== 'audio' ? (
                  <div className="min-h-0 flex-1" />
                ) : null}
              </>
            )}
          </div>
          <span
            className="absolute left-0 top-0 h-full w-0.5 cursor-ew-resize"
            style={{ background: isResizing || selected ? borderColor : 'transparent' }}
          />
          <span
            className="absolute right-0 top-0 h-full w-0.5 cursor-ew-resize"
            style={{ background: isResizing || selected ? borderColor : 'transparent' }}
          />
          {trackType === 'audio' || trackType === 'video' ? (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                  type="button"
                  size="icon"
                  variant={sharedReference ? 'secondary' : 'ghost'}
                  data-testid={`${trackType}-shared-reference`}
                  aria-label={sharedReference ? t('multitrack.disableSharedReference') : t('multitrack.enableSharedReference')}
                  aria-pressed={sharedReference}
                  className={`absolute ${audioLockEnabled ? 'right-8' : 'right-2'} top-0.5 z-20 h-5 w-5 cursor-pointer bg-background/80 shadow-sm [&_svg]:!size-3 ${sharedReference ? 'text-highlight' : 'text-muted-foreground'}`}
                  onMouseDown={(event) => {
                    event.preventDefault()
                    event.stopPropagation()
                  }}
                  onClick={(event) => {
                    event.preventDefault()
                    event.stopPropagation()
                    onSharedReferenceToggle?.(!sharedReference)
                  }}
                  >
                    <Share2 />
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-72">
                  {t('multitrack.sharedReferenceTooltip')}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
          ) : null}
          {(trackType === 'audio' || trackType === 'video') && audioLockEnabled ? (
            <>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    type="button"
                    size="icon"
                    variant={audioLocked ? 'secondary' : 'ghost'}
                    data-testid={trackType === 'audio' ? 'audio-segment-lock' : 'video-segment-audio-lock'}
                    aria-label={audioLocked ? t('multitrack.unlockAudio') : t('multitrack.lockAudio')}
                    aria-pressed={audioLocked}
                    className={`absolute right-2 top-0.5 z-20 h-5 w-5 cursor-pointer bg-background/80 shadow-sm [&_svg]:!size-3 ${audioLocked ? 'text-highlight' : 'text-muted-foreground'}`}
                    onMouseDown={(event) => {
                      event.preventDefault()
                      event.stopPropagation()
                    }}
                    onClick={(event) => {
                      event.preventDefault()
                      event.stopPropagation()
                      onAudioLockToggle?.(!audioLocked)
                    }}
                  >
                    {audioLocked ? <Lock/> : <Unlock/>}
                  </Button>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-72">
                  {t(
                    trackType === 'video'
                      ? 'multitrack.videoAudioLockTooltip'
                      : 'multitrack.audioLockTooltip',
                  )}
                </TooltipContent>
              </Tooltip>
            </>
          ) : null}
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        {trackType === 'video' && onSmartSplit ? (
          <ContextMenuItem onClick={() => onSmartSplit(segment.id)}>
            {t('multitrack.smartSplit')}
          </ContextMenuItem>
        ) : null}
        {trackType === 'video' && onSmartSplitTasks ? (
          <ContextMenuItem onClick={() => onSmartSplitTasks(segment.id)}>
            {t('multitrack.smartSplitTasksOnly')}
          </ContextMenuItem>
        ) : null}
        {(trackType === 'video' || trackType === 'audio') && onRecognizeSubtitles ? (
          <ContextMenuSub>
            <ContextMenuSubTrigger>
              {t('multitrack.recognizeSubtitles')}
            </ContextMenuSubTrigger>
            <ContextMenuSubContent>
              <ContextMenuItem onClick={() => onRecognizeSubtitles(segment.id, 'whisper-large-v3')}>
                {t('multitrack.recognizeSubtitlesWhisper')}
              </ContextMenuItem>
              <ContextMenuItem onClick={() => onRecognizeSubtitles(segment.id, 'qwen3-asr')}>
                {t('multitrack.recognizeSubtitlesQwen3')}
              </ContextMenuItem>
            </ContextMenuSubContent>
          </ContextMenuSub>
        ) : null}
        {trackType === 'task' && onDistribute ? (
          <ContextMenuItem onClick={onDistribute}>
            {t('multitrack.distributeTaskSegments')}
          </ContextMenuItem>
        ) : null}
        {onClone ? (
          <ContextMenuItem onClick={() => onClone(segment.id)}>
            {cloneLabel}
          </ContextMenuItem>
        ) : null}
        {trackType === 'task' && onSplitTask ? (
          <ContextMenuItem onClick={() => onSplitTask(segment.id)}>
            {t('multitrack.splitTaskSegment')}
          </ContextMenuItem>
        ) : null}
        <ContextMenuItem onClick={() => onDelete(segment.id)}>
          {t('multitrack.deleteSegment')}
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
    </>
  )
}
