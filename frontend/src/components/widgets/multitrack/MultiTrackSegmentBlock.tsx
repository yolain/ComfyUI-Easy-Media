import { useEffect, useRef, useState } from 'react'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'
import { AudioWaveform } from '@/components/widgets/timeline/AudioWaveform'
import { mediaContentToViewUrl } from '@/lib/media-url'
import { useT } from '@/lib/i18n'
import { formatMultiTrackTime, getMultiTrackTaskModeLabel } from '@/lib/multitrack-utils'
import { getSegmentTrackPresentation } from '@/lib/multitrack-segment-style'
import { captureVideoPosterFrame } from '@/lib/video-utils'
import type { MultiTrackSegment, MultiTrackType } from '@/types/multitrack'

const RESIZE_ZONE_DEFAULT = 8
const RESIZE_ZONE_SELECTED = 12
const DRAG_START_DISTANCE = 4

interface MultiTrackSegmentBlockProps {
  trackType: MultiTrackType
  segmentIndex: number
  segment: MultiTrackSegment
  totalLength: number
  frameRate: number
  areaWidth: number
  canvasScale: number
  selected: boolean
  onSelect: (segmentId: string) => void
  onDelete: (segmentId: string) => void
  onResize: (segmentId: string, edge: 'start' | 'end', nextTime: number) => void
  onMove: (segmentId: string, nextStartTime: number, clientY: number) => void
  onDragPreviewChange?: (segmentId: string, nextStartTime: number, clientY: number) => void
  onDragPreviewEnd?: () => void
  onDoubleClick?: (segmentId: string, event: React.MouseEvent) => void
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
  onResize,
  onMove,
  onDragPreviewChange,
  onDragPreviewEnd,
  onDoubleClick,
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
  const label = trackType === 'task'
    ? t('multitrackSegment.taskLabel', {
      n: segmentIndex,
      mode: getMultiTrackTaskModeLabel(segment.content.task_mode ?? 'default', t),
    })
    : segment.content.file_name ?? segment.id
  const durationLabel = formatMultiTrackTime(segmentDuration, { frameRate, showFrames: true })
  const presentation = getSegmentTrackPresentation(trackType)
  const borderColor = isResizing
    ? 'var(--warning)'
    : selected
      ? presentation.borderColor
      : presentation.backgroundColorStrong

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

    captureVideoPosterFrame(mediaUrl)
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
  }, [mediaUrl, presentation.showThumbnail])

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

  function resizeDeltaFromClientX(clientX: number, startX: number): number {
    const adjustedDelta = (clientX - startX) / Math.max(canvasScale, 0.01)
    return (adjustedDelta / Math.max(areaWidth, 1)) * Math.max(totalLength, 1)
  }

  function handleMouseMove(event: React.MouseEvent) {
    const rect = event.currentTarget.getBoundingClientRect()
    const relX = (event.clientX - rect.left) / Math.max(canvasScale, 0.01)
    const zone = resizeZone()
    setCursorStyle(relX <= zone || relX >= rect.width / Math.max(canvasScale, 0.01) - zone ? 'ew-resize' : 'grab')
  }

  function handleMouseDown(event: React.MouseEvent) {
    if (event.button !== 0) return
    event.preventDefault()
    event.stopPropagation()
    didDragRef.current = false
    isDraggingRef.current = false
    onSelect(segment.id)

    const rect = event.currentTarget.getBoundingClientRect()
    const relX = (event.clientX - rect.left) / Math.max(canvasScale, 0.01)
    const visualWidth = rect.width / Math.max(canvasScale, 0.01)
    const zone = resizeZone()
    const edge = relX <= zone ? 'start' : relX >= visualWidth - zone ? 'end' : null

    const startX = event.clientX
    const startY = event.clientY
    if (!edge) {
      const originalStart = segment.start_frame
      const startRect = event.currentTarget.getBoundingClientRect()
      const currentTarget = event.currentTarget as HTMLElement
      const containerRect = currentTarget.offsetParent?.getBoundingClientRect()
      const scale = Math.max(canvasScale, 0.01)
      const pointerOffsetX = (event.clientX - startRect.left) / scale
      const pointerOffsetY = (event.clientY - startRect.top) / scale
      const previewWidth = startRect.width / scale
      const previewHeight = startRect.height / scale

      function handleMove(moveEvent: MouseEvent) {
        const deltaFrames = resizeDeltaFromClientX(moveEvent.clientX, startX)
        const distance = Math.hypot(moveEvent.clientX - startX, moveEvent.clientY - startY)
        if (distance < DRAG_START_DISTANCE) return
        if (!containerRect) return
        didDragRef.current = true
        if (!isDraggingRef.current) {
          isDraggingRef.current = true
          setIsDragging(true)
        }
        const x = (moveEvent.clientX - containerRect.left) / scale - pointerOffsetX
        const y = (moveEvent.clientY - containerRect.top) / scale - pointerOffsetY
        updateDragPreview({
          x,
          y,
          width: previewWidth,
          height: previewHeight,
          nextStartTime: originalStart + deltaFrames,
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
    setIsResizing(true)

    function handleMove(moveEvent: MouseEvent) {
      const deltaFrames = resizeDeltaFromClientX(moveEvent.clientX, startX)
      if (Math.abs(deltaFrames) > 0) didDragRef.current = true
      onResize(segment.id, resizeEdge, originalTime + deltaFrames)
    }

    function handleUp() {
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
          tabIndex={0}
          className="absolute top-1 bottom-1 flex items-center overflow-hidden rounded select-none active:opacity-70"
          style={{
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
            cursor: cursorStyle ?? 'grab',
            zIndex: isDragging ? 9999 : selected ? 30 : 1,
            boxShadow: isDragging ? '0 8px 24px rgb(0 0 0 / 0.35)' : undefined,
            pointerEvents: dragPreview ? 'none' : undefined,
          }}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseLeave={() => setCursorStyle(null)}
          onClick={(event) => {
            event.stopPropagation()
            if (!didDragRef.current) onSelect(segment.id)
          }}
          onContextMenu={(event) => {
            event.stopPropagation()
            onSelect(segment.id)
          }}
          onDoubleClick={(event) => {
            event.preventDefault()
            event.stopPropagation()
            onDoubleClick?.(segment.id, event)
          }}
          onKeyDown={(event) => {
            if (event.key === 'Enter') onSelect(segment.id)
            if (event.key === 'Delete' || event.key === 'Backspace') onDelete(segment.id)
          }}
        >
          <div className="pointer-events-none flex h-full min-w-0 flex-1 flex-col gap-0.5">
            <div className={`flex h-3.5 min-w-0 items-center gap-1 leading-none ${presentation.textClassName}`}>
              <span
                className="truncate rounded-sm px-1 font-medium"
              >
                {label}
              </span>
              <span
                className="shrink-0 rounded-sm px-1 tabular-nums"
              >
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
                className="relative h-2.5 overflow-hidden rounded-sm"
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
                  className="h-full w-full"
                />
              </div>
            ) : (
              <div className="min-h-0 flex-1" />
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
        </div>
      </ContextMenuTrigger>
      <ContextMenuContent>
        <ContextMenuItem onClick={() => onDelete(segment.id)}>
          Delete segment
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
    </>
  )
}
