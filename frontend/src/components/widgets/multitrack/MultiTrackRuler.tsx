import { useRef, useState } from 'react'
import { formatMultiTrackTime } from '@/lib/multitrack-utils'
import type { MultiTrackTaskMarker } from '@/types/multitrack'
import { MultiTrackTaskMarker as TaskMarker } from './MultiTrackTaskMarker'

export const MULTITRACK_LEFT_GUTTER = 28
export const MULTITRACK_RIGHT_RESERVE = 48
const MIN_LABEL_SPACING_PX = 56
const SECOND_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300] as const

interface MultiTrackRulerProps {
  totalLength: number
  frameRate: number
  width: number
  canvasScale: number
  currentTime: number
  taskMarkers: MultiTrackTaskMarker[]
  selectedTaskMarkerId: string | null
  onSeek: (time: number) => void
  onSelectTaskMarker: (markerId: string) => void
  onMoveTaskMarker: (markerId: string, frame: number) => void
  onDeleteTaskMarker: (markerId: string) => void
}

export function buildTicks(totalLength: number, frameRate: number, timelineWidth: number) {
  const safeFrameRate = Math.max(1, frameRate)
  const safeLength = Math.max(totalLength, 1)
  const pixelsPerFrame = timelineWidth / safeLength
  const showFrames = pixelsPerFrame >= 6
  const ticks: Array<{ time: number; label?: string; major: boolean }> = []
  let lastLabelTime = Number.NEGATIVE_INFINITY

  function hasLabelSpacing(time: number): boolean {
    return (time - lastLabelTime) * pixelsPerFrame >= MIN_LABEL_SPACING_PX
  }

  function pushTick(time: number, label: string | undefined, major: boolean) {
    if (label) lastLabelTime = time
    ticks.push({ time, label, major })
  }

  if (showFrames) {
    const minLabelFrameStep = Math.max(1, Math.ceil(MIN_LABEL_SPACING_PX / Math.max(pixelsPerFrame, 1)))
    const frameStepUnit = Math.max(1, Math.round(safeFrameRate / 8))
    const frameLabelStep = Math.max(1, Math.ceil(minLabelFrameStep / frameStepUnit) * frameStepUnit)
    for (let frame = 0; frame <= totalLength; frame += 1) {
      const isStepLabel = frame % frameLabelStep === 0
      const isEndLabel = frame === totalLength && hasLabelSpacing(frame)
      const major = isStepLabel || isEndLabel
      pushTick(
        frame,
        major ? formatMultiTrackTime(frame, { frameRate: safeFrameRate, showFrames: true }) : undefined,
        major,
      )
    }
    return ticks
  }

  const minStepSeconds = MIN_LABEL_SPACING_PX / Math.max(pixelsPerFrame * safeFrameRate, 1)
  const stepSeconds = SECOND_STEPS.find((candidate) => candidate >= minStepSeconds) ?? SECOND_STEPS[SECOND_STEPS.length - 1]
  const stepFrames = Math.max(1, Math.round(stepSeconds * safeFrameRate))
  for (let frame = 0; frame <= totalLength; frame += stepFrames) {
    pushTick(frame, formatMultiTrackTime(frame, { frameRate: safeFrameRate, showFrames: true }), true)
  }
  if (ticks.at(-1)?.time !== totalLength && hasLabelSpacing(totalLength)) {
    pushTick(totalLength, formatMultiTrackTime(totalLength, { frameRate: safeFrameRate, showFrames: true }), true)
  }
  return ticks
}

export function MultiTrackRuler({
  totalLength,
  frameRate,
  width,
  canvasScale,
  currentTime,
  taskMarkers,
  selectedTaskMarkerId,
  onSeek,
  onSelectTaskMarker,
  onMoveTaskMarker,
  onDeleteTaskMarker,
}: Readonly<MultiTrackRulerProps>) {
  const rulerRef = useRef<HTMLDivElement>(null)
  const [dragPreview, setDragPreview] = useState<{ markerId: string; frame: number } | null>(null)
  const timelineWidth = Math.max(1, width - MULTITRACK_LEFT_GUTTER)
  const playableWidth = Math.max(1, timelineWidth - MULTITRACK_RIGHT_RESERVE)
  const ticks = buildTicks(totalLength, frameRate, playableWidth)
  const safeLength = Math.max(totalLength, 1)
  const playheadLeft = MULTITRACK_LEFT_GUTTER + (currentTime / safeLength) * playableWidth
  const reserveLeft = MULTITRACK_LEFT_GUTTER + playableWidth

  function renderedCoordinateScale(rectWidth: number): number {
    const measuredScale = rectWidth / Math.max(width, 1)
    return Number.isFinite(measuredScale) && measuredScale > 0.01
      ? measuredScale
      : Math.max(canvasScale, 0.01)
  }

  function timeFromClientX(clientX: number, rectLeft: number, coordinateScale: number): number {
    const x = Math.max(0, (clientX - rectLeft) / coordinateScale - MULTITRACK_LEFT_GUTTER)
    const ratio = x / playableWidth
    return Math.max(0, Math.min(totalLength, Math.round(ratio * totalLength)))
  }

  function handleMouseDown(event: React.MouseEvent<HTMLDivElement>) {
    event.preventDefault()
    const rect = event.currentTarget.getBoundingClientRect()
    const coordinateScale = renderedCoordinateScale(rect.width)
    onSeek(timeFromClientX(event.clientX, rect.left, coordinateScale))

    function handleMouseMove(moveEvent: MouseEvent) {
      onSeek(timeFromClientX(moveEvent.clientX, rect.left, coordinateScale))
    }

    function handleMouseUp() {
      globalThis.removeEventListener('mousemove', handleMouseMove)
      globalThis.removeEventListener('mouseup', handleMouseUp)
    }

    globalThis.addEventListener('mousemove', handleMouseMove)
    globalThis.addEventListener('mouseup', handleMouseUp)
  }

  function handleTaskMarkerDragStart(markerId: string, clientX: number) {
    const marker = taskMarkers.find((candidate) => candidate.id === markerId)
    if (!marker) return
    const rulerRect = rulerRef.current?.getBoundingClientRect()
    const rectLeft = rulerRect?.left ?? 0
    const coordinateScale = renderedCoordinateScale(rulerRect?.width ?? 0)
    const originalFrame = marker.frame
    let lastValidFrame = originalFrame

    function updateDragPreview(nextClientX: number) {
      const nextFrame = Math.max(1, timeFromClientX(nextClientX, rectLeft, coordinateScale))
      const occupied = taskMarkers.some((candidate) => (
        candidate.id !== markerId && candidate.frame === nextFrame
      ))
      if (occupied) return
      lastValidFrame = nextFrame
      setDragPreview({ markerId, frame: nextFrame })
    }

    function handleMouseMove(event: MouseEvent) {
      updateDragPreview(event.clientX)
    }

    function handleMouseUp(event: MouseEvent) {
      updateDragPreview(event.clientX)
      globalThis.removeEventListener('mousemove', handleMouseMove)
      globalThis.removeEventListener('mouseup', handleMouseUp)
      setDragPreview(null)
      if (lastValidFrame !== originalFrame) onMoveTaskMarker(markerId, lastValidFrame)
    }

    updateDragPreview(clientX)
    globalThis.addEventListener('mousemove', handleMouseMove)
    globalThis.addEventListener('mouseup', handleMouseUp)
  }

  return (
    <div
      ref={rulerRef}
      className={`relative h-6 shrink-0 select-none border-b border-border ${dragPreview ? 'cursor-grabbing' : 'cursor-col-resize'}`}
      style={{ width }}
      onMouseDown={handleMouseDown}
    >
      <div className="absolute left-0 top-0 h-full border-r border-border" style={{ width: MULTITRACK_LEFT_GUTTER }} />
      <div
        className="pointer-events-none absolute top-0 h-full bg-black/30"
        style={{ left: reserveLeft, width: MULTITRACK_RIGHT_RESERVE }}
      />
      {ticks.map((tick) => (
        <div
          key={`${tick.time}-${tick.major ? 'major' : 'minor'}`}
          className="absolute bottom-0 flex h-full flex-col justify-end"
          style={{ left: MULTITRACK_LEFT_GUTTER + (tick.time / safeLength) * playableWidth }}
        >
          <div className={tick.major ? 'h-3 w-px bg-border' : 'h-1.5 w-px bg-border'} />
          {tick.label && <span className="absolute left-1 top-1 text-[8px] text-muted-foreground">{tick.label}</span>}
        </div>
      ))}
      {taskMarkers.map((marker, index) => {
        const dragging = dragPreview?.markerId === marker.id
        const displayMarker = dragging
          ? { ...marker, frame: dragPreview.frame }
          : marker
        return (
          <TaskMarker
            key={displayMarker.id}
            marker={displayMarker}
            markerNumber={index + 1}
            frameRate={frameRate}
            left={MULTITRACK_LEFT_GUTTER + (displayMarker.frame / safeLength) * playableWidth}
            selected={displayMarker.id === selectedTaskMarkerId}
            dragging={dragging}
            onSelect={onSelectTaskMarker}
            onDragStart={handleTaskMarkerDragStart}
            onDelete={onDeleteTaskMarker}
          />
        )
      })}
      <div className="absolute top-0 h-full w-px bg-destructive" style={{ left: playheadLeft }} />
    </div>
  )
}
