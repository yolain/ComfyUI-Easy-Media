import { useEffect, useMemo, useRef } from 'react'
import type { WheelEvent } from 'react'
import type { TimeDisplayFormat } from '@/types/timeline'

interface TimelineRulerProps {
  totalFrames: number
  frameRate: number
  displayFormat: TimeDisplayFormat
  /** pixel width of the ruler (matches track content area) */
  width: number
  /** Current scale of the comfyui canvas */
  canvasScale: number
  /** Current playhead position in frames */
  playheadFrame?: number
  /** Show seek label (click or playback stop) */
  showLabel: boolean
  /** Called when the user clicks or drags on the ruler to seek */
  onSeek?: (frame: number) => void
  /** Called when the user scrolls on the ruler to move the visible time range */
  onWheel?: (e: WheelEvent<HTMLDivElement>) => void
}

/**
 * Compute a tick interval (in frames) that gives roughly `targetTicks` ticks.
 * In seconds mode the interval is snapped to a round seconds value so labels
 * always land on clean multiples of 0.5 s, 1 s, 5 s, etc.
 */
function computeTickInterval(
  totalFrames: number,
  targetTicks: number,
  frameRate: number,
  displayFormat: TimeDisplayFormat,
): number {
  if (displayFormat === 'seconds') {
    const totalSecs = (totalFrames - 1) / frameRate
    const rawSecs = totalSecs / targetTicks
    const secCandidates = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]
    const bestSec = secCandidates.reduce((prev, curr) =>
      Math.abs(curr - rawSecs) < Math.abs(prev - rawSecs) ? curr : prev,
      secCandidates[0],
    )
    return Math.max(1, Math.round(bestSec * frameRate))
  }
  const raw = totalFrames / targetTicks
  const candidates = [1, 2, 4, 5, 6, 8, 10, 12, 15, 20, 24, 25, 30, 48, 50, 60, 100, 120, 240, 300]
  return candidates.reduce((prev, curr) =>
    Math.abs(curr - raw) < Math.abs(prev - raw) ? curr : prev,
    candidates[0],
  )
}

export function TimelineRuler({
  totalFrames,
  frameRate,
  displayFormat,
  width,
  canvasScale,
  playheadFrame,
  showLabel,
  onSeek,
  onWheel,
}: Readonly<TimelineRulerProps>) {
  const isDragging = useRef(false)

  useEffect(() => {
    return () => {
      isDragging.current = false
    }
  }, [])

  function frameFromX(x: number): number {
    return Math.max(
      0,
      Math.min(totalFrames - 1, Math.round((x / width) * (totalFrames - 1))),
    )
  }

  function handleMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    if (!onSeek) return
    e.preventDefault()
    isDragging.current = true
    const rect = e.currentTarget.getBoundingClientRect()
    const x = (e.clientX - rect.left) / canvasScale
    onSeek(frameFromX(x))

    function handleMouseMove(ev: MouseEvent) {
      if (!isDragging.current) return
      const x = (ev.clientX - rect.left) / canvasScale
      onSeek!(frameFromX(x))
    }
    function handleMouseUp() {
      isDragging.current = false
      globalThis.removeEventListener('mousemove', handleMouseMove)
      globalThis.removeEventListener('mouseup', handleMouseUp)
    }
    globalThis.addEventListener('mousemove', handleMouseMove)
    globalThis.addEventListener('mouseup', handleMouseUp)
  }

  const playheadX = playheadFrame === undefined
    ? undefined
    : (playheadFrame / Math.max(totalFrames - 1, 1)) * width 
  const ticks = useMemo(() => {
    const targetTicks = Math.max(5, Math.floor(width / 60))
    const interval = computeTickInterval(totalFrames, targetTicks, frameRate, displayFormat)
    const result: Array<{ frame: number; label: string; major: boolean }> = []
    for (let f = 0; f < totalFrames; f += interval) {
      const label =
        displayFormat === 'seconds'
          ? `${Math.max(0, f / frameRate).toFixed(1)}s`
          : `${f}f`
      result.push({ frame: f, label, major: true })
    }
    // Always include a tick at the last frame
    const last = totalFrames - 1
    if (!result.some((t) => t.frame === last)) {
      const label =
        displayFormat === 'seconds'
          ? `${Math.max(0, last / frameRate).toFixed(1)}s`
          : `${last}f`
      result.push({ frame: last, label, major: false })
    }
    return result
  }, [totalFrames, frameRate, displayFormat, width])

  function frameToX(frame: number) {
    return (frame / Math.max(totalFrames - 1, 1)) * width 
  }

  return (
    <div
      role="slider"
      aria-label="Playhead position"
      aria-valuenow={playheadFrame ?? 0}
      aria-valuemin={0}
      aria-valuemax={totalFrames - 1}
      tabIndex={onSeek ? 0 : -1}
      className="relative h-6 shrink-0 select-none border-b border-border overflow-hidden cursor-col-resize"
      style={{ width }}
      onMouseDown={handleMouseDown}
      onWheel={onWheel}
      onKeyDown={(e) => {
        if (!onSeek) return
        const step = e.shiftKey ? 10 : 1
        if (e.key === 'ArrowLeft') onSeek(Math.max(0, (playheadFrame ?? 0) - step))
        else if (e.key === 'ArrowRight') onSeek(Math.min(totalFrames - 1, (playheadFrame ?? 0) + step))
      }}
    >
      {ticks.map(({ frame, label, major }) => (
        <div
          key={frame}
          className="absolute top-0 flex flex-col items-start pointer-events-none"
          style={{ left: frameToX(frame) }}
        >
          <div
            className={`w-px ${major ? 'h-2 bg-border' : 'h-1.5 bg-border'}`}
          />
          {major && (
            <span className="text-[9px] text-muted-foreground/70 whitespace-nowrap translate-x-1">
              {label}
            </span>
          )}
        </div>
      ))}

      {/* Playhead indicator */}
      {playheadX !== undefined && (
        <div
          className="absolute top-0 h-full w-px bg-red-400 pointer-events-none z-10"
          style={{ left: playheadX }}
        >
          {/* Triangle head */}
          <div
            className="absolute top-0 -translate-x-1/2 w-0 h-0 pointer-events-none"
            style={{
              borderLeft: '4px solid transparent',
              borderRight: '4px solid transparent',
              borderTop: '6px solid rgb(248 113 113)',
            }}
          />
        </div>
      )}

      {/* Seek label */}
      {showLabel && playheadX !== undefined && (
        <div
          className="absolute top-0 -translate-x-1/2 z-20 pointer-events-none"
          style={{ left: playheadX, top: '-2px' }}
        >
          <div className="bg-red-400 text-white text-[8px] font-medium px-1.5 py-0.5 rounded whitespace-nowrap shadow">
            {displayFormat === 'seconds'
              ? `${Math.max(0, (playheadFrame ?? 0) / frameRate).toFixed(1)}s`
              : `${playheadFrame ?? 0}f`}
          </div>
        </div>
      )}
    </div>
  )
}
