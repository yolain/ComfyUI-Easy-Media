import { formatMultiTrackTime } from '@/lib/multitrack-utils'

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
  onSeek: (time: number) => void
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
  onSeek,
}: Readonly<MultiTrackRulerProps>) {
  const timelineWidth = Math.max(1, width - MULTITRACK_LEFT_GUTTER)
  const playableWidth = Math.max(1, timelineWidth - MULTITRACK_RIGHT_RESERVE)
  const ticks = buildTicks(totalLength, frameRate, playableWidth)
  const safeLength = Math.max(totalLength, 1)
  const playheadLeft = MULTITRACK_LEFT_GUTTER + (currentTime / safeLength) * playableWidth
  const reserveLeft = MULTITRACK_LEFT_GUTTER + playableWidth

  function timeFromClientX(clientX: number, rectLeft: number): number {
    const x = Math.max(0, (clientX - rectLeft) / canvasScale - MULTITRACK_LEFT_GUTTER)
    const ratio = x / playableWidth
    return Math.max(0, Math.min(totalLength, Math.round(ratio * totalLength)))
  }

  function handleMouseDown(event: React.MouseEvent<HTMLDivElement>) {
    event.preventDefault()
    const rect = event.currentTarget.getBoundingClientRect()
    onSeek(timeFromClientX(event.clientX, rect.left))

    function handleMouseMove(moveEvent: MouseEvent) {
      onSeek(timeFromClientX(moveEvent.clientX, rect.left))
    }

    function handleMouseUp() {
      globalThis.removeEventListener('mousemove', handleMouseMove)
      globalThis.removeEventListener('mouseup', handleMouseUp)
    }

    globalThis.addEventListener('mousemove', handleMouseMove)
    globalThis.addEventListener('mouseup', handleMouseUp)
  }

  return (
    <div
      className="relative h-6 shrink-0 cursor-col-resize select-none border-b border-border"
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
      <div className="absolute top-0 h-full w-px bg-destructive" style={{ left: playheadLeft }} />
    </div>
  )
}
