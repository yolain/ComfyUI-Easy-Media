import React, { useRef, useState } from 'react'
import type { Segment, TimeDisplayFormat } from '@/types/timeline'
import { cn } from '@/lib/utils'
import { useT } from '@/lib/i18n'

/** Resize trigger zone: 8px normally, 20px when selected */
const RESIZE_ZONE_DEFAULT = 8
const RESIZE_ZONE_SELECTED = 12

interface SegmentBlockProps {
  segment: Segment
  totalFrames: number
  areaWidth: number
  /** Whether dragging/resizing this segment is allowed */
  interactive?: boolean
  /** Hide the left resize handle */
  hideLeftHandle?: boolean
  /** Disable left resize functionality (handle still visible if not hidden) */
  disableLeftResize?: boolean
  /** Minimum start frame for left resize (default: 0) */
  minStart?: number
  selected?: boolean
  onClick?: (segment: Segment) => void
  onDoubleClick?: (segment: Segment) => void
  onSelect?: (segment: Segment) => void
  onContextMenu?: (e: React.MouseEvent, segment: Segment) => void
  /** Called on every drag move; origStart is the frame position at mousedown */
  onDragEnd?: (segmentId: string, deltaFrames: number, origStart: number) => void
  onResizeEnd?: (segmentId: string, edge: 'start' | 'end', deltaFrames: number, minStart?: number) => void
  children?: React.ReactNode
  /** Absolutely-positioned background content (e.g. thumbnail image) */
  backgroundSlot?: React.ReactNode
  className?: string
  /** Override inline background/border styles (e.g. bg-card for maintain track) */
  bgStyle?: React.CSSProperties
  /** When provided, shows a time-position label on click/drag */
  frameRate?: number
  displayFormat?: TimeDisplayFormat
}

/** Returns pixel left and width for a segment in the track area. */
function segmentRect(start: number, end: number, total: number, areaW: number) {
  const scale = areaW / Math.max(total - 1, 1)
  const left = start * scale
  const right = Math.min((end + 1) * scale, areaW)
  return { left, width: Math.max(right - left, 2) }
}

/** Lightens a hex color by mixing with white. Returns border fallback for non-hex values. */
function lightenColor(hex: string, amount: number = 0.3): string {
  if (!hex.startsWith('#') || hex.length < 7) return 'var(--border)'
  const r = Number.parseInt(hex.slice(1, 3), 16)
  const g = Number.parseInt(hex.slice(3, 5), 16)
  const b = Number.parseInt(hex.slice(5, 7), 16)

  const newR = Math.round(r + (255 - r) * amount)
  const newG = Math.round(g + (255 - g) * amount)
  const newB = Math.round(b + (255 - b) * amount)

  return `#${newR.toString(16).padStart(2, '0')}${newG.toString(16).padStart(2, '0')}${newB.toString(16).padStart(2, '0')}`
}

export const SegmentBlock = React.forwardRef<HTMLDivElement, SegmentBlockProps>(function SegmentBlock({
  segment,
  totalFrames,
  areaWidth,
  interactive = true,
  hideLeftHandle = false,
  disableLeftResize = false,
  minStart = 0,
  selected = false,
  onClick,
  onDoubleClick,
  onSelect,
  onContextMenu,
  onDragEnd,
  onResizeEnd,
  children,
  backgroundSlot,
  className,
  bgStyle,
  frameRate,
  displayFormat,
}: Readonly<SegmentBlockProps>, ref) {
  const t = useT()
  const [labelVisible, setLabelVisible] = useState(false)
  const labelTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  function showLabelPersistent() {
    setLabelVisible(true)
    if (labelTimerRef.current) clearTimeout(labelTimerRef.current)
  }

  function hideLabelAfterDelay() {
    if (labelTimerRef.current) clearTimeout(labelTimerRef.current)
    labelTimerRef.current = setTimeout(() => setLabelVisible(false), 1500)
  }
  const { left, width } = segmentRect(
    segment.start_frame,
    segment.end_frame,
    totalFrames,
    areaWidth,
  )

  const dragStart = useRef<{ x: number; startFrame: number } | null>(null)
  // Track whether a drag actually happened so we can suppress the click
  const didDrag = useRef(false)
  const [isResizing, setIsResizing] = useState(false)
  // Hover state for cursor change - larger zone when selected
  const [cursorStyle, setCursorStyle] = useState<'ew-resize' | 'grab' | null>(null)

  function getResizeZone(): number {
    return selected ? RESIZE_ZONE_SELECTED : RESIZE_ZONE_DEFAULT
  }

  function handleMouseMove(e: React.MouseEvent) {
    if (!interactive) return
    const rect = e.currentTarget.getBoundingClientRect()
    const relX = e.clientX - rect.left
    const zone = getResizeZone()
    const nearStart = relX <= zone && !disableLeftResize
    const nearEnd = relX >= rect.width - zone

    if (nearStart || nearEnd) {
      setCursorStyle('ew-resize')
    } else {
      setCursorStyle('grab')
    }
  }

  function handleMouseLeave() {
    setCursorStyle(null)
  }

  function handleMouseDown(e: React.MouseEvent) {
    if (!interactive) return
    e.preventDefault()
    didDrag.current = false
    const scale = areaWidth / Math.max(totalFrames - 1, 1)
    const startX = e.clientX
    const origStart = segment.start_frame
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const relX = e.clientX - rect.left
    const zone = getResizeZone()
    const isResizingStart = relX <= zone && !disableLeftResize
    const isResizingEnd = relX >= rect.width - zone
    // Detect initial resize intent based on where the click started
    let resizeMode: 'start' | 'end' | 'move' = 'move'
    if (isResizingStart) {
      resizeMode = 'start'
    } else if (isResizingEnd) {
      resizeMode = 'end'
    }
    showLabelPersistent()

    if (resizeMode !== 'move') setIsResizing(true)

    function onMove(ev: MouseEvent) {
      const deltaFrames = Math.round((ev.clientX - startX) / scale)
      if (deltaFrames !== 0) didDrag.current = true

      if (resizeMode === 'start' && onResizeEnd) {
        onResizeEnd(segment.id, 'start', deltaFrames, minStart)
      } else if (resizeMode === 'end' && onResizeEnd) {
        onResizeEnd(segment.id, 'end', deltaFrames)
      } else if (resizeMode === 'move' && onDragEnd) {
        onDragEnd(segment.id, deltaFrames, origStart)
      }
    }

    function onUp() {
      dragStart.current = null
      if (resizeMode !== 'move') setIsResizing(false)
      hideLabelAfterDelay()
      globalThis.removeEventListener('mousemove', onMove)
      globalThis.removeEventListener('mouseup', onUp)
    }

    globalThis.addEventListener('mousemove', onMove)
    globalThis.addEventListener('mouseup', onUp)
  }

  function makeResizeHandler(edge: 'start' | 'end') {
    return (e: React.MouseEvent) => {
      if (!interactive || !onResizeEnd) return
      e.preventDefault()
      e.stopPropagation()
      const startX = e.clientX
      const scale = areaWidth / Math.max(totalFrames - 1, 1)
      setIsResizing(true)
      showLabelPersistent()

      function onMove(ev: MouseEvent) {
        const deltaFrames = Math.round((ev.clientX - startX) / scale)
        onResizeEnd?.(segment.id, edge, deltaFrames, edge === 'start' ? minStart : undefined)
      }

      function onUp() {
        setIsResizing(false)
        hideLabelAfterDelay()
        globalThis.removeEventListener('mousemove', onMove)
        globalThis.removeEventListener('mouseup', onUp)
      }

      globalThis.addEventListener('mousemove', onMove)
      globalThis.addEventListener('mouseup', onUp)
    }
  }

  return (
    <div
      ref={ref}
      role="button"
      tabIndex={0}
      data-segment-block=""
      className={cn('absolute top-0 h-full flex items-center overflow-hidden rounded select-none active:opacity-70', className)}
      style={{
        left,
        width,
        backgroundColor: segment.color,
        // opacity: 1,
        border: isResizing ? '1px solid #eab308' : selected ? '1px solid var(--foreground)' : `1px solid ${lightenColor(segment.color)}`,
        cursor: cursorStyle ?? (interactive ? 'grab' : 'default'),
        // outlineOffset: selected ? '-1px' : '0px',
        ...bgStyle,
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={(e) => {
        if (didDrag.current) return
        e.stopPropagation()
        // Show label briefly on click
        showLabelPersistent()
        hideLabelAfterDelay()
        onSelect?.(segment); onClick?.(segment)
      }}
      onDoubleClick={() => onDoubleClick?.(segment)}
      onContextMenu={(e) => onContextMenu?.(e, segment)}
      onKeyDown={(e) => { if (e.key === 'Enter') { onSelect?.(segment); onClick?.(segment) } }}
    >
      {/* Background slot (e.g. cover thumbnail) */}
      {backgroundSlot && (
        <div className="absolute inset-0 pointer-events-none overflow-hidden rounded z-1">
          {backgroundSlot}
        </div>
      )}

      {/* Time-position label: shows on click/drag, auto-hides after 2s */}
      {labelVisible && frameRate !== undefined && displayFormat !== undefined && (
        <div className="absolute top-0.5 left-1 z-20 pointer-events-none">
          <span className="text-[9px] bg-black/70 text-white px-1 py-px rounded leading-none">
            {displayFormat === 'seconds'
              ? `${(segment.start_frame / frameRate).toFixed(1)}s`
              : `${segment.start_frame}f`}
          </span>
        </div>
      )}

      {/* Left resize handle */}
      {interactive && onResizeEnd && !hideLeftHandle && (
        <button
          type="button"
          aria-label={t('segmentBlock.resizeStart')}
          className="absolute left-0 top-0 h-full w-0.5 cursor-ew-resize z-20"
          style={{ background: isResizing ? '#eab308' : selected ? 'var(--foreground)' : 'transparent' }}
          onMouseDown={(e) => {
            if (disableLeftResize) return
            makeResizeHandler('start')(e)
          }}
        />
      )}

      {/* Content slot */}
      <div className="flex-1 overflow-hidden min-w-0 px-1.5 text-xs text-foreground font-medium pointer-events-none">
        {children}
      </div>

      {/* Right resize handle */}
      {interactive && onResizeEnd  && (
        <button
          type="button"
          aria-label={t('segmentBlock.resizeEnd')}
          className="absolute right-0 top-0 h-full w-0.5 cursor-ew-resize z-20"
          style={{ background: isResizing ? '#eab308' : selected ? 'var(--foreground)' : 'transparent' }}
          onMouseDown={(e) => {
            e.stopPropagation()
            makeResizeHandler('end')(e)
          }}
        />
      )}
    </div>
  )
})
