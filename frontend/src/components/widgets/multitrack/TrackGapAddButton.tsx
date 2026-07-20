import { forwardRef } from 'react'
import { Plus } from 'lucide-react'
import { Button, type ButtonProps } from '@/components/ui/button'
import { cn } from '@/lib/utils'

export interface TrackSegmentGap {
  startFrame: number
  endFrame: number
}

interface TrackGapAddButtonProps extends Omit<ButtonProps, 'children' | 'aria-label'> {
  gap: TrackSegmentGap
  totalLength: number
  width: number
  ariaLabel: string
}

export function getTrackSegmentGaps(
  segments: ReadonlyArray<{ start_frame: number; end_frame: number }>,
): TrackSegmentGap[] {
  const sorted = [...segments].sort((left, right) => left.start_frame - right.start_frame)
  if (sorted.length === 0) return []

  const gaps: TrackSegmentGap[] = sorted[0].start_frame > 0
    ? [{ startFrame: 0, endFrame: sorted[0].start_frame }]
    : []
  let coveredEnd = sorted[0].end_frame
  for (const segment of sorted.slice(1)) {
    if (segment.start_frame > coveredEnd) {
      gaps.push({ startFrame: coveredEnd, endFrame: segment.start_frame })
    }
    coveredEnd = Math.max(coveredEnd, segment.end_frame)
  }
  return gaps
}

export const TrackGapAddButton = forwardRef<HTMLButtonElement, TrackGapAddButtonProps>(function TrackGapAddButton({
  gap,
  totalLength,
  width,
  ariaLabel,
  onClick,
  className,
  style,
  ...buttonProps
}, ref) {
  const centerFrame = gap.startFrame + (gap.endFrame - gap.startFrame) / 2
  const left = centerFrame / Math.max(totalLength, 1) * width

  return (
    <Button
      {...buttonProps}
      ref={ref}
      type="button"
      variant="secondary"
      size="icon"
      data-testid={`track-gap-add-${gap.startFrame}-${gap.endFrame}`}
      className={cn(
        'absolute top-1/2 z-10 h-5 w-5 -translate-x-1/2 -translate-y-1/2 cursor-pointer',
        className,
      )}
      style={{ ...style, left }}
      aria-label={ariaLabel}
      onClick={(event) => {
        event.stopPropagation()
        onClick?.(event)
      }}
    >
      <Plus className="h-2.5 w-2.5" />
    </Button>
  )
})
