import { useEffect, useMemo, useState, type FormEvent } from 'react'
import { Scissors } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { NumberInput } from '@/components/ui/number-input'
import { useT } from '@/lib/i18n'
import { frameToSeconds, secondsToFrame, segmentDuration } from '@/lib/multitrack-utils'
import type { MultiTrackSegment } from '@/types/multitrack'

const DEFAULT_SPLIT_SECONDS = 5

interface SplitTaskSegmentDialogProps {
  segment: MultiTrackSegment | null
  frameRate: number
  open: boolean
  onOpenChange: (open: boolean) => void
  onConfirm: (segmentId: string, frames: number) => void
}

export function SplitTaskSegmentDialog({
  segment,
  frameRate,
  open,
  onOpenChange,
  onConfirm,
}: Readonly<SplitTaskSegmentDialogProps>) {
  const t = useT()
  const [seconds, setSeconds] = useState(DEFAULT_SPLIT_SECONDS)
  const durationFrames = segment ? segmentDuration(segment) : 0
  const safeFrameRate = Math.max(1, Math.round(frameRate))

  useEffect(() => {
    if (!open || !segment) return
    setSeconds(DEFAULT_SPLIT_SECONDS)
  }, [open, segment?.id])

  const targetFrames = useMemo(() => {
    return secondsToFrame(seconds, safeFrameRate)
  }, [safeFrameRate, seconds])

  const splitCount = targetFrames > 0 && targetFrames < durationFrames
    ? Math.max(2, Math.ceil(durationFrames / targetFrames))
    : 0
  const isValid = Boolean(segment) && targetFrames >= 1 && targetFrames < durationFrames && splitCount >= 2
  const durationSeconds = frameToSeconds(durationFrames, safeFrameRate)
  const maxSeconds = Math.max(1, Math.floor(frameToSeconds(Math.max(1, durationFrames - 1), safeFrameRate)))

  function handleValueChange(nextValue: number) {
    setSeconds(Math.max(1, Math.round(nextValue)))
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (!segment || !isValid) return
    onConfirm(segment.id, targetFrames)
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xs">
        <DialogHeader>
          <DialogTitle>{t('multitrack.splitTaskSegment')}</DialogTitle>
          <DialogDescription>
            {t('multitrack.splitTaskSegmentDescription', {
              frames: durationFrames,
              seconds: durationSeconds.toFixed(2),
            })}
          </DialogDescription>
        </DialogHeader>
        <form className="space-y-4" onSubmit={handleSubmit}>
          <div className="flex w-full items-center gap-2">
            <NumberInput
              min={1}
              max={maxSeconds}
              step={1}
              value={seconds}
              className="h-9 min-w-0 flex-1"
              aria-label={t('multitrack.splitTaskSegmentSeconds')}
              onChange={handleValueChange}
            />
            <span className="flex h-9 shrink-0 items-center rounded-md border border-border bg-muted px-3 text-sm text-muted-foreground">
              {t('multitrack.splitTaskSegmentSecondsUnit')}
            </span>
          </div>
          <p className={`text-xs leading-5 ${isValid ? 'text-muted-foreground' : 'text-destructive'}`}>
            {isValid
              ? t('multitrack.splitTaskSegmentPreview', { count: splitCount })
              : t('multitrack.splitTaskSegmentInvalid')}
          </p>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              {t('common.cancel')}
            </Button>
            <Button type="submit" disabled={!isValid}>
              <Scissors className="h-4 w-4" />
              {t('multitrack.splitTaskSegmentConfirm')}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
