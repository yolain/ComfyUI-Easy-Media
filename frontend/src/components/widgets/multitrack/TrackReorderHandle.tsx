import type { ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { useT } from '@/lib/i18n'
import type { MultiTrack } from '@/types/multitrack'

interface TrackReorderHandleProps {
  track: MultiTrack
  children: ReactNode
}

export function TrackReorderHandle({ track, children }: Readonly<TrackReorderHandleProps>) {
  const t = useT()

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      draggable
      className="h-5 w-5 cursor-grab p-0 active:cursor-grabbing"
      aria-label={t('multitrack.reorderTrack', { name: track.name })}
      data-multitrack-track-drag-handle={track.id}
    >
      {children}
    </Button>
  )
}
