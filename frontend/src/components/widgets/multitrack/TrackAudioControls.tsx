import type { ReactNode } from 'react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useT } from '@/lib/i18n'
import type { MultiTrack } from '@/types/multitrack'

interface TrackAudioControlsProps {
  track: MultiTrack
  icon: ReactNode
  preserveSelection?: boolean
  onChange: (patch: Partial<Pick<MultiTrack, 'muted' | 'solo'>>) => void
}

export function TrackAudioControls({ track, icon, preserveSelection = false, onChange }: Readonly<TrackAudioControlsProps>) {
  const t = useT()
  return (
    <div className="flex h-full flex-col items-center justify-center gap-0.5">
      {icon}
      <div className="flex gap-px mt-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className={`h-3 w-3 cursor-pointer rounded-sm bg-card p-0 text-[8px] ${track.muted ? 'text-destructive' : 'text-muted-foreground'}`}
              aria-label={t('multitrack.muteTrack', { name: track.name })}
              aria-pressed={track.muted}
              onClick={(event) => {
                if (preserveSelection) event.stopPropagation()
                onChange({ muted: !track.muted })
              }}
            >
              {t('multitrack.muteShort')}
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('multitrack.muteTrack', { name: track.name })}</TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="icon"
              className={`h-3 w-3 cursor-pointer rounded-sm bg-card p-0 text-[8px] ${track.solo ? 'text-highlight' : 'text-muted-foreground'}`}
              aria-label={t('multitrack.soloTrack', { name: track.name })}
              aria-pressed={track.solo === true}
              onClick={(event) => {
                if (preserveSelection) event.stopPropagation()
                onChange({ solo: !track.solo })
              }}
            >
              {t('multitrack.soloShort')}
            </Button>
          </TooltipTrigger>
          <TooltipContent>{t('multitrack.soloTrack', { name: track.name })}</TooltipContent>
        </Tooltip>
      </div>
    </div>
  )
}
