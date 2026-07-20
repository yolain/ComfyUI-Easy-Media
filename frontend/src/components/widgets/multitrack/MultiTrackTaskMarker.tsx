import { Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { TaskMarkerIcon } from '@/components/ui/custom-lucide-icon'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { useT } from '@/lib/i18n'
import { formatMultiTrackTime } from '@/lib/multitrack-utils'
import type { MultiTrackTaskMarker as TaskMarker } from '@/types/multitrack'

interface MultiTrackTaskMarkerProps {
  marker: TaskMarker
  markerNumber: number
  frameRate: number
  left: number
  selected: boolean
  dragging: boolean
  onSelect: (markerId: string) => void
  onDragStart: (markerId: string, clientX: number) => void
  onDelete: (markerId: string) => void
}

export function MultiTrackTaskMarker({
  marker,
  markerNumber,
  frameRate,
  left,
  selected,
  dragging,
  onSelect,
  onDragStart,
  onDelete,
}: Readonly<MultiTrackTaskMarkerProps>) {
  const t = useT()
  const markerLabel = t('multitrack.taskMarkerLabel', {
    n: markerNumber.toString().padStart(2, '0'),
  })
  const timecode = formatMultiTrackTime(marker.frame, { frameRate, showFrames: true, showHours: false })

  return (
    <ContextMenu>
      <Tooltip>
        <TooltipTrigger asChild>
          <ContextMenuTrigger asChild>
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className={`absolute top-0 z-30 h-6 w-5 -translate-x-1/2 rounded-none p-0 hover:bg-transparent ${dragging ? 'cursor-grabbing' : 'cursor-grab'} ${selected ? 'opacity-100' : 'opacity-90'}`}
              style={{ left, color: 'var(--highlight)' }}
              aria-label={t('multitrack.taskMarkerAtFrame', { frame: marker.frame })}
              aria-pressed={selected}
              onMouseDown={(event) => {
                event.preventDefault()
                event.stopPropagation()
                onSelect(marker.id)
                onDragStart(marker.id, event.clientX)
              }}
              onContextMenu={(event) => {
                event.stopPropagation()
                onSelect(marker.id)
              }}
              onClick={(event) => {
                event.stopPropagation()
              }}
            >
              <span className="absolute left-1/2 top-0 h-2 w-px -translate-x-1/2 bg-border" />
              <TaskMarkerIcon className="absolute left-1/2 top-1 h-4 w-4 -translate-x-1/2 fill-current stroke-current" />
            </Button>
          </ContextMenuTrigger>
        </TooltipTrigger>
        <TooltipContent
          side="bottom"
          sideOffset={3}
          className="min-w-24 rounded-md border border-border bg-popover px-2 py-1.5 text-popover-foreground shadow-md"
        >
          <div className="text-[10px] font-medium leading-4">{markerLabel}</div>
          <div className="text-[11px] leading-4 tabular-nums">{timecode}</div>
        </TooltipContent>
      </Tooltip>
      <ContextMenuContent>
        <ContextMenuItem onClick={() => onDelete(marker.id)}>
          <Trash2 className="mr-2 h-3.5 w-3.5" />
          {t('multitrack.deleteTaskMarker')}
        </ContextMenuItem>
      </ContextMenuContent>
    </ContextMenu>
  )
}
