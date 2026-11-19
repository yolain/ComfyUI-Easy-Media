import { Eye, Share2, Trash2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { cn } from '@/lib/utils'

export type TaskReferenceControlLayout = 'editor' | 'flow' | 'split'

export interface TaskReferenceControlMetrics {
  buttonSize: number
  iconSize: number
  muteFontSize: number
}

interface ReferenceAction {
  label: string
  tooltip?: string
  testId?: string
  onClick: () => void
}

interface TaskReferenceControlsProps {
  layout: TaskReferenceControlLayout
  metrics?: TaskReferenceControlMetrics
  testId?: string
  shared?: boolean
  muted?: boolean
  canShare: boolean
  canPreview?: boolean
  share: ReferenceAction
  mute: ReferenceAction
  preview: ReferenceAction
  remove: ReferenceAction
}

export function TaskReferenceControls({
  layout, metrics, testId, shared = false, muted = false, canShare, canPreview = true,
  share, mute, preview, remove,
}: Readonly<TaskReferenceControlsProps>) {
  const inEditor = layout === 'editor'
  const { buttonSize, iconSize, muteFontSize } = metrics ?? {
    buttonSize: shared ? 16 : 18,
    iconSize: 16,
    muteFontSize: 14,
  }
  const iconStyle = { width: iconSize, height: iconSize }
  const DeleteIcon = inEditor ? Trash2 : X
  const controls = [
    {
      key: 'share', action: share, content: <Share2 style={iconStyle} />,
      className: cn(!inEditor && 'rounded-none', shared ? 'text-highlight' : 'text-muted-foreground'),
      disabled: !canShare, pressed: shared,
    },
    {
      key: 'mute', action: mute, content: 'M', fontSize: muteFontSize,
      className: cn('font-bold', muted ? 'text-destructive hover:text-destructive' : 'text-muted-foreground'),
      pressed: muted,
    },
    {
      key: 'preview', action: preview, content: <Eye style={iconStyle} />,
      className: 'font-bold text-muted-foreground', disabled: !canPreview,
    },
    {
      key: 'delete', action: remove, content: <DeleteIcon style={iconStyle} />,
      className: cn(!inEditor && 'rounded-none', 'text-destructive hover:text-destructive'),
    },
  ]

  return (
    <div
      data-testid={testId}
      className={cn(
        'absolute z-10 flex opacity-0 transition-opacity group-hover/task-image:opacity-100 group-focus-within/task-image:opacity-100',
        inEditor ? 'right-0.5 top-1 gap-1' : layout === 'flow' ? 'right-0 top-1 gap-1' : 'right-0 top-0 gap-0.5',
      )}
    >
      <TooltipProvider>
        {controls.map(({ key, action, content, className, disabled, pressed, fontSize }) => (
          <Tooltip key={key}>
            <TooltipTrigger asChild>
              <Button
                type="button"
                size="icon"
                variant={pressed ? 'secondary' : 'ghost'}
                data-testid={action.testId}
                className={cn('h-auto w-auto cursor-pointer bg-background/70 hover:bg-background/90', className)}
                style={{ width: buttonSize, height: buttonSize, fontSize }}
                aria-label={action.label}
                aria-pressed={pressed}
                disabled={disabled}
                onClick={(event) => {
                  event.stopPropagation()
                  action.onClick()
                }}
              >
                {content}
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom" className="max-w-64">{action.tooltip ?? action.label}</TooltipContent>
          </Tooltip>
        ))}
      </TooltipProvider>
    </div>
  )
}
