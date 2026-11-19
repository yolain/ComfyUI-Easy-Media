import { Box, ImagePlus } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { useT } from '@/lib/i18n'
import { MAX_TASK_IMAGES } from '@/lib/task-image-utils'
import { MAX_TASK_MODS } from '@/lib/task-mod-utils'

export type TaskReferencePicker = 'image' | 'mod'

interface TaskReferenceActionsProps {
  imageCount: number
  modCount: number
  openPicker: TaskReferencePicker | null
  onPick: (type: TaskReferencePicker) => void
}

export function TaskReferenceActions({
  imageCount,
  modCount,
  openPicker,
  onPick,
}: Readonly<TaskReferenceActionsProps>) {
  const t = useT()
  const buttonClass = 'h-auto min-h-7 w-full justify-start gap-1.5 px-1 py-0.5 text-left text-xs leading-tight whitespace-nowrap @max-[5rem]:min-h-6 @max-[5rem]:gap-1 @max-[5rem]:px-0.5 @max-[5rem]:text-[10px] @max-[5rem]:[&_svg]:size-3'

  return (
    <div className="@container flex w-full min-w-0 flex-col gap-1">
      <div className="mb-2 px-1 text-left text-[10px] font-medium leading-tight whitespace-nowrap text-muted-foreground @max-[5rem]:mb-1 @max-[5rem]:px-0.5">
        {t('multitrack.addReference')}
      </div>
      <Button
        type="button" variant="ghost" size="sm" className={buttonClass}
        aria-haspopup="dialog" aria-expanded={openPicker === 'image'}
        disabled={imageCount >= MAX_TASK_IMAGES}
        onClick={() => onPick('image')}
      >
        <ImagePlus className="shrink-0" />
        <span>{t('multitrack.addReferenceImage')}</span>
      </Button>
      <Button
        type="button" variant="ghost" size="sm" className={buttonClass}
        aria-haspopup="dialog" aria-expanded={openPicker === 'mod'}
        disabled={modCount >= MAX_TASK_MODS}
        onClick={() => onPick('mod')}
      >
        <Box className="shrink-0" />
        <span>{t('multitrack.addMod')}</span>
      </Button>
    </div>
  )
}
