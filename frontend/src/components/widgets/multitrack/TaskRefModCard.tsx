import { useState } from 'react'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { useT } from '@/lib/i18n'
import { taskModName, taskModPreviewUrl } from '@/lib/task-mod-utils'
import { cn } from '@/lib/utils'
import type { MultiTrackTaskMod } from '@/types/multitrack'
import { TaskReferenceControls, type TaskReferenceControlLayout, type TaskReferenceControlMetrics } from './TaskReferenceControls'
import { TaskReferenceIndex } from './TaskReferenceIndex'

interface TaskRefModCardProps {
  mod: MultiTrackTaskMod
  index: number
  canShare: boolean
  className?: string
  layout: TaskReferenceControlLayout
  controlMetrics?: TaskReferenceControlMetrics
  onShare: () => void
  onMute: () => void
  onDelete: () => void
  onReselect: (path: string) => boolean
}

export function TaskRefModCard({
  mod, index, canShare, className, layout, controlMetrics, onShare, onMute, onDelete, onReselect,
}: Readonly<TaskRefModCardProps>) {
  const t = useT()
  const [failedPreviewUrl, setFailedPreviewUrl] = useState<string | null>(null)
  const [previewOpen, setPreviewOpen] = useState(false)
  const [selectorOpen, setSelectorOpen] = useState(false)
  const [replacementConflict, setReplacementConflict] = useState(false)
  const name = taskModName(mod.file_path)
  const previewUrl = taskModPreviewUrl(mod.file_path)
  const hasPreview = failedPreviewUrl !== previewUrl
  const shareLabel = t(mod.shared_reference ? 'multitrack.unshareMod' : 'multitrack.shareMod')
  const muteLabel = t(mod.muted ? 'multitrack.enableMod' : 'multitrack.disableMod')

  return (
    <div
      data-testid={`task-mod-${mod.id}`}
      className={cn(
        'group/task-image relative flex aspect-square min-w-0 items-center justify-center overflow-hidden rounded-md border bg-muted',
        mod.shared_reference ? 'border-highlight' : 'border-border',
        mod.muted && 'opacity-60',
        className,
      )}
    >
      <Popover open={selectorOpen} onOpenChange={(open) => {
        setSelectorOpen(open)
        if (!open) setReplacementConflict(false)
      }}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="ghost"
            className="absolute inset-0 size-full cursor-pointer rounded-none p-0 text-foreground hover:bg-transparent hover:text-foreground focus-visible:ring-inset"
            aria-label={t('multitrack.reselectMod', { name })}
          >
            {hasPreview ? (
              <img
                key={previewUrl}
                src={previewUrl}
                alt={name}
                loading="lazy"
                draggable={false}
                className="h-full w-full object-cover"
                onError={() => setFailedPreviewUrl(previewUrl)}
              />
            ) : (
              <span className="line-clamp-2 max-w-full wrap-anywhere px-2 text-center text-xs leading-tight font-normal whitespace-normal" title={name}>
                {name}
              </span>
            )}
            <TaskReferenceIndex kind="mod" index={index} testId={`task-mod-index-${mod.id}`} />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="end">
          <MediaSelector
            value={mod.file_path}
            mediaType="mod"
            maxSelectionCount={1}
            onChange={(path) => {
              if (!onReselect(path)) {
                setReplacementConflict(true)
                return
              }
              setReplacementConflict(false)
              setSelectorOpen(false)
            }}
          />
          {replacementConflict && (
            <p role="alert" className="px-2 py-1.5 text-xs text-destructive">
              {t('multitrack.reselectModAlreadyAdded')}
            </p>
          )}
        </PopoverContent>
      </Popover>
      {layout === 'editor' && (
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 z-[5] bg-black/30 opacity-0 transition-opacity group-hover/task-image:opacity-100 group-focus-within/task-image:opacity-100"
        />
      )}
      <TaskReferenceControls
        layout={layout}
        metrics={controlMetrics}
        testId={`task-mod-actions-${mod.id}`}
        shared={mod.shared_reference}
        muted={mod.muted}
        canShare={canShare}
        share={{ label: shareLabel, onClick: onShare }}
        mute={{ label: muteLabel, onClick: onMute }}
        preview={{ label: t('multitrack.previewMod'), onClick: () => setPreviewOpen(true) }}
        remove={{ label: t('multitrack.deleteMod'), onClick: onDelete }}
      />
      <Dialog open={previewOpen} onOpenChange={setPreviewOpen}>
        <DialogContent className="max-w-xl">
          <DialogHeader>
            <DialogTitle>{name}</DialogTitle>
            <DialogDescription>{mod.file_path}</DialogDescription>
          </DialogHeader>
          {hasPreview && <img src={previewUrl} alt={name} className="max-h-[65vh] w-full rounded-md object-contain" />}
        </DialogContent>
      </Dialog>
    </div>
  )
}
