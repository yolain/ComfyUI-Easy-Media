import { useEffect, useMemo, useRef, useState } from 'react'
import { CloudUpload, Eye, Plus, RotateCcw, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import { useT } from '@/lib/i18n'
import { mediaContentToViewUrl } from '@/lib/media-url'
import { getMultiTrackTaskModeLabel } from '@/lib/multitrack-utils'
import { cn } from '@/lib/utils'
import { uuid } from '@/lib/uuid'
import type {
  MultiTrackSegment,
  MultiTrackSegmentContent,
  MultiTrackSourceType,
  MultiTrackTaskImage,
  MultiTrackTaskMode,
} from '@/types/multitrack'

type PromptTab = 'user' | 'system'
type EditMode = 'individual' | 'combined'

interface TrackSegmentContentUpdate {
  segmentId: string
  patch: Partial<MultiTrackSegmentContent>
}

interface TaskSegmentEditorProps {
  segment: MultiTrackSegment
  trackSegments?: MultiTrackSegment[]
  videoSegments?: MultiTrackSegment[]
  onContentChange: (patch: Partial<MultiTrackSegmentContent>) => void
  onTrackSegmentsContentChange?: (updates: TrackSegmentContentUpdate[]) => void
}

interface SystemPromptResponse {
  items?: SystemPromptOption[]
}

interface SystemPromptOption {
  task_type?: string
  system_prompt?: string
}

const TASK_MODES: MultiTrackTaskMode[] = ['default', 'ref', 'edit']
const MAX_TASK_IMAGES = 9
const MULTIPLE_MEDIA_SEPARATOR = '|MULTIPLE|'
let cachedSystemPromptOptions: SystemPromptOption[] | undefined
let systemPromptOptionsRequest: Promise<SystemPromptOption[]> | null = null

function taskImages(segment: MultiTrackSegment): MultiTrackTaskImage[] {
  return Array.isArray(segment.content.images) ? segment.content.images : []
}

function moveImage(images: MultiTrackTaskImage[], sourceId: string, targetId: string): MultiTrackTaskImage[] {
  const sourceIndex = images.findIndex((item) => item.id === sourceId)
  const targetIndex = images.findIndex((item) => item.id === targetId)
  if (sourceIndex < 0 || targetIndex < 0 || sourceIndex === targetIndex) return images

  const nextImages = [...images]
  const [moved] = nextImages.splice(sourceIndex, 1)
  nextImages.splice(targetIndex, 0, moved)
  return nextImages
}

function imageDisplayName(image: MultiTrackTaskImage): string {
  return image.file_name ?? image.file_path ?? image.local_path ?? image.url ?? image.slot_name ?? image.id
}

function getTaskType(mode: MultiTrackTaskMode, imageCount: number, hasVideoInRange: boolean): string {
  if (mode === 'ref') return hasVideoInRange ? 'rv2v' : 'r2v'
  if (mode === 'edit') return imageCount > 0 ? 'vi2v' : 'v2v'
  return imageCount > 0 ? 'i2v' : 't2v'
}

function hasVideoInRange(segment: MultiTrackSegment, videoSegments: MultiTrackSegment[]): boolean {
  return videoSegments.some((videoSegment) => (
    videoSegment.content.media_type === 'video' &&
    videoSegment.content.source_type !== 'preset' &&
    videoSegment.start_frame < segment.end_frame &&
    videoSegment.end_frame > segment.start_frame
  ))
}

function getDefaultSystemPromptForSegment(
  segment: MultiTrackSegment,
  options: SystemPromptOption[],
  videoSegments: MultiTrackSegment[],
): string {
  const images = taskImages(segment)
  const taskType = getTaskType(
    segment.content.task_mode ?? 'default',
    images.length,
    hasVideoInRange(segment, videoSegments),
  )
  return options.find((option) => option.task_type === taskType)?.system_prompt ?? ''
}

function createTaskImage(filePath: string, source: MultiTrackSourceType): MultiTrackTaskImage {
  const fileName = filePath.split(/[\\/]/).pop() ?? filePath
  return {
    id: uuid(),
    source_type: source,
    file_path: source === 'local' || source === 'url' ? undefined : filePath,
    local_path: source === 'local' ? filePath : undefined,
    url: source === 'url' ? filePath : undefined,
    file_name: fileName,
  }
}

function splitSelectedMedia(value: string): string[] {
  return value.split(MULTIPLE_MEDIA_SEPARATOR).filter((item) => item.length > 0)
}

function renderSystemPromptHighlight(value: string) {
  return value.split(/(\{[^{}]*\})/g).map((part, index) => (
    /^\{[^{}]*\}$/.test(part) ? (
      <span key={`${part}-${index}`} className="text-highlight" data-system-prompt-variable="true">
        {part}
      </span>
    ) : part
  ))
}

async function uploadImageFile(file: File): Promise<MultiTrackTaskImage> {
  const formData = new FormData()
  formData.append('image', file)
  const response = await fetch('/upload/image', {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error(`Failed to upload image: ${file.name}`)
  }
  const result = await response.json() as { name?: string; subfolder?: string }
  const name = result.name ?? file.name
  const subfolder = result.subfolder ?? ''
  return {
    id: uuid(),
    source_type: 'input',
    file_path: subfolder ? `${subfolder}/${name}` : name,
    file_name: name,
  }
}

async function loadSystemPromptOptions(): Promise<SystemPromptOption[]> {
  if (cachedSystemPromptOptions) return cachedSystemPromptOptions
  if (!systemPromptOptionsRequest) {
    systemPromptOptionsRequest = (async () => {
      const response = await fetch('/easy-media/prompt/system-prompts')
      if (!response.ok) {
        throw new Error(`Failed to load system prompt options: ${response.status}`)
      }
      const result = await response.json() as SystemPromptResponse
      const options = Array.isArray(result.items) ? result.items : []
      cachedSystemPromptOptions = options
      return options
    })().catch((error: unknown) => {
      systemPromptOptionsRequest = null
      throw error
    })
  }
  return systemPromptOptionsRequest
}

export function TaskSegmentEditor({
  segment,
  trackSegments,
  videoSegments = [],
  onContentChange,
  onTrackSegmentsContentChange,
}: Readonly<TaskSegmentEditorProps>) {
  const t = useT()
  const draggedImageIdRef = useRef<string | null>(null)
  const systemPromptOverlayRef = useRef<HTMLDivElement>(null)
  const [promptTab, setPromptTab] = useState<PromptTab>('user')
  const [editMode, setEditMode] = useState<EditMode>('individual')
  const [mediaSelectorOpen, setMediaSelectorOpen] = useState(false)
  const [isImageDragOver, setIsImageDragOver] = useState(false)
  const [systemPromptOptions, setSystemPromptOptions] = useState<SystemPromptOption[] | null>(cachedSystemPromptOptions ?? null)
  const [systemPromptLoading, setSystemPromptLoading] = useState(false)
  const images = taskImages(segment)
  const mode = segment.content.task_mode ?? 'default'
  const editableSegments = useMemo(() => (
    trackSegments && trackSegments.length > 0 ? trackSegments : [segment]
  ), [segment, trackSegments])
  const taskIndex = Math.max(0, editableSegments.findIndex((item) => item.id === segment.id))
  const promptValue = editMode === 'combined'
    ? editableSegments.map((item) => item.content.text ?? '').join('|')
    : segment.content.text ?? ''
  const segmentHasVideoInRange = hasVideoInRange(segment, videoSegments)
  const systemPromptDefault = getDefaultSystemPromptForSegment(segment, systemPromptOptions ?? [], videoSegments)
  const systemPromptValue = editMode === 'combined'
    ? editableSegments.map((item) => (
        item.content.system_prompt ?? getDefaultSystemPromptForSegment(item, systemPromptOptions ?? [], videoSegments)
      )).join('|')
    : segment.content.system_prompt || systemPromptDefault

  useEffect(() => {
    if (promptTab !== 'system') return
    if (systemPromptOptions !== null) return
    let cancelled = false
    setSystemPromptLoading(true)
    loadSystemPromptOptions()
      .then((options) => {
        if (!cancelled) setSystemPromptOptions(options)
      })
      .catch((error: unknown) => {
        console.error('[TaskSegmentEditor] failed to load system prompt options:', error)
        if (!cancelled) setSystemPromptOptions([])
      })
      .finally(() => {
        if (!cancelled) setSystemPromptLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [promptTab, systemPromptOptions])

  function hasDraggedImageFile(dataTransfer: DataTransfer): boolean {
    const files = Array.from(dataTransfer.files)
    if (files.some((file) => file.type.startsWith('image/'))) return true
    return Array.from(dataTransfer.items ?? []).some((item) => item.kind === 'file' && item.type.startsWith('image/'))
  }

  function handleImageDragEnter(event: React.DragEvent<HTMLDivElement>) {
    if (!event.dataTransfer || !hasDraggedImageFile(event.dataTransfer)) return
    event.preventDefault()
    setIsImageDragOver(true)
  }

  function handleImageDragOver(event: React.DragEvent<HTMLDivElement>) {
    if (!event.dataTransfer || !hasDraggedImageFile(event.dataTransfer)) return
    event.preventDefault()
    setIsImageDragOver(true)
  }

  function handleImageDragLeave(event: React.DragEvent<HTMLDivElement>) {
    const relatedTarget = event.relatedTarget
    if (relatedTarget instanceof Node && event.currentTarget.contains(relatedTarget)) return
    setIsImageDragOver(false)
  }

  async function handleDrop(event: React.DragEvent<HTMLDivElement | HTMLButtonElement>) {
    event.preventDefault()
    setIsImageDragOver(false)
    if (!event.dataTransfer) return
    const remainingSlots = MAX_TASK_IMAGES - images.length
    if (remainingSlots <= 0) return
    const files = Array.from(event.dataTransfer.files)
      .filter((file) => file.type.startsWith('image/'))
      .slice(0, remainingSlots)
    if (files.length === 0) return

    try {
      const uploaded = await Promise.all(files.map((file) => uploadImageFile(file)))
      onContentChange({ images: [...images, ...uploaded] })
    } catch (error) {
      console.error('[TaskSegmentEditor] failed to upload task images:', error)
    }
  }

  function handleSelectedMedia(filePath: string, source?: 'input' | 'output' | 'local') {
    const remainingSlots = MAX_TASK_IMAGES - images.length
    if (remainingSlots <= 0) return
    const selectedPaths = splitSelectedMedia(filePath).slice(0, remainingSlots)
    const sourceType = source ?? (filePath.startsWith('http://') || filePath.startsWith('https://') ? 'url' : 'input')
    const nextImages = selectedPaths.map((path) => createTaskImage(path, sourceType))
    if (nextImages.length === 0) return
    onContentChange({ images: [...images, ...nextImages] })
    setMediaSelectorOpen(false)
  }

  function handlePromptChange(value: string) {
    if (editMode === 'combined') {
      const parts = value.split('|')
      onTrackSegmentsContentChange?.(editableSegments.map((item, index) => ({
        segmentId: item.id,
        patch: { text: parts[index] ?? '' },
      })))
      return
    }
    onContentChange({ text: value })
  }

  function handleSystemPromptChange(value: string) {
    if (editMode === 'combined') {
      const parts = value.split('|')
      onTrackSegmentsContentChange?.(editableSegments.map((item, index) => ({
        segmentId: item.id,
        patch: { system_prompt: parts[index] ?? '' },
      })))
      return
    }
    onContentChange({ system_prompt: value })
  }

  function handleSystemPromptScroll(event: React.UIEvent<HTMLTextAreaElement>) {
    if (!systemPromptOverlayRef.current) return
    systemPromptOverlayRef.current.scrollTop = event.currentTarget.scrollTop
    systemPromptOverlayRef.current.scrollLeft = event.currentTarget.scrollLeft
  }

  function handleDeleteImage(imageId: string) {
    onContentChange({ images: images.filter((image) => image.id !== imageId) })
  }

  function handlePreviewImage(image: MultiTrackTaskImage) {
    const url = mediaContentToViewUrl({
      source_type: image.source_type ?? 'input',
      file_path: image.file_path,
      local_path: image.local_path,
      url: image.url,
      slot_name: image.slot_name,
    })
    if (!url) return
    window.open(url, '_blank', 'noopener,noreferrer')
  }

  const imageGridColumns = images.length > 0 && images.length < 4 ? 'grid-cols-2' : 'grid-cols-3'
  const imagePickerSurfaceClass = isImageDragOver ? 'border-primary bg-accent/20' : 'border-border bg-muted/20'

  return (
    <div className="flex h-full min-h-24 w-full flex-col overflow-hidden rounded-sm bg-background text-foreground">
      <div className="flex min-h-0 flex-1 gap-4 p-4">
        {editMode === 'individual' && (
          <div
            data-testid="task-image-drop-zone"
            aria-label={t('multitrack.taskImageDropZone')}
            className={cn(
              'flex aspect-square h-full min-h-0 shrink-0 items-center justify-center rounded-md border border-dashed transition-colors',
              isImageDragOver ? 'border-primary bg-accent/20' : 'border-border bg-muted/30',
            )}
            onDragEnter={handleImageDragEnter}
            onDragOver={handleImageDragOver}
            onDragLeave={handleImageDragLeave}
            onDrop={handleDrop}
          >
            <Popover open={mediaSelectorOpen} onOpenChange={setMediaSelectorOpen}>
              {images.length === 0 ? (
                <PopoverTrigger asChild>
                  <div
                    role="button"
                    tabIndex={0}
                    className={cn(
                      'flex aspect-square h-full max-h-full min-h-24 min-w-24 max-w-full cursor-pointer flex-col items-center justify-center gap-1 rounded-md px-4 py-2 text-foreground shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                      imagePickerSurfaceClass,
                    )}
                    aria-label={t('multitrack.taskImageDropZone')}
                    onKeyDown={(event) => {
                      if (event.key !== 'Enter' && event.key !== ' ') return
                      event.preventDefault()
                      setMediaSelectorOpen(true)
                    }}
                  >
                    <CloudUpload />
                    <span className="text-[10px] font-semibold mt-1">{t('multitrack.selectImage')}</span>
                    <span className="max-w-full whitespace-normal text-center text-[9px] text-muted-foreground">
                      {t('multitrack.imageDropHint')}
                    </span>
                  </div>
                </PopoverTrigger>
              ) : (
                <div className={cn('grid h-full content-start gap-2 overflow-y-auto rounded-md p-3 transition-colors', imageGridColumns, imagePickerSurfaceClass)}>
                  {images.map((image) => {
                    const imageUrl = mediaContentToViewUrl({
                      source_type: image.source_type ?? 'input',
                      file_path: image.file_path,
                      local_path: image.local_path,
                      url: image.url,
                      slot_name: image.slot_name,
                    })
                    return (
                      <div
                        key={image.id}
                        draggable
                        data-testid={`task-image-${image.id}`}
                        className="group relative flex aspect-square items-center justify-center overflow-hidden rounded-md border border-border bg-background"
                        onDragStart={() => {
                          draggedImageIdRef.current = image.id
                        }}
                        onDragOver={(event) => event.preventDefault()}
                        onDrop={() => {
                          const sourceId = draggedImageIdRef.current
                          draggedImageIdRef.current = null
                          if (!sourceId) return
                          onContentChange({ images: moveImage(images, sourceId, image.id) })
                        }}
                      >
                        {imageUrl ? (
                          <img
                            src={imageUrl}
                            alt={imageDisplayName(image)}
                            className="h-full w-full object-cover"
                            draggable={false}
                          />
                        ) : (
                          <div className="flex h-full w-full items-center justify-center px-2 text-center text-[8px] text-muted-foreground">
                            {imageDisplayName(image)}
                          </div>
                        )}
                        <div
                          data-testid={`task-image-actions-${image.id}`}
                          className="absolute inset-0 flex items-center justify-center gap-1 bg-black/30 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
                        >
                          <Button
                            type="button"
                            size="icon"
                            variant="ghost"
                            className="h-6 w-6 bg-background/70 text-foreground hover:bg-background/90 [&_svg]:!size-3"
                            aria-label={t('multitrack.previewImage')}
                            onClick={() => handlePreviewImage(image)}
                          >
                            <Eye />
                          </Button>
                          <Button
                            type="button"
                            size="icon"
                            variant="ghost"
                            className="h-6 w-6 bg-background/70 text-destructive hover:bg-background/90 hover:text-destructive [&_svg]:!size-3"
                            aria-label={t('multitrack.deleteImage')}
                            onClick={() => handleDeleteImage(image.id)}
                          >
                            <Trash2 />
                          </Button>
                        </div>
                      </div>
                    )
                  })}
                  {images.length < MAX_TASK_IMAGES && (
                    <PopoverTrigger asChild>
                      <Button
                        type="button"
                        variant="outline"
                        className="aspect-square h-auto border-dashed text-muted-foreground"
                        aria-label={t('multitrack.selectImage')}
                      >
                        <Plus className="h-7 w-7" />
                      </Button>
                    </PopoverTrigger>
                  )}
                </div>
              )}
              <PopoverContent className="w-auto p-0" align="end">
                <MediaSelector
                  value=""
                  mediaType="image"
                  onChange={handleSelectedMedia}
                />
              </PopoverContent>
            </Popover>
          </div>
        )}

        <div
          data-testid="task-prompt-panel"
          className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden rounded-md border border-border bg-muted/30 shadow-sm"
        >
          {editMode === 'individual' && (
            <div className="flex h-11 shrink-0 items-center px-2 justify-between">
              <div className="flex h-7 items-center rounded-md bg-card p-1">
                <Button
                  type="button"
                  variant="ghost"
                  className={cn(
                    'h-full rounded-sm px-3 py-1 text-[10px] shadow-none cursor-pointer',
                    promptTab === 'user' ? 'bg-background text-primary hover:bg-background/90 hover:text-primary' : 'text-muted-foreground',
                  )}
                  onClick={() => setPromptTab('user')}
                >
                  {t('multitrack.userPrompt')}
                </Button>
                <Button
                  type="button"
                  variant="ghost"
                  className={cn(
                    'h-full rounded-sm px-3 py-1 text-[10px] shadow-none cursor-pointer',
                    promptTab === 'system' ? 'bg-background text-primary hover:bg-background/90 hover:text-primary' : 'text-muted-foreground',
                  )}
                  onClick={() => setPromptTab('system')}
                >
                  {t('multitrack.systemPrompt')}
                </Button>
              </div>
              {promptTab === 'system' && Boolean(segment.content.system_prompt) && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      type="button"
                      size="icon"
                      variant="ghost"
                      className="h-7 w-7 text-muted-foreground [&_svg]:!size-3"
                      aria-label={t('multitrack.resetSystemPrompt')}
                      onClick={() => onContentChange({ system_prompt: '' })}
                    >
                      <RotateCcw />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">{t('multitrack.resetSystemPrompt')}</TooltipContent>
                </Tooltip>
              )}
            </div>
          )}
          {editMode === 'combined' ? (
            <div className="flex min-h-0 flex-1 flex-col gap-1 p-2">
              <Textarea
                aria-label={t('multitrack.prompt')}
                placeholder={t('multitrack.promptPlaceholder')}
                className="min-h-24 flex-1 resize-none border-none bg-card text-[10px] shadow-none focus-visible:ring-1"
                value={promptValue}
                onChange={(event) => handlePromptChange(event.currentTarget.value)}
              />
              <p className="shrink-0 text-[9px] text-muted-foreground mt-1">{t('maintainTrack.combinedHint')}</p>
            </div>
          ) : promptTab === 'user' ? (
            <div className="min-h-0 flex-1 pb-2 px-2">
              <Textarea
                aria-label={t('multitrack.prompt')}
                placeholder={t('multitrack.promptPlaceholder')}
                className="h-full min-h-24 resize-none border-none bg-card text-[10px] shadow-none focus-visible:ring-1"
                value={promptValue}
                onChange={(event) => handlePromptChange(event.currentTarget.value)}
              />
            </div>
          ) : (
            <div className="relative min-h-0 flex-1 pb-2 px-2">
              <div className="relative h-full min-h-24 overflow-hidden rounded-md bg-card">
                <div
                  ref={systemPromptOverlayRef}
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-0 overflow-hidden whitespace-pre-wrap wrap-break-word px-3 py-2 text-[10px] leading-normal text-foreground"
                >
                  {renderSystemPromptHighlight(systemPromptValue)}
                </div>
                <Textarea
                  aria-label={t('multitrack.systemPrompt')}
                  placeholder={systemPromptLoading ? t('multitrack.loadingSystemPrompt') : t('multitrack.systemPromptPlaceholder')}
                  className="absolute inset-0 h-full min-h-0 resize-none border-none bg-transparent text-[10px] leading-normal shadow-none focus-visible:ring-1"
                  style={{ color: 'transparent', caretColor: 'var(--foreground)' }}
                  value={systemPromptValue}
                  onChange={(event) => handleSystemPromptChange(event.currentTarget.value)}
                  onScroll={handleSystemPromptScroll}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="relative flex shrink-0 items-center justify-between border-t border-dashed border-border p-2">
        <Tabs value={editMode} onValueChange={(value) => setEditMode(value as EditMode)}>
          <TabsList className="h-8 bg-card">
            <TabsTrigger value="individual" className="text-[10px]">{t('multitrack.individualEdit')}</TabsTrigger>
            <TabsTrigger value="combined" className="text-[10px]">{t('multitrack.combinedEdit')}</TabsTrigger>
          </TabsList>
        </Tabs>

        <span className="absolute left-1/2 -translate-x-1/2 text-[10px] text-muted-foreground">
          {t('multitrack.taskNumber', { n: taskIndex })}
        </span>

        <Select value={mode} onValueChange={(value) => onContentChange({ task_mode: value as MultiTrackTaskMode })}>
          <SelectTrigger className="w-36 h-8 text-[10px] bg-card">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {TASK_MODES.map((taskMode) => (
              <SelectItem key={taskMode} value={taskMode}>
                <span className="text-[10px]">
                  {getMultiTrackTaskModeLabel(taskMode, t)} ({getTaskType(taskMode, images.length, segmentHasVideoInRange)})
                </span>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  )
}
