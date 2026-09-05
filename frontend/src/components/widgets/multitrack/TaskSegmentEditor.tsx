import { useEffect, useMemo, useRef, useState } from 'react'
import { CloudUpload, Eye, Pencil, Plus, RotateCcw, Share2, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Popover, PopoverAnchor, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import type { MediaTab } from '@/components/widgets/mediaSelector/MediaSelector'
import { useT } from '@/lib/i18n'
import { mediaContentToViewUrl } from '@/lib/media-url'
import { getSegmentTrackPresentation } from '@/lib/multitrack-segment-style'
import { computeSlotItems } from '@/lib/timeline-utils'
import {
  applyCombinedTaskTexts,
  formatMultiTrackDurationTimecode,
  frameToSeconds,
  getSelectedTaskUserPrompt,
  getSelectedTaskUserPromptPatch,
  getMultiTrackTaskType,
  MULTITRACK_CONTINUITY_MODES,
  MULTITRACK_DEFAULT_CONTINUITY_MODE,
  MULTITRACK_DEFAULT_FRAME_RATE,
  MULTITRACK_DEFAULT_REF_IMAGE_SIZE,
  MULTITRACK_REF_IMAGE_SIZES,
  MULTITRACK_TASK_MODES,
  parseMultiTrackDurationTimecode,
  segmentDuration,
} from '@/lib/multitrack-utils'
import { cn } from '@/lib/utils'
import {
  createTaskImage,
  canEnableSharedTaskImage,
  MAX_TASK_IMAGES,
  sharedTaskImageUpdates,
  splitSelectedTaskMedia,
  taskImageIdentity,
  taskImageSlotNumber,
  taskImagesFromContent,
  uploadTaskImageFile,
} from '@/lib/task-image-utils'
import { invalidateMediaListCache } from '@/stores/media-list-store'
import type {
  MultiTrack,
  MultiTrackContinuityMode,
  MultiTrackRefImageSize,
  MultiTrackSegment,
  MultiTrackSegmentContent,
  MultiTrackSourceType,
  MultiTrackTaskImage,
  MultiTrackTaskMode,
  MultiTrackUserPromptVariant,
} from '@/types/multitrack'
import { PanoramaImagePreview } from '@/components/widgets/panorama/PanoramaImagePreview'
import { PromptContentEditor, type PromptReferenceResource } from './PromptContentEditor'

type PromptTab = 'user' | 'system'
type EditMode = 'individual' | 'combined'

interface TrackSegmentContentUpdate {
  segmentId: string
  patch: Partial<MultiTrackSegmentContent>
}

interface TaskSegmentEditorProps {
  segment: MultiTrackSegment
  node?: unknown
  app?: unknown
  trackSegments?: MultiTrackSegment[]
  selectedSegments?: MultiTrackSegment[]
  videoSegments?: MultiTrackSegment[]
  mediaTracks?: MultiTrack[]
  frameRate?: number
  totalFrames?: number
  imageIndexOffset?: number
  format?: string
  onContentChange: (patch: Partial<MultiTrackSegmentContent>) => void
  onTrackSegmentsContentChange?: (updates: TrackSegmentContentUpdate[]) => void
  onTrackSegmentsChange?: (segments: MultiTrackSegment[]) => void
  onDurationChange?: (duration: number) => void
  onOpenImagePreview?: (imageId: string) => void
}

interface SystemPromptResponse {
  items?: SystemPromptOption[]
}

interface SystemPromptOption {
  task_type?: string
  system_prompt?: string
  format?: string
  modes?: MultiTrackTaskMode[]
}

let cachedSystemPromptOptions: SystemPromptOption[] | undefined
let systemPromptOptionsRequest: Promise<SystemPromptOption[]> | null = null

function taskImages(segment: MultiTrackSegment): MultiTrackTaskImage[] {
  return taskImagesFromContent(segment.content.images)
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

function imageSelectorValue(image: MultiTrackTaskImage | undefined): string {
  if (!image) return ''
  if (image.slot_name) return `__slot__:${image.slot_name}`
  return image.file_path ?? image.local_path ?? image.url ?? ''
}

function imageSelectorTab(image: MultiTrackTaskImage | undefined): MediaTab {
  if (image?.source_type === 'output') return 'outputs'
  if (image?.source_type === 'local') return 'local'
  if (image?.source_type === 'url') return 'url'
  if (image?.source_type === 'slot') return 'slot'
  return 'inputs'
}

function selectedImageSource(
  filePath: string,
  source?: 'input' | 'output' | 'local',
): MultiTrackSourceType {
  if (source) return source
  if (filePath.startsWith('__slot__:')) return 'slot'
  if (filePath.startsWith('http://') || filePath.startsWith('https://')) return 'url'
  return 'input'
}

function hasVideoInRange(segment: MultiTrackSegment, videoSegments: MultiTrackSegment[]): boolean {
  return videoSegments.some((videoSegment) => (
    videoSegment.content.media_type === 'video' &&
    videoSegment.start_frame < segment.end_frame &&
    videoSegment.end_frame > segment.start_frame
  ))
}

function getDefaultSystemPromptForSegment(
  segment: MultiTrackSegment,
  options: SystemPromptOption[],
  videoSegments: MultiTrackSegment[],
  format?: string,
): string {
  const images = taskImages(segment)
  const taskType = getMultiTrackTaskType(
    segment.content.task_mode ?? 'default',
    images.length,
    hasVideoInRange(segment, videoSegments),
  )
  const mode = segment.content.task_mode ?? 'default'
  const promptTaskType = taskType === 'fl2v' || taskType === 'fmlf2v' ? 'i2v' : taskType
  const formatMatch = options.find((option) => (
    option.format === format
    && (option.modes?.includes(mode) || option.task_type === taskType)
  ))
  return formatMatch?.system_prompt
    ?? options.find((option) => option.task_type === promptTaskType && option.format === undefined)?.system_prompt
    ?? ''
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
  node,
  app,
  trackSegments,
  selectedSegments,
  videoSegments = [],
  mediaTracks = [],
  frameRate = MULTITRACK_DEFAULT_FRAME_RATE,
  totalFrames,
  imageIndexOffset = 0,
  format,
  onContentChange,
  onTrackSegmentsContentChange,
  onTrackSegmentsChange,
  onDurationChange,
  onOpenImagePreview,
}: Readonly<TaskSegmentEditorProps>) {
  const t = useT()
  const draggedImageIdRef = useRef<string | null>(null)
  const [promptTab, setPromptTab] = useState<PromptTab>('user')
  const [editMode, setEditMode] = useState<EditMode>('individual')
  const [mediaSelectorOpen, setMediaSelectorOpen] = useState(false)
  const [reselectImageId, setReselectImageId] = useState<string | null>(null)
  const [isImageDragOver, setIsImageDragOver] = useState(false)
  const [systemPromptOptions, setSystemPromptOptions] = useState<SystemPromptOption[] | null>(cachedSystemPromptOptions ?? null)
  const [systemPromptLoading, setSystemPromptLoading] = useState(false)
  const [isDurationEditing, setIsDurationEditing] = useState(false)
  const duration = frameToSeconds(segmentDuration(segment), frameRate)
  const formattedDuration = formatMultiTrackDurationTimecode(duration, frameRate)
  const [durationInput, setDurationInput] = useState(formattedDuration)
  const images = taskImages(segment)
  const slotItems = useMemo(
    () => computeSlotItems(node, app, 'image'),
    [node, app, mediaSelectorOpen],
  )
  const reselectImage = images.find((image) => image.id === reselectImageId)
  const mode = segment.content.task_mode ?? 'default'
  const continuityMode = segment.content.continuity_mode ?? MULTITRACK_DEFAULT_CONTINUITY_MODE
  const refImageSize = segment.content.ref_image_size ?? MULTITRACK_DEFAULT_REF_IMAGE_SIZE
  const editableSegments = useMemo(() => (
    trackSegments && trackSegments.length > 0 ? trackSegments : [segment]
  ), [segment, trackSegments])
  const combinedPromptValue = editableSegments.map((item) => (
    getSelectedTaskUserPrompt(item.content)
  )).join('|')
  const [combinedPromptInput, setCombinedPromptInput] = useState(combinedPromptValue)
  const taskIndex = Math.max(0, editableSegments.findIndex((item) => item.id === segment.id))
  const firstTaskSegmentId = editableSegments.reduce((first, item) => (
    item.start_frame < first.start_frame ? item : first
  )).id
  const hasSelectedContinuityTargets = selectedSegments?.some((item) => item.id !== firstTaskSegmentId) === true
  const promptVariant: MultiTrackUserPromptVariant = segment.content.user_prompt_variant === 'b' ? 'b' : 'a'
  const promptValue = editMode === 'combined'
    ? combinedPromptInput
    : getSelectedTaskUserPrompt(segment.content)
  const usesMiniMaxPromptPlaceholder = format === 'MiniMax'
  const promptPlaceholder = usesMiniMaxPromptPlaceholder
    ? t('multitrack.minimaxPromptPlaceholder')
    : t('multitrack.promptPlaceholder')
  const segmentHasVideoInRange = hasVideoInRange(segment, videoSegments)
  const systemPromptDefault = getDefaultSystemPromptForSegment(segment, systemPromptOptions ?? [], videoSegments, format)
  const systemPromptValue = editMode === 'combined'
    ? editableSegments.map((item) => (
        item.content.system_prompt ?? getDefaultSystemPromptForSegment(item, systemPromptOptions ?? [], videoSegments, format)
      )).join('|')
    : segment.content.system_prompt || systemPromptDefault
  const promptResources = useMemo<PromptReferenceResource[]>(() => {
    const imageResources = images.map((image, index) => ({
      id: `image:${image.id}`,
      type: 'image' as const,
      index: index + 1,
      label: t('multitrack.referencePicture', { n: index + 1 }),
      detail: imageDisplayName(image),
      token: `@${t('multitrack.referencePicture', { n: index + 1 })}`,
      color: getSegmentTrackPresentation('task').backgroundColor,
      thumbnailUrl: mediaContentToViewUrl({
        source_type: image.source_type ?? 'input',
        file_path: image.file_path,
        local_path: image.local_path,
        url: image.url,
        slot_name: image.slot_name,
      }) ?? undefined,
    }))
    const videoTracks = mediaTracks.filter((track) => (
      track.type === 'video'
      && track.segments.some((item) => item.content.media_type === 'video')
    ))
    const audioTracks = mediaTracks.filter((track) => (
      track.type === 'audio'
      && track.segments.some((item) => item.content.media_type === 'audio')
    ))
    const videoAudioResources = videoTracks.map((track, index) => {
      const resourceIndex = index + 1
      const label = t('multitrack.referenceAudioItem', { n: resourceIndex })
      return {
        id: `video-audio:${track.id}`,
        type: 'audio' as const,
        index: resourceIndex,
        label,
        detail: track.name,
        token: `@${label}`,
        color: getSegmentTrackPresentation('video').waveformColor
          ?? getSegmentTrackPresentation('video').backgroundColor,
      }
    })
    const audioResources = audioTracks.map((track, index) => {
      const resourceIndex = videoTracks.length + index + 1
      const label = t('multitrack.referenceAudioItem', { n: resourceIndex })
      return {
        id: `audio:${track.id}`,
        type: 'audio' as const,
        index: resourceIndex,
        label,
        detail: track.name,
        token: `@${label}`,
        color: getSegmentTrackPresentation('audio').waveformColor
          ?? getSegmentTrackPresentation('audio').backgroundColor,
      }
    })
    const videoResources = videoTracks.map((track, index) => {
      const resourceIndex = index + 1
      const label = t('multitrack.referenceVideoItem', { n: resourceIndex })
      return {
        id: `video:${track.id}`,
        type: 'video' as const,
        index: resourceIndex,
        label,
        detail: track.name,
        token: `@${label}`,
        color: getSegmentTrackPresentation('video').waveformColor
          ?? getSegmentTrackPresentation('video').backgroundColor,
      }
    })
    return [...imageResources, ...videoAudioResources, ...audioResources, ...videoResources]
  }, [images, mediaTracks, t])

  useEffect(() => {
    setDurationInput(formattedDuration)
  }, [formattedDuration])

  useEffect(() => {
    setIsDurationEditing(false)
  }, [segment.id])

  useEffect(() => {
    setCombinedPromptInput((current) => current === combinedPromptValue ? current : combinedPromptValue)
  }, [combinedPromptValue])

  function commitDuration() {
    const nextDuration = parseMultiTrackDurationTimecode(durationInput, frameRate)
    if (nextDuration === null) {
      setDurationInput(formattedDuration)
      setIsDurationEditing(false)
      return
    }
    setDurationInput(formatMultiTrackDurationTimecode(nextDuration, frameRate))
    setIsDurationEditing(false)
    if (nextDuration !== duration) onDurationChange?.(nextDuration)
  }

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

    const results = await Promise.allSettled(files.map((file) => uploadTaskImageFile(file)))
    const uploaded = results.flatMap((result) => {
      if (result.status === 'fulfilled') return [result.value]
      console.error('[TaskSegmentEditor] failed to upload task image:', result.reason)
      return []
    })
    if (uploaded.length > 0) {
      invalidateMediaListCache('inputs')
      onContentChange({ images: [...images, ...uploaded] })
    }
  }

  function handleSelectedMedia(filePath: string, source?: 'input' | 'output' | 'local') {
    const remainingSlots = reselectImageId ? 1 : MAX_TASK_IMAGES - images.length
    if (remainingSlots <= 0) return
    const selectedPaths = splitSelectedTaskMedia(filePath).slice(0, remainingSlots)
    const sourceType = selectedImageSource(filePath, source)
    const nextImages = selectedPaths.map((path) => createTaskImage(path, sourceType))
    if (nextImages.length === 0) return
    if (reselectImageId) {
      const replacement = { ...nextImages[0], id: reselectImageId }
      if (reselectImage?.shared_reference === true && onTrackSegmentsContentChange) {
        const oldIdentity = taskImageIdentity(reselectImage)
        const unsharedUpdates = new Map(
          sharedTaskImageUpdates(editableSegments, reselectImage, false)
            .map((update) => [update.segmentId, update.images]),
        )
        const replacementSourceSegments = editableSegments.map((item) => ({
          ...item,
          content: {
            ...item.content,
            images: item.id === segment.id
              ? [replacement, ...(unsharedUpdates.get(item.id) ?? []).filter((image) => (
                  taskImageIdentity(image) !== oldIdentity
                ))]
              : unsharedUpdates.get(item.id) ?? taskImages(item),
          },
        }))
        onTrackSegmentsContentChange(
          sharedTaskImageUpdates(replacementSourceSegments, replacement, true).map((update) => ({
            segmentId: update.segmentId,
            patch: { images: update.images },
          })),
        )
        setReselectImageId(null)
        setMediaSelectorOpen(false)
        return
      }
      onContentChange({
        images: images.map((image) => image.id === reselectImageId ? replacement : image),
      })
    } else {
      onContentChange({ images: [...images, ...nextImages] })
    }
    setReselectImageId(null)
    setMediaSelectorOpen(false)
  }

  function commitCombinedPrompt(value: string) {
    const parts = value === '' ? [''] : value.split(/[|｜]/)
    if (onTrackSegmentsChange) {
      const trackEndFrame = editableSegments.reduce((max, item) => Math.max(max, item.end_frame), 0)
      onTrackSegmentsChange(applyCombinedTaskTexts(
        parts,
        editableSegments,
        totalFrames ?? trackEndFrame,
        segment.color,
      ))
      return
    }
    onTrackSegmentsContentChange?.(editableSegments.map((item, index) => ({
      segmentId: item.id,
      patch: getSelectedTaskUserPromptPatch(item.content, parts[index] ?? ''),
    })))
  }

  function handlePromptChange(value: string) {
    if (editMode === 'combined') {
      setCombinedPromptInput(value)
      commitCombinedPrompt(value)
      return
    }
    onContentChange(getSelectedTaskUserPromptPatch(segment.content, value))
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

  function handleDropdownContentChange(patch: Partial<MultiTrackSegmentContent>) {
    if (!selectedSegments || selectedSegments.length <= 1 || !onTrackSegmentsContentChange) {
      onContentChange(patch)
      return
    }

    onTrackSegmentsContentChange(selectedSegments.map((item) => ({
      segmentId: item.id,
      patch: patch.continuity_mode !== undefined && item.id === firstTaskSegmentId
        ? { ...patch, continuity_mode: MULTITRACK_DEFAULT_CONTINUITY_MODE }
        : patch,
    })))
  }

  function handleDeleteImage(imageId: string) {
    const image = images.find((item) => item.id === imageId)
    if (image?.shared_reference === true && onTrackSegmentsContentChange) {
      const identity = taskImageIdentity(image)
      const updates = sharedTaskImageUpdates(editableSegments, image, false).map((update) => ({
        segmentId: update.segmentId,
        patch: {
          images: update.segmentId === segment.id
            ? update.images.filter((item) => taskImageIdentity(item) !== identity)
            : update.images,
        },
      }))
      onTrackSegmentsContentChange(updates)
      return
    }
    onContentChange({ images: images.filter((image) => image.id !== imageId) })
  }

  function handleSharedImageChange(image: MultiTrackTaskImage, enabled: boolean) {
    if (enabled && !canEnableSharedTaskImage(editableSegments, image)) return
    const updates = sharedTaskImageUpdates(editableSegments, image, enabled)
    if (onTrackSegmentsContentChange) {
      onTrackSegmentsContentChange(updates.map((update) => ({
        segmentId: update.segmentId,
        patch: { images: update.images },
      })))
      return
    }
    const current = updates.find((update) => update.segmentId === segment.id)
    if (current) onContentChange({ images: current.images })
  }

  const imageGridColumns = images.length > 0 && images.length < 4 ? 'grid-cols-2' : 'grid-cols-3'
  const imagePickerSurfaceClass = isImageDragOver ? 'border-primary bg-accent/20' : 'border-border bg-muted/20'
  const containerRef = useRef<HTMLDivElement>(null)
  const [showEditModeToggle, setShowEditModeToggle] = useState(true)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? container.clientWidth
      setShowEditModeToggle(width >= 600)
    })

    observer.observe(container)
    // Initial check
    setShowEditModeToggle(container.clientWidth >= 600)

    return () => observer.disconnect()
  }, [])

  return (
    <div
      ref={containerRef}
      data-capture-wheel="true"
      className="task-segment-editor flex h-full min-h-24 w-full flex-col overflow-hidden rounded-sm bg-background text-foreground"
    >
      <div className="flex min-h-0 flex-1 gap-4 p-4">
        {editMode === 'individual' && (
          <div
            data-testid="task-image-drop-zone"
            aria-label={t('multitrack.taskImageDropZone')}
            className={cn(
              'task-image-drop-zone flex aspect-square h-full min-h-0 shrink-0 items-center justify-center rounded-md border border-dashed transition-colors',
              isImageDragOver ? 'border-primary bg-accent/20' : 'border-border bg-muted/30',
            )}
            onDragEnter={handleImageDragEnter}
            onDragOver={handleImageDragOver}
            onDragLeave={handleImageDragLeave}
            onDrop={handleDrop}
          >
            <Popover
              open={mediaSelectorOpen}
              onOpenChange={(open) => {
                setMediaSelectorOpen(open)
                if (!open) setReselectImageId(null)
              }}
            >
              {images.length === 0 ? (
                <PopoverTrigger asChild>
                  <div
                    role="button"
                    tabIndex={0}
                    className={cn(
                      'task-image-picker-empty flex aspect-square h-full max-h-full min-h-24 min-w-24 max-w-full cursor-pointer flex-col items-center justify-center gap-1 rounded-md px-4 py-2 text-foreground shadow-sm transition-colors hover:bg-accent hover:text-accent-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
                      imagePickerSurfaceClass,
                    )}
                    aria-label={t('multitrack.taskImageDropZone')}
                    onClick={() => setReselectImageId(null)}
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
                <PopoverAnchor asChild>
                  <div
                    data-testid="task-image-grid"
                    className={cn(
                      'task-image-grid relative grid h-full w-full content-start gap-2 overflow-y-auto rounded-md p-3 transition-colors',
                      imageGridColumns,
                      imagePickerSurfaceClass,
                    )}
                  >
                  {images.map((image, index) => {
                    const imageUrl = mediaContentToViewUrl({
                      source_type: image.source_type ?? 'input',
                      file_path: image.file_path,
                      local_path: image.local_path,
                      url: image.url,
                      slot_name: image.slot_name,
                    })
                    const canShareImage = image.shared_reference === true
                      || canEnableSharedTaskImage(editableSegments, image)
                    return (
                      <div
                        key={image.id}
                        draggable
                        data-testid={`task-image-${image.id}`}
                        className={`task-image-grid-item group relative flex aspect-square w-full self-start cursor-pointer items-center justify-center overflow-hidden rounded-md border bg-black ${
                          image.shared_reference ? 'border-highlight' : 'border-border'
                        }`}
                        role="button"
                        tabIndex={0}
                        aria-label={t('multitrack.reselectImage', { name: imageDisplayName(image) })}
                        onClick={() => {
                          setReselectImageId(image.id)
                          setMediaSelectorOpen(true)
                        }}
                        onKeyDown={(event) => {
                          if (event.target !== event.currentTarget) return
                          if (event.key !== 'Enter' && event.key !== ' ') return
                          event.preventDefault()
                          setReselectImageId(image.id)
                          setMediaSelectorOpen(true)
                        }}
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
                        {imageUrl && image.panorama_view ? (
                          <PanoramaImagePreview
                            imageId={image.id}
                            imageUrl={imageUrl}
                            alt={imageDisplayName(image)}
                            view={image.panorama_view}
                            className="absolute inset-0 m-auto"
                          />
                        ) : imageUrl ? (
                          <img
                            src={imageUrl}
                            alt={imageDisplayName(image)}
                            className="absolute inset-0 h-full w-full object-contain"
                            draggable={false}
                          />
                        ) : (
                          <div className="flex h-full w-full items-center justify-center px-2 text-center text-[8px] text-muted-foreground">
                            {image.source_type === 'slot'
                              ? t('mediaSelector.slotImage', { n: taskImageSlotNumber(image, index) })
                              : imageDisplayName(image)}
                          </div>
                        )}
                        <div
                          data-testid={`task-image-actions-${image.id}`}
                          className="absolute right-1 top-1 z-10 flex gap-1 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100"
                        >
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Button
                                type="button"
                                size="icon"
                                variant={image.shared_reference === true ? 'secondary' : 'ghost'}
                                data-testid={`task-image-shared-${image.id}`}
                                className={`h-6 w-6 cursor-pointer bg-background/70 hover:bg-background/90 [&_svg]:!size-3 ${image.shared_reference === true ? 'text-highlight' : 'text-muted-foreground'}`}
                                aria-label={image.shared_reference === true
                                  ? t('multitrack.disableSharedReference')
                                  : t('multitrack.enableSharedReference')}
                                aria-pressed={image.shared_reference === true}
                                disabled={!canShareImage}
                                onClick={(event) => {
                                  event.stopPropagation()
                                  handleSharedImageChange(image, image.shared_reference !== true)
                                }}
                                >
                                  <Share2 />
                                </Button>
                              </TooltipTrigger>
                              <TooltipContent side="bottom" className="max-w-64">
                                {canShareImage
                                  ? t('multitrack.sharedImageReferenceTooltip')
                                  : t('multitrack.sharedImageLimitTooltip')}
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                          <Button
                            type="button"
                            size="icon"
                            variant="ghost"
                            className="h-6 w-6 cursor-pointer bg-background/70 text-foreground hover:bg-background/90 [&_svg]:!size-3"
                            aria-label={t('multitrack.previewImage')}
                            disabled={!imageUrl}
                            onClick={(event) => {
                              event.stopPropagation()
                              onOpenImagePreview?.(image.id)
                            }}
                          >
                            <Eye />
                          </Button>
                          <Button
                            type="button"
                            size="icon"
                            variant="ghost"
                            className="h-6 w-6 cursor-pointer bg-background/70 text-destructive hover:bg-background/90 hover:text-destructive [&_svg]:!size-3"
                            aria-label={t('multitrack.deleteImage')}
                            onClick={(event) => {
                              event.stopPropagation()
                              handleDeleteImage(image.id)
                            }}
                          >
                            <Trash2 />
                          </Button>
                        </div>
                        <span
                          data-testid={`task-image-index-${image.id}`}
                          className="absolute bottom-0 left-0 z-10 min-w-5 rounded-sm bg-black/50 px-1.5 py-0.5 text-center text-[9px] font-semibold leading-none text-white"
                        >
                          {index + imageIndexOffset}
                        </span>
                      </div>
                    )
                  })}
                  {images.length < MAX_TASK_IMAGES && (
                    <Button
                      type="button"
                      variant="outline"
                      className="task-image-grid-add aspect-square h-auto w-full self-start border-dashed text-muted-foreground"
                      aria-label={t('multitrack.selectImage')}
                      onClick={() => {
                        setReselectImageId(null)
                        setMediaSelectorOpen(true)
                      }}
                    >
                      <Plus className="h-7 w-7" />
                    </Button>
                  )}
                  </div>
                </PopoverAnchor>
              )}
              <PopoverContent className="w-auto p-0" align="end">
                <MediaSelector
                  key={reselectImageId ?? 'add-images'}
                  value={imageSelectorValue(reselectImage)}
                  mediaType="image"
                  defaultTab={imageSelectorTab(reselectImage)}
                  allowMultipleSelection={!reselectImageId}
                  maxSelectionCount={reselectImageId ? 1 : MAX_TASK_IMAGES - images.length}
                  slotItems={slotItems}
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
              {promptTab === 'user' ? (
                <TooltipProvider delayDuration={300}>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <div aria-label={t('multitrack.userPromptVariantTooltip')}>
                        <Tabs
                          value={promptVariant}
                          onValueChange={(value) => onContentChange({
                            user_prompt_variant: value as MultiTrackUserPromptVariant,
                          })}
                        >
                          <TabsList className="h-7 bg-card p-1">
                            <TabsTrigger value="a" className="h-full min-w-7 px-2 text-[10px]">A</TabsTrigger>
                            <TabsTrigger value="b" className="h-full min-w-7 px-2 text-[10px]">B</TabsTrigger>
                          </TabsList>
                        </Tabs>
                      </div>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="max-w-72">
                      {t('multitrack.userPromptVariantTooltip')}
                    </TooltipContent>
                  </Tooltip>
                </TooltipProvider>
              ) : Boolean(segment.content.system_prompt) && (
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
              <div className="relative h-full min-h-0 flex-1 overflow-hidden rounded-md bg-card">
                <PromptContentEditor
                  ariaLabel={t('multitrack.prompt')}
                  placeholder={promptPlaceholder}
                  interactivePlaceholder={usesMiniMaxPromptPlaceholder}
                  testId="combined-prompt-highlight"
                  className="border-none bg-card shadow-none"
                  value={promptValue}
                  resources={promptResources}
                  mentionsEnabled
                  highlightPromptSemantics
                  highlightPipes
                  onChange={handlePromptChange}
                />
              </div>
              <p className="shrink-0 text-[9px] text-muted-foreground mt-1">{t('maintainTrack.combinedHint')}</p>
            </div>
          ) : promptTab === 'user' ? (
            <div className="relative h-full min-h-0 flex-1 overflow-hidden pb-2 px-2">
              <PromptContentEditor
                placeholder={promptPlaceholder}
                interactivePlaceholder={usesMiniMaxPromptPlaceholder}
                ariaLabel={t('multitrack.prompt')}
                className="border-none bg-card shadow-none"
                value={promptValue}
                resources={promptResources}
                mentionsEnabled
                highlightPromptSemantics
                onChange={handlePromptChange}
              />
            </div>
          ) : (
            <div className="relative h-full min-h-0 flex-1 overflow-hidden pb-2 px-2">
              <div className="relative h-full min-h-0 overflow-hidden rounded-md bg-card">
                <PromptContentEditor
                  ariaLabel={t('multitrack.systemPrompt')}
                  placeholder={systemPromptLoading ? t('multitrack.loadingSystemPrompt') : t('multitrack.systemPromptPlaceholder')}
                  className="border-none bg-card shadow-none"
                  value={systemPromptValue}
                  resources={promptResources}
                  highlightSystemVariables
                  onChange={handleSystemPromptChange}
                />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="relative flex shrink-0 items-center justify-between border-t border-dashed border-border p-2">
        <Tabs value={editMode} onValueChange={(value) => setEditMode(value as EditMode)} className={showEditModeToggle ? undefined : 'hidden'}>
          <TabsList className="h-8 bg-card">
            <TabsTrigger value="individual" className="text-[10px]">{t('multitrack.individualEdit')}</TabsTrigger>
            <TabsTrigger value="combined" className="text-[10px]">{t('multitrack.combinedEdit')}</TabsTrigger>
          </TabsList>
        </Tabs>

        <div className={cn(
          showEditModeToggle ? 'absolute left-1/2 -translate-x-1/2' : 'flex items-center',
          isDurationEditing && 'w-28'
        )}>
          {isDurationEditing ? (
            <Input
              autoFocus
              aria-label={t('multitrack.duration')}
              type="text"
              inputMode="numeric"
              placeholder="00:00:00"
              className="tabular-nums"
              value={durationInput}
              onChange={(event) => setDurationInput(event.currentTarget.value)}
              onBlur={commitDuration}
              onKeyDown={(event) => {
                if (event.key !== 'Enter') return
                event.preventDefault()
                commitDuration()
              }}
            />
          ) : (
            <div className="flex items-center gap-1 text-muted-foreground">
              <div className="flex flex-col items-center">
                <span className="text-[10px] text-primary">{t('multitrack.taskNumber', { n: taskIndex + 1 })}</span>
                <span className="text-[10px] mt-0.5 tabular-nums">{formattedDuration}</span>
              </div>
              {onDurationChange && (
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="h-5 w-5 cursor-pointer"
                  aria-label={t('multitrack.editTaskDuration')}
                  onClick={() => setIsDurationEditing(true)}
                >
                  <Pencil className="h-3 w-3" />
                </Button>
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          {format === 'MiniMax' && (taskIndex > 0 || hasSelectedContinuityTargets) && (
            <Select
              value={continuityMode}
              onValueChange={(value) => handleDropdownContentChange({
                continuity_mode: value as MultiTrackContinuityMode,
              })}
            >
              <TooltipProvider delayDuration={300}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <SelectTrigger aria-label={t('multitrack.continuityMode')} className="h-8 w-28 bg-card text-[10px]">
                      <SelectValue />
                    </SelectTrigger>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="max-w-72">
                    {t('multitrack.continuityModeTooltip')}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <SelectContent>
                {MULTITRACK_CONTINUITY_MODES.map((continuityOption) => (
                  <SelectItem key={continuityOption} value={continuityOption}>
                    <span className="text-[10px]">{t(`multitrackContinuityModes.${continuityOption}`)}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          {format === 'MiniMax' && ['ref','edit'].includes(mode) && (
            <Select
              value={refImageSize}
              onValueChange={(value) => handleDropdownContentChange({ ref_image_size: value as MultiTrackRefImageSize })}
            >
              <TooltipProvider delayDuration={300}>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <SelectTrigger aria-label={t('multitrack.refImageSize')} className="h-8 w-20 bg-card text-[10px]">
                      <SelectValue />
                    </SelectTrigger>
                  </TooltipTrigger>
                  <TooltipContent side="bottom" className="max-w-72">
                    {t('multitrack.refImageSizeTooltip')}
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
              <SelectContent>
                {MULTITRACK_REF_IMAGE_SIZES.map((refImageSizeOption) => (
                  <SelectItem key={refImageSizeOption} value={refImageSizeOption}>
                    <span className="text-[10px]">{t(`multitrackRefImageSizes.${refImageSizeOption}`)}</span>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}
          <Select value={mode} onValueChange={(value) => handleDropdownContentChange({ task_mode: value as MultiTrackTaskMode })}>
            <TooltipProvider delayDuration={300}>
              <Tooltip>
                <TooltipTrigger asChild>
                  <SelectTrigger aria-label={t('multitrack.taskMode')} className="h-8 w-24 bg-card text-[10px]">
                    <SelectValue />
                  </SelectTrigger>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-72">
                  {t('multitrack.taskModeTooltip')}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <SelectContent>
              {MULTITRACK_TASK_MODES.map((taskMode) => (
                <SelectItem key={taskMode} value={taskMode}>
                  <span className="text-[10px]">
                    {t(`multitrackTaskModes.${getMultiTrackTaskType(taskMode, images.length, segmentHasVideoInRange)}`)}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    </div>
  )
}
