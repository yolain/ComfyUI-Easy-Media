import { useEffect, useMemo, useRef, useState } from 'react'
import { ArrowLeft, Captions, Clapperboard, Eye, ListTree, PanelLeft, PanelRight, Plus, Volume2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import {
  collectMultiTrackPreviewResolutionInput,
  frameToSeconds,
  getActivePreviewAudioSources,
  getActivePreviewVideoSegment,
  getMultiTrackTaskModeLabel,
  MULTITRACK_TASK_MODES,
  parseMultiTrackPreviewResolution,
  segmentDuration,
  snapTimeToFrame,
  type MultiTrackVideoMetadata,
  type SelectedMultiTrackSegment,
} from '@/lib/multitrack-utils'
import { useT } from '@/lib/i18n'
import { mediaContentToViewUrl } from '@/lib/media-url'
import {
  DEFAULT_SUBTITLE_SPEECH_SETTINGS,
  type SubtitleSpeechSettings,
} from '@/lib/subtitle-speech'
import {
  createTaskImage,
  MAX_TASK_IMAGES,
  splitSelectedTaskMedia,
  taskImagesFromContent,
  uploadTaskImageFile,
} from '@/lib/task-image-utils'
import { loadBrowserVideoMetadata } from '@/lib/video-utils'
import { DEFAULT_SUBTITLE_STYLE } from '@/lib/subtitle-recognition'
import { invalidateMediaListCache } from '@/stores/media-list-store'
import type { MultiTrackSegment, MultiTrackSegmentContent, MultiTrackSubtitleStyle, MultiTrackTaskImage, MultiTrackTaskMode, TrackData } from '@/types/multitrack'
import { PreviewFloatingToolbar } from './PreviewFloatingToolbar'
import { PreviewAudioPlayback } from './PreviewAudioPlayback'
import { PanoramaImagePreview } from '@/components/widgets/panorama/PanoramaImagePreview'
import { TaskSegmentEditor } from './TaskSegmentEditor'
import { VideoPreview } from './VideoPreview'
import { SubtitleSettingsPanel } from './SubtitleSettingsPanel'

const RESOLUTION_POLL_INTERVAL_MS = 250
const SUBTITLE_SETTINGS_PANEL_TOOLBAR_INSET = 'calc(max(35%, 18rem) + 12px)'
const EXPANDED_IMAGE_MIN_ZOOM = 1
const EXPANDED_IMAGE_MAX_ZOOM = 6
const EXPANDED_IMAGE_ZOOM_STEP = 1.15

interface PreviewAreaProps {
  data: TrackData
  currentTime: number
  selectedSegment: SelectedMultiTrackSegment | null
  isPlaying: boolean
  playbackNonce?: number
  node: unknown
  editingSubtitleSegmentId?: string | null
  onSubtitleEditRequestHandled?: () => void
  onSelectSegment?: (segmentId: string) => void
  onGlobalSettingsChange: (patch: Partial<Pick<TrackData, 'muted' | 'volume_db' | 'frame_rate'>>) => void
  onSelectedSegmentContentChange: (patch: Partial<MultiTrackSegmentContent>) => void
  taskSegments?: MultiTrackSegment[]
  onTrackSegmentsContentChange?: (updates: Array<{ segmentId: string; patch: Partial<MultiTrackSegmentContent> }>) => void
  onTaskTrackSegmentsChange?: (segments: MultiTrackSegment[]) => void
  onSelectedSegmentDurationChange: (duration: number) => void
  onGenerateSubtitleSpeech?: (
    segment: MultiTrackSegment,
    settings: SubtitleSpeechSettings,
  ) => Promise<void>
}

function resolutionInputSignature(input: unknown): string {
  try {
    return JSON.stringify(input)
  } catch {
    return String(Date.now())
  }
}

interface ActiveTaskImages {
  index: number
  segmentId: string
  allImages: MultiTrackTaskImage[]
  images: Array<{ image: MultiTrackTaskImage; url: string }>
}

interface ActiveTaskPrompt {
  segmentId: string
  prompt: string
  taskMode: MultiTrackTaskMode
}

type PreviewLayoutMode = 'balanced' | 'image-large'

interface ExpandedTaskImageTarget {
  imageId: string
  segmentId: string
  source: 'active' | 'selected'
}

function moveTaskImage(
  images: MultiTrackTaskImage[],
  sourceId: string,
  targetId: string,
): MultiTrackTaskImage[] {
  const sourceIndex = images.findIndex((image) => image.id === sourceId)
  const targetIndex = images.findIndex((image) => image.id === targetId)
  if (sourceIndex < 0 || targetIndex < 0 || sourceIndex === targetIndex) return images

  const nextImages = [...images]
  const [movedImage] = nextImages.splice(sourceIndex, 1)
  nextImages.splice(targetIndex, 0, movedImage)
  return nextImages
}

function getActiveTaskImages(data: TrackData, currentTime: number): ActiveTaskImages | null {
  const currentFrame = snapTimeToFrame(currentTime, data.frame_rate)

  for (const track of data.tracks) {
    if (track.type !== 'task') continue
    const index = track.segments.findIndex((segment) => (
      currentFrame >= segment.start_frame && currentFrame < segment.end_frame
    ))
    if (index < 0) continue

    const segment = track.segments[index]
    const allImages = taskImagesFromContent(segment.content.images)
    const images = allImages.flatMap((image) => {
      const url = mediaContentToViewUrl({
        source_type: image.source_type ?? 'input',
        file_path: image.file_path,
        local_path: image.local_path,
        url: image.url,
        slot_name: image.slot_name,
      })
      return url ? [{ image, url }] : []
    })
    return { index, segmentId: segment.id, allImages, images }
  }

  return null
}

function getActiveTaskPrompt(data: TrackData, currentTime: number): ActiveTaskPrompt | null {
  const currentFrame = snapTimeToFrame(currentTime, data.frame_rate)

  for (const track of data.tracks) {
    if (track.type !== 'task') continue
    const segment = track.segments.find((item) => (
      currentFrame >= item.start_frame && currentFrame < item.end_frame
    ))
    if (!segment) continue

    const prompt = segment.content.user_prompt ?? ''
    return { segmentId: segment.id, prompt, taskMode: segment.content.task_mode ?? 'default' }
  }

  return null
}

function getActiveSubtitleSegments(data: TrackData, currentTime: number): MultiTrackSegment[] {
  const currentFrame = snapTimeToFrame(currentTime, data.frame_rate)
  return data.tracks.flatMap((track) => {
    if (track.type !== 'subtitle' || track.visible === false) return []
    return track.segments.filter((item) => (
      currentFrame >= item.start_frame && currentFrame < item.end_frame
    ))
  })
}

function clampSubtitleStyle(style: MultiTrackSubtitleStyle): MultiTrackSubtitleStyle {
  const width = Math.max(0.1, Math.min(1, style.width))
  const x = Math.max(0, Math.min(1 - width, style.x))
  return {
    ...style,
    font_size: Math.max(8, Math.min(96, style.font_size)),
    outline_color: style.outline_color || '#000000',
    background_opacity: Math.max(0, Math.min(1, style.background_opacity ?? 0.7)),
    x,
    y: Math.max(0, Math.min(0.95, style.y)),
    width,
  }
}

function isTransparentSubtitleBackground(backgroundColor: string): boolean {
  const value = backgroundColor.trim().toLowerCase()
  return value === 'transparent' || value === 'rgba(0, 0, 0, 0)' || value === 'rgba(0,0,0,0)'
}

function subtitleBackgroundColor(backgroundColor: string, opacity: number): string {
  if (isTransparentSubtitleBackground(backgroundColor)) return 'transparent'
  const alpha = Math.max(0, Math.min(1, opacity))
  const hex = backgroundColor.trim().match(/^#?([0-9a-fA-F]{6})$/)
  if (hex) {
    const raw = hex[1]
    const red = Number.parseInt(raw.slice(0, 2), 16)
    const green = Number.parseInt(raw.slice(2, 4), 16)
    const blue = Number.parseInt(raw.slice(4, 6), 16)
    return `rgba(${red}, ${green}, ${blue}, ${alpha})`
  }
  const rgb = backgroundColor.trim().match(/^rgba?\(\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)(?:\s*,\s*[0-9.]+)?\s*\)$/)
  if (rgb) return `rgba(${rgb[1]}, ${rgb[2]}, ${rgb[3]}, ${alpha})`
  return backgroundColor
}

function subtitleTextShadow(outlineColor: string): string {
  return [
    `1px 0 0 ${outlineColor}`,
    `-1px 0 0 ${outlineColor}`,
    `0 1px 0 ${outlineColor}`,
    `0 -1px 0 ${outlineColor}`,
    `1px 1px 0 ${outlineColor}`,
    `-1px 1px 0 ${outlineColor}`,
    `1px -1px 0 ${outlineColor}`,
    `-1px -1px 0 ${outlineColor}`,
  ].join(', ')
}

export function PreviewArea({
  data,
  currentTime,
  selectedSegment,
  isPlaying,
  playbackNonce = 0,
  node,
  editingSubtitleSegmentId,
  onSubtitleEditRequestHandled,
  onSelectSegment,
  onGlobalSettingsChange,
  onSelectedSegmentContentChange,
  taskSegments,
  onTrackSegmentsContentChange,
  onTaskTrackSegmentsChange,
  onSelectedSegmentDurationChange,
  onGenerateSubtitleSpeech,
}: Readonly<PreviewAreaProps>) {
  const t = useT()
  const draggedTaskImageIdRef = useRef<string | null>(null)
  const expandedTaskImageRef = useRef<HTMLImageElement>(null)
  const [expandedTaskImageTarget, setExpandedTaskImageTarget] = useState<ExpandedTaskImageTarget | null>(null)
  const [expandedTaskImageZoom, setExpandedTaskImageZoom] = useState(EXPANDED_IMAGE_MIN_ZOOM)
  const [expandedTaskImageOrigin, setExpandedTaskImageOrigin] = useState({ x: 50, y: 50 })
  const [activeTaskPreviewImageId, setActiveTaskPreviewImageId] = useState<string | null>(null)
  const [previewLayoutMode, setPreviewLayoutMode] = useState<PreviewLayoutMode>('balanced')
  const [activeTaskMediaSelectorOpen, setActiveTaskMediaSelectorOpen] = useState(false)
  const [activeTaskImageDragOver, setActiveTaskImageDragOver] = useState(false)
  const [editingTaskPrompt, setEditingTaskPrompt] = useState<{ segmentId: string; text: string } | null>(null)
  const [editingSubtitle, setEditingSubtitle] = useState<{ segmentId: string; text: string } | null>(null)
  const [subtitleSpeechSettings, setSubtitleSpeechSettings] = useState<SubtitleSpeechSettings>(DEFAULT_SUBTITLE_SPEECH_SETTINGS)
  const [firstVideoMetadata, setFirstVideoMetadata] = useState<MultiTrackVideoMetadata | null>(null)
  const [resolutionInput, setResolutionInput] = useState(() => collectMultiTrackPreviewResolutionInput(node))
  const videoSegments = useMemo(() => (
    data.tracks
      .filter((track) => track.type === 'video')
      .flatMap((track) => track.segments)
  ), [data.tracks])
  const hasSegments = useMemo(
    () => data.tracks.some((track) => track.segments.length > 0),
    [data.tracks],
  )
  const firstVideoUrl = useMemo(() => {
    const firstVideoSegment = videoSegments.find((segment) => segment.content.media_type === 'video')
    if (!firstVideoSegment) return null
    return mediaContentToViewUrl({
      source_type: firstVideoSegment.content.source_type ?? 'input',
      file_path: firstVideoSegment.content.file_path,
      local_path: firstVideoSegment.content.local_path,
      url: firstVideoSegment.content.url,
      slot_name: firstVideoSegment.content.slot_name,
    })
  }, [videoSegments])
  const previewSelectedSegment = selectedSegment?.trackType === 'audio' ? null : selectedSegment
  const activeVideo = previewSelectedSegment?.trackType === 'task'
    ? null
    : getActivePreviewVideoSegment(data, currentTime, previewSelectedSegment?.trackType === 'video' ? previewSelectedSegment.segment.id : null)
  const activeAudioSources = getActivePreviewAudioSources(data, currentTime, previewSelectedSegment)
  const activeTaskImages = previewSelectedSegment === null ? getActiveTaskImages(data, currentTime) : null
  const activeTaskPrompt = previewSelectedSegment === null ? getActiveTaskPrompt(data, currentTime) : null
  const activeSubtitleSegments = useMemo(() => {
    if (previewSelectedSegment?.trackType === 'task') return []
    const segments = getActiveSubtitleSegments(data, currentTime)
    if (previewSelectedSegment?.trackType !== 'subtitle') return segments
    const selectedSubtitle = previewSelectedSegment.segment
    const selectedIndex = segments.findIndex((segment) => segment.id === selectedSubtitle.id)
    if (selectedIndex < 0) return [...segments, selectedSubtitle]
    return segments.map((segment, index) => index === selectedIndex ? selectedSubtitle : segment)
  }, [currentTime, data, previewSelectedSegment])
  const resolution = parseMultiTrackPreviewResolution(resolutionInput, firstVideoMetadata)
  const selectedMediaDuration = selectedSegment?.trackType === 'video' || selectedSegment?.trackType === 'audio'
    ? frameToSeconds(segmentDuration(selectedSegment.segment), data.frame_rate)
    : null
  const selectedSubtitleStyle = selectedSegment?.trackType === 'subtitle'
    ? clampSubtitleStyle({
        ...DEFAULT_SUBTITLE_STYLE,
        ...selectedSegment.segment.content.subtitle_style,
      })
    : null
  const selectedTaskImages = selectedSegment?.trackType === 'task'
    ? selectedSegment.segment.content.images ?? []
    : []
  const expandedTaskImage = expandedTaskImageTarget?.source === 'selected'
    ? selectedTaskImages.find((image) => image.id === expandedTaskImageTarget.imageId) ?? null
    : expandedTaskImageTarget?.source === 'active'
      ? activeTaskImages?.allImages.find((image) => image.id === expandedTaskImageTarget.imageId) ?? null
      : null
  const activeTaskPreviewImage = activeTaskImages?.images.find(({ image }) => image.id === activeTaskPreviewImageId)
    ?? activeTaskImages?.images[0]
    ?? null
  const usesTaskImageOnlyPreview = Boolean(activeTaskImages && !activeVideo)
  const activeTaskImagePanelClassName = previewLayoutMode === 'image-large'
    ? 'w-32 flex-[0_0_8rem]'
    : 'w-20 flex-[0_0_5rem]'
  const activeTaskVideoPanelClassName = previewLayoutMode === 'image-large'
    ? 'h-full flex-none'
    : 'h-full flex-none'
  const previewMediaGroupClassName = selectedSubtitleStyle
    ? 'mx-auto flex h-full min-h-24 w-full max-w-full items-stretch justify-center gap-0'
    : usesTaskImageOnlyPreview
    ? 'mx-auto flex h-full min-h-24 w-full max-w-full items-center justify-center gap-3'
    : 'mx-auto flex h-full min-h-24 w-fit max-w-full items-center justify-center gap-3'

  useEffect(() => {
    setExpandedTaskImageTarget(null)
  }, [selectedSegment?.segment.id])

  useEffect(() => {
    if (!editingSubtitleSegmentId) return
    const segment = activeSubtitleSegments.find((item) => item.id === editingSubtitleSegmentId)
    if (!segment) return
    setEditingSubtitle({
      segmentId: segment.id,
      text: segment.content.text ?? '',
    })
    onSubtitleEditRequestHandled?.()
  }, [activeSubtitleSegments, editingSubtitleSegmentId, onSubtitleEditRequestHandled])

  useEffect(() => {
    if (!activeTaskImages) {
      setActiveTaskPreviewImageId(null)
      return
    }
    if (activeTaskImages.images.some(({ image }) => image.id === activeTaskPreviewImageId)) return
    setActiveTaskPreviewImageId(activeTaskImages.images[0]?.image.id ?? null)
  }, [activeTaskImages, activeTaskPreviewImageId])

  useEffect(() => {
    if (!activeTaskPrompt) {
      setEditingTaskPrompt(null)
      return
    }
    if (editingTaskPrompt?.segmentId === activeTaskPrompt.segmentId) return
    setEditingTaskPrompt(null)
  }, [activeTaskPrompt, editingTaskPrompt?.segmentId])

  useEffect(() => {
    if (expandedTaskImageTarget?.source === 'active' && expandedTaskImageTarget.segmentId !== activeTaskImages?.segmentId) {
      setExpandedTaskImageTarget(null)
    }
  }, [expandedTaskImageTarget, activeTaskImages?.segmentId])

  useEffect(() => {
    if (!expandedTaskImageTarget) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setExpandedTaskImageTarget(null)
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [expandedTaskImageTarget])

  useEffect(() => {
    setExpandedTaskImageZoom(EXPANDED_IMAGE_MIN_ZOOM)
    setExpandedTaskImageOrigin({ x: 50, y: 50 })
  }, [expandedTaskImageTarget?.imageId, expandedTaskImageTarget?.segmentId])

  useEffect(() => {
    if (!firstVideoUrl) {
      setFirstVideoMetadata(null)
      return
    }

    let cancelled = false
    loadBrowserVideoMetadata(firstVideoUrl)
      .then((metadata) => {
        if (!cancelled) setFirstVideoMetadata({ width: metadata.width, height: metadata.height })
      })
      .catch((error: unknown) => {
        console.error('[PreviewArea] failed to read first video metadata:', error)
        if (!cancelled) setFirstVideoMetadata(null)
      })

    return () => {
      cancelled = true
    }
  }, [firstVideoUrl])

  useEffect(() => {
    let currentSignature = resolutionInputSignature(collectMultiTrackPreviewResolutionInput(node))
    setResolutionInput((current) => (
      resolutionInputSignature(current) === currentSignature
        ? current
        : collectMultiTrackPreviewResolutionInput(node)
    ))

    const timer = window.setInterval(() => {
      const nextInput = collectMultiTrackPreviewResolutionInput(node)
      const nextSignature = resolutionInputSignature(nextInput)
      if (nextSignature === currentSignature) return
      currentSignature = nextSignature
      setResolutionInput(nextInput)
    }, RESOLUTION_POLL_INTERVAL_MS)

    return () => window.clearInterval(timer)
  }, [node])

  function handleActiveTaskImageReorderDrop(targetImageId: string) {
    if (!activeTaskImages) return
    const sourceId = draggedTaskImageIdRef.current
    draggedTaskImageIdRef.current = null
    if (!sourceId) return
    onTrackSegmentsContentChange?.([{
      segmentId: activeTaskImages.segmentId,
      patch: { images: moveTaskImage(activeTaskImages.allImages, sourceId, targetImageId) },
    }])
  }

  function handleActiveTaskImageDelete(imageId: string) {
    if (!activeTaskImages) return
    onTrackSegmentsContentChange?.([{
      segmentId: activeTaskImages.segmentId,
      patch: {
        images: activeTaskImages.allImages.filter((item) => item.id !== imageId),
      },
    }])
  }

  function hasDraggedImageFile(dataTransfer: DataTransfer): boolean {
    const files = Array.from(dataTransfer.files)
    if (files.some((file) => file.type.startsWith('image/'))) return true
    return Array.from(dataTransfer.items ?? []).some((item) => item.kind === 'file' && item.type.startsWith('image/'))
  }

  function handleActiveTaskImageDragEnter(event: React.DragEvent<HTMLDivElement | HTMLButtonElement>) {
    if (!event.dataTransfer || !hasDraggedImageFile(event.dataTransfer)) return
    event.preventDefault()
    setActiveTaskImageDragOver(true)
  }

  function handleActiveTaskImageDragOver(event: React.DragEvent<HTMLDivElement | HTMLButtonElement>) {
    if (!event.dataTransfer || !hasDraggedImageFile(event.dataTransfer)) return
    event.preventDefault()
    setActiveTaskImageDragOver(true)
  }

  function handleActiveTaskImageDragLeave(event: React.DragEvent<HTMLDivElement | HTMLButtonElement>) {
    const relatedTarget = event.relatedTarget
    if (relatedTarget instanceof Node && event.currentTarget.contains(relatedTarget)) return
    setActiveTaskImageDragOver(false)
  }

  async function handleActiveTaskImageDrop(event: React.DragEvent<HTMLDivElement | HTMLButtonElement>) {
    event.preventDefault()
    setActiveTaskImageDragOver(false)
    if (!event.dataTransfer || !activeTaskImages) return
    const remainingSlots = MAX_TASK_IMAGES - activeTaskImages.allImages.length
    if (remainingSlots <= 0) return
    const files = Array.from(event.dataTransfer.files)
      .filter((file) => file.type.startsWith('image/'))
      .slice(0, remainingSlots)
    if (files.length === 0) return

    const results = await Promise.allSettled(files.map((file) => uploadTaskImageFile(file)))
    const uploaded = results.flatMap((result) => {
      if (result.status === 'fulfilled') return [result.value]
      console.error('[PreviewArea] failed to upload active task image:', result.reason)
      return []
    })
    if (uploaded.length === 0) return
    invalidateMediaListCache('inputs')
    onTrackSegmentsContentChange?.([{
      segmentId: activeTaskImages.segmentId,
      patch: { images: [...activeTaskImages.allImages, ...uploaded] },
    }])
  }

  function handleActiveTaskSelectedMedia(filePath: string, source?: 'input' | 'output' | 'local') {
    if (!activeTaskImages) return
    const remainingSlots = MAX_TASK_IMAGES - activeTaskImages.allImages.length
    if (remainingSlots <= 0) return
    const selectedPaths = splitSelectedTaskMedia(filePath).slice(0, remainingSlots)
    const sourceType = source ?? (filePath.startsWith('http://') || filePath.startsWith('https://') ? 'url' : 'input')
    const nextImages = selectedPaths.map((path) => createTaskImage(path, sourceType))
    if (nextImages.length === 0) return
    onTrackSegmentsContentChange?.([{
      segmentId: activeTaskImages.segmentId,
      patch: { images: [...activeTaskImages.allImages, ...nextImages] },
    }])
    setActiveTaskMediaSelectorOpen(false)
  }

  function handleTaskPromptInput(text: string) {
    if (!editingTaskPrompt) return
    setEditingTaskPrompt({ ...editingTaskPrompt, text })
    onTrackSegmentsContentChange?.([{
      segmentId: editingTaskPrompt.segmentId,
      patch: { user_prompt: text },
    }])
  }

  function commitTaskPromptEditing() {
    setEditingTaskPrompt(null)
  }

  function handleActiveTaskModeChange(segmentId: string, taskMode: MultiTrackTaskMode) {
    onTrackSegmentsContentChange?.([{
      segmentId,
      patch: { task_mode: taskMode },
    }])
  }

  function commitSubtitleEditing() {
    if (!editingSubtitle) return
    if (selectedSegment?.trackType === 'subtitle' && selectedSegment.segment.id === editingSubtitle.segmentId) {
      onSelectedSegmentContentChange({ text: editingSubtitle.text })
    } else {
      onTrackSegmentsContentChange?.([{
        segmentId: editingSubtitle.segmentId,
        patch: { text: editingSubtitle.text },
      }])
    }
    setEditingSubtitle(null)
  }

  function renderActiveTaskImagePreview(
    image: MultiTrackTaskImage,
    url: string,
    imageName: string,
    className: string,
  ) {
    return image.panorama_view ? (
      <PanoramaImagePreview
        imageId={image.id}
        imageUrl={url}
        alt={imageName}
        view={image.panorama_view}
        className={className}
      />
    ) : (
      <img
        className={className}
        src={url}
        alt={imageName}
        draggable={false}
      />
    )
  }

  function renderActiveTaskImageAddSlot(
    className = 'h-10 w-10 aspect-square',
    showLabel = false,
  ) {
    if (!activeTaskImages || activeTaskImages.allImages.length >= MAX_TASK_IMAGES) return null
    return (
      <Popover open={activeTaskMediaSelectorOpen} onOpenChange={setActiveTaskMediaSelectorOpen}>
        <PopoverTrigger asChild>
          <Button
            type="button"
            variant="outline"
            data-testid={activeTaskImages.images.length === 0
              ? 'task-preview-empty-add-image'
              : 'task-preview-add-image'}
            className={`cursor-pointer shrink-0 border-dashed bg-background/70 text-muted-foreground transition-colors hover:bg-background/90 ${
              activeTaskImageDragOver ? 'border-primary bg-accent/20 text-foreground' : 'border-border'
            } ${showLabel ? 'flex-col gap-1' : ''} ${className}`}
            aria-label={t('multitrack.addImage')}
            onClick={(event) => {
              event.stopPropagation()
              setActiveTaskMediaSelectorOpen(true)
            }}
            onDragEnter={handleActiveTaskImageDragEnter}
            onDragOver={handleActiveTaskImageDragOver}
            onDragLeave={handleActiveTaskImageDragLeave}
            onDrop={handleActiveTaskImageDrop}
          >
            <Plus className="h-5 w-5" />
            {showLabel ? <span className="text-[10px] font-semibold">{t('multitrack.addImage')}</span> : null}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-auto p-0" align="end">
          <MediaSelector
            value=""
            mediaType="image"
            onChange={handleActiveTaskSelectedMedia}
          />
        </PopoverContent>
      </Popover>
    )
  }

  function renderActiveTaskImageThumbnail(
    image: MultiTrackTaskImage,
    url: string,
    selected: boolean,
    className = 'aspect-square w-full',
    showControls = true,
    layout: 'thumbnail' | 'flow' = 'thumbnail',
  ) {
    if (!activeTaskImages) return null
    const imageName = image.file_name ?? image.file_path ?? image.local_path ?? image.url ?? image.id
    return (
      <div
        key={image.id}
        data-testid={`task-preview-image-${image.id}`}
        className={`group relative shrink-0 cursor-pointer overflow-hidden rounded-sm border bg-black ${
          selected ? 'border-primary' : 'border-border'
        } ${className}`}
        draggable
        onDragStart={() => {
          draggedTaskImageIdRef.current = image.id
        }}
        onDragEnd={() => {
          draggedTaskImageIdRef.current = null
        }}
        onDragOver={(event) => event.preventDefault()}
        onDrop={() => handleActiveTaskImageReorderDrop(image.id)}
      >
        <Button
          type="button"
          variant="ghost"
          className={layout === 'flow'
            ? 'h-full w-auto max-w-full rounded-none p-0 hover:bg-transparent'
            : 'h-full w-full rounded-none p-0 hover:bg-transparent'}
          aria-label={imageName}
          onClick={() => setActiveTaskPreviewImageId(image.id)}
        >
          {renderActiveTaskImagePreview(
            image,
            url,
            imageName,
            layout === 'flow'
              ? 'h-full w-auto max-w-full object-contain'
              : 'h-full w-full object-contain',
          )}
        </Button>
        {showControls ? renderActiveTaskImageControls(image, imageName, layout === 'flow' ? 'flow' : 'split') : null}
      </div>
    )
  }

  function renderSubtitleOverlay(activeSegment: MultiTrackSegment) {
    const selected = selectedSegment?.segment.id === activeSegment.id && selectedSegment.trackType === 'subtitle'
    const style = clampSubtitleStyle({
      ...DEFAULT_SUBTITLE_STYLE,
      ...activeSegment.content.subtitle_style,
    })
    const editing = editingSubtitle?.segmentId === activeSegment.id ? editingSubtitle : null
    const renderedBackgroundColor = subtitleBackgroundColor(style.background_color, style.background_opacity)
    const hasSubtitleBackground = !isTransparentSubtitleBackground(renderedBackgroundColor)
    const updateStyle = (patch: Partial<MultiTrackSubtitleStyle>) => {
      if (!selected) return
      onSelectedSegmentContentChange({
        subtitle_style: clampSubtitleStyle({
          ...style,
          ...patch,
        }),
      })
    }

    function handleOverlayMouseDown(event: React.MouseEvent<HTMLDivElement>) {
      if (editing) {
        event.stopPropagation()
        return
      }
      event.preventDefault()
      event.stopPropagation()
      if (!selected) {
        onSelectSegment?.(activeSegment.id)
        return
      }

      const previewRect = event.currentTarget.closest('[data-multitrack-preview-area]')?.getBoundingClientRect()
      if (!previewRect) return
      const previewWidth = previewRect.width
      const previewHeight = previewRect.height
      const startX = event.clientX
      const startY = event.clientY
      const startStyle = style

      function handleMove(moveEvent: MouseEvent) {
        updateStyle({
          x: startStyle.x + ((moveEvent.clientX - startX) / previewWidth),
          y: startStyle.y + ((moveEvent.clientY - startY) / previewHeight),
        })
      }

      function handleUp() {
        globalThis.removeEventListener('mousemove', handleMove)
        globalThis.removeEventListener('mouseup', handleUp)
      }

      globalThis.addEventListener('mousemove', handleMove)
      globalThis.addEventListener('mouseup', handleUp)
    }

    function handleSubtitleDoubleClick(event: React.MouseEvent<HTMLDivElement>) {
      event.preventDefault()
      event.stopPropagation()
      onSelectSegment?.(activeSegment.id)
      setEditingSubtitle({
        segmentId: activeSegment.id,
        text: activeSegment.content.text ?? '',
      })
    }

    function handleResizeMouseDown(edge: 'left' | 'right', event: React.MouseEvent<HTMLSpanElement>) {
      if (!selected) return
      event.preventDefault()
      event.stopPropagation()
      const previewRect = event.currentTarget.closest('[data-multitrack-preview-area]')?.getBoundingClientRect()
      if (!previewRect) return
      const previewWidth = previewRect.width
      const startX = event.clientX
      const startStyle = style

      function handleMove(moveEvent: MouseEvent) {
        const delta = (moveEvent.clientX - startX) / previewWidth
        if (edge === 'left') {
          const nextX = startStyle.x + delta
          updateStyle({
            x: nextX,
            width: startStyle.width - delta,
          })
          return
        }
        updateStyle({ width: startStyle.width + delta })
      }

      function handleUp() {
        globalThis.removeEventListener('mousemove', handleMove)
        globalThis.removeEventListener('mouseup', handleUp)
      }

      globalThis.addEventListener('mousemove', handleMove)
      globalThis.addEventListener('mouseup', handleUp)
    }

    return (
      <div
        data-testid="subtitle-preview-overlay"
        className={`absolute z-10 text-center leading-snug ${selected ? 'ring-1 ring-primary cursor-move' : 'cursor-pointer'}`}
        style={{
          left: `${(style.x + style.width / 2) * 100}%`,
          top: `${style.y * 100}%`,
          width: 'max-content',
          maxWidth: `${style.width * 100}%`,
          transform: 'translateX(-50%)',
          color: style.color,
          fontSize: style.font_size,
          textShadow: subtitleTextShadow(style.outline_color || '#000000'),
        }}
        onMouseDown={handleOverlayMouseDown}
        onDoubleClick={handleSubtitleDoubleClick}
      >
        {editing ? (
          <Input
            data-testid="subtitle-preview-editor"
            autoFocus
            aria-label={t('multitrack.editSubtitle')}
            value={editing.text}
            className={`inline-block h-auto max-w-full rounded-sm border-primary px-3 py-1.5 text-center leading-snug text-inherit shadow-none focus-visible:ring-1 ${hasSubtitleBackground ? 'shadow-sm' : ''}`}
            style={{
              width: `${Math.max(editing.text.length + 2, 5)}em`,
              backgroundColor: renderedBackgroundColor,
              color: style.color,
              fontSize: style.font_size,
              textShadow: subtitleTextShadow(style.outline_color || '#000000'),
            }}
            onMouseDown={(event) => event.stopPropagation()}
            onClick={(event) => event.stopPropagation()}
            onDoubleClick={(event) => event.stopPropagation()}
            onChange={(event) => setEditingSubtitle({ ...editing, text: event.currentTarget.value })}
            onBlur={commitSubtitleEditing}
            onKeyDown={(event) => {
              if (event.key === 'Enter') {
                event.preventDefault()
                commitSubtitleEditing()
              }
              if (event.key === 'Escape') {
                event.preventDefault()
                setEditingSubtitle(null)
              }
            }}
          />
        ) : (
          <span
            data-testid="subtitle-preview-text"
            className={`inline-block max-w-full whitespace-pre-wrap break-words rounded-sm px-2 py-1 ${hasSubtitleBackground ? 'shadow-sm' : ''}`}
            style={{
              backgroundColor: renderedBackgroundColor,
            }}
          >
            {activeSegment.content.text}
          </span>
        )}
        {selected ? (
          <>
            <span
              className="absolute left-0 top-0 h-full w-1 cursor-ew-resize bg-primary"
              onMouseDown={(event) => handleResizeMouseDown('left', event)}
            />
            <span
              className="absolute right-0 top-0 h-full w-1 cursor-ew-resize bg-primary"
              onMouseDown={(event) => handleResizeMouseDown('right', event)}
            />
          </>
        ) : null}
      </div>
    )
  }

  function renderActiveTaskImageControls(
    image: MultiTrackTaskImage,
    imageName: string,
    layout: 'split' | 'corner' | 'flow',
  ) {
    if (!activeTaskImages) return null
    const wrapperClassName = layout === 'corner'
      ? 'absolute right-2 top-2 flex gap-1'
      : layout === 'flow'
        ? 'absolute right-1 top-1 flex gap-0.5'
      : `absolute left-0 top-0 ${image.panorama_view ? '' : ' opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100'}`
    const deleteWrapperClassName = layout === 'corner'
      ? ''
      : 'absolute right-0 top-0 opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100'
    const controlClassName = layout === 'corner' ? 'h-7 w-7' : 'h-5 w-5'
    const iconClassName = layout === 'corner' ? '[&_svg]:!size-4' : '[&_svg]:!size-3'

    return (
      <>
        <div className={wrapperClassName}>
          <Button
            type="button"
            size="icon"
            variant="ghost"
            className={`${controlClassName} cursor-pointer rounded-none bg-background/70 text-foreground hover:bg-background/90 ${iconClassName}`}
            aria-label={t('multitrack.previewImage')}
            onClick={(event) => {
              event.stopPropagation()
              setExpandedTaskImageTarget({
                imageId: image.id,
                segmentId: activeTaskImages.segmentId,
                source: 'active',
              })
            }}
          >
            <Eye />
          </Button>
          {layout !== 'split' ? (
            <Button
              type="button"
              size="icon"
              variant="ghost"
              className={`${controlClassName} cursor-pointer rounded-none bg-background/70 text-destructive hover:bg-background/90 hover:text-destructive ${iconClassName}`}
              aria-label={`${t('multitrack.deleteImage')} ${imageName}`}
              onClick={(event) => {
                event.stopPropagation()
                handleActiveTaskImageDelete(image.id)
              }}
            >
              <X />
            </Button>
          ) : null}
        </div>
        {layout === 'split' ? (
          <div className={deleteWrapperClassName}>
            <Button
              type="button"
              size="icon"
              variant="ghost"
              className={`${controlClassName} cursor-pointer rounded-none bg-background/70 text-destructive hover:bg-background/90 hover:text-destructive ${iconClassName}`}
              aria-label={`${t('multitrack.deleteImage')} ${imageName}`}
              onClick={(event) => {
                event.stopPropagation()
                handleActiveTaskImageDelete(image.id)
              }}
            >
              <X />
            </Button>
          </div>
        ) : null}
      </>
    )
  }

  function renderActiveTaskPromptBar() {
    if (!activeTaskPrompt) return null
    const editing = editingTaskPrompt?.segmentId === activeTaskPrompt.segmentId ? editingTaskPrompt : null
    const promptText = editing?.text ?? activeTaskPrompt.prompt
    const hasPromptText = promptText.trim().length > 0
    const layoutLabel = previewLayoutMode === 'image-large'
      ? t('multitrack.previewLayoutBalanced')
      : t('multitrack.previewLayoutImageLarge')

    return (
      <div
        data-testid="task-prompt-overlay"
        className="flex w-full items-center bg-black/70 px-2 text-primary-foreground"
      >
        { !usesTaskImageOnlyPreview ? (<TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                size="icon"
                variant="ghost"
                className="h-6 w-6 shrink-0 cursor-pointer rounded-none text-primary [&_svg]:!size-3"
                aria-label={layoutLabel}
                onClick={(event) => {
                  event.stopPropagation()
                  setPreviewLayoutMode((mode) => mode === 'balanced' ? 'image-large' : 'balanced')
                }}
              >
              {previewLayoutMode === 'image-large' ? <PanelRight /> : <PanelLeft />}
              </Button>
            </TooltipTrigger>
            <TooltipContent side="top">{layoutLabel}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
        ) : null}
        {editing ? (
          <Textarea
            data-testid="task-prompt-editor"
            autoFocus
            aria-label={t('multitrack.userPrompt')}
            placeholder={t('multitrack.promptPlaceholder')}
            className="min-h-6 flex-1 resize-none border-none bg-transparent px-1 py-0 text-[9px] leading-4 text-primary shadow-none focus-visible:ring-0"
            value={editing.text}
            onMouseDown={(event) => event.stopPropagation()}
            onClick={(event) => event.stopPropagation()}
            onDoubleClick={(event) => event.stopPropagation()}
            onChange={(event) => handleTaskPromptInput(event.currentTarget.value)}
            onBlur={commitTaskPromptEditing}
            onKeyDown={(event) => {
              if (event.key === 'Escape') {
                event.preventDefault()
                setEditingTaskPrompt(null)
              }
              if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                event.preventDefault()
                commitTaskPromptEditing()
              }
            }}
          />
        ) : (
          <div
            data-testid="task-prompt-text"
            className={`min-w-0 flex-1 truncate px-1 text-[9px] leading-4 ${
              hasPromptText ? 'text-primary' : 'text-muted-foreground'
            }`}
            title={hasPromptText ? activeTaskPrompt.prompt : t('multitrack.promptPlaceholder')}
            onDoubleClick={(event) => {
              event.preventDefault()
              event.stopPropagation()
              setEditingTaskPrompt({
                segmentId: activeTaskPrompt.segmentId,
                text: activeTaskPrompt.prompt,
              })
            }}
          >
            {hasPromptText ? activeTaskPrompt.prompt : t('multitrack.promptPlaceholder')}
          </div>
        )}
        <span className="shrink-0 px-1 text-[9px] leading-4 text-secondary">|</span>
        <select
          data-testid="task-mode-select"
          aria-label="task_mode"
          className="h-5 shrink-0 cursor-pointer border-none bg-transparent px-1 text-[9px] leading-4 text-primary outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50"
          value={activeTaskPrompt.taskMode}
          disabled={!onTrackSegmentsContentChange}
          onMouseDown={(event) => event.stopPropagation()}
          onClick={(event) => event.stopPropagation()}
          onDoubleClick={(event) => event.stopPropagation()}
          onChange={(event) => handleActiveTaskModeChange(
            activeTaskPrompt.segmentId,
            event.currentTarget.value as MultiTrackTaskMode,
          )}
        >
          {MULTITRACK_TASK_MODES.map((taskMode) => (
            <option key={taskMode} value={taskMode}>
              {getMultiTrackTaskModeLabel(taskMode, t)}
            </option>
          ))}
        </select>
      </div>
    )
  }

  function renderEmptyPreview() {
    const trackGuides = [
      [ListTree, 'multitrack.emptyTaskTrack', 'multitrack.emptyTaskTrackPurpose'],
      [Clapperboard, 'multitrack.emptyVideoTrack', 'multitrack.emptyVideoTrackPurpose'],
      [Volume2, 'multitrack.emptyAudioTrack', 'multitrack.emptyAudioTrackPurpose'],
      [Captions, 'multitrack.emptySubtitleTrack', 'multitrack.emptySubtitleTrackPurpose'],
    ] as const

    return (
      <div
        data-testid="multitrack-empty-preview"
        className="flex h-full w-full flex-col items-center justify-center px-6 py-5 text-center"
      >
        <Clapperboard className="mb-2 h-9 w-9 text-muted-foreground" aria-hidden="true" />
        <h3 className="text-sm font-semibold text-foreground">
          {t('multitrack.emptyPreviewTitle')}
        </h3>
        <p className="mt-1 text-xs text-muted-foreground">
          {t('multitrack.emptyPreviewDescription')}
        </p>
        <dl className="mt-4 grid max-w-xl grid-cols-[auto_1fr] gap-x-3 gap-y-1.5 text-left text-[11px] leading-4">
          {trackGuides.map(([Icon, trackKey, purposeKey]) => (
            <div key={trackKey} className="contents">
              <dt className="flex items-center gap-1.5 font-medium text-foreground">
                <Icon className="h-3.5 w-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
                <span>{t(trackKey)}</span>
              </dt>
              <dd className="text-muted-foreground">{t(purposeKey)}</dd>
            </div>
          ))}
        </dl>
      </div>
    )
  }

  function renderExpandedTaskImage(image: MultiTrackTaskImage) {
    const imageUrl = mediaContentToViewUrl({
      source_type: image.source_type ?? 'input',
      file_path: image.file_path,
      local_path: image.local_path,
      url: image.url,
      slot_name: image.slot_name,
    })
    if (!imageUrl) return null
    const imageName = image.file_name ?? image.file_path ?? image.local_path ?? image.url ?? image.id

    function handleWheel(event: React.WheelEvent<HTMLDivElement>) {
      event.preventDefault()
      event.stopPropagation()
      const imageElement = expandedTaskImageRef.current
      if (imageElement) {
        const rect = imageElement.getBoundingClientRect()
        if (rect.width > 0 && rect.height > 0) {
          setExpandedTaskImageOrigin({
            x: Math.max(0, Math.min(100, (event.clientX - rect.left) / rect.width * 100)),
            y: Math.max(0, Math.min(100, (event.clientY - rect.top) / rect.height * 100)),
          })
        }
      }
      const zoomFactor = event.deltaY < 0
        ? EXPANDED_IMAGE_ZOOM_STEP
        : 1 / EXPANDED_IMAGE_ZOOM_STEP
      setExpandedTaskImageZoom((current) => Math.max(
        EXPANDED_IMAGE_MIN_ZOOM,
        Math.min(EXPANDED_IMAGE_MAX_ZOOM, current * zoomFactor),
      ))
    }

    return (
      <div
        data-testid="task-image-expanded-preview"
        data-zoom={expandedTaskImageZoom.toFixed(2)}
        className="absolute inset-0 z-30 flex items-center justify-center overflow-hidden bg-black p-3 pt-12"
        onWheel={handleWheel}
      >
        <Button
          type="button"
          variant="secondary"
          className="absolute left-2 top-2 z-20 h-8 cursor-pointer gap-1.5 px-2.5 text-xs"
          aria-label={t('multitrack.backToPreview')}
          onClick={() => setExpandedTaskImageTarget(null)}
        >
          <ArrowLeft className="h-4 w-4" />
          <span>{t('multitrack.backToPreview')}</span>
        </Button>
        <span className="pointer-events-none absolute right-2 top-2 z-20 rounded bg-background/70 px-2 py-1 text-[10px] tabular-nums text-foreground">
          {Math.round(expandedTaskImageZoom * 100)}%
        </span>
        <img
          ref={expandedTaskImageRef}
          src={imageUrl}
          alt={imageName}
          className="max-h-full max-w-full object-contain transition-transform duration-100 ease-out"
          style={{
            transform: `scale(${expandedTaskImageZoom})`,
            transformOrigin: `${expandedTaskImageOrigin.x}% ${expandedTaskImageOrigin.y}%`,
          }}
          draggable={false}
        />
      </div>
    )
  }

  if (selectedSegment?.trackType === 'task') {
    return (
      <div
        data-multitrack-preview-area
        className="relative flex min-h-24 flex-1 overflow-hidden rounded-sm bg-background"
        onClick={(event) => event.stopPropagation()}
      >
        {expandedTaskImage ? (
          renderExpandedTaskImage(expandedTaskImage)
        ) : (
          <TaskSegmentEditor
            segment={selectedSegment.segment}
            trackSegments={taskSegments}
            videoSegments={videoSegments}
            frameRate={data.frame_rate}
            totalFrames={data.total_length}
            onContentChange={onSelectedSegmentContentChange}
            onTrackSegmentsContentChange={onTrackSegmentsContentChange}
            onTrackSegmentsChange={onTaskTrackSegmentsChange}
            onDurationChange={onSelectedSegmentDurationChange}
            onOpenImagePreview={(imageId) => setExpandedTaskImageTarget({
              imageId,
              segmentId: selectedSegment.segment.id,
              source: 'selected',
            })}
          />
        )}
      </div>
    )
  }

  return (
    <div
      data-multitrack-preview-area
      className="relative flex min-h-24 flex-1 items-center justify-center overflow-hidden rounded-sm bg-black text-xs text-muted-foreground"
      onClick={(event) => event.stopPropagation()}
    >
      {expandedTaskImage && expandedTaskImageTarget?.source === 'active'
        ? renderExpandedTaskImage(expandedTaskImage)
        : null}
      <div className="flex h-full min-h-24 w-full flex-col items-center justify-center gap-0.5">
        {!hasSegments ? renderEmptyPreview() : null}
        <div className={!hasSegments ? 'hidden' : previewMediaGroupClassName}>
          {usesTaskImageOnlyPreview && activeTaskImages ? (
            <div
              data-testid="task-preview-images"
              data-layout="flow"
              className="flex h-full min-h-0 w-full flex-wrap content-center items-center justify-center gap-2 overflow-y-auto bg-black p-2"
            >
              {activeTaskImages.images.map(({ image, url }) => (
                renderActiveTaskImageThumbnail(
                  image,
                  url,
                  image.id === activeTaskPreviewImage?.image.id,
                  'h-32 w-fit max-w-full',
                  true,
                  'flow',
                )
              ))}
              {renderActiveTaskImageAddSlot('h-32 w-32 aspect-square', true)}
            </div>
          ) : activeTaskImages && (
            <div
              data-testid="task-preview-images"
              className={`flex max-h-full shrink-0 flex-col gap-2 overflow-hidden transition-all duration-300 ease-in-out ${activeTaskImagePanelClassName}`}
            >
              <div className="shrink-0 text-center text-[10px] font-medium text-foreground pt-2">
                {t('multitrack.previewTaskLabel', { n: activeTaskImages.index })}
              </div>
              <div className="flex min-h-0 flex-col gap-2 overflow-y-auto pb-2">
                {activeTaskImages.images.map(({ image, url }) => {
                  return renderActiveTaskImageThumbnail(image, url, image.id === activeTaskPreviewImage?.image.id)
                })}
                {renderActiveTaskImageAddSlot('aspect-square h-auto w-full')}
              </div>
            </div>
          )}
          {!usesTaskImageOnlyPreview ? (
            <div className={selectedSubtitleStyle
              ? 'min-w-0 h-full flex-1 flex-col items-center justify-center flex'
              : `min-w-0 transition-all duration-300 ease-in-out ${activeTaskImages ? activeTaskVideoPanelClassName : 'h-full flex-[1_1_100%]'}`
            }>
              <VideoPreview
                activeVideo={activeVideo}
                resolution={resolution}
                isPlaying={isPlaying}
                playbackNonce={playbackNonce}
                muted
                volume={0}
              >
                {activeSubtitleSegments.map((segment) => (
                  <div key={segment.id}>
                    {renderSubtitleOverlay(segment)}
                  </div>
                ))}
              </VideoPreview>
            </div>
          ) : null}
          {selectedSegment?.trackType === 'subtitle' && selectedSubtitleStyle ? (
            <SubtitleSettingsPanel
              text={selectedSegment.segment.content.text ?? ''}
              style={selectedSubtitleStyle}
              onTextChange={(text) => onSelectedSegmentContentChange({ text })}
              onStyleChange={(patch) => {
                onSelectedSegmentContentChange({
                  subtitle_style: clampSubtitleStyle({
                    ...selectedSubtitleStyle,
                    ...patch,
                  }),
                })
              }}
              speechSettings={subtitleSpeechSettings}
              onSpeechSettingsChange={setSubtitleSpeechSettings}
              onGenerateSpeech={(settings) => onGenerateSubtitleSpeech?.(selectedSegment.segment, settings) ?? Promise.resolve()}
            />
          ) : null}
        </div>
        {hasSegments ? renderActiveTaskPromptBar() : null}
      </div>
      <PreviewAudioPlayback sources={activeAudioSources} isPlaying={isPlaying} playbackNonce={playbackNonce} />
      <PreviewFloatingToolbar
        globalMuted={data.muted === true}
        globalVolumeDb={data.volume_db ?? 0}
        frameRate={data.frame_rate}
        selectedMediaVolumeDb={selectedSegment?.trackType === 'video' || selectedSegment?.trackType === 'audio'
          ? selectedSegment.segment.content.volume_db ?? 0
          : null}
        selectedMediaMuted={selectedSegment?.trackType === 'video' || selectedSegment?.trackType === 'audio'
          ? selectedSegment.segment.content.muted === true
          : false}
        selectedMediaDuration={selectedMediaDuration}
        rightInset={selectedSubtitleStyle
          ? SUBTITLE_SETTINGS_PANEL_TOOLBAR_INSET
          : 12}
        onGlobalSettingsChange={onGlobalSettingsChange}
        onSelectedSegmentContentChange={onSelectedSegmentContentChange}
        onSelectedSegmentDurationChange={onSelectedSegmentDurationChange}
      />
    </div>
  )
}
