import { useEffect, useMemo, useRef, useState } from 'react'
import { ChevronLeft, ChevronRight, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  collectMultiTrackPreviewResolutionInput,
  frameToSeconds,
  getActivePreviewAudioSources,
  getActivePreviewVideoSegment,
  parseMultiTrackPreviewResolution,
  segmentDuration,
  snapTimeToFrame,
  type MultiTrackVideoMetadata,
  type SelectedMultiTrackSegment,
} from '@/lib/multitrack-utils'
import { useT } from '@/lib/i18n'
import { AudioWaveform } from '@/components/widgets/timeline/AudioWaveform'
import { mediaContentToViewUrl } from '@/lib/media-url'
import { loadBrowserVideoMetadata } from '@/lib/video-utils'
import { DEFAULT_SUBTITLE_STYLE } from '@/lib/subtitle-recognition'
import type { MultiTrackPanoramaView, MultiTrackSegment, MultiTrackSegmentContent, MultiTrackSubtitleStyle, MultiTrackTaskImage, TrackData } from '@/types/multitrack'
import { PreviewFloatingToolbar } from './PreviewFloatingToolbar'
import { PreviewAudioPlayback } from './PreviewAudioPlayback'
import { PanoramaViewerOverlay } from '@/components/widgets/panorama/PanoramaViewerOverlay'
import { PanoramaIcon } from '@/components/widgets/panorama/PanoramaIcon'
import { PanoramaImagePreview } from '@/components/widgets/panorama/PanoramaImagePreview'
import { TaskSegmentEditor } from './TaskSegmentEditor'
import { VideoPreview } from './VideoPreview'

const RESOLUTION_POLL_INTERVAL_MS = 250

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
}

interface ActivePanoramaTarget {
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

function updateTaskImagePanoramaView(
  images: MultiTrackTaskImage[],
  imageId: string,
  panoramaView: MultiTrackPanoramaView | undefined,
): MultiTrackTaskImage[] {
  return images.map((image) => {
    if (image.id !== imageId) return image
    if (panoramaView) return { ...image, panorama_view: panoramaView }
    const restored = { ...image }
    delete restored.panorama_view
    return restored
  })
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
    const allImages = segment.content.images ?? []
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
    return images.length > 0 ? { index, segmentId: segment.id, allImages, images } : null
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

    const prompt = (segment.content.user_prompt ?? segment.content.text ?? '').trim()
    if (prompt.length > 0) return { segmentId: segment.id, prompt }
  }

  return null
}

function getActiveSubtitleSegments(data: TrackData, currentTime: number): MultiTrackSegment[] {
  const currentFrame = snapTimeToFrame(currentTime, data.frame_rate)
  return data.tracks.flatMap((track) => {
    if (track.type !== 'subtitle') return []
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
    x,
    y: Math.max(0, Math.min(0.95, style.y)),
    width,
  }
}

function isTransparentSubtitleBackground(backgroundColor: string): boolean {
  const value = backgroundColor.trim().toLowerCase()
  return value === 'transparent' || value === 'rgba(0, 0, 0, 0)' || value === 'rgba(0,0,0,0)'
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
}: Readonly<PreviewAreaProps>) {
  const t = useT()
  const draggedTaskImageIdRef = useRef<string | null>(null)
  const [activePanoramaTarget, setActivePanoramaTarget] = useState<ActivePanoramaTarget | null>(null)
  const [activeTaskPreviewImageId, setActiveTaskPreviewImageId] = useState<string | null>(null)
  const [taskPromptExpanded, setTaskPromptExpanded] = useState(true)
  const [editingSubtitle, setEditingSubtitle] = useState<{ segmentId: string; text: string } | null>(null)
  const [firstVideoMetadata, setFirstVideoMetadata] = useState<MultiTrackVideoMetadata | null>(null)
  const [resolutionInput, setResolutionInput] = useState(() => collectMultiTrackPreviewResolutionInput(node))
  const videoSegments = useMemo(() => (
    data.tracks
      .filter((track) => track.type === 'video')
      .flatMap((track) => track.segments)
  ), [data.tracks])
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
  const activeVideo = selectedSegment?.trackType === 'task' || selectedSegment?.trackType === 'audio'
    ? null
    : getActivePreviewVideoSegment(data, currentTime, selectedSegment?.trackType === 'video' ? selectedSegment.segment.id : null)
  const activeAudioSources = getActivePreviewAudioSources(data, currentTime, selectedSegment)
  const activeTaskImages = selectedSegment === null ? getActiveTaskImages(data, currentTime) : null
  const activeTaskPrompt = selectedSegment === null ? getActiveTaskPrompt(data, currentTime) : null
  const activeSubtitleSegments = useMemo(() => {
    if (selectedSegment?.trackType === 'task' || selectedSegment?.trackType === 'audio') return []
    const segments = getActiveSubtitleSegments(data, currentTime)
    if (selectedSegment?.trackType !== 'subtitle') return segments
    const selectedSubtitle = selectedSegment.segment
    const selectedIndex = segments.findIndex((segment) => segment.id === selectedSubtitle.id)
    if (selectedIndex < 0) return [...segments, selectedSubtitle]
    return segments.map((segment, index) => index === selectedIndex ? selectedSubtitle : segment)
  }, [currentTime, data, selectedSegment])
  const resolution = parseMultiTrackPreviewResolution(resolutionInput, firstVideoMetadata)
  const selectedMediaDuration = selectedSegment?.trackType === 'video' || selectedSegment?.trackType === 'audio'
    ? frameToSeconds(segmentDuration(selectedSegment.segment), data.frame_rate)
    : null
  const selectedAudio = selectedSegment?.trackType === 'audio' ? selectedSegment.segment : null
  const selectedTaskImages = selectedSegment?.trackType === 'task'
    ? selectedSegment.segment.content.images ?? []
    : []
  const activePanoramaImage = activePanoramaTarget?.source === 'selected'
    ? selectedTaskImages.find((image) => image.id === activePanoramaTarget.imageId) ?? null
    : activePanoramaTarget?.source === 'active'
      ? activeTaskImages?.allImages.find((image) => image.id === activePanoramaTarget.imageId) ?? null
      : null
  const activeTaskPreviewImage = activeTaskImages?.images.find(({ image }) => image.id === activeTaskPreviewImageId)
    ?? activeTaskImages?.images[0]
    ?? null
  const usesTaskImageOnlyPreview = Boolean(activeTaskImages && !selectedAudio && !activeVideo)

  useEffect(() => {
    setActivePanoramaTarget(null)
  }, [selectedSegment?.segment.id])

  useEffect(() => {
    setTaskPromptExpanded(true)
  }, [activeTaskPrompt?.segmentId])

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
    if (activePanoramaTarget?.source === 'active' && activePanoramaTarget.segmentId !== activeTaskImages?.segmentId) {
      setActivePanoramaTarget(null)
    }
  }, [activePanoramaTarget, activeTaskImages?.segmentId])

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

  function handleActiveTaskImageDrop(targetImageId: string) {
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

  function renderActiveTaskImageThumbnail(
    image: MultiTrackTaskImage,
    url: string,
    selected: boolean,
    className = 'aspect-square w-full',
    showControls = true,
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
        onDrop={() => handleActiveTaskImageDrop(image.id)}
      >
        <Button
          type="button"
          variant="ghost"
          className="h-full w-full rounded-none p-0 hover:bg-transparent"
          aria-label={imageName}
          onClick={() => setActiveTaskPreviewImageId(image.id)}
        >
          {renderActiveTaskImagePreview(image, url, imageName, 'h-full w-full object-contain')}
        </Button>
        {showControls ? renderActiveTaskImageControls(image, imageName, 'split') : null}
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
            className={`inline-block h-auto max-w-full rounded-sm border-primary px-3 py-1.5 text-center leading-snug text-inherit shadow-none focus-visible:ring-1 ${isTransparentSubtitleBackground(style.background_color) ? '' : 'shadow-sm'}`}
            style={{
              width: `${Math.max(editing.text.length + 2, 5)}em`,
              backgroundColor: style.background_color,
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
            className={`inline-block max-w-full whitespace-pre-wrap break-words rounded-sm px-2 py-1 ${isTransparentSubtitleBackground(style.background_color) ? '' : 'shadow-sm'}`}
            style={{
              backgroundColor: style.background_color,
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
    layout: 'split' | 'corner',
  ) {
    if (!activeTaskImages) return null
    const panoramaClassName = image.panorama_view ? 'text-highlight' : 'text-foreground'
    const wrapperClassName = layout === 'corner'
      ? 'absolute right-2 top-2 flex gap-1'
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
            className={`${controlClassName} cursor-pointer rounded-none bg-background/70 hover:bg-background/90 ${iconClassName} ${panoramaClassName}`}
            aria-label={t('panorama.preview')}
            onClick={(event) => {
              event.stopPropagation()
              setActivePanoramaTarget({
                imageId: image.id,
                segmentId: activeTaskImages.segmentId,
                source: 'active',
              })
            }}
          >
            <PanoramaIcon />
          </Button>
          {layout === 'corner' ? (
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

  if (selectedSegment?.trackType === 'task') {
    return (
      <div
        data-multitrack-preview-area
        className="relative flex min-h-24 flex-1 overflow-hidden rounded-sm bg-background"
        onClick={(event) => event.stopPropagation()}
      >
        {activePanoramaImage ? (
          <PanoramaViewerOverlay
            key={`${selectedSegment.segment.id}:${activePanoramaImage.id}`}
            imageUrl={mediaContentToViewUrl({
              source_type: activePanoramaImage.source_type ?? 'input',
              file_path: activePanoramaImage.file_path,
              local_path: activePanoramaImage.local_path,
              url: activePanoramaImage.url,
              slot_name: activePanoramaImage.slot_name,
            }) ?? ''}
            savedView={activePanoramaImage.panorama_view}
            onPanoramaViewChange={(panoramaView) => {
              onSelectedSegmentContentChange({
                images: updateTaskImagePanoramaView(
                  selectedTaskImages,
                  activePanoramaImage.id,
                  panoramaView,
                ),
              })
            }}
            onExit={() => setActivePanoramaTarget(null)}
          />
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
            onOpenPanorama={(imageId) => setActivePanoramaTarget({
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
      {activePanoramaImage && activePanoramaTarget?.source === 'active' ? (
        <PanoramaViewerOverlay
          key={`${activePanoramaTarget.segmentId}:${activePanoramaImage.id}`}
          imageUrl={mediaContentToViewUrl({
            source_type: activePanoramaImage.source_type ?? 'input',
            file_path: activePanoramaImage.file_path,
            local_path: activePanoramaImage.local_path,
            url: activePanoramaImage.url,
            slot_name: activePanoramaImage.slot_name,
          }) ?? ''}
          savedView={activePanoramaImage.panorama_view}
          onPanoramaViewChange={(panoramaView) => {
            if (!activeTaskImages) return
            onTrackSegmentsContentChange?.([{
              segmentId: activePanoramaTarget.segmentId,
              patch: {
                images: updateTaskImagePanoramaView(
                  activeTaskImages.allImages,
                  activePanoramaImage.id,
                  panoramaView,
                ),
              },
            }])
          }}
          onExit={() => setActivePanoramaTarget(null)}
        />
      ) : null}
      <div className="flex h-full min-h-24 w-full flex-col items-center justify-center gap-0.5">
        <div className="flex h-full min-h-24 w-full items-center justify-center gap-3">
          {selectedAudio ? (
            <div data-testid="selected-audio-waveform" className="h-20 max-h-full w-full overflow-hidden px-2">
              <AudioWaveform
                content={{
                  source_type: selectedAudio.content.source_type ?? 'input',
                  file_path: selectedAudio.content.file_path,
                  local_path: selectedAudio.content.local_path,
                  url: selectedAudio.content.url,
                  slot_name: selectedAudio.content.slot_name,
                }}
                className="h-full w-full"
              />
            </div>
          ) : usesTaskImageOnlyPreview && activeTaskImages && activeTaskPreviewImage ? (
            <div
              data-testid="task-preview-images"
              className="flex h-full min-h-0 w-full flex-col bg-black"
            >
              <div className="relative flex min-h-0 flex-[9] items-center justify-center overflow-hidden p-2">
                {(() => {
                  const imageName = activeTaskPreviewImage.image.file_name
                    ?? activeTaskPreviewImage.image.file_path
                    ?? activeTaskPreviewImage.image.local_path
                    ?? activeTaskPreviewImage.image.url
                    ?? activeTaskPreviewImage.image.id
                  return (
                    <>
                      {renderActiveTaskImagePreview(
                        activeTaskPreviewImage.image,
                        activeTaskPreviewImage.url,
                        imageName,
                        'max-h-full h-full max-w-full w-auto object-contain',
                      )}
                      {renderActiveTaskImageControls(activeTaskPreviewImage.image, imageName, 'corner')}
                    </>
                  )
                })()}
              </div>
              <div className="flex min-h-0 flex-[1] items-center justify-center gap-2 overflow-x-auto p-2">
                {activeTaskImages.images.map(({ image, url }) => (
                  renderActiveTaskImageThumbnail(
                    image,
                    url,
                    image.id === activeTaskPreviewImage.image.id,
                    'h-full aspect-square',
                    false,
                  )
                ))}
              </div>
            </div>
          ) : activeTaskImages && (
            <div
              data-testid="task-preview-images"
              className="flex max-h-full w-20 shrink-0 flex-col gap-2 overflow-hidden"
            >
              <div className="shrink-0 text-center text-[10px] font-medium text-foreground pt-2">
                {t('multitrack.previewTaskLabel', { n: activeTaskImages.index })}
              </div>
              <div className="flex min-h-0 flex-col gap-2 overflow-y-auto pb-2">
                {activeTaskImages.images.map(({ image, url }) => {
                  return renderActiveTaskImageThumbnail(image, url, image.id === activeTaskPreviewImage?.image.id)
                })}
              </div>
            </div>
          )}
          {!selectedAudio && !usesTaskImageOnlyPreview ? (
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
          ) : null}
        </div>
        {activeTaskPrompt ? (
          <div
            data-testid="task-prompt-overlay"
            className={`flex w-full items-center bg-black/70 text-primary-foreground ${
              taskPromptExpanded ? 'px-2' : 'w-8 justify-center'
            }`}
          >
            <Button
              type="button"
              size="icon"
              variant="ghost"
              className="h-6 w-6 shrink-0 rounded-none text-primary [&_svg]:!size-3 cursor-pointer"
              aria-label={taskPromptExpanded ? t('multitrack.hideTaskPrompt') : t('multitrack.showTaskPrompt')}
              onClick={(event) => {
                event.stopPropagation()
                setTaskPromptExpanded((expanded) => !expanded)
              }}
            >
              {taskPromptExpanded ? <ChevronLeft /> : <ChevronRight />}
            </Button>
            {taskPromptExpanded ? (
              <div
                data-testid="task-prompt-text"
                className="min-w-0 flex-1 truncate text-primary px-1 text-[9px] leading-4"
                title={activeTaskPrompt.prompt}
              >
                {activeTaskPrompt.prompt}
              </div>
            ) : null}
          </div>
        ) : null}
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
        selectedSubtitleStyle={selectedSegment?.trackType === 'subtitle'
          ? {
              ...DEFAULT_SUBTITLE_STYLE,
              ...selectedSegment.segment.content.subtitle_style,
            }
          : null}
        onGlobalSettingsChange={onGlobalSettingsChange}
        onSelectedSegmentContentChange={onSelectedSegmentContentChange}
        onSelectedSegmentDurationChange={onSelectedSegmentDurationChange}
      />
    </div>
  )
}
