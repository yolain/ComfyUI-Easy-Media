import { useEffect, useMemo, useRef, useState } from 'react'
import { X } from 'lucide-react'
import { Button } from '@/components/ui/button'
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
import type { MultiTrackPanoramaView, MultiTrackSegment, MultiTrackSegmentContent, MultiTrackTaskImage, TrackData } from '@/types/multitrack'
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
  node: unknown
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

export function PreviewArea({
  data,
  currentTime,
  selectedSegment,
  isPlaying,
  node,
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
            muted
            volume={0}
          />
        ) : null}
      </div>
      <PreviewAudioPlayback sources={activeAudioSources} isPlaying={isPlaying} />
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
        onGlobalSettingsChange={onGlobalSettingsChange}
        onSelectedSegmentContentChange={onSelectedSegmentContentChange}
        onSelectedSegmentDurationChange={onSelectedSegmentDurationChange}
      />
    </div>
  )
}
