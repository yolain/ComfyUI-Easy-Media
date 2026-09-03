import { type ReactNode, useEffect, useMemo, useRef } from 'react'
import type { ActivePreviewVideoSegment, MultiTrackPreviewResolution } from '@/lib/multitrack-utils'
import { useT } from '@/lib/i18n'
import { mediaContentToViewUrl, mediaSlotNumber } from '@/lib/media-url'
import { cn } from '@/lib/utils'

interface VideoPreviewProps {
  activeVideo: ActivePreviewVideoSegment | null
  resolution: MultiTrackPreviewResolution
  isPlaying: boolean
  playbackNonce?: number
  muted: boolean
  volume: number
  className?: string
  children?: ReactNode
}

function objectFitForResizeMethod(method: MultiTrackPreviewResolution['resizeMethod']): React.CSSProperties['objectFit'] {
  if (method === 'stretch') return 'fill'
  if (method === 'crop') return 'cover'
  return 'contain'
}

function seekVideo(video: HTMLVideoElement, targetTime: number) {
  if (!Number.isFinite(targetTime) || Math.abs(video.currentTime - targetTime) <= 0.001) return
  try {
    video.currentTime = targetTime
  } catch (error) {
    console.error('[VideoPreview] failed to seek preview video:', error)
  }
}

export function VideoPreview({
  activeVideo,
  resolution,
  isPlaying,
  playbackNonce = 0,
  muted,
  volume,
  className,
  children,
}: Readonly<VideoPreviewProps>) {
  const t = useT()
  const videoRef = useRef<HTMLVideoElement>(null)
  const videoUrl = useMemo(() => {
    if (!activeVideo) return null
    return mediaContentToViewUrl({
      source_type: activeVideo.segment.content.source_type ?? 'input',
      file_path: activeVideo.segment.content.file_path,
      local_path: activeVideo.segment.content.local_path,
      url: activeVideo.segment.content.url,
      slot_name: activeVideo.segment.content.slot_name,
    })
  }, [activeVideo])
  const safeVolume = Math.max(0, Math.min(volume, 1))
  const fit = objectFitForResizeMethod(resolution.resizeMethod)
  const activeSegmentId = activeVideo?.segment.id ?? null
  const activeLocalTime = activeVideo?.localTime ?? 0

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    video.muted = muted
    video.volume = safeVolume
  }, [muted, safeVolume])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !activeVideo || isPlaying) return
    seekVideo(video, activeVideo.localTime)
  }, [activeVideo, activeLocalTime, isPlaying])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !activeVideo) return
    if (isPlaying) {
      seekVideo(video, activeVideo.localTime)
      const playResult = video.play()
      if (playResult) {
        playResult.catch((error: unknown) => {
          console.error('[VideoPreview] failed to play preview video:', error)
        })
      }
      return
    }
    video.pause()
  }, [isPlaying, videoUrl, activeSegmentId, playbackNonce])

  return (
    <div
      data-testid="multitrack-video-stage"
      className={cn(
        'relative flex h-full min-w-0 max-h-full max-w-full items-center justify-center overflow-hidden bg-black',
        className,
      )}
      style={{ aspectRatio: `${resolution.width} / ${resolution.height}` }}
    >
      {videoUrl ? (
        <video
          ref={videoRef}
          data-testid="multitrack-video-preview"
          className="absolute inset-0 h-full min-h-0 w-full min-w-0 max-h-full max-w-full"
          src={videoUrl}
          muted={muted}
          playsInline
          preload="auto"
          style={{ objectFit: fit }}
        />
      ) : activeVideo?.segment.content.source_type === 'slot' ? (
        <div
          data-testid="multitrack-video-slot-placeholder"
          className="flex h-full w-full items-center justify-center bg-muted/30 px-3 text-center text-xs font-medium text-muted-foreground"
        >
          {t('mediaSelector.slotVideo', {
            n: mediaSlotNumber(activeVideo.segment.content.slot_name),
          })}
        </div>
      ) : (
        <div data-testid="multitrack-black-frame" className="h-full w-full bg-black" />
      )}
      {children}
    </div>
  )
}
