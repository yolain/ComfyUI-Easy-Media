import { type ReactNode, useEffect, useMemo, useRef } from 'react'
import type { ActivePreviewVideoSegment, MultiTrackPreviewResolution } from '@/lib/multitrack-utils'
import { mediaContentToViewUrl } from '@/lib/media-url'

interface VideoPreviewProps {
  activeVideo: ActivePreviewVideoSegment | null
  resolution: MultiTrackPreviewResolution
  isPlaying: boolean
  playbackNonce?: number
  muted: boolean
  volume: number
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
  children,
}: Readonly<VideoPreviewProps>) {
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
      className="relative flex h-full max-h-full items-center justify-center overflow-hidden bg-black"
      style={{ aspectRatio: `${resolution.width} / ${resolution.height}` }}
    >
      {videoUrl ? (
        <video
          ref={videoRef}
          data-testid="multitrack-video-preview"
          className="h-full w-full"
          src={videoUrl}
          muted={muted}
          playsInline
          preload="auto"
          style={{ objectFit: fit }}
        />
      ) : (
        <div data-testid="multitrack-black-frame" className="h-full w-full bg-black" />
      )}
      {children}
    </div>
  )
}
