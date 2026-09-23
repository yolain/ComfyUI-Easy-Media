import { type ReactNode, useEffect, useMemo, useRef, useState } from 'react'
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
  frameRate: number
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
  frameRate,
  className,
  children,
}: Readonly<VideoPreviewProps>) {
  const t = useT()
  const videoRef = useRef<HTMLVideoElement>(null)
  const [still, setStill] = useState<{ key: string; url: string } | null>(null)
  const [failedStillKey, setFailedStillKey] = useState<string | null>(null)
  const [sourceRate, setSourceRate] = useState<{ url: string; fps: number | null } | null>(null)
  const [nativeReadyKey, setNativeReadyKey] = useState<string | null>(null)
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
  const isUrlSource = activeVideo?.segment.content.source_type === 'url'
  const activeLocalTime = activeVideo?.localTime ?? 0
  const sourceFrame = Math.round(activeLocalTime * frameRate)
  const stillKey = activeVideo && videoUrl
    ? `${videoUrl}:${frameRate}:${sourceFrame}`
    : null
  const sourceRateKnown = sourceRate?.url === videoUrl
  const useNativeFrame = sourceRateKnown && sourceRate.fps !== null
    && Math.abs(sourceRate.fps - frameRate) <= 0.01

  useEffect(() => {
    if (!activeVideo || !videoUrl || isUrlSource) return
    const controller = new AbortController()
    const content = activeVideo.segment.content
    const probedUrl = videoUrl
    async function probeFrameRate() {
      try {
        const response = await fetch('/easy-media/video/source-fps', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_type: content.source_type ?? 'input',
            file_path: content.file_path,
            local_path: content.local_path,
            url: content.url,
          }),
          signal: controller.signal,
        })
        if (!response.ok) throw new Error(`Video frame rate request failed (${response.status})`)
        const result: unknown = await response.json()
        const fps = result && typeof result === 'object' && 'fps' in result
          ? (result as { fps: unknown }).fps : null
        if (typeof fps !== 'number' || !Number.isFinite(fps) || fps <= 0) {
          throw new Error('Video frame rate response is invalid')
        }
        if (!controller.signal.aborted) setSourceRate({ url: probedUrl, fps })
      } catch (error) {
        if (!controller.signal.aborted) {
          console.error('[VideoPreview] failed to probe source frame rate:', error)
          setSourceRate({ url: probedUrl, fps: null })
        }
      }
    }
    void probeFrameRate()
    return () => controller.abort()
  }, [videoUrl, isUrlSource])

  useEffect(() => {
    if (isPlaying || isUrlSource || useNativeFrame || !sourceRateKnown || !activeVideo || !videoUrl || stillKey === null) {
      setStill(null)
      setFailedStillKey(null)
      return
    }
    const requestedKey = stillKey
    const controller = new AbortController()
    let objectUrl: string | null = null
    const content = activeVideo.segment.content
    async function loadFrame() {
      try {
        const response = await fetch('/easy-media/video/preview-frame', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_type: content.source_type ?? 'input',
            file_path: content.file_path,
            local_path: content.local_path,
            url: content.url,
            frame: sourceFrame,
            fps: frameRate,
          }),
          signal: controller.signal,
        })
        if (!response.ok) throw new Error(`Preview frame request failed (${response.status})`)
        objectUrl = URL.createObjectURL(await response.blob())
        if (!controller.signal.aborted) {
          setFailedStillKey(null)
          setStill({ key: requestedKey, url: objectUrl })
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          console.error('[VideoPreview] failed to load timeline frame:', error)
          setFailedStillKey(requestedKey)
        }
      }
    }
    const timer = setTimeout(() => { void loadFrame() }, 60)
    return () => {
      clearTimeout(timer)
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [activeVideo?.segment.id, frameRate, isPlaying, isUrlSource, sourceFrame, sourceRateKnown, stillKey, useNativeFrame, videoUrl])

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
    if (
      useNativeFrame && !isPlaying && video && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
      && !video.seeking && Math.abs(video.currentTime - activeLocalTime) < 1 / frameRate
    ) {
      setNativeReadyKey(stillKey)
    }
  }, [activeLocalTime, frameRate, isPlaying, stillKey, useNativeFrame])

  function markNativeFrameReady() {
    const video = videoRef.current
    if (
      useNativeFrame && !isPlaying && video && !video.seeking
      && Math.abs(video.currentTime - activeLocalTime) < 1 / frameRate
    ) {
      setNativeReadyKey(stillKey)
    }
  }

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
          onSeeked={markNativeFrameReady}
          onLoadedData={markNativeFrameReady}
          style={{
            objectFit: fit,
            visibility: isPlaying || isUrlSource || failedStillKey === stillKey
              || (useNativeFrame && nativeReadyKey === stillKey) ? 'visible' : 'hidden',
          }}
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
      {!isPlaying && still?.key === stillKey && (
        <img
          src={still.url}
          alt=""
          data-testid="multitrack-timeline-frame"
          className="absolute inset-0 h-full w-full"
          style={{ objectFit: fit }}
        />
      )}
      {children}
    </div>
  )
}
