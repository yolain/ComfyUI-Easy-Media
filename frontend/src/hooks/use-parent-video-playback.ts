import { useCallback, useEffect, type RefObject } from 'react'

export interface ParentVideoPlayback {
  isPlaying: boolean
  currentTime: number
  startTime: number
  endTime: number
  muted: boolean
  onPause: () => void
  onSeek: (time: number) => void
}

/** Follow a parent clock without taking over locally started playback while it is paused. */
export function useParentVideoPlayback(
  playback: ParentVideoPlayback | undefined,
  sourceRef: RefObject<HTMLVideoElement | null>,
  outputRef: RefObject<HTMLVideoElement | null>,
  setCurrentTime: (time: number) => void,
  playVideos: () => void,
  pauseVideos: () => void,
) {
  const time = playback
    ? Math.max(playback.startTime, Math.min(playback.endTime, playback.currentTime))
    : undefined
  const playing = playback?.isPlaying

  const syncVideo = useCallback((video: HTMLVideoElement | null) => {
    if (!video || time === undefined) return
    const target = Number.isFinite(video.duration) ? Math.min(time, video.duration) : time
    try {
      if (!playing || Math.abs(video.currentTime - target) > 0.12) video.currentTime = target
      if (playing && Number.isFinite(video.duration) && time >= video.duration) {
        video.pause()
      } else if (playing && video.paused) {
        // A parent seek or loop can bring an already-ended child back into range.
        void video.play()?.catch((error: unknown) => {
          console.error('[CompareVideoWidget] failed to resume parent playback:', error)
          pauseVideos()
        })
      }
    } catch (error) {
      console.error('[CompareVideoWidget] failed to follow parent playback:', error)
    }
  }, [pauseVideos, playing, time])

  useEffect(() => {
    if (time === undefined) return
    setCurrentTime(time)
    syncVideo(sourceRef.current)
    syncVideo(outputRef.current)
  }, [outputRef, setCurrentTime, sourceRef, syncVideo, time])

  useEffect(() => {
    if (playing === undefined) return
    if (playing) playVideos()
    else pauseVideos()
  }, [pauseVideos, playVideos, playing])

  return syncVideo
}
