import { useEffect, useMemo, useRef } from 'react'
import { mediaContentToViewUrl } from '@/lib/media-url'
import { isSameOriginBrowserMedia, resolveBrowserAudioPreviewGain } from '@/lib/audio-utils'
import type { ActivePreviewAudioSource } from '@/lib/multitrack-utils'

interface PreviewAudioSourceProps {
  source: ActivePreviewAudioSource
  isPlaying: boolean
  playbackNonce?: number
}

function PreviewAudioSource({ source, isPlaying, playbackNonce = 0 }: Readonly<PreviewAudioSourceProps>) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const gainNodeRef = useRef<GainNode | null>(null)
  const url = useMemo(() => mediaContentToViewUrl({
    source_type: source.segment.content.source_type ?? 'input',
    file_path: source.segment.content.file_path,
    local_path: source.segment.content.local_path,
    url: source.segment.content.url,
    slot_name: source.segment.content.slot_name,
  }), [source.segment.content])
  const pageHref = globalThis.location?.href
  const gain = url ? resolveBrowserAudioPreviewGain(source.volumeDb, url, pageHref) : 0
  const canUseGainNode = url ? isSameOriginBrowserMedia(url, pageHref) : false

  useEffect(() => () => {
    const audio = audioRef.current
    if (!audio) return
    audio.pause()
    audio.src = ''
  }, [])

  useEffect(() => {
    const audio = audioRef.current
    const AudioContextConstructor = globalThis.AudioContext
    if (!audio || !url || !AudioContextConstructor) return
    if (!canUseGainNode) return

    try {
      const context = new AudioContextConstructor()
      const gainNode = context.createGain()
      context.createMediaElementSource(audio).connect(gainNode).connect(context.destination)
      audioContextRef.current = context
      gainNodeRef.current = gainNode
      audio.volume = 1
      return () => {
        gainNode.disconnect()
        gainNodeRef.current = null
        audioContextRef.current = null
        context.close().catch((error: unknown) => {
          console.error('[PreviewAudioPlayback] failed to close audio context:', error)
        })
      }
    } catch (error) {
      console.error('[PreviewAudioPlayback] failed to create audio gain graph:', error)
    }
  }, [canUseGainNode, url])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    if (gainNodeRef.current) {
      gainNodeRef.current.gain.value = gain
      return
    }
    audio.volume = Math.max(0, Math.min(1, gain))
  }, [gain])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio || isPlaying || Math.abs(audio.currentTime - source.localTime) <= 0.001) return
    try {
      audio.currentTime = source.localTime
    } catch (error) {
      console.error('[PreviewAudioPlayback] failed to seek audio:', error)
    }
  }, [isPlaying, source.localTime])

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    if (!isPlaying) {
      audio.pause()
      return
    }
    audioContextRef.current?.resume().catch((error: unknown) => {
      console.error('[PreviewAudioPlayback] failed to resume audio context:', error)
    })
    try {
      audio.currentTime = source.localTime
    } catch (error) {
      console.error('[PreviewAudioPlayback] failed to seek audio before playback:', error)
    }
    const playResult = audio.play()
    playResult?.catch((error: unknown) => {
      console.error('[PreviewAudioPlayback] failed to play audio:', error)
    })
  }, [isPlaying, source.segment.id, url, playbackNonce])

  return url ? (
    <audio
      ref={audioRef}
      data-testid={`preview-audio-${source.segment.id}`}
      src={url}
      preload="auto"
      className="hidden"
    />
  ) : null
}

interface PreviewAudioPlaybackProps {
  sources: ActivePreviewAudioSource[]
  isPlaying: boolean
  playbackNonce?: number
}

export function PreviewAudioPlayback({ sources, isPlaying, playbackNonce = 0 }: Readonly<PreviewAudioPlaybackProps>) {
  return sources.map((source) => (
    <PreviewAudioSource
      key={`${source.trackId}:${source.segment.id}`}
      source={source}
      isPlaying={isPlaying}
      playbackNonce={playbackNonce}
    />
  ))
}
