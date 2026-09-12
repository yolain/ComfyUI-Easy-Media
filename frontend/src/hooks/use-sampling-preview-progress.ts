import { useEffect, useRef, useState } from 'react'

export interface SamplingPreviewFrame {
  bitmap: ImageBitmap
  durationMs: number
}

export function useSamplingPreviewFrames(
  blob: Blob | null,
  revision: string | undefined,
  enabled = true,
) {
  const [frames, setFrames] = useState<SamplingPreviewFrame[]>([])
  const activeFrames = useRef<SamplingPreviewFrame[]>([])

  useEffect(() => {
    let disposed = false
    let committed = false
    let decoder: ImageDecoder | null = null
    const decodedFrames: SamplingPreviewFrame[] = []

    const replaceFrames = (nextFrames: SamplingPreviewFrame[]) => {
      const previousFrames = activeFrames.current
      activeFrames.current = nextFrames
      setFrames(nextFrames)
      previousFrames.forEach(({ bitmap }) => bitmap.close())
    }

    async function decodeFrames() {
      if (!enabled || !blob || blob.type !== 'image/webp' || typeof ImageDecoder === 'undefined') {
        if (!disposed) replaceFrames([])
        return
      }
      const supported = await ImageDecoder.isTypeSupported(blob.type)
      if (disposed) return
      if (!supported) {
        replaceFrames([])
        return
      }

      const data = await blob.arrayBuffer()
      if (disposed) return
      decoder = new ImageDecoder({ data, type: blob.type, preferAnimation: true })
      await decoder.tracks.ready
      const frameCount = decoder.tracks.selectedTrack?.frameCount ?? 0

      for (let frameIndex = 0; frameIndex < frameCount; frameIndex += 1) {
        const result = await decoder.decode({ frameIndex, completeFramesOnly: true })
        const durationMs = Math.max(1, (result.image.duration ?? 0) / 1000)
        let bitmap: ImageBitmap
        try {
          bitmap = await createImageBitmap(result.image)
        } finally {
          result.image.close()
        }
        if (disposed) {
          bitmap.close()
          return
        }
        decodedFrames.push({ bitmap, durationMs })
      }

      if (!disposed) {
        committed = true
        replaceFrames([...decodedFrames])
        decoder.close()
        decoder = null
      }
    }

    void decodeFrames().catch((error: unknown) => {
      if (!disposed) console.error('[SamplingPreviewWidget] failed to decode preview:', error)
    })

    return () => {
      disposed = true
      decoder?.close()
      if (!committed) decodedFrames.forEach(({ bitmap }) => bitmap.close())
    }
  }, [blob, enabled, revision])

  useEffect(() => () => {
    activeFrames.current.forEach(({ bitmap }) => bitmap.close())
    activeFrames.current = []
  }, [])

  return frames
}

export function useSamplingPreviewPlayback(
  revision: string | undefined,
  frameCount: number,
  fps: number | null,
  isPlaying: boolean,
) {
  const [progress, setProgress] = useState(0)
  const [frameIndex, setFrameIndex] = useState(0)
  const elapsedMs = useRef(0)

  useEffect(() => {
    elapsedMs.current = 0
    setProgress(0)
    setFrameIndex(0)
  }, [revision])

  useEffect(() => {
    const durationMs = fps && frameCount > 1 ? frameCount / fps * 1000 : 0
    if (!revision || durationMs <= 0) {
      setProgress(revision ? 100 : 0)
      setFrameIndex(0)
      return
    }
    if (!isPlaying) return

    let animationFrame = 0
    let previousTime = performance.now()
    const update = (now: number) => {
      elapsedMs.current = (elapsedMs.current + now - previousTime) % durationMs
      previousTime = now
      const nextProgress = elapsedMs.current / durationMs * 100
      setProgress(nextProgress)
      setFrameIndex(Math.min(frameCount - 1, Math.floor(nextProgress / 100 * frameCount)))
      animationFrame = requestAnimationFrame(update)
    }
    animationFrame = requestAnimationFrame(update)
    return () => cancelAnimationFrame(animationFrame)
  }, [fps, frameCount, isPlaying, revision])

  return { frameIndex, progress }
}
