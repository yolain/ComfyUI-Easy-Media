import { useEffect, useRef, useCallback } from 'react'
import { mediaContentToViewUrl, type ViewableMediaContent } from '@/lib/media-url'

interface AudioWaveformProps {
  content: ViewableMediaContent
  /** Fraction [0,1] of the full audio where the visible window starts */
  startRatio?: number
  /** Fraction [0,1] of the full audio where the visible window ends */
  endRatio?: number
  className?: string
}

const waveformPeaksCache = new Map<string, Promise<Float32Array> | Float32Array>()

function drawWaveform(canvas: HTMLCanvasElement, peaks: Float32Array, startRatio: number, endRatio: number) {
  const dpr = window.devicePixelRatio || 1
  const rect = canvas.getBoundingClientRect()
  if (rect.width === 0 || rect.height === 0) return

  canvas.width = Math.round(rect.width * dpr)
  canvas.height = Math.round(rect.height * dpr)

  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.scale(dpr, dpr)
  const w = rect.width
  const h = rect.height
  const mid = h / 2

  // Only render the slice of peaks that falls within [startRatio, endRatio]
  const sliceStart = Math.floor(startRatio * peaks.length)
  const sliceEnd = Math.ceil(endRatio * peaks.length)
  const sliceLength = Math.max(sliceEnd - sliceStart, 1)
  const step = sliceLength / w

  ctx.clearRect(0, 0, w, h)
  ctx.fillStyle = '#1D7456'

  for (let x = 0; x < w; x++) {
    const idx = Math.min(sliceStart + Math.floor(x * step), sliceEnd - 1)
    const amp = peaks[idx] * mid * 0.9
    ctx.fillRect(x, mid - amp, 1, Math.max(amp * 2, 1))
  }
}

async function loadWaveformPeaks(audioUrl: string): Promise<Float32Array> {
  const cached = waveformPeaksCache.get(audioUrl)
  if (cached) return cached

  const promise = (async () => {
    const response = await fetch(audioUrl)
    if (!response.ok) throw new Error(`Unable to fetch waveform media: ${audioUrl}`)

    const arrayBuffer = await response.arrayBuffer()
    const audioCtx = new AudioContext()
    try {
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
      const channelData = audioBuffer.getChannelData(0)
      const numPeaks = 2000
      const step = Math.ceil(channelData.length / numPeaks)
      const peaks = new Float32Array(numPeaks)
      for (let i = 0; i < numPeaks; i++) {
        let max = 0
        const start = i * step
        for (let j = start; j < start + step && j < channelData.length; j++) {
          const v = Math.abs(channelData[j])
          if (v > max) max = v
        }
        peaks[i] = max
      }
      waveformPeaksCache.set(audioUrl, peaks)
      return peaks
    } finally {
      await audioCtx.close()
    }
  })()

  waveformPeaksCache.set(audioUrl, promise)
  try {
    return await promise
  } catch (error) {
    waveformPeaksCache.delete(audioUrl)
    throw error
  }
}

export function AudioWaveform({ content, startRatio = 0, endRatio = 1, className }: Readonly<AudioWaveformProps>) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const peaksRef = useRef<Float32Array | null>(null)
  const audioUrl = mediaContentToViewUrl(content)
  const startRatioRef = useRef(startRatio)
  const endRatioRef = useRef(endRatio)
  startRatioRef.current = startRatio
  endRatioRef.current = endRatio

  const redraw = useCallback(() => {
    const canvas = canvasRef.current
    if (!canvas || !peaksRef.current) return
    drawWaveform(canvas, peaksRef.current, startRatioRef.current, endRatioRef.current)
  }, [])

  useEffect(() => {
    if (!audioUrl) {
      peaksRef.current = null
      return
    }
    const currentAudioUrl = audioUrl

    let cancelled = false

    async function load() {
      try {
        const peaks = await loadWaveformPeaks(currentAudioUrl)
        if (cancelled) return
        peaksRef.current = peaks
        redraw()
      } catch (error: unknown) {
        if (!cancelled) {
          console.warn('[AudioWaveform] failed to decode waveform:', error)
        }
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [audioUrl, redraw])

  // Redraw whenever trim ratios change
  useEffect(() => {
    redraw()
  }, [startRatio, endRatio, redraw])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ro = new ResizeObserver(redraw)
    ro.observe(canvas)
    return () => ro.disconnect()
  }, [redraw])

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ display: 'block', width: '100%', height: '100%' }}
    />
  )
}
