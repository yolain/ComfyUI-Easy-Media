import { useEffect, useRef, useCallback } from 'react'
import type { AudioContent } from '@/types/timeline'
import { audioContentToViewUrl } from '@/lib/media-url'

interface AudioWaveformProps {
  content: AudioContent
  /** Fraction [0,1] of the full audio where the visible window starts */
  startRatio?: number
  /** Fraction [0,1] of the full audio where the visible window ends */
  endRatio?: number
  className?: string
}

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

export function AudioWaveform({ content, startRatio = 0, endRatio = 1, className }: Readonly<AudioWaveformProps>) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const peaksRef = useRef<Float32Array | null>(null)
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
    const url = audioContentToViewUrl(content)
    if (!url) return
    const audioUrl = url

    let cancelled = false
    const controller = new AbortController()

    async function load() {
      try {
        const response = await fetch(audioUrl, { signal: controller.signal })
        if (!response.ok || cancelled) return

        const arrayBuffer = await response.arrayBuffer()
        if (cancelled) return

        const audioCtx = new AudioContext()
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer)
        await audioCtx.close()
        if (cancelled) return

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

        peaksRef.current = peaks
        redraw()
      } catch {
        // Silently ignore fetch/decode errors (e.g. network unavailable)
      }
    }

    load()
    return () => {
      cancelled = true
      controller.abort()
    }
  }, [content, redraw])

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
