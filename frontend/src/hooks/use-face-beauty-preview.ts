import { useEffect, useRef, useState } from 'react'
import { createBeautyRenderer } from '@/lib/face-beauty-renderer'
import type { BeautyPreset, BeautyPreview } from '@/lib/face-beauty'
import { mediaPathToViewUrl } from '@/lib/media-url'
import { downloadEasyMediaModel } from '@/lib/model-download'
export interface BeautyNode {
  onExecuted?: (output: unknown) => void
  onConnectionsChange?: (...args: unknown[]) => void
  __easyBeautyPreview?: BeautyPreview
  __easyBeautyPreviewLink?: number | null
  inputs?: { name?: string; link?: number | null }[]
}
export function useFaceBeautyPreview(node: BeautyNode, preset: BeautyPreset) {
  const [, setConnectionRevision] = useState(0)
  useEffect(() => {
    const original = node.onConnectionsChange
    const handler = (...args: unknown[]) => { original?.apply(node, args); setConnectionRevision(value => value + 1) }
    node.onConnectionsChange = handler
    return () => { if (node.onConnectionsChange === handler) node.onConnectionsChange = original }
  }, [node])
  const imageLink = node.inputs?.find(input => input.name === 'image')?.link ?? null
  const hasImageConnection = imageLink !== null
  const connected = hasImageConnection && preset.source_mode !== 'picker'
  const [preview, setPreview] = useState<BeautyPreview | null>(connected && node.__easyBeautyPreviewLink === imageLink ? node.__easyBeautyPreview ?? null : null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [missingModel, setMissingModel] = useState(false)
  const [revision, setRevision] = useState(0)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<ReturnType<typeof createBeautyRenderer> | null>(null)
  const parametersRef = useRef(preset)
  parametersRef.current = preset
  const sourceKey = JSON.stringify(preset.source)
  const selectionRef = useRef({ connected, source: preset.source, imageLink })
  selectionRef.current = { connected, source: preset.source, imageLink }
  useEffect(() => {
    const original = node.onExecuted
    const handler = (output: unknown) => {
      original?.call(node, output)
      if (!output || typeof output !== 'object') return
      const next = (output as { face_beauty?: BeautyPreview[] }).face_beauty?.[0]
      if (!next) return
      const current = selectionRef.current
      if (next.input_connected !== current.connected) return
      if (!current.connected && (next.source?.source_type !== current.source?.source_type || next.source?.file_path !== current.source?.file_path)) return
      node.__easyBeautyPreview = next
      node.__easyBeautyPreviewLink = current.connected ? current.imageLink : null
      setPreview(next); setError('')
    }
    node.onExecuted = handler
    return () => { if (node.onExecuted === handler) node.onExecuted = original }
  }, [node])
  const activeImageLink = connected ? imageLink : null
  useEffect(() => {
    if (connected) {
      const cached = node.__easyBeautyPreview
      setPreview(cached?.input_connected && node.__easyBeautyPreviewLink === activeImageLink ? cached : null)
      setError(''); setMissingModel(false); setLoading(false)
      return
    }
    setPreview(null); setError(''); setMissingModel(false); setLoading(false)
    if (sourceKey === 'null') return
    const controller = new AbortController()
    setLoading(true)
    void (async () => {
      try {
        const response = await fetch('/easy-media/face-beauty/prepare', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ version: 1, source: JSON.parse(sourceKey) as BeautyPreset['source'] }), signal: controller.signal,
        })
        const payload = await response.json() as BeautyPreview & { error?: string; model?: unknown }
        if (controller.signal.aborted) return
        if (!response.ok) {
          setMissingModel(response.status === 409 && Boolean(payload.model))
          throw new Error(payload.error ?? `Face Beauty: ${response.status}`)
        }
        setPreview(payload)
      } catch (cause) {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : String(cause))
      } finally { if (!controller.signal.aborted) setLoading(false) }
    })()
    return () => controller.abort()
  }, [connected, activeImageLink, node, sourceKey, revision])
  useEffect(() => {
    if (!preview || !canvasRef.current) return
    let cancelled = false
    const source = new Image()
    source.onload = () => {
      if (cancelled || !canvasRef.current) return
      try {
        const renderer = createBeautyRenderer(canvasRef.current, source, preview.geometry)
        rendererRef.current = renderer
        renderer.draw(parametersRef.current.parameters, parametersRef.current.colors)
      } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)) }
    }
    source.onerror = () => { if (!cancelled) setError('Unable to load face beauty preview image') }
    source.src = mediaPathToViewUrl(preview.filename, preview.type)
    return () => {
      cancelled = true; source.onload = null; source.onerror = null
      rendererRef.current?.dispose(); rendererRef.current = null
    }
  }, [preview])
  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      try { rendererRef.current?.draw(preset.parameters, preset.colors) }
      catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)) }
    })
    return () => cancelAnimationFrame(frame)
  }, [preset.parameters, preset.colors])
  async function downloadModel() {
    setLoading(true); setError('')
    try { await downloadEasyMediaModel('face-landmarker'); setMissingModel(false); setRevision(value => value + 1) }
    catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)) }
    finally { setLoading(false) }
  }
  return { preview, error, loading, missingModel, canvasRef, connected, hasImageConnection, downloadModel }
}
