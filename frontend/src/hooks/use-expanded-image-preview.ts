import { useEffect, useRef, useState } from 'react'

const MIN_ZOOM = 1
const MAX_ZOOM = 6
const ZOOM_STEP = 1.15
const CENTER = { x: 50, y: 50 }

export function useExpandedImagePreview(activeKey: string | null, onClose: () => void) {
  const imageRef = useRef<HTMLImageElement>(null)
  const [zoom, setZoom] = useState(MIN_ZOOM)
  const [origin, setOrigin] = useState(CENTER)

  useEffect(() => {
    setZoom(MIN_ZOOM)
    setOrigin(CENTER)
  }, [activeKey])

  useEffect(() => {
    if (!activeKey) return
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [activeKey, onClose])

  function onWheel(event: React.WheelEvent<HTMLDivElement>) {
    event.preventDefault()
    event.stopPropagation()
    const rect = imageRef.current?.getBoundingClientRect()
    if (rect && rect.width > 0 && rect.height > 0) {
      setOrigin({
        x: Math.max(0, Math.min(100, (event.clientX - rect.left) / rect.width * 100)),
        y: Math.max(0, Math.min(100, (event.clientY - rect.top) / rect.height * 100)),
      })
    }
    const factor = event.deltaY < 0 ? ZOOM_STEP : 1 / ZOOM_STEP
    setZoom((current) => Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, current * factor)))
  }

  return { imageRef, zoom, origin, onWheel }
}
