import { useEffect, useState } from 'react'
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'

type DrawForeground = NonNullable<ComfyApp['canvas']['onDrawForeground']>

export function useCanvasScale(app: ComfyApp | null | undefined): number {
  const [canvasScale, setCanvasScale] = useState<number>(() => app?.canvas?.ds?.scale ?? 1)

  useEffect(() => {
    if (!app?.canvas) return

    const canvas = app.canvas
    const origOnDrawForeground = canvas.onDrawForeground?.bind(canvas)
    canvas.onDrawForeground = ((...args: Parameters<DrawForeground>) => {
      origOnDrawForeground?.(...args)
      const scale = canvas.ds?.scale ?? 1
      setCanvasScale((prev) => (prev !== scale ? scale : prev))
    }) as DrawForeground

    return () => {
      canvas.onDrawForeground = origOnDrawForeground
    }
  }, [app])

  return canvasScale
}
