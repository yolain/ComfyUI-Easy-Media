import { useEffect, useRef } from 'react'
import {
  BackSide,
  Mesh,
  MeshBasicMaterial,
  PerspectiveCamera,
  RepeatWrapping,
  Scene,
  SphereGeometry,
  SRGBColorSpace,
  Texture,
  TextureLoader,
  WebGLRenderer,
} from 'three'
import { horizontalToVerticalFov, normalizePanoramaView } from '@/lib/panorama-camera'
import type { MultiTrackPanoramaView } from '@/types/multitrack'

interface PanoramaCanvasProps {
  imageUrl: string
  view: MultiTrackPanoramaView
  onViewChange: (view: MultiTrackPanoramaView) => void
  onAspectRatioChange: (aspectRatio: number) => void
  onLoad?: () => void
  onError: (error: Error) => void
}

interface PanoramaResources {
  camera: PerspectiveCamera
  geometry: SphereGeometry
  material: MeshBasicMaterial
  renderer: WebGLRenderer
  scene: Scene
  texture: Texture
}

interface DragState {
  pointerId: number
  x: number
  y: number
  timestamp: number
  velocityPitch: number
  velocityYaw: number
}

const DRAG_DEGREES_PER_PIXEL = 0.15
const WHEEL_FOV_DEGREES_PER_PIXEL = 0.1
const MOMENTUM_FRICTION_PER_MS = 0.006
const MIN_MOMENTUM_DEGREES_PER_MS = 0.002

function updateCamera(resources: PanoramaResources, view: MultiTrackPanoramaView): void {
  const pitchRadians = view.pitch * Math.PI / 180
  const yawRadians = view.yaw * Math.PI / 180
  resources.camera.fov = horizontalToVerticalFov(view.hfov, resources.camera.aspect)
  resources.camera.rotation.set(pitchRadians, -yawRadians - Math.PI / 2, 0, 'YXZ')
  resources.camera.updateProjectionMatrix()
  resources.renderer.render(resources.scene, resources.camera)
}

export function PanoramaCanvas({
  imageUrl,
  view,
  onViewChange,
  onAspectRatioChange,
  onLoad,
  onError,
}: Readonly<PanoramaCanvasProps>) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const resourcesRef = useRef<PanoramaResources | null>(null)
  const viewRef = useRef(view)
  const onAspectRatioChangeRef = useRef(onAspectRatioChange)
  const onLoadRef = useRef(onLoad)
  const onErrorRef = useRef(onError)
  const dragRef = useRef<DragState | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  viewRef.current = view
  onAspectRatioChangeRef.current = onAspectRatioChange
  onLoadRef.current = onLoad
  onErrorRef.current = onError

  const cancelMomentum = () => {
    if (animationFrameRef.current !== null) {
      window.cancelAnimationFrame(animationFrameRef.current)
      animationFrameRef.current = null
    }
  }

  const emitView = (nextView: MultiTrackPanoramaView) => {
    viewRef.current = nextView
    onViewChange(nextView)
  }

  const startMomentum = (
    velocityYaw: number,
    velocityPitch: number,
    velocityFov: number,
    startTime: number,
  ) => {
    cancelMomentum()
    if (Math.max(Math.abs(velocityYaw), Math.abs(velocityPitch), Math.abs(velocityFov)) < MIN_MOMENTUM_DEGREES_PER_MS) {
      return
    }
    let previousTime = startTime
    const animate = (timestamp: number) => {
      const elapsed = Math.min(32, Math.max(1, timestamp - previousTime))
      previousTime = timestamp
      const decay = Math.exp(-MOMENTUM_FRICTION_PER_MS * elapsed)
      velocityYaw *= decay
      velocityPitch *= decay
      velocityFov *= decay
      const nextView = normalizePanoramaView({
        ...viewRef.current,
        yaw: viewRef.current.yaw + velocityYaw * elapsed,
        pitch: viewRef.current.pitch + velocityPitch * elapsed,
        hfov: viewRef.current.hfov + velocityFov * elapsed,
      })
      emitView(nextView)

      if (Math.max(Math.abs(velocityYaw), Math.abs(velocityPitch), Math.abs(velocityFov)) < MIN_MOMENTUM_DEGREES_PER_MS) {
        animationFrameRef.current = null
        return
      }
      animationFrameRef.current = window.requestAnimationFrame(animate)
    }
    animationFrameRef.current = window.requestAnimationFrame(animate)
  }

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    let resources: PanoramaResources
    let cancelled = false
    let observer: ResizeObserver | null = null

    try {
      const scene = new Scene()
      const camera = new PerspectiveCamera(90, 1, 0.1, 100)
      camera.position.set(0, 0, 0)
      const renderer = new WebGLRenderer({ canvas, antialias: true })
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
      const geometry = new SphereGeometry(10, 64, 32)
      const texture = new TextureLoader().load(
        imageUrl,
        (loadedTexture) => {
          if (cancelled) return
          loadedTexture.colorSpace = SRGBColorSpace
          updateCamera(resources, viewRef.current)
          onLoadRef.current?.()
        },
        undefined,
        (cause) => {
          if (!cancelled) onErrorRef.current(new Error('Failed to load panorama texture', { cause }))
        },
      )
      texture.colorSpace = SRGBColorSpace
      texture.wrapS = RepeatWrapping
      texture.repeat.set(-1, 1)
      texture.offset.x = 1
      const material = new MeshBasicMaterial({ map: texture, side: BackSide })
      scene.add(new Mesh(geometry, material))
      resources = { camera, geometry, material, renderer, scene, texture }
      resourcesRef.current = resources

      observer = new ResizeObserver((entries) => {
        const entry = entries[0]
        const width = entry?.contentRect.width ?? canvas.clientWidth
        const height = entry?.contentRect.height ?? canvas.clientHeight
        if (width <= 0 || height <= 0) return
        resources.camera.aspect = width / height
        resources.renderer.setSize(width, height, false)
        onAspectRatioChangeRef.current(resources.camera.aspect)
        updateCamera(resources, viewRef.current)
      })
      observer.observe(canvas)
    } catch (cause) {
      onErrorRef.current(new Error('WebGL panorama renderer is unavailable', { cause }))
      return
    }

    return () => {
      cancelled = true
      observer?.disconnect()
      dragRef.current = null
      cancelMomentum()
      resourcesRef.current = null
      resources.texture.dispose()
      resources.material.dispose()
      resources.geometry.dispose()
      resources.renderer.dispose()
    }
  }, [imageUrl])

  useEffect(() => {
    const resources = resourcesRef.current
    if (resources) updateCamera(resources, view)
  }, [view])

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full touch-none cursor-grab bg-black active:cursor-grabbing"
      onPointerDown={(event) => {
        cancelMomentum()
        event.currentTarget.setPointerCapture?.(event.pointerId)
        dragRef.current = {
          pointerId: event.pointerId,
          x: event.clientX,
          y: event.clientY,
          timestamp: event.timeStamp,
          velocityPitch: 0,
          velocityYaw: 0,
        }
      }}
      onPointerMove={(event) => {
        const drag = dragRef.current
        if (!drag || drag.pointerId !== event.pointerId) return
        const elapsed = Math.max(1, event.timeStamp - drag.timestamp)
        const yawDelta = -(event.clientX - drag.x) * DRAG_DEGREES_PER_PIXEL
        const pitchDelta = -(event.clientY - drag.y) * DRAG_DEGREES_PER_PIXEL
        const smoothing = 0.65
        drag.velocityYaw = drag.velocityYaw * (1 - smoothing) + yawDelta / elapsed * smoothing
        drag.velocityPitch = drag.velocityPitch * (1 - smoothing) + pitchDelta / elapsed * smoothing
        drag.x = event.clientX
        drag.y = event.clientY
        drag.timestamp = event.timeStamp
        emitView(normalizePanoramaView({
          ...viewRef.current,
          yaw: viewRef.current.yaw + yawDelta,
          pitch: viewRef.current.pitch + pitchDelta,
        }))
      }}
      onPointerUp={(event) => {
        const drag = dragRef.current
        if (drag?.pointerId !== event.pointerId) return
        dragRef.current = null
        event.currentTarget.releasePointerCapture?.(event.pointerId)
        const releaseDecay = Math.exp(
          -MOMENTUM_FRICTION_PER_MS * Math.max(0, event.timeStamp - drag.timestamp),
        )
        startMomentum(
          drag.velocityYaw * releaseDecay,
          drag.velocityPitch * releaseDecay,
          0,
          event.timeStamp,
        )
      }}
      onPointerCancel={() => {
        dragRef.current = null
      }}
      onWheel={(event) => {
        event.preventDefault()
        event.stopPropagation()
        const fovDelta = event.deltaY * WHEEL_FOV_DEGREES_PER_PIXEL
        emitView(normalizePanoramaView({
          ...viewRef.current,
          hfov: viewRef.current.hfov + fovDelta,
        }))
        startMomentum(0, 0, fovDelta * MOMENTUM_FRICTION_PER_MS, event.timeStamp)
      }}
    />
  )
}
