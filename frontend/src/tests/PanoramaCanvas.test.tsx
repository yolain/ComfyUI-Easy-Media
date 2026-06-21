import { fireEvent, render, waitFor } from '@testing-library/react'
import * as THREE from 'three'
import { DEFAULT_PANORAMA_VIEW } from '@/lib/panorama-camera'
import { PanoramaCanvas } from '@/components/widgets/panorama/PanoramaCanvas'

vi.mock('three', () => {
  const mocks = {
    rendererDispose: 0,
    geometryDispose: 0,
    materialDispose: 0,
    textureDispose: 0,
    render: 0,
    setSize: 0,
    repeatX: 0,
    offsetX: 0,
    wrapS: 0,
  }

  class Scene {
    add() {}
  }

  class PerspectiveCamera {
    aspect = 1
    fov = 90
    position = { set() {} }
    rotation = { set() {} }
    updateProjectionMatrix() {}
  }

  class WebGLRenderer {
    setPixelRatio() {}
    setSize() { mocks.setSize += 1 }
    render() { mocks.render += 1 }
    dispose() { mocks.rendererDispose += 1 }
  }

  class SphereGeometry {
    dispose() { mocks.geometryDispose += 1 }
  }

  class MeshBasicMaterial {
    map: unknown = null
    needsUpdate = false
    dispose() { mocks.materialDispose += 1 }
  }

  class TextureLoader {
    load(_url: string, onLoad: (texture: Texture) => void): Texture {
      const texture = new Texture()
      queueMicrotask(() => onLoad(texture))
      return texture
    }
  }

  class Texture {
    colorSpace = ''
    repeat = { set: (x: number) => { mocks.repeatX = x } }
    offset = {
      get x() { return mocks.offsetX },
      set x(value: number) { mocks.offsetX = value },
    }
    get wrapS() { return mocks.wrapS }
    set wrapS(value: number) { mocks.wrapS = value }
    dispose() { mocks.textureDispose += 1 }
  }

  class Mesh {}

  return {
    BackSide: 1,
    Mesh,
    MeshBasicMaterial,
    PerspectiveCamera,
    Scene,
    SphereGeometry,
    SRGBColorSpace: 'srgb',
    RepeatWrapping: 1000,
    Texture,
    TextureLoader,
    WebGLRenderer,
    __mocks: mocks,
  }
})

const threeMocks = (THREE as unknown as { __mocks: {
  rendererDispose: number
  geometryDispose: number
  materialDispose: number
  textureDispose: number
  render: number
  setSize: number
  repeatX: number
  offsetX: number
  wrapS: number
} }).__mocks

class ResizeObserverMock {
  constructor(private readonly callback: ResizeObserverCallback) {}

  observe(target: Element) {
    this.callback([{
      target,
      contentRect: { width: 400, height: 200 } as DOMRectReadOnly,
    } as ResizeObserverEntry], this as unknown as ResizeObserver)
  }

  disconnect = vi.fn()
  unobserve = vi.fn()
}

class PointerEventMock extends MouseEvent {
  readonly pointerId: number

  constructor(type: string, init: PointerEventInit = {}) {
    super(type, init)
    this.pointerId = init.pointerId ?? 0
  }
}

describe('PanoramaCanvas', () => {
  beforeEach(() => {
    Object.assign(threeMocks, {
      rendererDispose: 0,
      geometryDispose: 0,
      materialDispose: 0,
      textureDispose: 0,
      render: 0,
      setSize: 0,
      repeatX: 0,
      offsetX: 0,
      wrapS: 0,
    })
    vi.stubGlobal('ResizeObserver', ResizeObserverMock)
    vi.stubGlobal('PointerEvent', PointerEventMock)
    Object.defineProperty(HTMLCanvasElement.prototype, 'setPointerCapture', {
      configurable: true,
      value: vi.fn(),
    })
    Object.defineProperty(HTMLCanvasElement.prototype, 'releasePointerCapture', {
      configurable: true,
      value: vi.fn(),
    })
  })

  it('reports canvas aspect ratio and emits drag camera changes', async () => {
    const onViewChange = vi.fn()
    const onAspectRatioChange = vi.fn()
    const { container } = render(
      <PanoramaCanvas
        imageUrl="/view?filename=pano.png&type=input"
        view={DEFAULT_PANORAMA_VIEW}
        onViewChange={onViewChange}
        onAspectRatioChange={onAspectRatioChange}
        onError={vi.fn()}
      />,
    )
    const canvas = container.querySelector('canvas') as HTMLCanvasElement

    await waitFor(() => expect(onAspectRatioChange).toHaveBeenCalledWith(2))
    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 10, clientY: 20 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 30, clientY: 5 })
    fireEvent.pointerUp(canvas, { pointerId: 1 })

    expect(onViewChange).toHaveBeenCalledWith(expect.objectContaining({
      yaw: expect.any(Number),
      pitch: expect.any(Number),
    }))
    expect(onViewChange.mock.lastCall?.[0]).not.toMatchObject({ yaw: 0, pitch: 0 })
  })

  it('continues rotating with decaying momentum after a quick drag', () => {
    const onViewChange = vi.fn()
    const animationFrames: FrameRequestCallback[] = []
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((callback) => {
      animationFrames.push(callback)
      return animationFrames.length
    })
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => undefined)
    const { container } = render(
      <PanoramaCanvas
        imageUrl="/view?filename=pano.png&type=input"
        view={DEFAULT_PANORAMA_VIEW}
        onViewChange={onViewChange}
        onAspectRatioChange={vi.fn()}
        onError={vi.fn()}
      />,
    )
    const canvas = container.querySelector('canvas') as HTMLCanvasElement

    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 10, clientY: 10, timeStamp: 10 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 60, clientY: 10, timeStamp: 26 })
    fireEvent.pointerUp(canvas, { pointerId: 1, timeStamp: 30 })
    const callsAtRelease = onViewChange.mock.calls.length
    animationFrames.shift()?.(46)

    expect(onViewChange.mock.calls.length).toBeGreaterThan(callsAtRelease)
  })

  it('allows vertical rotation past the former pole limit', () => {
    const onViewChange = vi.fn()
    const { container } = render(
      <PanoramaCanvas
        imageUrl="/view?filename=pano.png&type=input"
        view={DEFAULT_PANORAMA_VIEW}
        onViewChange={onViewChange}
        onAspectRatioChange={vi.fn()}
        onError={vi.fn()}
      />,
    )
    const canvas = container.querySelector('canvas') as HTMLCanvasElement

    fireEvent.pointerDown(canvas, { pointerId: 1, clientX: 0, clientY: 0 })
    fireEvent.pointerMove(canvas, { pointerId: 1, clientX: 0, clientY: -720 })

    expect(onViewChange).toHaveBeenCalledWith(expect.objectContaining({ pitch: 108 }))
  })

  it('changes FOV from wheel input only on the panorama canvas', () => {
    const onViewChange = vi.fn()
    const { container } = render(
      <PanoramaCanvas
        imageUrl="/view?filename=pano.png&type=input"
        view={DEFAULT_PANORAMA_VIEW}
        onViewChange={onViewChange}
        onAspectRatioChange={vi.fn()}
        onError={vi.fn()}
      />,
    )

    fireEvent.wheel(container.querySelector('canvas') as HTMLCanvasElement, { deltaY: 100 })

    expect(onViewChange).toHaveBeenCalledWith(expect.objectContaining({ hfov: 100 }))
  })

  it('disposes Three.js resources when unmounted', async () => {
    const { unmount } = render(
      <PanoramaCanvas
        imageUrl="/view?filename=pano.png&type=input"
        view={DEFAULT_PANORAMA_VIEW}
        onViewChange={vi.fn()}
        onAspectRatioChange={vi.fn()}
        onError={vi.fn()}
      />,
    )

    await waitFor(() => expect(threeMocks.render).toBeGreaterThan(0))
    unmount()

    expect(threeMocks.textureDispose).toBeGreaterThan(0)
    expect(threeMocks.materialDispose).toBeGreaterThan(0)
    expect(threeMocks.geometryDispose).toBeGreaterThan(0)
    expect(threeMocks.rendererDispose).toBeGreaterThan(0)
  })

  it('flips the sphere texture so positive yaw increases source U', async () => {
    render(
      <PanoramaCanvas
        imageUrl="/view?filename=pano.png&type=input"
        view={DEFAULT_PANORAMA_VIEW}
        onViewChange={vi.fn()}
        onAspectRatioChange={vi.fn()}
        onError={vi.fn()}
      />,
    )

    await waitFor(() => expect(threeMocks.render).toBeGreaterThan(0))
    expect(threeMocks.repeatX).toBe(-1)
    expect(threeMocks.offsetX).toBe(1)
    expect(threeMocks.wrapS).toBe(1000)
  })

  it('does not recreate WebGL resources when view callbacks change', async () => {
    const props = {
      imageUrl: '/view?filename=pano.png&type=input',
      view: DEFAULT_PANORAMA_VIEW,
      onViewChange: vi.fn(),
      onAspectRatioChange: vi.fn(),
      onError: vi.fn(),
    }
    const { rerender } = render(<PanoramaCanvas {...props} />)
    await waitFor(() => expect(threeMocks.render).toBeGreaterThan(0))

    rerender(
      <PanoramaCanvas
        {...props}
        view={{ ...DEFAULT_PANORAMA_VIEW, yaw: 20 }}
        onAspectRatioChange={vi.fn()}
        onError={vi.fn()}
      />,
    )

    expect(threeMocks.rendererDispose).toBe(0)
    expect(threeMocks.geometryDispose).toBe(0)
  })
})
