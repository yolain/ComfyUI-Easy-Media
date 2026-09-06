import { afterEach, describe, expect, it, vi } from 'vitest'
import { createReactWidget, type ReactWidgetProps } from '@/lib/create-react-widget'
import type { DOMWidgetOptions } from '@comfyorg/comfyui-frontend-types'

vi.mock('react-dom/client', () => ({
  createRoot: () => ({
    render: vi.fn(),
    unmount: vi.fn(),
  }),
}))

interface TestValue {
  label: string
}

interface TestDOMWidget {
  name: string
  type: string
  element: HTMLDivElement
  options: DOMWidgetOptions<string>
  y: number
  value: string
  inputEl?: HTMLDivElement
  callback: ReturnType<typeof vi.fn>
  width?: number
}

describe('createReactWidget', () => {
  afterEach(() => {
    document.body.replaceChildren()
    delete (globalThis as typeof globalThis & { LiteGraph?: unknown }).LiteGraph
  })

  it('restores a string value through reloaders that require the legacy inputEl field', () => {
    let widget: TestDOMWidget | undefined

    function TestWidget(_props: Readonly<ReactWidgetProps<TestValue>>) {
      return null
    }

    const node = {
      size: [400, 320],
      setDirtyCanvas: vi.fn(),
      addDOMWidget: vi.fn((
        name: string,
        type: string,
        element: HTMLDivElement,
        options: DOMWidgetOptions<string>,
      ) => {
        const createdWidget: TestDOMWidget = {
          name,
          type,
          element,
          options,
          y: 0,
          value: '',
          callback: vi.fn(),
        }
        Object.defineProperty(createdWidget, 'value', {
          configurable: true,
          get: () => options.getValue?.(),
          set: (value) => {
            options.setValue?.(value)
            createdWidget.callback(createdWidget.value)
          },
        })
        widget = createdWidget
        document.body.append(element)
        return createdWidget
      }),
    }
    const factory = createReactWidget<TestValue>(TestWidget, {
      defaultValue: JSON.stringify({ label: 'default' }),
    })
    const restoredValue = JSON.stringify({ label: 'restored' })

    factory(node, 'track_data', {} as never, {} as never)
    if (!widget) throw new Error('Expected the DOM widget to be created')
    const createdWidget = widget

    // ComfyUI-Easy-Use's Reload Node only restores string widgets that expose inputEl.
    if (createdWidget.inputEl && typeof restoredValue === 'string') {
      createdWidget.value = restoredValue
    }

    expect(createdWidget.value).toBe(restoredValue)
  })

  it('allows fixed-height widgets to reserve space for the default DOM widget margin', () => {
    let widget: TestDOMWidget | undefined

    function TestWidget(_props: Readonly<ReactWidgetProps<string>>) {
      return null
    }

    const node = {
      size: [400, 320],
      setDirtyCanvas: vi.fn(),
      addDOMWidget: vi.fn((
        name: string,
        type: string,
        element: HTMLDivElement,
        options: DOMWidgetOptions<string>,
      ) => {
        widget = {
          name,
          type,
          element,
          options,
          y: 0,
          value: '',
          callback: vi.fn(),
        }
        return widget
      }),
    }
    const factory = createReactWidget<string>(TestWidget, {
      height: 30,
      domWidgetOptions: {
        getMinHeight: () => 50,
        getMaxHeight: () => 50,
      },
    })

    factory(node, 'api_account', {} as never, {} as never)

    expect(widget?.element.style.height).toBe('30px')
    expect(widget?.options.getMinHeight?.()).toBe(50)
    expect(widget?.options.getMaxHeight?.()).toBe(50)
    expect(widget?.options.margin).toBeUndefined()
  })

  it('keeps opted-in widget widths responsive in LiteGraph mode', () => {
    let widget: TestDOMWidget | undefined

    function TestWidget(_props: Readonly<ReactWidgetProps<string>>) {
      return null
    }

    const liteGraph = { vueNodesMode: false }
    ;(globalThis as typeof globalThis & { LiteGraph?: typeof liteGraph }).LiteGraph = liteGraph

    const node = {
      size: [800, 700],
      setDirtyCanvas: vi.fn(),
      addDOMWidget: vi.fn((
        name: string,
        type: string,
        element: HTMLDivElement,
        options: DOMWidgetOptions<string>,
      ) => {
        widget = {
          name,
          type,
          element,
          options,
          y: 0,
          value: '',
          callback: vi.fn(),
          width: 640,
        }
        return widget
      }),
    }
    const factory = createReactWidget<string>(TestWidget, {
      keepResponsiveWidthInLiteGraph: true,
    })

    factory(node, 'track_data', {} as never, {} as never)
    if (!widget) throw new Error('Expected the DOM widget to be created')

    expect(widget.width).toBeUndefined()
    widget.width = 500
    expect(widget.width).toBeUndefined()

    liteGraph.vueNodesMode = true
    widget.width = 500
    expect(widget.width).toBe(500)
  })

  it('leaves widget width writes unchanged unless explicitly enabled', () => {
    let widget: TestDOMWidget | undefined

    function TestWidget(_props: Readonly<ReactWidgetProps<string>>) {
      return null
    }

    ;(globalThis as typeof globalThis & { LiteGraph?: { vueNodesMode: boolean } }).LiteGraph = {
      vueNodesMode: false,
    }
    const node = {
      size: [400, 320],
      setDirtyCanvas: vi.fn(),
      addDOMWidget: vi.fn((
        name: string,
        type: string,
        element: HTMLDivElement,
        options: DOMWidgetOptions<string>,
      ) => {
        widget = {
          name,
          type,
          element,
          options,
          y: 0,
          value: '',
          callback: vi.fn(),
          width: 320,
        }
        return widget
      }),
    }

    createReactWidget<string>(TestWidget)(node, 'timeline_data', {} as never, {} as never)
    if (!widget) throw new Error('Expected the DOM widget to be created')

    widget.width = 280
    expect(widget.width).toBe(280)
  })
})
