import React from 'react'
import { createRoot, Root } from 'react-dom/client'
import { CUSTOM_NODE_CLASS } from './constants'
import type {
  ComfyApp,
  DOMWidget,
  DOMWidgetOptions,
  InputSpec,
} from '@comfyorg/comfyui-frontend-types'

/**
 * Structural interface for the subset of LGraphNode methods we use.
 * LGraphNode is not exported from @comfyorg/comfyui-frontend-types, so we
 * declare only what we need and cast to this type inside the factory.
 */
interface ComfyNode {
  setDirtyCanvas(fg: boolean, bg?: boolean): void
  addDOMWidget<T extends HTMLElement, V extends object | string>(
    name: string,
    type: string,
    element: T,
    options?: DOMWidgetOptions<V>,
  ): DOMWidget<T, V>
}

/** ComfyUI adds serializeValue to DOM widgets at runtime. */
type ComfyDOMWidget<T extends HTMLElement, V extends object | string> =
  DOMWidget<T, V> & { serializeValue?: () => V }

export interface ReactWidgetProps<T extends object | string = object> {
  value: T
  onChange: (value: T) => void
  /** The input name as declared in the node schema */
  inputName: string
  /** The LGraphNode instance this widget belongs to */
  node: any
  /** The DOMWidget instance for this widget */
  widget: DOMWidget<HTMLDivElement, string>
  /** The ComfyApp instance */
  app: ComfyApp
}

export interface ReactWidgetOptions {
  /** Default serialized value (JSON string) when the widget is first created */
  defaultValue?: string
  /** Custom height for the widget container in pixels */
  height?: number
  /** Extra options merged into DOMWidgetOptions passed to addDOMWidget */
  domWidgetOptions?: Omit<DOMWidgetOptions<string>, 'getValue' | 'setValue'>
}

/**
 * Creates a ComfyUI custom widget factory that renders a React component
 * inside a DOM widget slot.
 *
 * Usage in getCustomWidgets:
 *   return {
 *     TIMELINE: createReactWidget(TimelineWidget, { defaultValue: '' })
 *   }
 */
export function createReactWidget<T extends object | string = object>(
  Component: React.ComponentType<ReactWidgetProps<T>>,
  options: ReactWidgetOptions = {},
) {
  // node is typed as `any` so this function is assignable to ComfyWidgetConstructor
  // (LGraphNode is not exported from the package, and using a narrower structural
  //  type would break contravariant parameter compatibility).
  return function widgetFactory(
    node: any, // eslint-disable-line @typescript-eslint/no-explicit-any
    inputName: string,
    _inputData: InputSpec,
    _app: ComfyApp,
  ): { widget: DOMWidget<HTMLDivElement, string> } {
    // Always store the value as a JSON string so ComfyUI can serialize it directly.
    let currentValue: string = options.defaultValue ?? ''
    let root: Root | null = null

    const container = document.createElement('div')
    container.classList.add('comfyui-react-widget', CUSTOM_NODE_CLASS)
    if (options.height !== undefined) {
      container.style.height = `${options.height}px`
    }

    const comfyNode = node as ComfyNode

    function parseValue(): T {
      try {
        return JSON.parse(currentValue) as T
      } catch {
        return currentValue as unknown as T
      }
    }

    function render() {
      root?.render(
        React.createElement(Component, {
          value: parseValue(),
          onChange: (v: T) => {
            currentValue = typeof v === 'string' ? v : JSON.stringify(v)
            comfyNode.setDirtyCanvas(true, true)
            render()
          },
          inputName,
          widget,
          node: comfyNode,
          app: _app,
        }),
      )
    }

    const widget = comfyNode.addDOMWidget<HTMLDivElement, string>(
      inputName,
      'react-widget',
      container,
      {
        getValue: () => currentValue,
        setValue: (v: string) => {
          currentValue = v
          render()
        },
        getMinHeight: () => 30,
        getMaxHeight: () => node.size[1],
        hideOnZoom: true,
        serialize: true,
        ...options.domWidgetOptions,
      },
    ) as ComfyDOMWidget<HTMLDivElement, string>

    // serializeValue is called by ComfyUI when building the API prompt payload
    widget.serializeValue = () => currentValue
    root = createRoot(container)
    render()

    return { widget }
  }
}
