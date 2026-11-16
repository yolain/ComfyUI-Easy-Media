import { useLayoutEffect, type RefObject } from 'react'

/** Keep wheel scrolling local, before node canvas capture listeners can intercept it. */
export function useVerticalWheelScroll(ref: RefObject<HTMLElement | null>): void {
  useLayoutEffect(() => {
    const element = ref.current
    const view = element?.ownerDocument.defaultView
    if (!element || !view) return

    function handleWheel(event: WheelEvent) {
      if (!element || event.ctrlKey || event.metaKey || event.deltaY === 0) return
      // Native event paths exclude portaled color pickers and other nodes. Focus
      // alone must never redirect an outside wheel gesture into this panel.
      if (!event.composedPath().includes(element)) return
      const limit = element.scrollHeight - element.clientHeight
      if (limit <= 0) return

      // Run before document/node capture listeners. In particular, a focused
      // slider must not let nodeV2 consume the gesture before it reaches us.
      event.preventDefault()
      event.stopImmediatePropagation()
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? element.clientHeight : 1
      element.scrollTop = Math.max(0, Math.min(limit, element.scrollTop + event.deltaY * unit))
    }

    const options = { capture: true, passive: false } as const
    view.addEventListener('wheel', handleWheel, options)
    return () => view.removeEventListener('wheel', handleWheel, options)
  }, [ref])
}
