import { useLayoutEffect, type RefObject } from 'react'

/** Scroll inside DOM widgets even when the node canvas cancels native wheel scrolling. */
export function useVerticalWheelScroll(ref: RefObject<HTMLElement | null>): void {
  useLayoutEffect(() => {
    const element = ref.current
    if (!element) return

    function handleWheel(event: WheelEvent) {
      if (!element || event.ctrlKey || event.metaKey || event.deltaY === 0) return
      const limit = element.scrollHeight - element.clientHeight
      if (limit <= 0) return

      // nodeV2 can prevent the default action in an ancestor's capture listener.
      // Apply scrolling explicitly; cancelling here also avoids double scrolling
      // in the legacy node renderer, where the native default still works.
      event.preventDefault()
      event.stopPropagation()
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? element.clientHeight : 1
      element.scrollTop = Math.max(0, Math.min(limit, element.scrollTop + event.deltaY * unit))
    }

    element.addEventListener('wheel', handleWheel, { passive: false })
    return () => element.removeEventListener('wheel', handleWheel)
  }, [ref])
}
