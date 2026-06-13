import { useLayoutEffect, useState } from 'react'
import type { RefObject } from 'react'

export function useElementWidth<T extends HTMLElement>(ref: RefObject<T | null>): number {
  const [width, setWidth] = useState(0)

  useLayoutEffect(() => {
    const el = ref.current
    if (!el) return

    const ro = new ResizeObserver(() => setWidth(el.clientWidth))
    ro.observe(el)
    setWidth(el.clientWidth)

    return () => ro.disconnect()
  }, [ref])

  return width
}
