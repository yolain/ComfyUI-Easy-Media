import { useEffect, useState, type RefObject } from 'react'

export const DEFAULT_LAZY_INTERSECTION_DELAY_MS = 180
const DEFAULT_INTERSECTION_OBSERVER_INIT: IntersectionObserverInit = { threshold: 0 }

interface UseDelayedIntersectionOptions {
  delayMs?: number
  observerInit?: IntersectionObserverInit
}

export function useDelayedIntersection<T extends Element>(
  ref: RefObject<T | null>,
  {
    delayMs = DEFAULT_LAZY_INTERSECTION_DELAY_MS,
    observerInit = DEFAULT_INTERSECTION_OBSERVER_INIT,
  }: UseDelayedIntersectionOptions = {},
): boolean {
  const [intersected, setIntersected] = useState(false)

  useEffect(() => {
    if (intersected) return

    const element = ref.current
    if (!element) return

    if (typeof IntersectionObserver === 'undefined') {
      setIntersected(true)
      return
    }

    let timer: ReturnType<typeof setTimeout> | undefined
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        timer ??= setTimeout(() => {
          setIntersected(true)
          observer.disconnect()
        }, delayMs)
        return
      }

      if (timer) {
        clearTimeout(timer)
        timer = undefined
      }
    }, observerInit)

    observer.observe(element)

    return () => {
      if (timer) clearTimeout(timer)
      observer.disconnect()
    }
  }, [delayMs, intersected, observerInit, ref])

  return intersected
}
