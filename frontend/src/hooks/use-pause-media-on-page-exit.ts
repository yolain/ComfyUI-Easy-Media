import { useEffect, useRef } from 'react'

/** Pause active media when the browser hides or leaves the current page. */
export function usePauseMediaOnPageExit(pauseMedia: () => void) {
  const pauseMediaRef = useRef(pauseMedia)
  pauseMediaRef.current = pauseMedia

  useEffect(() => {
    function handleVisibilityChange() {
      if (document.visibilityState === 'hidden') pauseMediaRef.current()
    }

    function handlePageHide() {
      pauseMediaRef.current()
    }

    document.addEventListener('visibilitychange', handleVisibilityChange)
    window.addEventListener('pagehide', handlePageHide)
    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange)
      window.removeEventListener('pagehide', handlePageHide)
    }
  }, [])
}
