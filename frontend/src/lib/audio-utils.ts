export interface BrowserAudioMetadata {
  duration: number
}

export function isSameOriginBrowserMedia(src: string, pageHref: string | undefined): boolean {
  if (!pageHref) return false
  try {
    return new URL(src, pageHref).origin === new URL(pageHref).origin
  } catch (error) {
    console.error('[audio-utils] failed to resolve media origin:', error)
    return false
  }
}

export function resolveBrowserAudioPreviewGain(
  volumeDb: number,
  src: string,
  pageHref: string | undefined = globalThis.location?.href,
): number {
  const safeVolumeDb = Number.isFinite(volumeDb) ? volumeDb : 0
  const gain = 10 ** (safeVolumeDb / 20)
  return isSameOriginBrowserMedia(src, pageHref) ? gain : Math.min(1, gain)
}

export function loadBrowserAudioMetadata(src: string): Promise<BrowserAudioMetadata> {
  return new Promise((resolve, reject) => {
    const audio = new Audio()
    const cleanup = () => {
      audio.removeEventListener('loadedmetadata', handleLoaded)
      audio.removeEventListener('error', handleError)
      audio.src = ''
    }
    const handleLoaded = () => {
      const duration = audio.duration
      cleanup()
      if (!Number.isFinite(duration) || duration <= 0) {
        reject(new Error(`Invalid audio duration for ${src}`))
        return
      }
      resolve({ duration })
    }
    const handleError = () => {
      cleanup()
      reject(new Error(`Unable to load audio metadata for ${src}`))
    }
    audio.addEventListener('loadedmetadata', handleLoaded)
    audio.addEventListener('error', handleError)
    audio.preload = 'metadata'
    audio.src = src
  })
}
