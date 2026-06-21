export interface BrowserVideoMetadata {
  duration: number
  width: number
  height: number
}

export function loadBrowserVideoMetadata(src: string): Promise<BrowserVideoMetadata> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'metadata'
    video.muted = true
    video.onloadedmetadata = () => {
      resolve({
        duration: Number.isFinite(video.duration) ? video.duration : 0,
        width: video.videoWidth,
        height: video.videoHeight,
      })
    }
    video.onerror = () => reject(new Error(`Unable to load video metadata: ${src}`))
    video.src = src
  })
}

export function captureVideoPosterFrame(src: string, sourceTime = 0.1): Promise<string> {
  return new Promise((resolve, reject) => {
    const video = document.createElement('video')
    video.preload = 'auto'
    video.muted = true
    video.playsInline = true

    function drawFrame() {
      try {
        const width = video.videoWidth
        const height = video.videoHeight
        if (width <= 0 || height <= 0) {
          reject(new Error(`Unable to capture video frame: ${src}`))
          return
        }

        const canvas = document.createElement('canvas')
        canvas.width = width
        canvas.height = height
        const context = canvas.getContext('2d')
        if (!context) {
          reject(new Error('Unable to create canvas context for video frame'))
          return
        }

        context.drawImage(video, 0, 0, width, height)
        canvas.toBlob((blob) => {
          if (!blob) {
            reject(new Error(`Unable to encode video frame: ${src}`))
            return
          }
          resolve(URL.createObjectURL(blob))
        }, 'image/jpeg', 0.8)
      } catch (error) {
        reject(error instanceof Error ? error : new Error(`Unable to capture video frame: ${src}`))
      }
    }

    video.onloadeddata = () => {
      const targetTime = Number.isFinite(video.duration)
        ? Math.max(0, Math.min(sourceTime, Math.max(0, video.duration - 0.001)))
        : Math.max(0, sourceTime)
      if (targetTime > 0) {
        video.currentTime = targetTime
        return
      }
      drawFrame()
    }
    video.onseeked = drawFrame
    video.onerror = () => reject(new Error(`Unable to load video frame: ${src}`))
    video.src = src
  })
}
