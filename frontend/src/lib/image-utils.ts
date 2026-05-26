import type React from 'react'
import type { ImageItem } from '@/types/timeline'

export function getImageSrc(img: ImageItem): string | null {
  if (img.url) return img.url
  if (img.local_path) return img.local_path
  if (img.file_path) {
    // Backward-compat: URL images stored via old code may have url in file_path
    if (/^https?:\/\//i.test(img.file_path)) return img.file_path
    // Handle both forward slash and backslash as path separator
    const lastSlash = Math.max(img.file_path.lastIndexOf('/'), img.file_path.lastIndexOf('\\'))
    const filename = lastSlash >= 0 ? img.file_path.slice(lastSlash + 1) : img.file_path
    const subfolder = lastSlash >= 0 ? img.file_path.slice(0, lastSlash) : ''
    const typeParam = img.source_type === 'output' || img.source_type === 'local' ? img.source_type : 'input'
    return `/view?filename=${encodeURIComponent(filename)}&type=${typeParam}&subfolder=${encodeURIComponent(subfolder)}`
  }
  return null
}

export function imageItemFromPath(filePath: string, sourceType: 'input' | 'output' | 'local' = 'input'): ImageItem {
  return {
    source_type: sourceType,
    file_path: filePath,
    file_name: filePath.split('/').pop() ?? filePath,
  }
}

export function imageItemFromUrl(url: string): ImageItem {
  return {
    source_type: 'url',
    url,
    file_name: url.split('/').pop() ?? url,
  }
}

/** Build a CSS tiled background from an array of images. */
export function tiledImageBackground(images: ImageItem[]): React.CSSProperties {
  const srcs = images.map((img) => getImageSrc(img)).filter((s): s is string => s !== null)
  if (srcs.length === 0) return {}
  // Use double-quoted url() to safely handle URLs that contain single quotes.
  // Encode any literal double quotes in the URL as %22 (valid URL encoding).
  return {
    backgroundImage: srcs.map((s) => `url("${s.replaceAll('"', '%22')}")`).join(', '),
    backgroundRepeat: 'repeat',
    backgroundSize: srcs.map(() => 'auto 100%').join(', '),
    backgroundPosition: '0 0',
  }
}
