import type { AudioContent } from '@/types/timeline'

type ViewSourceType = 'input' | 'output'

function splitMediaPath(filePath: string): { filename: string; subfolder: string } {
  const lastSlash = Math.max(filePath.lastIndexOf('/'), filePath.lastIndexOf('\\'))
  return {
    filename: lastSlash >= 0 ? filePath.slice(lastSlash + 1) : filePath,
    subfolder: lastSlash >= 0 ? filePath.slice(0, lastSlash) : '',
  }
}

export function mediaPathToViewUrl(filePath: string, sourceType: ViewSourceType): string {
  const { filename, subfolder } = splitMediaPath(filePath)
  return `/view?filename=${encodeURIComponent(filename)}&type=${sourceType}&subfolder=${encodeURIComponent(subfolder)}`
}

export function audioContentToViewUrl(content: AudioContent): string | null {
  if (content.url) return content.url
  if (content.local_path) return `file://${content.local_path}`
  if (!content.file_path) return null

  const typeParam = content.source_type === 'output' ? 'output' : 'input'
  return mediaPathToViewUrl(content.file_path, typeParam)
}
