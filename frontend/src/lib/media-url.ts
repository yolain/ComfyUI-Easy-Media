import type { AudioContent } from '@/types/timeline'

type ViewSourceType = 'input' | 'output'

export interface ViewableMediaContent {
  source_type: 'preset' | 'input' | 'output' | 'local' | 'url' | 'slot'
  file_path?: string
  local_path?: string
  url?: string
  slot_name?: string
}

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

export function mediaContentToViewUrl(content: ViewableMediaContent): string | null {
  if (content.url) return content.url
  if (content.local_path) return `file://${content.local_path}`
  if (content.source_type !== 'input' && content.source_type !== 'output') return null
  if (!content.file_path) return null

  const typeParam = content.source_type === 'output' ? 'output' : 'input'
  return mediaPathToViewUrl(content.file_path, typeParam)
}

export function audioContentToViewUrl(content: AudioContent): string | null {
  return mediaContentToViewUrl(content)
}
