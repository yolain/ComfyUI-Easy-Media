import { uuid } from '@/lib/uuid'
import type { MultiTrackSourceType, MultiTrackTaskImage } from '@/types/multitrack'

export const MAX_TASK_IMAGES = 9
export const MULTIPLE_MEDIA_SEPARATOR = '|MULTIPLE|'

export function taskImagesFromContent(images: MultiTrackTaskImage[] | undefined): MultiTrackTaskImage[] {
  return Array.isArray(images) ? images : []
}

export function createTaskImage(filePath: string, source: MultiTrackSourceType): MultiTrackTaskImage {
  const normalizedSource = filePath.startsWith('__slot__:') ? 'slot' : source
  const slotName = normalizedSource === 'slot' ? filePath.replace(/^__slot__:/, '') : undefined
  const fileName = slotName ?? filePath.split(/[\\/]/).pop() ?? filePath
  return {
    id: uuid(),
    source_type: normalizedSource,
    file_path: normalizedSource === 'input' || normalizedSource === 'output' ? filePath : undefined,
    local_path: normalizedSource === 'local' ? filePath : undefined,
    url: normalizedSource === 'url' ? filePath : undefined,
    slot_name: slotName,
    file_name: fileName,
  }
}

export function splitSelectedTaskMedia(value: string): string[] {
  return value.split(MULTIPLE_MEDIA_SEPARATOR).filter((item) => item.length > 0)
}

export async function uploadTaskImageFile(file: File): Promise<MultiTrackTaskImage> {
  const formData = new FormData()
  formData.append('image', file)
  const response = await fetch('/upload/image', {
    method: 'POST',
    body: formData,
  })
  if (!response.ok) {
    throw new Error(`Failed to upload image: ${file.name}`)
  }
  const result = await response.json() as { name?: string; subfolder?: string }
  const name = result.name ?? file.name
  const subfolder = result.subfolder ?? ''
  return {
    id: uuid(),
    source_type: 'input',
    file_path: subfolder ? `${subfolder}/${name}` : name,
    file_name: name,
  }
}
