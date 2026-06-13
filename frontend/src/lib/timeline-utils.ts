import type { ImageItem, TimelineData, Track, TrackType } from '@/types/timeline'
import { uuid } from './uuid'
import { traceToRootSourceViaLink } from './graph-utils'

// ---------------------------------------------------------------------------
// Slot item types (used by MediaSelector's slot tab)
// ---------------------------------------------------------------------------

export interface SlotItem {
  /** Display label, e.g. "image", "image_0", "audio_1" */
  label: string
  /** Encoded value passed to onChange, e.g. "__slot__:image_0" */
  value: string
  /** Preview image src for image slots */
  img?: string
  /** Audio name for audio slots */
  audio_name?: string
}

/**
 * Inspect a ComfyUI node's optional inputs and the upstream graph to build
 * a list of selectable slot items for the given media type.
 *
 * Logic:
 * - Find node.inputs whose type matches mediaType (case-insensitive)
 * - For each connected input (link !== null):
 *   - Trace the link to the source node's output slot
 *   - If the source output is_list → enumerate this node's image_N / audio_N inputs
 *   - Otherwise → one item for that input name
 */
export function computeSlotItems(
  node: any,
  app: any,
  mediaType: 'image' | 'audio',
): SlotItem[] {
  if (!node?.inputs || !app?.graph) return []

  const result: SlotItem[] = []
  const type = mediaType.toUpperCase()

  for (const input of node.inputs as Array<{ name: string; type: string; link: number | null }>) {
    if ((input.type ?? '').toUpperCase() !== type) continue
    if (input.link === null || input.link === undefined) continue

    const link = app.graph.links[input.link]
    if (!link) continue

    const sourceNodeId: number = link.origin_id
    const sourceSlotIdx: number = link.origin_slot
    const sourceNode = app.graph.getNodeById(sourceNodeId)
    if (!sourceNode) continue
    const sourceOutputSlot = sourceNode.outputs?.[sourceSlotIdx]
    const isList = sourceOutputSlot.shape == 6
    if (isList) {
      // Collect this node's indexed inputs: image_0, image_1, ... / audio_0, audio_1, ...
      const indexedInputs = (sourceNode.inputs as Array<{ name: string; type: string; link: number | null, isConnected: boolean }>)
        .filter((inp) => inp.isConnected)
        .sort((a, b) => {
          const ia = parseInt(a.name.split('_').pop() ?? '0', 10)
          const ib = parseInt(b.name.split('_').pop() ?? '0', 10)
          return ia - ib
        })
      if (indexedInputs.length > 0) {
        for (const inp of indexedInputs) {
          let imgsrc: string | undefined
          let audio_name: string | undefined
          if (inp.link) {
            const currentNodeId = traceToRootSourceViaLink(inp.link, app.graph)
            if (currentNodeId !== null) {
              const currentNode = app.graph.getNodeById(currentNodeId)
              if (mediaType == 'image') {
                imgsrc = currentNode?.imgs?.[0]?.currentSrc
              } else if (mediaType == 'audio') {
                audio_name = (currentNode.widgets_values?.[0] as string) ?? (currentNode.widgets?.[0]?.value as string)
              }
            }
          }
          result.push({ label: inp.name, img: imgsrc, audio_name: audio_name, value: `__slot__:${inp.name}` })
        }
      } else {
        // Fallback: represent base input as a single list reference
      }
    } else {
      // Non-list: sourceNode is the output node (e.g., VAEDecode, LoadImage)
      // Try to trace to find the actual image source
      let imgsrc: string | undefined
      let audio_name: string | undefined
      const currentNodeId = traceToRootSourceViaLink(link, app.graph)
      const node = currentNodeId !== null
        ? app.graph.getNodeById(currentNodeId)
        : sourceNode
      if (mediaType == 'image') {
        imgsrc = node?.imgs?.[0]?.currentSrc
      } else if (mediaType == 'audio' && node?.type == 'LoadAudio') {
        audio_name = (node.widgets_values?.[0] as string) ?? (node.widgets?.[0]?.value as string)
      }
      result.push({ label: input.name, img: imgsrc, audio_name: audio_name, value: `__slot__:${input.name}` })
    }

    // Only process the first connected input of this type
    break
  }

  return result
}

export function scaleImageItemsToDuration(
  images: ImageItem[],
  oldDuration: number,
  newDuration: number,
): ImageItem[] {
  if (oldDuration <= 0 || newDuration <= 0) return images

  return images.map((image) => {
    if (image.start_frame === undefined || image.end_frame === undefined) {
      return image
    }

    const start = Math.max(
      0,
      Math.min(newDuration - 1, Math.round((image.start_frame / oldDuration) * newDuration)),
    )
    const exclusiveEnd = Math.round(((image.end_frame + 1) / oldDuration) * newDuration)
    const end = Math.max(start, Math.min(newDuration - 1, exclusiveEnd - 1))

    return {
      ...image,
      start_frame: start,
      end_frame: end,
    }
  })
}

/**
 * Converts seconds to frame count using the formula from the spec:
 *   frames = ceil(seconds * frameRate / 4) * 4 + 1
 */
export function secondsToFrames(seconds: number, frameRate: number): number {
  return Math.ceil((seconds * frameRate) / 4) * 4 + 1
}

/**
 * Converts seconds to frame count for audio playback (no +1 offset).
 * Use this when computing audio segment duration or seeking.
 *   frames = ceil(seconds * frameRate)
 */
export function secondsToAudioFrames(seconds: number, frameRate: number): number {
  return Math.ceil(seconds * frameRate)
}

/**
 * Converts frame count back to seconds.
 *   seconds = (frames - 1) / frameRate
 */
export function framesToSeconds(frames: number, frameRate: number): number {
  return (frames - 1) / frameRate
}

/**
 * Format a frame index as a display string.
 * mode='frames' → "121f"
 * mode='seconds' → "5.00s"
 */
export function formatTime(
  frames: number,
  frameRate: number,
  mode: 'frames' | 'seconds',
): string {
  if (mode === 'frames') return `${frames}f`
  const secs = framesToSeconds(frames, frameRate)
  return `${secs.toFixed(2)}s`
}

/**
 * Parse a user-entered time string to frames.
 * Accepts "121" (treated as frames), "121f", "5s", "5.0s".
 * Returns NaN if unparseable.
 */
export function parseTimeInput(input: string, frameRate: number): number {
  const trimmed = input.trim()
  if (trimmed.endsWith('s')) {
    const secs = Number.parseFloat(trimmed.slice(0, -1))
    if (Number.isNaN(secs)) return Number.NaN
    return secondsToFrames(secs, frameRate)
  }
  const framesStr = trimmed.endsWith('f') ? trimmed.slice(0, -1) : trimmed
  const frames = Number.parseInt(framesStr, 10)
  return Number.isNaN(frames) ? Number.NaN : frames
}

/** Default colors per track type */
export const TRACK_DEFAULT_COLORS: Record<TrackType, string> = {
  audio: '#34d399',
  prompt: '#a78bfa',
  image: '#fb923c',
  video: '#60a5fa',
  maintain: 'var(--muted)',
}

/** Build a default empty TimelineData with one maintain track and one audio track */
export function createDefaultTimelineData(t?: (path: string, params?: Record<string, string | number>) => string): TimelineData {
  const _t = t ?? ((_: string) => '')
  const total_length = 121
  const frame_rate = 24

  const tracks: Track[] = [
    {
      id: uuid(),
      name: _t('timeline.defaultTrackName', { n: 1 }),
      type: 'maintain',
      color: TRACK_DEFAULT_COLORS.maintain,
      muted: false,
      locked: false,
      segments: [],
    },
    {
      id: uuid(),
      name: _t('timeline.defaultAudioTrackName', { n: 1 }),
      type: 'audio',
      color: TRACK_DEFAULT_COLORS.audio,
      muted: false,
      locked: false,
      segments: [],
    },
  ]

  return { tracks, total_length, frame_rate }
}

/**
 * Returns the pixel left offset and width for a segment on the track content area.
 * @param startFrame  segment start frame (inclusive)
 * @param endFrame    segment end frame (inclusive)
 * @param totalFrames total timeline length in frames
 * @param areaWidth   pixel width of the track content area
 */
export function segmentPixelRect(
  startFrame: number,
  endFrame: number,
  totalFrames: number,
  areaWidth: number,
): { left: number; width: number } {
  const left = (startFrame / (totalFrames - 1)) * areaWidth
  const right = ((endFrame + 1) / (totalFrames - 1)) * areaWidth
  return { left, width: Math.max(right - left, 2) }
}
