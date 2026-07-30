import type { MultiTrackSegment, MultiTrackSubtitleStyle } from '@/types/multitrack'

interface SubtitleSrtCue {
  startSeconds: number
  endSeconds: number
  text: string
}

interface CreateSubtitleSegmentsFromSrtOptions {
  color: string
  createId: (index: number) => string
  frameRate: number
  startFrame: number
  subtitleStyle: MultiTrackSubtitleStyle
}

const SRT_TIMESTAMP_PATTERN = /^(\d+):([0-5]\d):([0-5]\d)[,.](\d{3})\s*-->\s*(\d+):([0-5]\d):([0-5]\d)[,.](\d{3})(?:\s+.*)?$/

function timestampPartsToSeconds(parts: string[]): number {
  const [hours, minutes, seconds, milliseconds] = parts.map(Number)
  return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000
}

function parseSrtCues(value: string): SubtitleSrtCue[] {
  const lines = value.replace(/^\uFEFF/, '').replaceAll('\r\n', '\n').replaceAll('\r', '\n').split('\n')
  const cues: SubtitleSrtCue[] = []

  for (let index = 0; index < lines.length; index += 1) {
    const match = lines[index]?.trim().match(SRT_TIMESTAMP_PATTERN)
    if (!match) continue

    const startSeconds = timestampPartsToSeconds(match.slice(1, 5))
    const endSeconds = timestampPartsToSeconds(match.slice(5, 9))
    const textLines: string[] = []
    index += 1
    while (index < lines.length && lines[index]?.trim() !== '') {
      textLines.push(lines[index] ?? '')
      index += 1
    }
    const text = textLines.join('\n').trim()
    if (text && endSeconds > startSeconds) {
      cues.push({ startSeconds, endSeconds, text })
    }
  }

  return cues
}

export function createSubtitleSegmentsFromSrt(
  value: string,
  options: CreateSubtitleSegmentsFromSrtOptions,
): MultiTrackSegment[] {
  const cues = parseSrtCues(value)
    .sort((left, right) => left.startSeconds - right.startSeconds || left.endSeconds - right.endSeconds)
  if (cues.length === 0) throw new Error('No valid SRT subtitle cues found')

  const cueStartSeconds = cues[0].startSeconds
  return cues.map((cue, index) => {
    const startFrame = options.startFrame + Math.round((cue.startSeconds - cueStartSeconds) * options.frameRate)
    const endFrame = Math.max(
      startFrame + 1,
      options.startFrame + Math.round((cue.endSeconds - cueStartSeconds) * options.frameRate),
    )
    return {
      id: options.createId(index),
      start_frame: startFrame,
      end_frame: endFrame,
      color: options.color,
      content: {
        media_type: 'subtitle',
        text: cue.text,
        subtitle_style: { ...options.subtitleStyle },
      },
    }
  })
}
