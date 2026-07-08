import { throwIfMissingModelResponse } from '@/lib/model-download'
import {
  calculateTotalLength,
  MULTITRACK_DEFAULT_VOLUME_DB,
  MULTITRACK_TRACK_COLORS,
  secondsToFrame,
} from '@/lib/multitrack-utils'
import { uuid } from '@/lib/uuid'
import type { MultiTrack, MultiTrackSegment, MultiTrackSourceType, TrackData } from '@/types/multitrack'

export interface SubtitleSpeechSettings {
  model: 'VoxCPM2'
  prompt: string
  cfg: number
  steps: number
  referenceAudio: string
  referenceAudioSourceType?: Extract<MultiTrackSourceType, 'input' | 'output' | 'local'>
}

export const DEFAULT_SUBTITLE_SPEECH_SETTINGS: SubtitleSpeechSettings = {
  model: 'VoxCPM2',
  prompt: '',
  cfg: 2,
  steps: 10,
  referenceAudio: '',
  referenceAudioSourceType: 'input',
}

export interface SubtitleSpeechRequest extends SubtitleSpeechSettings {
  text: string
}

export interface SubtitleSpeechResult {
  filePath: string
  sourceType: 'output'
  absolutePath: string
  message: string
  duration?: number
}

export interface ApplySubtitleSpeechAudioOptions {
  subtitleSegmentId: string
  startFrame: number
  endFrame: number
  filePath: string
  duration?: number
}

function parseSubtitleSpeechResult(value: unknown): SubtitleSpeechResult {
  if (!value || typeof value !== 'object') throw new Error('Invalid subtitle speech response')
  const result = value as {
    error?: unknown
    file_path?: unknown
    source_type?: unknown
    absolute_path?: unknown
    message?: unknown
    duration?: unknown
  }
  if (typeof result.error === 'string') throw new Error(result.error)
  if (typeof result.file_path !== 'string' || result.source_type !== 'output') {
    throw new Error('Invalid subtitle speech response')
  }
  return {
    filePath: result.file_path,
    sourceType: 'output',
    absolutePath: typeof result.absolute_path === 'string' ? result.absolute_path : '',
    message: typeof result.message === 'string' ? result.message : '',
    duration: typeof result.duration === 'number' && Number.isFinite(result.duration)
      ? result.duration
      : undefined,
  }
}

export async function requestSubtitleSpeechAudio(request: SubtitleSpeechRequest): Promise<SubtitleSpeechResult> {
  const response = await fetch('/easy-media/subtitles/speech', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      text: request.text,
      model: request.model,
      prompt: request.prompt,
      cfg: request.cfg,
      steps: request.steps,
      reference_audio_path: request.referenceAudio || undefined,
      reference_audio_source_type: request.referenceAudio
        ? request.referenceAudioSourceType ?? 'input'
        : undefined,
    }),
  })
  let payload: unknown
  try {
    payload = await response.json()
  } catch (error) {
    throw new Error(`Subtitle speech returned invalid JSON: ${String(error)}`)
  }
  if (!response.ok) {
    throwIfMissingModelResponse(payload)
    const message = payload && typeof payload === 'object' && 'error' in payload
      ? String((payload as { error: unknown }).error)
      : `Subtitle speech failed (${response.status})`
    throw new Error(message)
  }
  return parseSubtitleSpeechResult(payload)
}

function rangesOverlap(startA: number, endA: number, startB: number, endB: number): boolean {
  return Math.max(startA, startB) < Math.min(endA, endB)
}

function createSpeechAudioTrack(index: number, segment: MultiTrackSegment): MultiTrack {
  return {
    id: uuid(),
    name: `Audio ${index}`,
    type: 'audio',
    color: MULTITRACK_TRACK_COLORS.audio,
    muted: false,
    solo: false,
    volume_db: MULTITRACK_DEFAULT_VOLUME_DB,
    locked: false,
    segments: [segment],
  }
}

export function applySubtitleSpeechAudio(
  data: TrackData,
  options: ApplySubtitleSpeechAudioOptions,
): TrackData {
  if (options.endFrame <= options.startFrame) return data
  const durationFrames = options.endFrame - options.startFrame
  const contentDuration = options.duration ?? durationFrames / data.frame_rate
  const speechDurationFrames = options.duration && Number.isFinite(options.duration)
    ? Math.max(1, secondsToFrame(options.duration, data.frame_rate))
    : durationFrames
  const endFrame = options.startFrame + speechDurationFrames
  const nextSegment: MultiTrackSegment = {
    id: uuid(),
    start_frame: options.startFrame,
    end_frame: endFrame,
    color: MULTITRACK_TRACK_COLORS.audio,
    content: {
      media_type: 'audio',
      source_type: 'output',
      file_path: options.filePath,
      file_name: options.filePath.split(/[\\/]/).pop() ?? options.filePath,
      duration: contentDuration,
      muted: false,
      volume_db: MULTITRACK_DEFAULT_VOLUME_DB,
    },
  }

  let inserted = false
  const tracks = data.tracks.map((track) => {
    if (track.type !== 'audio' || inserted) return track
    const overlaps = track.segments.some((segment) => (
      rangesOverlap(options.startFrame, endFrame, segment.start_frame, segment.end_frame)
    ))
    if (overlaps) return track
    inserted = true
    return {
      ...track,
      segments: [...track.segments, nextSegment]
        .sort((left, right) => left.start_frame - right.start_frame),
    }
  })

  const nextTracks = inserted
    ? tracks
    : [
        ...tracks,
        createSpeechAudioTrack(data.tracks.filter((track) => track.type === 'audio').length, nextSegment),
      ]

  return {
    ...data,
    tracks: nextTracks,
    total_length: calculateTotalLength(nextTracks, data.frame_rate),
  }
}
