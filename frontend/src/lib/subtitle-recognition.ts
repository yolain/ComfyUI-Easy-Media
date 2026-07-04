import { throwIfMissingModelResponse } from '@/lib/model-download'
import {
  calculateTotalLength,
  MULTITRACK_DEFAULT_VOLUME_DB,
  secondsToFrame,
} from '@/lib/multitrack-utils'
import { uuid } from '@/lib/uuid'
import type {
  MultiTrack,
  MultiTrackSegment,
  MultiTrackSubtitleStyle,
  TrackData,
} from '@/types/multitrack'

export const MULTITRACK_SUBTITLE_COLOR = '#9D4937'

export const DEFAULT_SUBTITLE_STYLE: MultiTrackSubtitleStyle = {
  font_size: 12,
  color: '#ffffff',
  outline_color: '#000000',
  background_color: 'rgba(0, 0, 0, 0)',
  x: 0.125,
  y: 0.8,
  width: 0.75,
}

export interface SubtitleRecognitionSegment {
  start: number
  end: number
  text: string
}

export interface SubtitleRecognitionResult {
  segments: SubtitleRecognitionSegment[]
}

function parseRecognitionSegment(value: unknown): SubtitleRecognitionSegment {
  if (!value || typeof value !== 'object') throw new Error('Invalid subtitle segment')
  const segment = value as Partial<Record<keyof SubtitleRecognitionSegment, unknown>>
  const start = Number(segment.start)
  const end = Number(segment.end)
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    throw new Error('Invalid subtitle timestamp')
  }
  if (typeof segment.text !== 'string') throw new Error('Invalid subtitle text')
  return { start, end, text: segment.text }
}

function parseSubtitleRecognitionResult(value: unknown): SubtitleRecognitionResult {
  if (!value || typeof value !== 'object') throw new Error('Invalid subtitle recognition response')
  const result = value as Partial<SubtitleRecognitionResult> & { error?: unknown }
  if (typeof result.error === 'string') throw new Error(result.error)
  if (!Array.isArray(result.segments)) throw new Error('Invalid subtitle recognition response')
  return {
    segments: result.segments.map(parseRecognitionSegment),
  }
}

function findMediaSegment(data: TrackData, segmentId: string): MultiTrackSegment | undefined {
  return data.tracks
    .find((track) => (
      (track.type === 'video' || track.type === 'audio') &&
      track.segments.some((segment) => segment.id === segmentId)
    ))
    ?.segments.find((segment) => segment.id === segmentId)
}

function createSubtitleTrack(index: number): MultiTrack {
  return {
    id: uuid(),
    name: `Subtitle ${index + 1}`,
    type: 'subtitle',
    color: MULTITRACK_SUBTITLE_COLOR,
    muted: false,
    solo: false,
    volume_db: MULTITRACK_DEFAULT_VOLUME_DB,
    locked: false,
    segments: [],
  }
}

export async function requestSubtitleRecognition(
  segment: MultiTrackSegment,
  fps: number,
): Promise<SubtitleRecognitionResult> {
  const response = await fetch('/easy-media/subtitles/recognize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      media_type: segment.content.media_type,
      source_type: segment.content.source_type ?? 'input',
      file_path: segment.content.file_path,
      local_path: segment.content.local_path,
      url: segment.content.url,
      fps,
      start_frame: segment.start_frame,
      end_frame: segment.end_frame,
      origin_start_frame: segment.origin_start_frame,
    }),
  })
  let payload: unknown
  try {
    payload = await response.json()
  } catch (error) {
    throw new Error(`Subtitle recognition returned invalid JSON: ${String(error)}`)
  }
  if (!response.ok) {
    throwIfMissingModelResponse(payload)
    const message = payload && typeof payload === 'object' && 'error' in payload
      ? String((payload as { error: unknown }).error)
      : `Subtitle recognition failed (${response.status})`
    throw new Error(message)
  }
  return parseSubtitleRecognitionResult(payload)
}

export function applySubtitleRecognition(
  data: TrackData,
  sourceSegmentId: string,
  result: SubtitleRecognitionResult,
): TrackData {
  const source = findMediaSegment(data, sourceSegmentId)
  if (!source || result.segments.length === 0) return data

  const recognizedSegments = result.segments.flatMap((item): MultiTrackSegment[] => {
    const startFrame = Math.max(
      source.start_frame,
      source.start_frame + secondsToFrame(item.start, data.frame_rate),
    )
    const endFrame = Math.min(
      source.end_frame,
      source.start_frame + secondsToFrame(item.end, data.frame_rate),
    )
    if (endFrame <= startFrame) return []
    return [{
      id: uuid(),
      start_frame: startFrame,
      end_frame: endFrame,
      color: MULTITRACK_SUBTITLE_COLOR,
      content: {
        media_type: 'subtitle',
        text: item.text,
        subtitle_style: { ...DEFAULT_SUBTITLE_STYLE },
      },
    }]
  })
  if (recognizedSegments.length === 0) return data

  let addedTrack = false
  const updatedTracks = data.tracks.map((track) => {
    if (track.type !== 'subtitle') return track
    if (addedTrack) return track
    addedTrack = true
    return {
      ...track,
      color: track.color || MULTITRACK_SUBTITLE_COLOR,
      segments: [...track.segments, ...recognizedSegments]
        .sort((left, right) => left.start_frame - right.start_frame),
    }
  })

  const tracks = addedTrack
    ? updatedTracks
    : [
        ...data.tracks,
        {
          ...createSubtitleTrack(data.tracks.filter((track) => track.type === 'subtitle').length),
          segments: recognizedSegments,
        },
      ]

  return {
    ...data,
    tracks,
    total_length: calculateTotalLength(tracks, data.frame_rate),
  }
}
