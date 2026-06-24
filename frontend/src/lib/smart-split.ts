import { uuid } from '@/lib/uuid'
import { throwIfMissingModelResponse } from '@/lib/model-download'
import type { MultiTrackSegment, TrackData } from '@/types/multitrack'

export interface SmartSplitResult {
  ranges: Array<[number, number]>
}

function parseSmartSplitResult(value: unknown): SmartSplitResult {
  if (!value || typeof value !== 'object') throw new Error('Invalid smart split response')
  const result = value as Partial<SmartSplitResult> & { error?: unknown }
  if (typeof result.error === 'string') throw new Error(result.error)
  if (!Array.isArray(result.ranges)) {
    throw new Error('Invalid smart split response')
  }
  const ranges = result.ranges.map((range) => {
    if (!Array.isArray(range) || range.length !== 2 || !range.every(Number.isFinite)) {
      throw new Error('Invalid shot range in smart split response')
    }
    return [Number(range[0]), Number(range[1])] as [number, number]
  })
  return { ranges }
}

export async function requestSmartSplit(segment: MultiTrackSegment, fps: number): Promise<SmartSplitResult> {
  const response = await fetch('/easy-media/video/smart-split', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      source_type: segment.content.source_type ?? 'input',
      file_path: segment.content.file_path,
      local_path: segment.content.local_path,
      url: segment.content.url,
      fps,
    }),
  })
  let payload: unknown
  try {
    payload = await response.json()
  } catch (error) {
    throw new Error(`Smart split returned invalid JSON: ${String(error)}`)
  }
  if (!response.ok) {
    throwIfMissingModelResponse(payload)
    const message = payload && typeof payload === 'object' && 'error' in payload
      ? String((payload as { error: unknown }).error)
      : `Smart split failed (${response.status})`
    throw new Error(message)
  }
  return parseSmartSplitResult(payload)
}

function splitSegment(
  segment: MultiTrackSegment,
  boundaries: number[],
  originStartFrame?: number,
): MultiTrackSegment[] {
  const points = [segment.start_frame, ...boundaries, segment.end_frame]
  return points.slice(0, -1).map((startFrame, index) => ({
    ...segment,
    id: index === 0 ? segment.id : uuid(),
    start_frame: startFrame,
    end_frame: points[index + 1],
    ...(originStartFrame === undefined ? {} : { origin_start_frame: originStartFrame }),
    content: {
      ...segment.content,
      images: segment.content.images?.map((image) => ({ ...image })),
    },
  }))
}

function findVideoSegment(data: TrackData, videoSegmentId: string): MultiTrackSegment | undefined {
  return data.tracks
    .find((track) => track.type === 'video' && track.segments.some((segment) => segment.id === videoSegmentId))
    ?.segments.find((segment) => segment.id === videoSegmentId)
}

function smartSplitBoundaries(
  source: MultiTrackSegment,
  result: SmartSplitResult,
): number[] {
  const sourceOriginStart = source.origin_start_frame ?? source.start_frame
  return [...new Set(result.ranges
    .map((range) => range[1] + sourceOriginStart)
    .filter((frame) => frame > source.start_frame && frame < source.end_frame))]
    .sort((left, right) => left - right)
}

export function hasMatchingTaskSegment(data: TrackData, videoSegmentId: string): boolean {
  const source = findVideoSegment(data, videoSegmentId)
  if (!source) return false
  return data.tracks.some((track) => track.type === 'task' && track.segments.some((segment) => (
    segment.start_frame === source.start_frame && segment.end_frame === source.end_frame
  )))
}

export function applySmartSplit(
  data: TrackData,
  videoSegmentId: string,
  result: SmartSplitResult,
): TrackData {
  const source = findVideoSegment(data, videoSegmentId)
  if (!source) return data

  const duration = source.end_frame - source.start_frame
  const sourceOriginStart = source.origin_start_frame ?? source.start_frame
  const boundaries = smartSplitBoundaries(source, result)
  if (duration <= 1 || boundaries.length === 0) return data

  const tracks = data.tracks.map((track) => ({
    ...track,
    segments: track.segments.flatMap((segment) => {
      const isSource = segment.id === videoSegmentId
      const isLinkedTask = track.type === 'task'
        && segment.start_frame === source.start_frame
        && segment.end_frame === source.end_frame
      if (isSource) return splitSegment(segment, boundaries, sourceOriginStart)
      return isLinkedTask ? splitSegment(segment, boundaries) : [segment]
    }).sort((left, right) => left.start_frame - right.start_frame),
  }))
  return { ...data, tracks }
}

export function applySmartSplitToMatchingTasks(
  data: TrackData,
  videoSegmentId: string,
  result: SmartSplitResult,
): TrackData {
  const source = findVideoSegment(data, videoSegmentId)
  if (!source) return data
  const boundaries = smartSplitBoundaries(source, result)
  if (boundaries.length === 0) return data

  return {
    ...data,
    tracks: data.tracks.map((track) => ({
      ...track,
      segments: track.segments.flatMap((segment) => (
        track.type === 'task'
          && segment.start_frame === source.start_frame
          && segment.end_frame === source.end_frame
          ? splitSegment(segment, boundaries)
          : [segment]
      )).sort((left, right) => left.start_frame - right.start_frame),
    })),
  }
}

export function splitTrackSegmentAtFrame(
  data: TrackData,
  segmentId: string,
  splitFrame: number,
): TrackData {
  const target = data.tracks.flatMap((track) => track.segments).find((segment) => segment.id === segmentId)
  const frame = Math.round(splitFrame)
  if (!target || frame <= target.start_frame || frame >= target.end_frame) return data
  const keepsSourceOffset = target.content.media_type === 'video' || target.content.media_type === 'audio'
  const originStart = keepsSourceOffset ? target.origin_start_frame ?? target.start_frame : undefined

  return {
    ...data,
    tracks: data.tracks.map((track) => ({
      ...track,
      segments: track.segments.flatMap((segment) => (
        segment.id === segmentId ? splitSegment(segment, [frame], originStart) : [segment]
      )).sort((left, right) => left.start_frame - right.start_frame),
    })),
  }
}
