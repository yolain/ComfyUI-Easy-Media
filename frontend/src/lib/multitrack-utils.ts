import type {
  MultiTrack,
  MultiTrackSegment,
  MultiTrackSegmentContent,
  MultiTrackSourceType,
  MultiTrackTaskMode,
  MultiTrackType,
  TrackData,
} from '@/types/multitrack'
import { uuid } from './uuid'

export const MULTITRACK_DEFAULT_FRAME_RATE = 24
export const MULTITRACK_DEFAULT_TOTAL_LENGTH = 120
export const MULTITRACK_MIN_DURATION_SECONDS = 5
export const MULTITRACK_TASK_MODES = ['default', 'ref', 'edit'] as const
export const MULTITRACK_DEFAULT_TASK_MODE: MultiTrackTaskMode = 'default'
export const MULTITRACK_DEFAULT_VOLUME_DB = 0
export const MULTITRACK_MIN_VOLUME_DB = -20
export const MULTITRACK_MAX_VOLUME_DB = 6
export const MULTITRACK_FRAME_RATE_OPTIONS = [16, 20, 24, 25, 30, 50, 60] as const

export function createMultiTrackVideoContent(
  filePath: string,
  sourceType: MultiTrackSourceType,
): MultiTrackSegmentContent & { source_type: MultiTrackSourceType } {
  const normalizedSourceType = sourceType === 'input' && /^https?:\/\//i.test(filePath)
    ? 'url'
    : sourceType
  return {
    media_type: 'video',
    source_type: normalizedSourceType,
    file_path: normalizedSourceType === 'input' || normalizedSourceType === 'output' ? filePath : undefined,
    local_path: normalizedSourceType === 'local' ? filePath : undefined,
    url: normalizedSourceType === 'url' ? filePath : undefined,
    file_name: filePath.split(/[\\/]/).pop() ?? filePath,
  }
}

export function createMultiTrackAudioContent(
  filePath: string,
  sourceType: MultiTrackSourceType,
): MultiTrackSegmentContent & { source_type: MultiTrackSourceType } {
  const normalizedSourceType = filePath.startsWith('__slot__:')
    ? 'slot'
    : sourceType === 'input' && /^https?:\/\//i.test(filePath)
      ? 'url'
      : sourceType
  return {
    media_type: 'audio',
    source_type: normalizedSourceType,
    file_path: normalizedSourceType === 'input' || normalizedSourceType === 'output' ? filePath : undefined,
    local_path: normalizedSourceType === 'local' ? filePath : undefined,
    url: normalizedSourceType === 'url' ? filePath : undefined,
    slot_name: normalizedSourceType === 'slot' ? filePath.replace(/^__slot__:/, '') : undefined,
    file_name: normalizedSourceType === 'slot'
      ? filePath.replace(/^__slot__:/, '')
      : filePath.split(/[\\/]/).pop() ?? filePath,
    muted: false,
    volume_db: MULTITRACK_DEFAULT_VOLUME_DB,
  }
}

export const MULTITRACK_TRACK_COLORS: Record<MultiTrack['type'], string> = {
  task: 'var(--multitrack-task-bg)',
  video: 'var(--primary)',
  audio: 'var(--highlight)',
  subtitle: 'var(--accent)',
}

export function getMultiTrackTaskModeLabel(
  mode: MultiTrackTaskMode,
  t: (path: string) => string,
): string {
  return t(`multitrackTaskModes.${mode}`)
}

export function secondsToFrame(time: number, frameRate: number): number {
  if (frameRate <= 0) return 0
  return Math.round(time * frameRate)
}

export function frameToSeconds(frame: number, frameRate: number): number {
  if (frameRate <= 0) return 0
  return frame / frameRate
}

export function clampMultiTrackVolumeDb(value: number): number {
  if (!Number.isFinite(value)) return MULTITRACK_DEFAULT_VOLUME_DB
  return Math.max(MULTITRACK_MIN_VOLUME_DB, Math.min(MULTITRACK_MAX_VOLUME_DB, value))
}

export function multiTrackDbToLinearGain(value: number): number {
  return 10 ** ((Number.isFinite(value) ? value : MULTITRACK_DEFAULT_VOLUME_DB) / 20)
}

function normalizedVolumeDb(volumeDb: unknown): number {
  const db = finiteNumber(volumeDb)
  return db === null ? MULTITRACK_DEFAULT_VOLUME_DB : clampMultiTrackVolumeDb(db)
}

export function formatMultiTrackDurationTimecode(duration: number, frameRate: number): string {
  const safeFrameRate = frameRate > 0 ? frameRate : MULTITRACK_DEFAULT_FRAME_RATE
  const totalFrames = Math.max(0, Math.round(duration * safeFrameRate))
  const totalSeconds = Math.floor(totalFrames / safeFrameRate)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  const frames = totalFrames % safeFrameRate
  return [minutes, seconds, frames].map((value) => value.toString().padStart(2, '0')).join(':')
}

export function parseMultiTrackDurationTimecode(value: string, frameRate: number): number | null {
  if (!Number.isInteger(frameRate) || frameRate <= 0) return null
  const match = /^(\d{2,}):([0-5]\d):(\d{2})$/.exec(value.trim())
  if (!match) return null

  const minutes = Number.parseInt(match[1], 10)
  const seconds = Number.parseInt(match[2], 10)
  const frames = Number.parseInt(match[3], 10)
  if (frames >= frameRate) return null

  const duration = (minutes * 60) + seconds + (frames / frameRate)
  return duration > 0 ? duration : null
}

export function formatMultiTrackTime(
  frame: number,
  options: { frameRate?: number; showFrames?: boolean } = {},
): string {
  const frameRate = options.frameRate && options.frameRate > 0 ? options.frameRate : 0
  const totalFrames = Math.max(0, options.showFrames ? Math.round(frame) : Math.floor(frame))
  const totalSeconds = options.showFrames && frameRate > 0
    ? Math.floor(totalFrames / frameRate)
    : frameRate > 0
      ? Math.floor(totalFrames / frameRate)
      : Math.floor(totalFrames)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const wholeSeconds = totalSeconds % 60
  const minutesLabel = minutes.toString().padStart(2, '0')
  const secondsLabel = wholeSeconds.toString().padStart(2, '0')

  if (options.showFrames && frameRate > 0) {
    const frameLabel = (totalFrames % frameRate).toString().padStart(2, '0')
    if (hours > 0) {
      return `${hours.toString().padStart(2, '0')}:${minutesLabel}:${secondsLabel}:${frameLabel}`
    }
    return `${minutesLabel}:${secondsLabel}:${frameLabel}`
  }

  if (hours > 0) {
    return `${hours.toString().padStart(2, '0')}:${minutesLabel}:${secondsLabel}`
  }
  return `${minutesLabel}:${secondsLabel}`
}

export function snapTimeToFrame(time: number, frameRate: number): number {
  void frameRate
  return Math.max(0, Math.round(time))
}

export function snapSecondsToFrame(seconds: number, frameRate: number): number {
  return Math.max(0, secondsToFrame(seconds, frameRate))
}

export function remapFrameToRate(frame: number, fromFrameRate: number, toFrameRate: number): number {
  return snapSecondsToFrame(frameToSeconds(frame, fromFrameRate), toFrameRate)
}

export function segmentDuration(segment: MultiTrackSegment): number {
  return Math.max(0, segment.end_frame - segment.start_frame)
}

export function calculateTotalLength(
  tracks: MultiTrack[],
  frameRate: number = MULTITRACK_DEFAULT_FRAME_RATE,
): number {
  const safeFrameRate = Math.max(1, Math.round(frameRate))
  const minimumLength = MULTITRACK_MIN_DURATION_SECONDS * safeFrameRate
  const maxEnd = calculateMaxSegmentEnd(tracks)
  return Math.max(minimumLength, maxEnd)
}

export const calculateTotalDuration = calculateTotalLength

function calculateMaxSegmentEnd(tracks: MultiTrack[]): number {
  return tracks
    .flatMap((track) => track.segments)
    .reduce((max, segment) => Math.max(max, segment.end_frame), 0)
}

function rangesOverlap(startA: number, endA: number, startB: number, endB: number): boolean {
  return startA < endB && endA > startB
}

function isMultiTrackTaskMode(value: unknown): value is MultiTrackTaskMode {
  return typeof value === 'string' && (MULTITRACK_TASK_MODES as readonly string[]).includes(value)
}

function normalizeTaskMode(value: unknown): MultiTrackTaskMode {
  return isMultiTrackTaskMode(value) ? value : MULTITRACK_DEFAULT_TASK_MODE
}

function insertionIndexForCenter(segments: MultiTrackSegment[], startTime: number): number {
  return segments.filter((segment) => {
    const center = segment.start_frame + segmentDuration(segment) / 2
    return center < startTime
  }).length
}

function insertIndexForTrack(
  segments: MultiTrackSegment[],
  startFrame: number,
  frameRate: number,
): number {
  void frameRate
  const sorted = [...segments].sort((a, b) => a.start_frame - b.start_frame)
  const targetStart = Math.max(0, Math.round(startFrame))
  return insertionIndexForCenter(sorted, targetStart)
}

function repositionSegment(
  segment: MultiTrackSegment,
  startFrame: number,
  endFrame: number,
): MultiTrackSegment {
  const shift = startFrame - segment.start_frame
  return {
    ...segment,
    start_frame: startFrame,
    end_frame: endFrame,
    origin_start_frame: segment.origin_start_frame === undefined
      ? undefined
      : segment.origin_start_frame + shift,
  }
}

function packSegmentsFromZero(segments: MultiTrackSegment[], frameRate: number): MultiTrackSegment[] {
  void frameRate
  let cursor = 0
  return segments.map((segment) => {
    const duration = segmentDuration(segment)
    const startFrame = Math.round(cursor)
    const endFrame = Math.round(startFrame + duration)
    cursor = endFrame
    return repositionSegment(segment, startFrame, endFrame)
  })
}

function arrangeAudioDrop(
  targetSegments: MultiTrackSegment[],
  movingSegment: MultiTrackSegment,
  nextStartTime: number,
  frameRate: number,
): { segments: MultiTrackSegment[]; insertIndex: number } {
  const insertIndex = insertIndexForTrack(targetSegments, nextStartTime, frameRate)
  const duration = segmentDuration(movingSegment)
  const requestedStart = snapTimeToFrame(nextStartTime, frameRate)
  const previousEnd = targetSegments[insertIndex - 1]?.end_frame ?? 0
  const movedStart = Math.max(requestedStart, previousEnd)
  const arranged = [
    ...targetSegments.slice(0, insertIndex),
    repositionSegment(movingSegment, movedStart, movedStart + duration),
    ...targetSegments.slice(insertIndex),
  ]

  let cursor = 0
  return {
    insertIndex,
    segments: arranged.map((segment, index) => {
      if (index < insertIndex) {
        cursor = Math.max(cursor, segment.end_frame)
        return segment
      }
      const segmentStart = Math.max(segment.start_frame, cursor)
      const shifted = segmentStart === segment.start_frame
        ? segment
        : repositionSegment(segment, segmentStart, segmentStart + segmentDuration(segment))
      cursor = shifted.end_frame
      return shifted
    }),
  }
}

export function syncMatchingTasksToPrimaryVideoTrack(
  originalTracks: MultiTrack[],
  updatedTracks: MultiTrack[],
): MultiTrack[] {
  const originalVideoTrack = originalTracks.find((track) => track.type === 'video')
  if (!originalVideoTrack) return updatedTracks

  const updatedVideoSegments = updatedTracks
    .filter((track) => track.type === 'video')
    .flatMap((track) => track.segments)
  const rangeUpdates = originalVideoTrack.segments.flatMap((originalSegment) => {
    const updatedSegment = updatedVideoSegments.find((segment) => segment.id === originalSegment.id)
    return updatedSegment ? [{ originalSegment, updatedSegment }] : []
  })

  return updatedTracks.map((track) => {
    if (track.type !== 'task') return track
    return {
      ...track,
      segments: track.segments.map((segment) => {
        const match = rangeUpdates.find(({ originalSegment }) => (
          segment.start_frame === originalSegment.start_frame &&
          segment.end_frame === originalSegment.end_frame
        ))
        return match
          ? {
              ...segment,
              start_frame: match.updatedSegment.start_frame,
              end_frame: match.updatedSegment.end_frame,
            }
          : segment
      }),
    }
  })
}

export function addDefaultTaskSegmentIfRangeEmpty(
  tracks: MultiTrack[],
  startFrame: number,
  endFrame: number,
): MultiTrack[] {
  const safeStartFrame = snapTimeToFrame(startFrame, MULTITRACK_DEFAULT_FRAME_RATE)
  const safeEndFrame = Math.max(safeStartFrame + 1, snapTimeToFrame(endFrame, MULTITRACK_DEFAULT_FRAME_RATE))
  const taskTrack = tracks.find((track) => track.type === 'task')
  if (!taskTrack) return tracks
  const hasTaskCoverage = taskTrack.segments.some((segment) => (
    rangesOverlap(segment.start_frame, segment.end_frame, safeStartFrame, safeEndFrame)
  ))
  if (hasTaskCoverage) return tracks

  const taskSegment: MultiTrackSegment = {
    id: uuid(),
    start_frame: safeStartFrame,
    end_frame: safeEndFrame,
    color: MULTITRACK_TRACK_COLORS.task,
    content: {
      media_type: 'none',
      task_mode: MULTITRACK_DEFAULT_TASK_MODE,
    },
  }

  return tracks.map((track) => {
    if (track.id !== taskTrack.id) return track
    return {
      ...track,
      segments: [...track.segments, taskSegment].sort((a, b) => a.start_frame - b.start_frame),
    }
  })
}

export function applyCombinedTaskTexts(
  parts: string[],
  segments: MultiTrackSegment[],
  totalFrames: number,
  color: string,
): MultiTrackSegment[] {
  const normalizedParts = parts.length > 0 ? parts : ['']
  if (normalizedParts.length === segments.length) {
    return segments.map((segment, index) => ({
      ...segment,
      content: { ...segment.content, text: normalizedParts[index] },
    }))
  }

  const count = normalizedParts.length
  const span = Math.max(count, Math.round(totalFrames))
  const base = Math.floor(span / count)
  const remainder = span % count
  let cursor = 0

  return normalizedParts.map((text, index) => {
    const existing = segments[index]
    const size = base + (index < remainder ? 1 : 0)
    const startFrame = cursor
    cursor += size
    return {
      id: existing?.id ?? uuid(),
      start_frame: startFrame,
      end_frame: cursor,
      color: existing?.color ?? color,
      content: existing
        ? { ...existing.content, text }
        : {
            media_type: 'none',
            task_mode: MULTITRACK_DEFAULT_TASK_MODE,
            images: [],
            text,
          },
    }
  })
}

export function distributeMultiTrackSegmentsEvenly(
  segments: MultiTrackSegment[],
  totalFrames: number,
): MultiTrackSegment[] {
  if (segments.length === 0) return []
  const span = Math.max(segments.length, Math.round(totalFrames))
  const base = Math.floor(span / segments.length)
  const remainder = span % segments.length
  let cursor = 0

  return [...segments]
    .sort((left, right) => left.start_frame - right.start_frame)
    .map((segment, index) => {
      const size = base + (index < remainder ? 1 : 0)
      const startFrame = cursor
      cursor += size
      return { ...segment, start_frame: startFrame, end_frame: cursor }
    })
}

export interface ClonedMultiTrackSegmentResult {
  segments: MultiTrackSegment[]
  clonedSegmentId: string
  addedDuration: number
}

export function cloneMultiTrackSegment(
  segments: MultiTrackSegment[],
  segmentId: string,
): ClonedMultiTrackSegmentResult | null {
  const sorted = [...segments].sort((left, right) => left.start_frame - right.start_frame)
  const index = sorted.findIndex((segment) => segment.id === segmentId)
  if (index < 0) return null

  const source = sorted[index]
  const duration = segmentDuration(source)
  if (duration <= 0) return null
  const clonedSegmentId = uuid()
  const cloneStart = source.end_frame
  const clone: MultiTrackSegment = {
    ...structuredClone(source),
    id: clonedSegmentId,
    start_frame: cloneStart,
    end_frame: cloneStart + duration,
  }
  let nextStart = clone.end_frame
  const subsequent = sorted.slice(index + 1).map((segment) => {
    const nextDuration = segmentDuration(segment)
    const shifted = {
      ...segment,
      start_frame: nextStart,
      end_frame: nextStart + nextDuration,
    }
    nextStart = shifted.end_frame
    return shifted
  })

  return {
    segments: [...sorted.slice(0, index + 1), clone, ...subsequent],
    clonedSegmentId,
    addedDuration: duration,
  }
}

export function deleteSegmentWithLinkedTasks(
  tracks: MultiTrack[],
  segmentId: string,
  frameRate = MULTITRACK_DEFAULT_FRAME_RATE,
): MultiTrack[] {
  return deleteSegmentsWithLinkedTasks(tracks, [segmentId], frameRate)
}

export function deleteSegmentsWithLinkedTasks(
  tracks: MultiTrack[],
  segmentIds: Iterable<string>,
  frameRate = MULTITRACK_DEFAULT_FRAME_RATE,
): MultiTrack[] {
  const ids = new Set(segmentIds)
  if (ids.size === 0) return tracks

  const deletedVideoTrackIds = new Set<string>()
  const deletedVideoRanges = tracks.flatMap((track) => {
    if (track.type !== 'video') return []
    const ranges = track.segments
      .filter((segment) => ids.has(segment.id))
      .map((segment) => ({ start: segment.start_frame, end: segment.end_frame }))
    if (ranges.length > 0) deletedVideoTrackIds.add(track.id)
    return ranges
  })

  const updatedTracks = tracks.map((track) => {
    const remainingSegments = track.segments.filter((segment) => {
      if (ids.has(segment.id)) return false
      if (track.type !== 'task') return true
      return !deletedVideoRanges.some((range) => (
        rangesOverlap(segment.start_frame, segment.end_frame, range.start, range.end)
      ))
    })

    if (track.type === 'video' && deletedVideoTrackIds.has(track.id)) {
      return {
        ...track,
        segments: packSegmentsFromZero(remainingSegments.sort((a, b) => a.start_frame - b.start_frame), frameRate),
      }
    }
    return { ...track, segments: remainingSegments }
  })

  return deletedVideoRanges.length > 0
    ? syncMatchingTasksToPrimaryVideoTrack(tracks, updatedTracks)
    : updatedTracks
}

export function moveSegmentBetweenCompatibleTracks(
  tracks: MultiTrack[],
  segmentId: string,
  targetTrackId: string,
  nextStartTime: number,
  frameRate: number,
): MultiTrack[] {
  const sourceTrack = tracks.find((track) => track.segments.some((segment) => segment.id === segmentId))
  const targetTrack = tracks.find((track) => track.id === targetTrackId)
  if (!sourceTrack || !targetTrack || sourceTrack.type !== targetTrack.type) return tracks

  const movingSegment = sourceTrack.segments.find((segment) => segment.id === segmentId)
  if (!movingSegment) return tracks

  const duration = segmentDuration(movingSegment)
  if (duration <= 0) return tracks
  const targetSegments = targetTrack.segments
    .filter((segment) => segment.id !== segmentId)
    .sort((a, b) => a.start_frame - b.start_frame)
  const insertIndex = insertIndexForTrack(targetSegments, nextStartTime, frameRate)

  if (targetTrack.type === 'audio') {
    const arranged = arrangeAudioDrop(targetSegments, movingSegment, nextStartTime, frameRate)
    return tracks.map((track) => {
      if (track.id === sourceTrack.id && track.id !== targetTrack.id) {
        return {
          ...track,
          segments: track.segments.filter((segment) => segment.id !== segmentId),
        }
      }
      if (track.id !== targetTrack.id) return track
      return { ...track, segments: arranged.segments }
    })
  }

  const nextTargetSegments = packSegmentsFromZero([
    ...targetSegments.slice(0, insertIndex),
    movingSegment,
    ...targetSegments.slice(insertIndex),
  ], frameRate)

  const movedTracks = tracks.map((track) => {
    if (track.id === sourceTrack.id && track.id !== targetTrack.id) {
      return {
        ...track,
        segments: packSegmentsFromZero(
          track.segments.filter((segment) => segment.id !== segmentId),
          frameRate,
        ),
      }
    }
    if (track.id !== targetTrack.id) return track
    return {
      ...track,
      segments: nextTargetSegments,
    }
  })
  return syncMatchingTasksToPrimaryVideoTrack(tracks, movedTracks)
}

function moveSegmentsWithinOriginalTracks(
  tracks: MultiTrack[],
  segmentIds: Set<string>,
  anchorSegmentId: string,
  nextStartTime: number,
  frameRate: number,
): MultiTrack[] {
  const anchorSegment = tracks
    .flatMap((track) => track.segments)
    .find((segment) => segment.id === anchorSegmentId)
  if (!anchorSegment) return tracks

  const delta = snapTimeToFrame(nextStartTime - anchorSegment.start_frame, frameRate)
  if (delta === 0) return tracks

  const movedTracks = tracks.map((track) => {
    const selected = track.segments
      .filter((segment) => segmentIds.has(segment.id))
      .sort((left, right) => left.start_frame - right.start_frame)
    if (selected.length === 0) return track

    const remaining = track.segments
      .filter((segment) => !segmentIds.has(segment.id))
      .sort((left, right) => left.start_frame - right.start_frame)
    const moved = selected.map((segment) => {
      const duration = segmentDuration(segment)
      const startFrame = Math.max(0, segment.start_frame + delta)
      return repositionSegment(segment, startFrame, startFrame + duration)
    })

    if (track.type === 'audio') {
      const combined = [...remaining, ...moved].sort((left, right) => left.start_frame - right.start_frame)
      let cursor = 0
      return {
        ...track,
        segments: combined.map((segment) => {
          const segmentStart = segmentIds.has(segment.id)
            ? Math.max(segment.start_frame, cursor)
            : segment.start_frame
          const shifted = segmentStart === segment.start_frame
            ? segment
            : repositionSegment(segment, segmentStart, segmentStart + segmentDuration(segment))
          cursor = Math.max(cursor, shifted.end_frame)
          return shifted
        }),
      }
    }

    const insertStart = moved.reduce((min, segment) => Math.min(min, segment.start_frame), Number.POSITIVE_INFINITY)
    const insertIndex = insertIndexForTrack(remaining, insertStart, frameRate)
    return {
      ...track,
      segments: packSegmentsFromZero([
        ...remaining.slice(0, insertIndex),
        ...moved,
        ...remaining.slice(insertIndex),
      ], frameRate),
    }
  })

  return syncMatchingTasksToPrimaryVideoTrack(tracks, movedTracks)
}

export function moveSelectedSegments(
  tracks: MultiTrack[],
  segmentIds: Iterable<string>,
  anchorSegmentId: string,
  targetTrackId: string,
  nextStartTime: number,
  frameRate: number,
): MultiTrack[] {
  const selectedIds = new Set(segmentIds)
  if (selectedIds.size <= 1 || !selectedIds.has(anchorSegmentId)) {
    return moveSegmentBetweenCompatibleTracks(tracks, anchorSegmentId, targetTrackId, nextStartTime, frameRate)
  }
  return moveSegmentsWithinOriginalTracks(tracks, selectedIds, anchorSegmentId, nextStartTime, frameRate)
}

export interface SegmentDragPlaceholder {
  segmentId: string
  targetTrackId: string
  insertIndex: number
  start_frame: number
  end_frame: number
}

export function getSegmentDragPlaceholder(
  tracks: MultiTrack[],
  segmentId: string,
  targetTrackId: string,
  nextStartTime: number,
  frameRate: number,
): SegmentDragPlaceholder | null {
  const sourceTrack = tracks.find((track) => track.segments.some((segment) => segment.id === segmentId))
  const targetTrack = tracks.find((track) => track.id === targetTrackId)
  if (!sourceTrack || !targetTrack || sourceTrack.type !== targetTrack.type) return null

  const movingSegment = sourceTrack.segments.find((segment) => segment.id === segmentId)
  if (!movingSegment) return null

  const duration = segmentDuration(movingSegment)
  if (duration <= 0) return null

  const targetSegments = targetTrack.segments
    .filter((segment) => segment.id !== segmentId)
    .sort((a, b) => a.start_frame - b.start_frame)
  const insertIndex = insertIndexForTrack(targetSegments, nextStartTime, frameRate)
  if (targetTrack.type === 'audio') {
    const arranged = arrangeAudioDrop(targetSegments, movingSegment, nextStartTime, frameRate)
    const placeholder = arranged.segments[arranged.insertIndex]
    return {
      segmentId,
      targetTrackId,
      insertIndex: arranged.insertIndex,
      start_frame: placeholder.start_frame,
      end_frame: placeholder.end_frame,
    }
  }
  const packed = packSegmentsFromZero([
    ...targetSegments.slice(0, insertIndex),
    movingSegment,
    ...targetSegments.slice(insertIndex),
  ], frameRate)
  const placeholder = packed[insertIndex]
  if (!placeholder) return null

  return {
    segmentId,
    targetTrackId,
    insertIndex,
    start_frame: placeholder.start_frame,
    end_frame: placeholder.end_frame,
  }
}

export function getSegmentDragPreviewSegments(
  tracks: MultiTrack[],
  placeholder: SegmentDragPlaceholder,
  frameRate: number,
): MultiTrackSegment[] | null {
  const targetTrack = tracks.find((track) => track.id === placeholder.targetTrackId)
  const movingSegment = tracks
    .flatMap((track) => track.segments)
    .find((segment) => segment.id === placeholder.segmentId)
  if (!targetTrack || !movingSegment) return null

  const duration = segmentDuration(movingSegment)
  if (duration <= 0) return null

  const targetSegments = targetTrack.segments
    .filter((segment) => segment.id !== placeholder.segmentId)
    .sort((a, b) => a.start_frame - b.start_frame)
  if (targetTrack.type === 'audio') {
    return arrangeAudioDrop(
      targetSegments,
      movingSegment,
      placeholder.start_frame,
      frameRate,
    ).segments.filter((segment) => segment.id !== placeholder.segmentId)
  }
  const nextTargetSegments = packSegmentsFromZero([
    ...targetSegments.slice(0, placeholder.insertIndex),
    movingSegment,
    ...targetSegments.slice(placeholder.insertIndex),
  ], frameRate)
  const previewTracks = tracks.map((track) => (
    track.id === targetTrack.id ? { ...track, segments: nextTargetSegments } : track
  ))
  const syncedPreviewTracks = syncMatchingTasksToPrimaryVideoTrack(tracks, previewTracks)
  const originalTaskSegments = new Map(
    tracks
      .filter((track) => track.type === 'task')
      .flatMap((track) => track.segments)
      .map((segment) => [segment.id, segment]),
  )
  const changedTaskSegments = syncedPreviewTracks
    .filter((track) => track.type === 'task')
    .flatMap((track) => track.segments)
    .filter((segment) => {
      const original = originalTaskSegments.get(segment.id)
      return original && (
        original.start_frame !== segment.start_frame ||
        original.end_frame !== segment.end_frame
      )
    })
  return [
    ...nextTargetSegments.filter((segment) => segment.id !== placeholder.segmentId),
    ...changedTaskSegments,
  ]
}

export interface ActivePreviewVideoSegment {
  trackId: string
  segment: MultiTrackSegment
  localTime: number
}

function segmentContainsTime(segment: MultiTrackSegment, time: number): boolean {
  return time >= segment.start_frame && time < segment.end_frame
}

function segmentSourceFrame(segment: MultiTrackSegment, currentFrame: number): number {
  return Math.max(0, currentFrame - (segment.origin_start_frame ?? segment.start_frame))
}

export function getActivePreviewVideoSegment(
  data: TrackData,
  currentTime: number,
  selectedSegmentId: string | null,
): ActivePreviewVideoSegment | null {
  const currentFrame = snapTimeToFrame(currentTime, data.frame_rate)
  const videoTracks = data.tracks.filter((track) => track.type === 'video')
  if (selectedSegmentId) {
    for (const track of videoTracks) {
      const segment = track.segments.find((item) => item.id === selectedSegmentId)
      if (!segment || segment.content.media_type !== 'video' || !segmentContainsTime(segment, currentFrame)) continue
      return {
        trackId: track.id,
        segment,
        localTime: frameToSeconds(segmentSourceFrame(segment, currentFrame), data.frame_rate),
      }
    }
    return null
  }

  for (const track of videoTracks) {
    const segment = track.segments.find((item) => (
      item.content.media_type === 'video' && segmentContainsTime(item, currentFrame)
    ))
    if (!segment) continue
    return {
      trackId: track.id,
      segment,
      localTime: frameToSeconds(segmentSourceFrame(segment, currentFrame), data.frame_rate),
    }
  }
  return null
}

export interface SelectedMultiTrackSegment {
  trackId: string
  trackType: MultiTrackType
  segment: MultiTrackSegment
}

export interface ActivePreviewAudioSource {
  trackId: string
  segment: MultiTrackSegment
  localTime: number
  volumeDb: number
}

export function getActivePreviewAudioSources(
  data: TrackData,
  currentTime: number,
  selectedSegment: SelectedMultiTrackSegment | null,
): ActivePreviewAudioSource[] {
  if (
    data.muted === true ||
    (selectedSegment !== null && selectedSegment.trackType !== 'video' && selectedSegment.trackType !== 'audio')
  ) return []
  const currentFrame = snapTimeToFrame(currentTime, data.frame_rate)
  const audioTracks = data.tracks.filter((track) => track.type === 'video' || track.type === 'audio')
  const hasSolo = audioTracks.some((track) => track.solo === true)
  const selectedId = selectedSegment && (selectedSegment.trackType === 'video' || selectedSegment.trackType === 'audio')
    ? selectedSegment.segment.id
    : null

  return audioTracks.flatMap((track) => {
    if (track.muted || (hasSolo && track.solo !== true)) return []
    return track.segments.flatMap((segment) => {
      if (selectedId && segment.id !== selectedId) return []
      if (!segmentContainsTime(segment, currentFrame) || segment.content.muted === true) return []
      if (segment.content.media_type !== 'video' && segment.content.media_type !== 'audio') return []
      return [{
        trackId: track.id,
        segment,
        localTime: frameToSeconds(segmentSourceFrame(segment, currentFrame), data.frame_rate),
        volumeDb: (data.volume_db ?? 0) + (track.volume_db ?? 0) + (segment.content.volume_db ?? 0),
      }]
    })
  })
}

export function getSelectedMultiTrackSegment(
  data: TrackData,
  selectedSegmentId: string | null,
): SelectedMultiTrackSegment | null {
  if (!selectedSegmentId) return null
  for (const track of data.tracks) {
    const segment = track.segments.find((item) => item.id === selectedSegmentId)
    if (!segment) continue
    return {
      trackId: track.id,
      trackType: track.type,
      segment,
    }
  }
  return null
}

export function updateMultiTrackSegmentContent(
  data: TrackData,
  segmentId: string,
  patch: Partial<MultiTrackSegmentContent>,
): TrackData {
  return {
    ...data,
    tracks: data.tracks.map((track) => ({
      ...track,
      segments: track.segments.map((segment) => (
        segment.id === segmentId
          ? {
              ...segment,
              content: {
                ...segment.content,
                ...patch,
              },
            }
          : segment
      )),
    })),
  }
}

export function syncMatchingTasksToPrimaryVideoSegment(
  originalTracks: MultiTrack[],
  updatedTracks: MultiTrack[],
  segmentId: string,
): MultiTrack[] {
  const primaryVideoTrack = originalTracks.find((track) => track.type === 'video')
  const originalSegment = primaryVideoTrack?.segments.find((segment) => segment.id === segmentId)
  if (!primaryVideoTrack || !originalSegment) return updatedTracks

  const updatedSegment = updatedTracks
    .find((track) => track.id === primaryVideoTrack.id)
    ?.segments.find((segment) => segment.id === segmentId)
  if (!updatedSegment) return updatedTracks

  return updatedTracks.map((track) => {
    if (track.type !== 'task') return track
    return {
      ...track,
      segments: track.segments.map((segment) => (
        segment.start_frame === originalSegment.start_frame &&
        segment.end_frame === originalSegment.end_frame
          ? {
              ...segment,
              start_frame: updatedSegment.start_frame,
              end_frame: updatedSegment.end_frame,
            }
          : segment
      )),
    }
  })
}

export function updateMultiTrackSegmentDuration(
  data: TrackData,
  segmentId: string,
  duration: number,
  frameRate: number,
): TrackData {
  const nextDuration = Math.max(1, snapSecondsToFrame(duration, frameRate))
  const resizedTracks = data.tracks.map((track) => {
    const selected = track.segments.find((segment) => segment.id === segmentId)
    if (!selected) return track
    const nextSegmentStart = track.segments
      .filter((segment) => segment.id !== segmentId && segment.start_frame > selected.start_frame)
      .reduce((nearest, segment) => Math.min(nearest, segment.start_frame), Number.POSITIVE_INFINITY)
    const requestedEnd = snapTimeToFrame(selected.start_frame + nextDuration, frameRate)
    const endFrame = Math.max(selected.start_frame + 1, Math.min(requestedEnd, nextSegmentStart))

    return {
      ...track,
      segments: track.segments.map((segment) => (
        segment.id === segmentId ? { ...segment, end_frame: endFrame } : segment
      )),
    }
  })
  const tracks = syncMatchingTasksToPrimaryVideoSegment(data.tracks, resizedTracks, segmentId)

  return {
    ...data,
    tracks,
    total_length: calculateTotalLength(tracks, frameRate),
  }
}

export function remapTrackDataFrameRate(data: TrackData, nextFrameRate: number): TrackData {
  const safeNextFrameRate = Math.max(1, Math.round(nextFrameRate))
  if (safeNextFrameRate === data.frame_rate) return data

  const tracks = data.tracks.map((track) => ({
    ...track,
    segments: track.segments.map((segment) => {
      const startFrame = remapFrameToRate(segment.start_frame, data.frame_rate, safeNextFrameRate)
      const endFrame = Math.max(
        startFrame + 1,
        remapFrameToRate(segment.end_frame, data.frame_rate, safeNextFrameRate),
      )
      return {
        ...segment,
        start_frame: startFrame,
        end_frame: endFrame,
        origin_start_frame: segment.origin_start_frame === undefined
          ? undefined
          : Math.round(segment.origin_start_frame * safeNextFrameRate / data.frame_rate),
      }
    }),
  }))

  return {
    ...data,
    frame_rate: safeNextFrameRate,
    tracks,
    total_length: calculateTotalLength(tracks, safeNextFrameRate),
  }
}

export type MultiTrackPreviewResizeMethod =
  | 'stretch'
  | 'resize'
  | 'pad'
  | 'pad (white)'
  | 'pad_edge'
  | 'pad_edge_pixel'
  | 'crop'
  | 'pillarbox_blur'

export interface MultiTrackPreviewResolution {
  width: number
  height: number
  resizeMethod: MultiTrackPreviewResizeMethod
  mode: 'fixed' | 'custom' | 'auto' | 'longest' | 'shortest'
}

export interface MultiTrackVideoMetadata {
  width: number
  height: number
}

const DEFAULT_PREVIEW_RESOLUTION = {
  width: 544,
  height: 960,
}

function positiveNumber(value: unknown): number | null {
  const numeric = Number(value)
  return Number.isFinite(numeric) && numeric > 0 ? numeric : null
}

function parseResizeMethod(value: unknown): MultiTrackPreviewResizeMethod {
  const methods: readonly string[] = [
    'stretch',
    'resize',
    'pad',
    'pad (white)',
    'pad_edge',
    'pad_edge_pixel',
    'crop',
    'pillarbox_blur',
  ]
  return typeof value === 'string' && methods.includes(value)
    ? value as MultiTrackPreviewResizeMethod
    : 'stretch'
}

export interface MultiTrackPreviewResolutionInput {
  resolution?: string
  resize_method?: MultiTrackPreviewResizeMethod
  resize_to_pixel?: number
  width?: number
  height?: number
}

interface NodeResolutionWidget {
  name?: string
  value?: unknown
  serializeValue?: () => unknown
}

interface NodeWithResolutionWidgets {
  widgets?: NodeResolutionWidget[]
}

function unwrapDynamicValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.length > 0 ? unwrapDynamicValue(value[0]) : undefined
  return value
}

function readWidgetValue(widget: NodeResolutionWidget | undefined): unknown {
  if (!widget) return undefined
  if (widget.value !== undefined) return unwrapDynamicValue(widget.value)
  if (!widget.serializeValue) return undefined
  try {
    return unwrapDynamicValue(widget.serializeValue())
  } catch (error) {
    console.error('[multitrack-utils] failed to read resolution widget value:', error)
    return undefined
  }
}

export function collectMultiTrackPreviewResolutionInput(node: unknown): MultiTrackPreviewResolutionInput {
  const widgets = (node as NodeWithResolutionWidgets | null)?.widgets ?? []
  const widgetByName = new Map(widgets.map((widget) => [widget.name, widget]))
  const resolutionValue = readWidgetValue(widgetByName.get('resolution'))
  const collected: MultiTrackPreviewResolutionInput = {}

  if (typeof resolutionValue === 'string') {
    collected.resolution = resolutionValue
  } else if (resolutionValue && typeof resolutionValue === 'object') {
    const value = resolutionValue as Record<string, unknown>
    if (typeof value.resolution === 'string') collected.resolution = value.resolution
    const resizeMethod = parseResizeMethod(value.resize_method)
    if (value.resize_method !== undefined) collected.resize_method = resizeMethod
    const resizeToPixel = positiveNumber(value.resize_to_pixel)
    if (resizeToPixel !== null) collected.resize_to_pixel = resizeToPixel
    const width = positiveNumber(value.width)
    if (width !== null) collected.width = width
    const height = positiveNumber(value.height)
    if (height !== null) collected.height = height
  }

  const resizeMethodValue = readWidgetValue(widgetByName.get('resolution.resize_method'))
  if (resizeMethodValue !== undefined) collected.resize_method = parseResizeMethod(resizeMethodValue)
  const resizeToPixel = positiveNumber(readWidgetValue(widgetByName.get('resolution.resize_to_pixel')))
  if (resizeToPixel !== null) collected.resize_to_pixel = resizeToPixel
  const width = positiveNumber(readWidgetValue(widgetByName.get('resolution.width')))
  if (width !== null) collected.width = width
  const height = positiveNumber(readWidgetValue(widgetByName.get('resolution.height')))
  if (height !== null) collected.height = height

  return collected
}

function scaledMetadataResolution(
  metadata: MultiTrackVideoMetadata | null,
  mode: 'longest' | 'shortest',
  resizeToPixel: number,
): Pick<MultiTrackPreviewResolution, 'width' | 'height'> {
  if (!metadata || metadata.width <= 0 || metadata.height <= 0) return DEFAULT_PREVIEW_RESOLUTION
  const aspect = metadata.width / metadata.height
  if (mode === 'longest') {
    if (metadata.width >= metadata.height) {
      return { width: resizeToPixel, height: Math.round(resizeToPixel / aspect) }
    }
    return { width: Math.round(resizeToPixel * aspect), height: resizeToPixel }
  }

  if (metadata.width <= metadata.height) {
    return { width: resizeToPixel, height: Math.round(resizeToPixel / aspect) }
  }
  return { width: Math.round(resizeToPixel * aspect), height: resizeToPixel }
}

export function parseMultiTrackPreviewResolution(
  resolution: unknown,
  firstVideoMetadata: MultiTrackVideoMetadata | null,
): MultiTrackPreviewResolution {
  const values = typeof resolution === 'string'
    ? { resolution }
    : resolution && typeof resolution === 'object'
      ? resolution as Record<string, unknown>
      : {}
  const resolutionLabel = typeof values.resolution === 'string' ? values.resolution : ''
  const normalizedLabel = resolutionLabel.toLowerCase()
  const resizeMethod = parseResizeMethod(values.resize_method)
  const fixedMatch = resolutionLabel.match(/(\d+)\s*x\s*(\d+)/i)

  if (normalizedLabel.includes('custom')) {
    return {
      width: positiveNumber(values.width) ?? DEFAULT_PREVIEW_RESOLUTION.width,
      height: positiveNumber(values.height) ?? DEFAULT_PREVIEW_RESOLUTION.height,
      resizeMethod,
      mode: 'custom',
    }
  }

  if (normalizedLabel.includes('longest') || normalizedLabel.includes('shortest')) {
    const mode = normalizedLabel.includes('longest') ? 'longest' : 'shortest'
    const scaled = scaledMetadataResolution(
      firstVideoMetadata,
      mode,
      positiveNumber(values.resize_to_pixel) ?? 960,
    )
    return {
      ...scaled,
      resizeMethod,
      mode,
    }
  }

  if (fixedMatch) {
    return {
      width: Number(fixedMatch[1]),
      height: Number(fixedMatch[2]),
      resizeMethod,
      mode: 'fixed',
    }
  }

  if (normalizedLabel.includes('auto') && firstVideoMetadata && firstVideoMetadata.width > 0 && firstVideoMetadata.height > 0) {
    return {
      width: firstVideoMetadata.width,
      height: firstVideoMetadata.height,
      resizeMethod,
      mode: 'auto',
    }
  }

  return {
    ...DEFAULT_PREVIEW_RESOLUTION,
    resizeMethod,
    mode: 'auto',
  }
}

interface LegacyMultiTrackSegment extends Omit<MultiTrackSegment, 'start_frame' | 'end_frame'> {
  start_frame?: unknown
  end_frame?: unknown
}

interface LegacyMultiTrack extends Omit<MultiTrack, 'type' | 'segments'> {
  type: MultiTrack['type'] | 'segment'
  segments: LegacyMultiTrackSegment[]
}

interface LegacyTrackData extends Omit<Partial<TrackData>, 'tracks' | 'total_length'> {
  tracks: LegacyMultiTrack[]
  total_length?: unknown
}

function finiteNumber(value: unknown): number | null {
  const numeric = Number(value)
  return Number.isFinite(numeric) ? numeric : null
}

function normalizeFrameValue(value: unknown): number | null {
  const numeric = finiteNumber(value)
  return numeric === null ? null : Math.max(0, Math.round(numeric))
}

function normalizeSignedFrameValue(value: unknown): number | null {
  const numeric = finiteNumber(value)
  return numeric === null ? null : Math.round(numeric)
}

function omitLegacyVolume<T extends object>(value: T): Omit<T, 'volume'> {
  const copy = { ...value } as T & { volume?: unknown }
  delete copy.volume
  return copy
}

function normalizeLegacySegment(segment: LegacyMultiTrackSegment): MultiTrackSegment {
  const startFrame = normalizeFrameValue(segment.start_frame)
    ?? 0
  const endFrame = normalizeFrameValue(segment.end_frame)
    ?? startFrame + 1

  return {
    ...omitLegacyVolume(segment),
    start_frame: startFrame,
    end_frame: Math.max(startFrame + 1, endFrame),
    origin_start_frame: normalizeSignedFrameValue(segment.origin_start_frame) ?? undefined,
    content: {
      ...omitLegacyVolume(segment.content),
      muted: segment.content.muted === true,
      volume_db: normalizedVolumeDb(segment.content.volume_db),
    },
  }
}

function normalizeTrackSegments(track: LegacyMultiTrack): MultiTrackSegment[] {
  return track.segments
    .map((segment) => normalizeLegacySegment(segment))
    .sort((a, b) => a.start_frame - b.start_frame)
}

function normalizeTotalLength(tracks: MultiTrack[], frameRate: number): number {
  return calculateTotalLength(tracks, frameRate)
}

export function createDefaultTrackData(): TrackData {
  return {
    muted: false,
    volume_db: MULTITRACK_DEFAULT_VOLUME_DB,
    tracks: [
      {
        id: uuid(),
        name: 'Task 0',
        type: 'task',
        task_mode: MULTITRACK_DEFAULT_TASK_MODE,
        color: MULTITRACK_TRACK_COLORS.task,
        muted: false,
        solo: false,
        volume_db: MULTITRACK_DEFAULT_VOLUME_DB,
        locked: false,
        segments: [],
      },
      {
        id: uuid(),
        name: 'Video 0',
        type: 'video',
        color: MULTITRACK_TRACK_COLORS.video,
        muted: false,
        solo: false,
        volume_db: MULTITRACK_DEFAULT_VOLUME_DB,
        locked: false,
        segments: [],
      },
    ],
    total_length: MULTITRACK_DEFAULT_TOTAL_LENGTH,
    frame_rate: MULTITRACK_DEFAULT_FRAME_RATE,
  }
}

export function normalizeTrackData(raw: LegacyTrackData): TrackData {
  const frameRate = Math.max(1, Math.round(raw.frame_rate ?? MULTITRACK_DEFAULT_FRAME_RATE))
  const tracks = raw.tracks.map((track) => {
    const normalizedTrack = omitLegacyVolume(track)
    const segments = normalizeTrackSegments(track)
    const audioSettings = {
      muted: track.muted === true,
      solo: track.solo === true,
      volume_db: normalizedVolumeDb(track.volume_db),
    }
    if (track.type !== 'segment') {
      if (track.type === 'task') {
        return {
          ...normalizedTrack,
          ...audioSettings,
          type: 'task' as const,
          task_mode: normalizeTaskMode(track.task_mode),
          color: track.color === 'var(--muted)' ? MULTITRACK_TRACK_COLORS.task : track.color,
          segments: segments.map((segment) => ({
            ...segment,
            content: {
              ...segment.content,
              task_mode: normalizeTaskMode(segment.content.task_mode),
              images: Array.isArray(segment.content.images) ? segment.content.images : [],
            },
          })),
        }
      }
      return {
        ...normalizedTrack,
        ...audioSettings,
        type: track.type,
        segments,
      } as MultiTrack
    }

    return {
      ...normalizedTrack,
      ...audioSettings,
      name: track.name,
      type: 'task' as const,
      task_mode: MULTITRACK_DEFAULT_TASK_MODE,
      color: MULTITRACK_TRACK_COLORS.task,
      segments: segments.map((segment) => ({
        ...segment,
        content: {
          ...segment.content,
          task_mode: normalizeTaskMode(segment.content.task_mode),
          images: Array.isArray(segment.content.images) ? segment.content.images : [],
        },
      })),
    }
  })

  const trackCounts: Record<MultiTrackType, number> = {
    task: 0,
    video: 0,
    audio: 0,
    subtitle: 0,
  }
  const namedTracks = tracks.map((track) => {
    const index = trackCounts[track.type]
    trackCounts[track.type] += 1
    const typeName = track.type.charAt(0).toUpperCase() + track.type.slice(1)
    return { ...track, name: `${typeName} ${index}` }
  })

  return {
    ...omitLegacyVolume(raw),
    muted: raw.muted ?? false,
    volume_db: normalizedVolumeDb(raw.volume_db),
    frame_rate: frameRate,
    total_length: normalizeTotalLength(namedTracks, frameRate),
    tracks: namedTracks,
  }
}
