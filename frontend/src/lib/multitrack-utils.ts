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
export const MULTITRACK_TASK_MODES = ['default', 'ref', 'edit'] as const
export const MULTITRACK_DEFAULT_TASK_MODE: MultiTrackTaskMode = 'default'
export const MULTITRACK_DEFAULT_VOLUME = 1
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

export function calculateTotalLength(tracks: MultiTrack[]): number {
  const maxEnd = calculateMaxSegmentEnd(tracks)
  return Math.max(MULTITRACK_DEFAULT_TOTAL_LENGTH, maxEnd)
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

function packSegmentsFromZero(segments: MultiTrackSegment[], frameRate: number): MultiTrackSegment[] {
  void frameRate
  let cursor = 0
  return segments.map((segment) => {
    const duration = segmentDuration(segment)
    const startFrame = Math.round(cursor)
    const endFrame = Math.round(startFrame + duration)
    cursor = endFrame
    return {
      ...segment,
      start_frame: startFrame,
      end_frame: endFrame,
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

export function deleteSegmentWithLinkedTasks(tracks: MultiTrack[], segmentId: string): MultiTrack[] {
  const sourceTrack = tracks.find((track) => track.segments.some((segment) => segment.id === segmentId))
  const deletedSegment = sourceTrack?.segments.find((segment) => segment.id === segmentId)
  if (!sourceTrack || !deletedSegment) return tracks

  const shouldDeleteLinkedTask = (segment: MultiTrackSegment) => (
    sourceTrack.type === 'video' &&
    rangesOverlap(segment.start_frame, segment.end_frame, deletedSegment.start_frame, deletedSegment.end_frame)
  )

  return tracks.map((track) => {
    if (track.id === sourceTrack.id) {
      return {
        ...track,
        segments: track.segments.filter((segment) => segment.id !== segmentId),
      }
    }
    if (track.type !== 'task') return track
    return {
      ...track,
      segments: track.segments.filter((segment) => !shouldDeleteLinkedTask(segment)),
    }
  })
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

  const movedSegment = {
    ...movingSegment,
    start_frame: 0,
    end_frame: duration,
  }
  const nextTargetSegments = packSegmentsFromZero([
    ...targetSegments.slice(0, insertIndex),
    movedSegment,
    ...targetSegments.slice(insertIndex),
  ], frameRate)

  return tracks.map((track) => {
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
  const placeholderSegment: MultiTrackSegment = {
    ...movingSegment,
    start_frame: 0,
    end_frame: duration,
  }
  const packed = packSegmentsFromZero([
    ...targetSegments.slice(0, insertIndex),
    placeholderSegment,
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
  const placeholderSegment: MultiTrackSegment = {
    ...movingSegment,
    start_frame: 0,
    end_frame: duration,
  }

  return packSegmentsFromZero([
    ...targetSegments.slice(0, placeholder.insertIndex),
    placeholderSegment,
    ...targetSegments.slice(placeholder.insertIndex),
  ], frameRate).filter((segment) => segment.id !== placeholder.segmentId)
}

export interface ActivePreviewVideoSegment {
  trackId: string
  segment: MultiTrackSegment
  localTime: number
}

function segmentContainsTime(segment: MultiTrackSegment, time: number): boolean {
  return time >= segment.start_frame && time < segment.end_frame
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
        localTime: frameToSeconds(Math.max(0, currentFrame - segment.start_frame), data.frame_rate),
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
      localTime: frameToSeconds(Math.max(0, currentFrame - segment.start_frame), data.frame_rate),
    }
  }
  return null
}

export interface SelectedMultiTrackSegment {
  trackId: string
  trackType: MultiTrackType
  segment: MultiTrackSegment
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

export function updateMultiTrackSegmentDuration(
  data: TrackData,
  segmentId: string,
  duration: number,
  frameRate: number,
): TrackData {
  const nextDuration = Math.max(1, snapSecondsToFrame(duration, frameRate))
  const tracks = data.tracks.map((track) => {
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

  return {
    ...data,
    tracks,
    total_length: calculateTotalLength(tracks),
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
      }
    }),
  }))

  return {
    ...data,
    frame_rate: safeNextFrameRate,
    tracks,
    total_length: Math.max(
      calculateMaxSegmentEnd(tracks),
      remapFrameToRate(data.total_length, data.frame_rate, safeNextFrameRate),
    ),
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

function normalizeLegacySegment(segment: LegacyMultiTrackSegment): MultiTrackSegment {
  const startFrame = normalizeFrameValue(segment.start_frame)
    ?? 0
  const endFrame = normalizeFrameValue(segment.end_frame)
    ?? startFrame + 1

  return {
    ...segment,
    start_frame: startFrame,
    end_frame: Math.max(startFrame + 1, endFrame),
  }
}

function normalizeTrackSegments(track: LegacyMultiTrack): MultiTrackSegment[] {
  return track.segments
    .map((segment) => normalizeLegacySegment(segment))
    .sort((a, b) => a.start_frame - b.start_frame)
}

function normalizeTotalLength(raw: LegacyTrackData, tracks: MultiTrack[]): number {
  const totalLength = normalizeFrameValue(raw.total_length)
  if (totalLength !== null) return Math.max(1, totalLength)

  return calculateTotalLength(tracks)
}

export function createDefaultTrackData(): TrackData {
  return {
    muted: false,
    volume: MULTITRACK_DEFAULT_VOLUME,
    tracks: [
      {
        id: uuid(),
        name: 'Task 1',
        type: 'task',
        task_mode: MULTITRACK_DEFAULT_TASK_MODE,
        color: MULTITRACK_TRACK_COLORS.task,
        muted: false,
        locked: false,
        segments: [],
      },
      {
        id: uuid(),
        name: 'Video 1',
        type: 'video',
        color: MULTITRACK_TRACK_COLORS.video,
        muted: false,
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
    const segments = normalizeTrackSegments(track)
    if (track.type !== 'segment') {
      if (track.type === 'task') {
        return {
          ...track,
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
        ...track,
        type: track.type,
        segments,
      } as MultiTrack
    }

    return {
      ...track,
      name: track.name === 'Segment 1' ? 'Task 1' : track.name,
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

  return {
    ...raw,
    muted: raw.muted ?? false,
    volume: raw.volume ?? MULTITRACK_DEFAULT_VOLUME,
    frame_rate: frameRate,
    total_length: normalizeTotalLength(raw, tracks),
    tracks,
  }
}
