import { useEffect, useRef, useState } from 'react'
import { Download, ExternalLink, Loader2, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { TooltipProvider } from '@/components/ui/tooltip'
import { useCanvasScale } from '@/hooks/use-canvas-scale'
import { useElementWidth } from '@/hooks/use-element-width'
import { useMultiTrackHistory } from '@/hooks/use-multitrack-history'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import { LocaleContext, translate } from '@/lib/i18n'
import {
  addDefaultTaskSegmentIfRangeEmpty,
  createDefaultTrackData,
  calculateTotalLength,
  cloneMultiTrackSegment,
  createMultiTrackAudioContent,
  createMultiTrackVideoContent,
  deleteSegmentsWithLinkedTasks,
  deleteSegmentWithLinkedTasks,
  distributeMultiTrackSegmentsEvenly,
  getSelectedMultiTrackSegment,
  MULTITRACK_DEFAULT_VOLUME_DB,
  MULTITRACK_TRACK_COLORS,
  moveSelectedSegments,
  moveSegmentBetweenCompatibleTracks,
  normalizeTrackData,
  remapFrameToRate,
  remapTrackDataFrameRate,
  secondsToFrame,
  snapSecondsToFrame,
  snapTimeToFrame,
  syncMatchingTasksToPrimaryVideoTrack,
  syncMatchingTasksToPrimaryVideoSegment,
  updateMultiTrackSegmentContent,
  updateMultiTrackSegmentDuration,
} from '@/lib/multitrack-utils'
import { mediaContentToViewUrl } from '@/lib/media-url'
import {
  MODEL_MISSING_EVENT,
  MissingModelError,
  downloadEasyMediaModel,
  parseMissingModelPayload,
  type MissingModelInfo,
} from '@/lib/model-download'
import {
  applySmartSplit,
  applySmartSplitToMatchingTasks,
  hasMatchingTaskSegment,
  requestSmartSplit,
  splitTrackSegmentAtFrame,
} from '@/lib/smart-split'
import {
  applySubtitleRecognition,
  DEFAULT_SUBTITLE_STYLE,
  MULTITRACK_SUBTITLE_COLOR,
  requestSubtitleRecognition,
} from '@/lib/subtitle-recognition'
import { loadBrowserAudioMetadata } from '@/lib/audio-utils'
import { uuid } from '@/lib/uuid'
import { loadBrowserVideoMetadata } from '@/lib/video-utils'
import type { MultiTrack, MultiTrackSegment, MultiTrackSegmentContent, MultiTrackSourceType, MultiTrackType, TrackData } from '@/types/multitrack'
import { MultiTrackRuler } from './multitrack/MultiTrackRuler'
import { MultiTrackToolbar } from './multitrack/MultiTrackToolbar'
import { PreviewArea } from './multitrack/PreviewArea'
import { TrackArea } from './multitrack/TrackArea'

type CustomEventCallback = (event: CustomEvent<unknown>) => void

interface EasyMediaEventApi {
  addCustomEventListener?: (type: string, callback: CustomEventCallback) => void
  removeCustomEventListener?: (type: string, callback: CustomEventCallback) => void
}

function ensureTrackData(raw: unknown): TrackData {
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    const data = raw as Partial<TrackData>
    if (Array.isArray(data.tracks)) {
      return normalizeTrackData(raw as Parameters<typeof normalizeTrackData>[0])
    }
  }
  return createDefaultTrackData()
}

function insertSegmentAtFrame(
  segments: MultiTrackSegment[],
  nextSegment: MultiTrackSegment,
): MultiTrackSegment[] {
  const sorted = [...segments].sort((left, right) => left.start_frame - right.start_frame)
  const insertIndex = sorted.filter((segment) => (
    segment.start_frame + (segment.end_frame - segment.start_frame) / 2 < nextSegment.start_frame
  )).length
  const before = sorted.slice(0, insertIndex)
  const previousEnd = before.at(-1)?.end_frame ?? 0
  const duration = nextSegment.end_frame - nextSegment.start_frame
  const insertedStart = Math.max(nextSegment.start_frame, previousEnd)
  const inserted = { ...nextSegment, start_frame: insertedStart, end_frame: insertedStart + duration }
  let cursor = inserted.end_frame
  const after = sorted.slice(insertIndex).map((segment) => {
    const segmentDuration = segment.end_frame - segment.start_frame
    const shiftedStart = Math.max(segment.start_frame, cursor)
    cursor = shiftedStart + segmentDuration
    return { ...segment, start_frame: shiftedStart, end_frame: cursor }
  })
  return [...before, inserted, ...after]
}

export function MultiTrackWidget({ value, onChange, app, node }: Readonly<ReactWidgetProps<TrackData>>) {
  const committedData = ensureTrackData(value)
  const committedDataKey = JSON.stringify(committedData)
  const [resizePreviewData, setResizePreviewData] = useState<TrackData | null>(null)
  const data = resizePreviewData ?? committedData
  const dataRef = useRef(committedData)
  dataRef.current = committedData
  const [currentTime, setCurrentTime] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [timelineCollapsed, setTimelineCollapsed] = useState(false)
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null)
  const [selectedSegmentIds, setSelectedSegmentIds] = useState<Set<string>>(() => new Set())
  const [editingSubtitleSegmentId, setEditingSubtitleSegmentId] = useState<string | null>(null)
  const [isSmartSplitting, setIsSmartSplitting] = useState(false)
  const [isRecognizingSubtitles, setIsRecognizingSubtitles] = useState(false)
  const [syncPlayNonce, setSyncPlayNonce] = useState(0)
  const [missingModel, setMissingModel] = useState<MissingModelInfo | null>(null)
  const [isDownloadingModel, setIsDownloadingModel] = useState(false)
  const [modelDownloadError, setModelDownloadError] = useState<string | null>(null)
  const rafRef = useRef<number | null>(null)
  const timelineContainerRef = useRef<HTMLDivElement>(null)
  const startedAtRef = useRef(0)
  const startTimeRef = useRef(0)
  const currentTimeRef = useRef(0)
  const timelineWidth = Math.max(1, useElementWidth(timelineContainerRef))
  const scaledTimelineWidth = timelineWidth * zoom
  const canvasScale = useCanvasScale(app)
  const selectedSegment = selectedSegmentIds.size <= 1
    ? getSelectedMultiTrackSegment(data, selectedSegmentId)
    : null
  const selectedTaskTrackSegments = selectedSegment?.trackType === 'task'
    ? data.tracks.find((track) => track.id === selectedSegment.trackId && track.type === 'task')?.segments ?? [selectedSegment.segment]
    : undefined
  const {
    canUndo,
    canRedo,
    commitChange: commitTrackChange,
    undo: undoTrackChange,
    redo: redoTrackChange,
  } = useMultiTrackHistory(committedData, onChange)
  const locale = app?.ui?.settings?.settingsValues?.['Comfy.Locale']
  const t = (path: string, params?: Record<string, string | number>) => translate(locale, path, params)
  const missingModelDirectoryName = missingModel?.directory.split(/[\\/]/).filter(Boolean).at(-1) ?? ''

  function commitNormalizedTrackChange(nextData: TrackData) {
    commitTrackChange(normalizeTrackData(nextData))
  }

  function setSingleSelectedSegment(segmentId: string | null) {
    setSelectedSegmentId(segmentId)
    setSelectedSegmentIds(segmentId ? new Set([segmentId]) : new Set())
  }

  function handleSelectSegment(segmentId: string, mode: 'replace' | 'toggle' | 'add' = 'replace') {
    setSelectedSegmentIds((current) => {
      if (mode === 'replace') {
        setSelectedSegmentId(segmentId)
        return new Set([segmentId])
      }
      const next = new Set(current)
      if (mode === 'toggle' && next.has(segmentId)) {
        next.delete(segmentId)
        setSelectedSegmentId((active) => active === segmentId ? next.values().next().value ?? null : active)
        return next
      }
      next.add(segmentId)
      setSelectedSegmentId(segmentId)
      return next
    })
  }

  function handleSelectSegments(segmentIds: string[]) {
    const next = new Set(segmentIds)
    setSelectedSegmentIds(next)
    setSelectedSegmentId(segmentIds.at(-1) ?? null)
  }

  function handleClearSelection() {
    setSingleSelectedSegment(null)
  }

  useEffect(() => {
    setResizePreviewData(null)
    const validSegmentIds = new Set(committedData.tracks.flatMap((track) => track.segments.map((segment) => segment.id)))
    setSelectedSegmentIds((current) => {
      const next = new Set([...current].filter((segmentId) => validSegmentIds.has(segmentId)))
      setSelectedSegmentId((active) => active && !validSegmentIds.has(active) ? next.values().next().value ?? null : active)
      return next
    })
  }, [committedDataKey])

  useEffect(() => {
    currentTimeRef.current = currentTime
  }, [currentTime])

  useEffect(() => {
    const syncNode = node as { __easyMediaSyncPlay?: (startAt: number) => void }
    syncNode.__easyMediaSyncPlay = () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      currentTimeRef.current = 0
      startTimeRef.current = 0
      startedAtRef.current = performance.now()
      setCurrentTime(0)
      setIsPlaying(true)
      setSyncPlayNonce((value) => value + 1)
    }
    return () => {
      if (syncNode.__easyMediaSyncPlay) delete syncNode.__easyMediaSyncPlay
    }
  }, [node])

  useEffect(() => {
    const api = app.api as EasyMediaEventApi | undefined
    if (!api?.addCustomEventListener || !api.removeCustomEventListener) return
    const handleMissingModel: CustomEventCallback = (event) => {
      const model = parseMissingModelPayload(event.detail)
      if (!model) return
      setMissingModel(model)
      setModelDownloadError(null)
      setIsDownloadingModel(false)
    }
    api.addCustomEventListener(MODEL_MISSING_EVENT, handleMissingModel)
    return () => api.removeCustomEventListener?.(MODEL_MISSING_EVENT, handleMissingModel)
  }, [app])

  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
      return
    }

    startedAtRef.current = performance.now()
    startTimeRef.current = currentTimeRef.current

    function tick(now: number) {
      const elapsed = (now - startedAtRef.current) / 1000
      const next = snapTimeToFrame(startTimeRef.current + secondsToFrame(elapsed, data.frame_rate), data.frame_rate)
      if (next >= data.total_length) {
        currentTimeRef.current = data.total_length
        setCurrentTime(data.total_length)
        setIsPlaying(false)
        return
      }
      currentTimeRef.current = next
      setCurrentTime(next)
      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [isPlaying, syncPlayNonce, data.frame_rate, data.total_length])

  async function handleAddVideo(
    trackId: string,
    filePath: string,
    sourceType: MultiTrackSourceType,
    requestedStartFrame?: number,
  ) {
    const content = createMultiTrackVideoContent(filePath, sourceType)
    const src = mediaContentToViewUrl(content)
    let duration = 1
    if (src) {
      try {
        const metadata = await loadBrowserVideoMetadata(src)
        duration = Math.max(metadata.duration, 1)
      } catch (error) {
        console.error('[MultiTrackWidget] failed to read video metadata:', error)
      }
    }

    const latestData = dataRef.current
    const addedVideoRange = { added: false, startFrame: 0, endFrame: 0 }
    const videoUpdatedTracks = latestData.tracks.map((track) => {
      if (track.id !== trackId) return track
      const startFrame = requestedStartFrame === undefined
        ? snapTimeToFrame(track.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0), latestData.frame_rate)
        : Math.max(0, Math.round(requestedStartFrame))
      const endFrame = startFrame + Math.max(1, snapSecondsToFrame(duration, latestData.frame_rate))
      addedVideoRange.added = true
      addedVideoRange.startFrame = startFrame
      addedVideoRange.endFrame = endFrame
      return {
        ...track,
        segments: (() => {
          const nextSegment: MultiTrackSegment = {
            id: uuid(),
            start_frame: startFrame,
            end_frame: endFrame,
            color: track.color,
            content: {
              ...content,
              duration,
            },
          }
          if (requestedStartFrame === undefined) return [...track.segments, nextSegment]
          return insertSegmentAtFrame(track.segments, nextSegment)
        })(),
      }
    })
    const taskSyncedTracks = requestedStartFrame === undefined
      ? videoUpdatedTracks
      : syncMatchingTasksToPrimaryVideoTrack(latestData.tracks, videoUpdatedTracks)
    const updatedTracks = addedVideoRange.added
      ? addDefaultTaskSegmentIfRangeEmpty(taskSyncedTracks, addedVideoRange.startFrame, addedVideoRange.endFrame)
      : taskSyncedTracks

    commitNormalizedTrackChange({
      ...latestData,
      tracks: updatedTracks,
      total_length: calculateTotalLength(updatedTracks, latestData.frame_rate),
    })
  }

  async function handleAddAudio(
    trackId: string,
    filePath: string,
    sourceType: MultiTrackSourceType,
    previewUrl?: string,
    requestedStartFrame?: number,
  ) {
    const content = createMultiTrackAudioContent(filePath, sourceType)
    if (previewUrl) content.url = previewUrl
    const src = mediaContentToViewUrl(content)
    let duration = 5
    if (src) {
      try {
        duration = (await loadBrowserAudioMetadata(src)).duration
      } catch (error) {
        console.error('[MultiTrackWidget] failed to read audio metadata:', error)
      }
    }
    const latestData = dataRef.current
    const updatedTracks = latestData.tracks.map((track) => {
      if (track.id !== trackId || track.type !== 'audio') return track
      const startFrame = requestedStartFrame === undefined
        ? track.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0)
        : Math.max(0, Math.round(requestedStartFrame))
      const endFrame = startFrame + Math.max(1, snapSecondsToFrame(duration, latestData.frame_rate))
      return {
        ...track,
        segments: (() => {
          const nextSegment: MultiTrackSegment = {
            id: uuid(),
            start_frame: startFrame,
            end_frame: endFrame,
            color: track.color,
            content: { ...content, duration },
          }
          if (requestedStartFrame === undefined) return [...track.segments, nextSegment]
          return insertSegmentAtFrame(track.segments, nextSegment)
        })(),
      }
    })
    commitNormalizedTrackChange({ ...latestData, tracks: updatedTracks, total_length: calculateTotalLength(updatedTracks, latestData.frame_rate) })
  }

  function handleAddTrack(type: MultiTrackType) {
    if (type !== 'audio' && type !== 'subtitle') return
    const trackNumber = data.tracks.filter((track) => track.type === type).length
    if ((type === 'audio' || type === 'subtitle') && trackNumber >= 2) return
    const track: MultiTrack = {
      id: uuid(),
      name: `${type === 'audio' ? 'Audio' : 'Subtitle'} ${trackNumber}`,
      type,
      color: type === 'subtitle' ? '#9D4937' : MULTITRACK_TRACK_COLORS.audio,
      muted: false,
      solo: type === 'audio' ? false : undefined,
      volume_db: type === 'audio' ? MULTITRACK_DEFAULT_VOLUME_DB : undefined,
      locked: false,
      segments: [],
    }
    commitNormalizedTrackChange({ ...data, tracks: [...data.tracks, track] })
  }

  function handleDeleteTrack(trackId: string) {
    const target = data.tracks.find((track) => track.id === trackId)
    const firstVideoTrackId = data.tracks.find((track) => track.type === 'video')?.id
    if (!target || target.type === 'task' || target.id === firstVideoTrackId) return
    const segmentIds = new Set(target.segments.map((segment) => segment.id))
    const updatedTracks = data.tracks.filter((track) => track.id !== trackId)
    commitNormalizedTrackChange({ ...data, tracks: updatedTracks, total_length: calculateTotalLength(updatedTracks, data.frame_rate) })
    setSelectedSegmentIds((current) => {
      const next = new Set([...current].filter((segmentId) => !segmentIds.has(segmentId)))
      setSelectedSegmentId((active) => active && segmentIds.has(active) ? next.values().next().value ?? null : active)
      return next
    })
  }

  function handleTrackAudioSettingsChange(
    trackId: string,
    patch: Partial<Pick<MultiTrack, 'muted' | 'solo'>>,
  ) {
    commitNormalizedTrackChange({
      ...data,
      tracks: data.tracks.map((track) => track.id === trackId ? { ...track, ...patch } : track),
    })
  }

  async function handleReplaceVideo(
    trackId: string,
    segmentId: string,
    filePath: string,
    sourceType: MultiTrackSourceType,
  ) {
    const content = createMultiTrackVideoContent(filePath, sourceType)
    const src = mediaContentToViewUrl(content)
    let duration = 1
    if (src) {
      try {
        const metadata = await loadBrowserVideoMetadata(src)
        duration = Math.max(metadata.duration, 1)
      } catch (error) {
        console.error('[MultiTrackWidget] failed to read replacement video metadata:', error)
      }
    }

    const updatedTracks = data.tracks.map((track) => {
      if (track.id !== trackId) return track
      return {
        ...track,
        segments: track.segments.map((segment) => (
          segment.id === segmentId
            ? {
                ...segment,
                content: {
                  ...content,
                  duration,
                },
              }
            : segment
        )),
      }
    })

    commitNormalizedTrackChange({
      ...data,
      tracks: updatedTracks,
      total_length: calculateTotalLength(updatedTracks, data.frame_rate),
    })
  }

  function handleAddTaskSegment(trackId: string) {
    const updatedTracks = data.tracks.map((track) => {
      if (track.id !== trackId || track.type !== 'task') return track
      const startFrame = snapTimeToFrame(
        track.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0),
        data.frame_rate,
      )
      const endFrame = startFrame + Math.max(1, secondsToFrame(5, data.frame_rate))
      return {
        ...track,
        segments: [
          ...track.segments,
          {
            id: uuid(),
            start_frame: startFrame,
            end_frame: endFrame,
            color: track.color,
            content: {
              media_type: 'none' as const,
              task_mode: track.task_mode ?? 'default',
            },
          },
        ],
      }
    })

    commitNormalizedTrackChange({
      ...data,
      tracks: updatedTracks,
      total_length: calculateTotalLength(updatedTracks, data.frame_rate),
    })
  }

  function handleAddSubtitleSegment(trackId: string) {
    const updatedTracks = data.tracks.map((track) => {
      if (track.id !== trackId || track.type !== 'subtitle') return track
      const startFrame = snapTimeToFrame(
        track.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0),
        data.frame_rate,
      )
      const endFrame = startFrame + Math.max(1, secondsToFrame(5, data.frame_rate))
      const existingStyle = track.segments.find((segment) => segment.content.subtitle_style)?.content.subtitle_style
      return {
        ...track,
        segments: [
          ...track.segments,
          {
            id: uuid(),
            start_frame: startFrame,
            end_frame: endFrame,
            color: track.color || MULTITRACK_SUBTITLE_COLOR,
            content: {
              media_type: 'subtitle' as const,
              text: '默认文字',
              subtitle_style: { ...(existingStyle ?? DEFAULT_SUBTITLE_STYLE) },
            },
          },
        ],
      }
    })

    commitNormalizedTrackChange({
      ...data,
      tracks: updatedTracks,
      total_length: calculateTotalLength(updatedTracks, data.frame_rate),
    })
  }

  function handleDeleteSegment(segmentId: string) {
    const idsToDelete = selectedSegmentIds.has(segmentId) ? selectedSegmentIds : new Set([segmentId])
    const updatedTracks = idsToDelete.size > 1
      ? deleteSegmentsWithLinkedTasks(data.tracks, idsToDelete, data.frame_rate)
      : deleteSegmentWithLinkedTasks(data.tracks, segmentId, data.frame_rate)
    const totalLength = calculateTotalLength(updatedTracks, data.frame_rate)
    commitNormalizedTrackChange({
      ...data,
      tracks: updatedTracks,
      total_length: totalLength,
    })
    handleClearSelection()
    setCurrentTime((time) => Math.min(time, totalLength))
  }

  function handleDistributeTaskSegments(trackId: string) {
    const updatedTracks = data.tracks.map((track) => (
      track.id === trackId && track.type === 'task'
        ? { ...track, segments: distributeMultiTrackSegmentsEvenly(track.segments, data.total_length) }
        : track
    ))
    commitNormalizedTrackChange({ ...data, tracks: updatedTracks })
  }

  function handleCloneTaskSegment(trackId: string, segmentId: string) {
    let clonedSegmentId: string | null = null
    let taskTrackEnd = 0
    const updatedTracks = data.tracks.map((track) => {
      if (track.id !== trackId || track.type !== 'task') return track
      const result = cloneMultiTrackSegment(track.segments, segmentId)
      if (!result) return track
      clonedSegmentId = result.clonedSegmentId
      taskTrackEnd = result.segments.reduce((max, segment) => Math.max(max, segment.end_frame), 0)
      return { ...track, segments: result.segments }
    })
    if (!clonedSegmentId) return
    commitNormalizedTrackChange({
      ...data,
      tracks: updatedTracks,
      total_length: Math.max(data.total_length, taskTrackEnd),
    })
    setSingleSelectedSegment(clonedSegmentId)
  }

  function buildResizedTrackData(sourceData: TrackData, segmentId: string, edge: 'start' | 'end', nextTime: number): TrackData {
    const resizedTracks = sourceData.tracks.map((track) => ({
      ...track,
      segments: track.segments.map((segment) => {
        if (segment.id !== segmentId) return segment
        const sortedSegments = [...track.segments].sort((a, b) => a.start_frame - b.start_frame)
        const segmentIndex = sortedSegments.findIndex((item) => item.id === segmentId)
        const prevSegment = segmentIndex > 0 ? sortedSegments[segmentIndex - 1] : null
        const nextSegment = segmentIndex >= 0 && segmentIndex < sortedSegments.length - 1
          ? sortedSegments[segmentIndex + 1]
          : null
        const sourceDuration = segment.content.duration && segment.content.duration > 0
          ? Math.max(1, snapSecondsToFrame(segment.content.duration, sourceData.frame_rate))
          : Number.POSITIVE_INFINITY

        if (edge === 'start') {
          const nextStart = snapTimeToFrame(nextTime, sourceData.frame_rate)
          const minStart = Math.max(0, prevSegment?.end_frame ?? 0, segment.end_frame - sourceDuration)
          const maxStart = segment.end_frame - 1
          return {
            ...segment,
            start_frame: Math.max(minStart, Math.min(nextStart, maxStart)),
          }
        }

        const nextEnd = snapTimeToFrame(nextTime, sourceData.frame_rate)
        const minEnd = segment.start_frame + 1
        const maxEnd = Math.min(nextSegment?.start_frame ?? Number.POSITIVE_INFINITY, segment.start_frame + sourceDuration)
        return {
          ...segment,
          end_frame: Math.max(minEnd, Math.min(nextEnd, maxEnd)),
        }
      }).sort((a, b) => a.start_frame - b.start_frame),
    }))
    const updatedTracks = syncMatchingTasksToPrimaryVideoSegment(
      sourceData.tracks,
      resizedTracks,
      segmentId,
    )
    return {
      ...sourceData,
      tracks: updatedTracks,
      total_length: calculateTotalLength(updatedTracks, sourceData.frame_rate),
    }
  }

  function handleResizeSegmentPreview(segmentId: string, edge: 'start' | 'end', nextTime: number) {
    setResizePreviewData(buildResizedTrackData(committedData, segmentId, edge, nextTime))
  }

  function handleResizeSegment(segmentId: string, edge: 'start' | 'end', nextTime: number) {
    setResizePreviewData(null)
    commitNormalizedTrackChange(buildResizedTrackData(committedData, segmentId, edge, nextTime))
  }

  function handleMoveSegment(segmentId: string, targetTrackId: string, nextStartTime: number) {
    const updatedTracks = selectedSegmentIds.has(segmentId)
      ? moveSelectedSegments(data.tracks, selectedSegmentIds, segmentId, targetTrackId, nextStartTime, data.frame_rate)
      : moveSegmentBetweenCompatibleTracks(data.tracks, segmentId, targetTrackId, nextStartTime, data.frame_rate)

    commitNormalizedTrackChange({
      ...data,
      tracks: updatedTracks,
      total_length: calculateTotalLength(updatedTracks, data.frame_rate),
    })
  }

  function handleGlobalSettingsChange(patch: Partial<Pick<TrackData, 'muted' | 'volume_db' | 'frame_rate'>>) {
    const nextFrameRate = patch.frame_rate
    if (typeof nextFrameRate === 'number' && nextFrameRate > 0 && nextFrameRate !== data.frame_rate) {
      const remapped = remapTrackDataFrameRate(data, nextFrameRate)
      const nextCurrentTime = remapFrameToRate(currentTimeRef.current, data.frame_rate, remapped.frame_rate)
      currentTimeRef.current = nextCurrentTime
      setCurrentTime(nextCurrentTime)
      commitNormalizedTrackChange({
        ...remapped,
        ...patch,
        frame_rate: remapped.frame_rate,
      })
      return
    }

    commitNormalizedTrackChange({ ...data, ...patch, frame_rate: data.frame_rate })
  }

  function handleSelectedSegmentContentChange(patch: Partial<MultiTrackSegmentContent>) {
    if (!selectedSegmentId) return
    if (selectedSegment?.trackType === 'subtitle' && patch.subtitle_style) {
      const { subtitle_style: subtitleStyle, ...selectedSegmentPatch } = patch
      const hasSelectedSegmentPatch = Object.keys(selectedSegmentPatch).length > 0
      const updatedTracks = data.tracks.map((track) => {
        if (track.id !== selectedSegment.trackId || track.type !== 'subtitle') return track
        return {
          ...track,
          segments: track.segments.map((segment) => ({
            ...segment,
            content: {
              ...segment.content,
              ...(hasSelectedSegmentPatch && segment.id === selectedSegmentId ? selectedSegmentPatch : {}),
              subtitle_style: subtitleStyle,
            },
          })),
        }
      })
      commitNormalizedTrackChange({
        ...data,
        tracks: updatedTracks,
      })
      return
    }
    commitNormalizedTrackChange(updateMultiTrackSegmentContent(data, selectedSegmentId, patch))
  }

  function handleTrackSegmentsContentChange(updates: Array<{ segmentId: string; patch: Partial<MultiTrackSegmentContent> }>) {
    const updateMap = new Map(updates.map((update) => [update.segmentId, update.patch]))
    const updatedTracks = data.tracks.map((track) => ({
      ...track,
      segments: track.segments.map((segment) => {
        const patch = updateMap.get(segment.id)
        if (!patch) return segment
        return {
          ...segment,
          content: {
            ...segment.content,
            ...patch,
          },
        }
      }),
    }))
    commitNormalizedTrackChange({
      ...data,
      tracks: updatedTracks,
    })
  }

  function handleTaskTrackSegmentsChange(segments: MultiTrackSegment[]) {
    if (selectedSegment?.trackType !== 'task') return
    const updatedTracks = data.tracks.map((track) => (
      track.id === selectedSegment.trackId ? { ...track, segments } : track
    ))
    commitNormalizedTrackChange({
      ...data,
      tracks: updatedTracks,
      total_length: calculateTotalLength(updatedTracks, data.frame_rate),
    })
  }

  function handleSelectedSegmentDurationChange(duration: number) {
    if (!selectedSegmentId) return
    commitNormalizedTrackChange(updateMultiTrackSegmentDuration(data, selectedSegmentId, duration, data.frame_rate))
  }

  async function handleSmartSplit(segmentId: string) {
    const segment = data.tracks
      .find((track) => track.type === 'video' && track.segments.some((item) => item.id === segmentId))
      ?.segments.find((item) => item.id === segmentId)
    if (!segment || isSmartSplitting) return

    setIsPlaying(false)
    setIsSmartSplitting(true)
    try {
      const result = await requestSmartSplit(segment, data.frame_rate)
      commitNormalizedTrackChange(applySmartSplit(data, segmentId, result))
    } catch (error) {
      if (error instanceof MissingModelError) {
        setMissingModel(error.model)
        setModelDownloadError(null)
        return
      }
      console.error('[MultiTrackWidget] smart split failed:', error)
      const message = error instanceof Error ? error.message : String(error)
      try {
        app.extensionManager.toast.add({
          severity: 'error',
          summary: t('multitrack.smartSplitFailed'),
          detail: message,
          life: 5000,
        })
      } catch (toastError) {
        console.error('[MultiTrackWidget] failed to show smart split error:', toastError)
      }
    } finally {
      setIsSmartSplitting(false)
    }
  }

  async function handleSmartSplitTasks(segmentId: string) {
    const segment = data.tracks
      .find((track) => track.type === 'video' && track.segments.some((item) => item.id === segmentId))
      ?.segments.find((item) => item.id === segmentId)
    if (!segment || isSmartSplitting) return
    if (!hasMatchingTaskSegment(data, segmentId)) {
      try {
        app.extensionManager.toast.add({
          severity: 'warn',
          summary: t('multitrack.smartSplitTasksOnly'),
          detail: t('multitrack.noMatchingTaskSegment'),
          life: 4000,
        })
      } catch (toastError) {
        console.error('[MultiTrackWidget] failed to show task split warning:', toastError)
      }
      return
    }

    setIsPlaying(false)
    setIsSmartSplitting(true)
    try {
      const result = await requestSmartSplit(segment, data.frame_rate)
      commitNormalizedTrackChange(applySmartSplitToMatchingTasks(data, segmentId, result))
    } catch (error) {
      if (error instanceof MissingModelError) {
        setMissingModel(error.model)
        setModelDownloadError(null)
        return
      }
      console.error('[MultiTrackWidget] task-only smart split failed:', error)
      const message = error instanceof Error ? error.message : String(error)
      try {
        app.extensionManager.toast.add({
          severity: 'error',
          summary: t('multitrack.smartSplitFailed'),
          detail: message,
          life: 5000,
        })
      } catch (toastError) {
        console.error('[MultiTrackWidget] failed to show smart split error:', toastError)
      }
    } finally {
      setIsSmartSplitting(false)
    }
  }

  async function handleRecognizeSubtitles(segmentId: string) {
    const segment = data.tracks
      .find((track) => (track.type === 'video' || track.type === 'audio') && track.segments.some((item) => item.id === segmentId))
      ?.segments.find((item) => item.id === segmentId)
    if (!segment || isRecognizingSubtitles) return

    setIsPlaying(false)
    setIsRecognizingSubtitles(true)
    try {
      const result = await requestSubtitleRecognition(segment, data.frame_rate)
      commitNormalizedTrackChange(applySubtitleRecognition(data, segmentId, result))
    } catch (error) {
      if (error instanceof MissingModelError) {
        setMissingModel(error.model)
        setModelDownloadError(null)
        return
      }
      console.error('[MultiTrackWidget] subtitle recognition failed:', error)
      const message = error instanceof Error ? error.message : String(error)
      try {
        app.extensionManager?.toast?.add({
          severity: 'error',
          summary: t('multitrack.subtitleRecognitionFailed'),
          detail: message,
          life: 5000,
        })
      } catch (toastError) {
        console.error('[MultiTrackWidget] failed to show subtitle recognition error:', toastError)
      }
    } finally {
      setIsRecognizingSubtitles(false)
    }
  }

  function handleCutSegment(segmentId: string, splitFrame: number) {
    commitNormalizedTrackChange(splitTrackSegmentAtFrame(data, segmentId, splitFrame))
  }

  function handleCutAtCurrentTime() {
    const splitFrame = snapTimeToFrame(currentTime, data.frame_rate)
    const targetSegmentIds = selectedSegmentIds.size > 0
      ? Array.from(selectedSegmentIds)
      : data.tracks.flatMap((track) => (
        track.segments
          .filter((segment) => splitFrame > segment.start_frame && splitFrame < segment.end_frame)
          .map((segment) => segment.id)
      ))
    if (targetSegmentIds.length === 0) return

    const originalSegmentCount = data.tracks.reduce((count, track) => count + track.segments.length, 0)
    const nextData = targetSegmentIds.reduce(
      (currentData, segmentId) => splitTrackSegmentAtFrame(currentData, segmentId, splitFrame),
      data,
    )
    const nextSegmentCount = nextData.tracks.reduce((count, track) => count + track.segments.length, 0)
    if (nextSegmentCount === originalSegmentCount) return

    commitNormalizedTrackChange(nextData)
  }

  async function handleDownloadMissingModel() {
    if (!missingModel || isDownloadingModel) return
    setIsDownloadingModel(true)
    setModelDownloadError(null)
    try {
      const downloadedModel = await downloadEasyMediaModel(missingModel.name)
      setMissingModel(null)
      try {
        app.extensionManager?.toast?.add({
          severity: 'success',
          summary: t('modelDownload.downloadComplete'),
          detail: t('modelDownload.downloadCompleteDetail', { name: downloadedModel.display_name }),
          life: 5000,
        })
      } catch (toastError) {
        console.error('[MultiTrackWidget] failed to show model download success:', toastError)
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      setModelDownloadError(message)
    } finally {
      setIsDownloadingModel(false)
    }
  }

  function handleManualDownload() {
    if (!missingModel) return
    for (const url of missingModel.urls ?? [missingModel.url]) {
      globalThis.open(url, '_blank', 'noopener,noreferrer')
    }
  }

  return (
    <LocaleContext.Provider value={locale}>
      <TooltipProvider>
        <div
          className="relative flex h-full w-full flex-col overflow-hidden rounded text-foreground font-sans text-xs select-none"
          aria-busy={isSmartSplitting || isRecognizingSubtitles}
          onClick={() => {
            handleClearSelection()
          }}
        >
          <PreviewArea
            data={data}
            currentTime={currentTime}
            selectedSegment={selectedSegment}
            isPlaying={isPlaying}
            playbackNonce={syncPlayNonce}
            node={node}
            editingSubtitleSegmentId={editingSubtitleSegmentId}
            onSubtitleEditRequestHandled={() => setEditingSubtitleSegmentId(null)}
            onSelectSegment={(segmentId) => handleSelectSegment(segmentId)}
            onGlobalSettingsChange={handleGlobalSettingsChange}
            onSelectedSegmentContentChange={handleSelectedSegmentContentChange}
            taskSegments={selectedTaskTrackSegments}
            onTrackSegmentsContentChange={handleTrackSegmentsContentChange}
            onTaskTrackSegmentsChange={handleTaskTrackSegmentsChange}
            onSelectedSegmentDurationChange={handleSelectedSegmentDurationChange}
          />
          <MultiTrackToolbar
            currentTime={currentTime}
            totalLength={data.total_length}
            frameRate={data.frame_rate}
            isPlaying={isPlaying}
            zoom={zoom}
            timelineCollapsed={timelineCollapsed}
            onPlayPause={() => setIsPlaying((value) => !value)}
            onZoomChange={setZoom}
            onToggleTimeline={() => setTimelineCollapsed((collapsed) => !collapsed)}
            canDelete={selectedSegmentIds.size > 0}
            onDeleteSelected={() => {
              if (selectedSegmentId) handleDeleteSegment(selectedSegmentId)
            }}
            onCutAtCurrentTime={handleCutAtCurrentTime}
            canUndo={canUndo}
            canRedo={canRedo}
            onUndo={undoTrackChange}
            onRedo={redoTrackChange}
          />
          <div
            data-testid="multitrack-timeline-panel"
            aria-hidden={timelineCollapsed}
            className={`grid shrink-0 transition-[grid-template-rows] duration-300 ease-in-out ${timelineCollapsed ? 'grid-rows-[0fr]' : 'grid-rows-[1fr]'}`}
          >
            <div className="min-h-0 shrink-0 overflow-hidden">
              <div ref={timelineContainerRef} className="no-scrollbar shrink-0 overflow-x-auto overflow-y-hidden">
                <div className="min-h-full" style={{ width: scaledTimelineWidth, minWidth: '100%' }}>
                  <MultiTrackRuler
                    totalLength={data.total_length}
                    frameRate={data.frame_rate}
                    width={scaledTimelineWidth}
                    canvasScale={canvasScale}
                    currentTime={currentTime}
                    onSeek={(time) => setCurrentTime(snapTimeToFrame(time, data.frame_rate))}
                  />
                  <TrackArea
                    data={data}
                    node={node}
                    app={app}
                    width={scaledTimelineWidth}
                    currentTime={currentTime}
                    canvasScale={canvasScale}
                    selectedSegmentIds={selectedSegmentIds}
                    onAddVideo={handleAddVideo}
                    onAddAudio={handleAddAudio}
                    onAddTrack={handleAddTrack}
                    onAddSubtitleSegment={handleAddSubtitleSegment}
                    onReplaceVideo={handleReplaceVideo}
                    onAddTaskSegment={handleAddTaskSegment}
                    onSelectSegment={handleSelectSegment}
                    onSelectSegments={handleSelectSegments}
                    onClearSelection={handleClearSelection}
                    onDeleteSegment={handleDeleteSegment}
                    onDeleteTrack={handleDeleteTrack}
                    onTrackAudioSettingsChange={handleTrackAudioSettingsChange}
                    onDistributeTaskSegments={handleDistributeTaskSegments}
                    onCloneTaskSegment={handleCloneTaskSegment}
                    onResizeSegment={handleResizeSegment}
                    onResizeSegmentPreview={handleResizeSegmentPreview}
                    onMoveSegment={handleMoveSegment}
                    onSmartSplit={handleSmartSplit}
                    onSmartSplitTasks={handleSmartSplitTasks}
                    onRecognizeSubtitles={handleRecognizeSubtitles}
                    onEditSubtitleSegment={setEditingSubtitleSegmentId}
                    cutMode={false}
                    onCutSegment={handleCutSegment}
                  />
                </div>
              </div>
            </div>
          </div>
          {isSmartSplitting ? (
            <div
              className="absolute inset-0 z-50 flex items-center justify-center bg-background/80"
              data-testid="smart-split-overlay"
              role="status"
              aria-live="polite"
            >
              <div className="flex items-center gap-2 rounded border border-border bg-background px-3 py-2 text-foreground shadow-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>{t('multitrack.smartSplitting')}</span>
              </div>
            </div>
          ) : null}
          {isRecognizingSubtitles ? (
            <div
              className="absolute inset-0 z-50 flex items-center justify-center bg-background/80"
              data-testid="subtitle-recognition-overlay"
              role="status"
              aria-live="polite"
            >
              <div className="flex items-center gap-2 rounded border border-border bg-background px-3 py-2 text-foreground shadow-sm">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>{t('multitrack.recognizingSubtitles')}</span>
              </div>
            </div>
          ) : null}
          {missingModel ? (
            <div
              className="absolute inset-0 z-[60] flex items-center justify-center bg-background/90 p-4"
              data-testid="missing-model-overlay"
              role="dialog"
              aria-modal="true"
              aria-labelledby="missing-model-title"
            >
              <div className="w-full max-w-md rounded border border-border bg-background p-4 text-foreground shadow-sm">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h3 id="missing-model-title" className="text-sm font-semibold">
                      {t('modelDownload.title', { name: missingModel.display_name })}
                    </h3>
                    <p className="mt-2 text-xs leading-5 text-muted-foreground">
                      {t('modelDownload.description', {
                        name: missingModel.display_name,
                        directory: missingModelDirectoryName,
                      })}
                    </p>
                  </div>
                  <Button
                    type="button"
                    variant="ghost"
                    size="icon"
                    onClick={() => setMissingModel(null)}
                    aria-label={t('modelDownload.close')}
                    disabled={isDownloadingModel}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
                <div className="mt-3 rounded border border-border bg-muted px-2 py-1.5 font-mono text-[8px] leading-4 text-foreground break-all">
                  {missingModel.path}
                </div>
                {modelDownloadError ? (
                  <p className="mt-3 text-xs leading-5 text-destructive">{modelDownloadError}</p>
                ) : null}
                <div className="mt-4 flex flex-wrap justify-end gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleManualDownload}
                    disabled={isDownloadingModel}
                  >
                    <ExternalLink className="h-4 w-4" />
                    {t('modelDownload.manual')}
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    onClick={handleDownloadMissingModel}
                    disabled={isDownloadingModel}
                  >
                    {isDownloadingModel ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
                    {isDownloadingModel ? t('modelDownload.downloading') : t('modelDownload.auto')}
                  </Button>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </TooltipProvider>
    </LocaleContext.Provider>
  )
}
