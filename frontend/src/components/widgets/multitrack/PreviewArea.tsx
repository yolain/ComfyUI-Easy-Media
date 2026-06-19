import { useEffect, useMemo, useState } from 'react'
import {
  collectMultiTrackPreviewResolutionInput,
  frameToSeconds,
  getActivePreviewVideoSegment,
  parseMultiTrackPreviewResolution,
  segmentDuration,
  type MultiTrackVideoMetadata,
  type SelectedMultiTrackSegment,
} from '@/lib/multitrack-utils'
import { mediaContentToViewUrl } from '@/lib/media-url'
import { loadBrowserVideoMetadata } from '@/lib/video-utils'
import type { MultiTrackSegment, MultiTrackSegmentContent, TrackData } from '@/types/multitrack'
import { PreviewFloatingToolbar } from './PreviewFloatingToolbar'
import { TaskSegmentEditor } from './TaskSegmentEditor'
import { VideoPreview } from './VideoPreview'

const RESOLUTION_POLL_INTERVAL_MS = 250

interface PreviewAreaProps {
  data: TrackData
  currentTime: number
  selectedSegment: SelectedMultiTrackSegment | null
  isPlaying: boolean
  node: unknown
  onGlobalSettingsChange: (patch: Partial<Pick<TrackData, 'muted' | 'volume' | 'frame_rate'>>) => void
  onSelectedSegmentContentChange: (patch: Partial<MultiTrackSegmentContent>) => void
  taskSegments?: MultiTrackSegment[]
  onTrackSegmentsContentChange?: (updates: Array<{ segmentId: string; patch: Partial<MultiTrackSegmentContent> }>) => void
  onSelectedSegmentDurationChange: (duration: number) => void
}

function resolutionInputSignature(input: unknown): string {
  try {
    return JSON.stringify(input)
  } catch {
    return String(Date.now())
  }
}

export function PreviewArea({
  data,
  currentTime,
  selectedSegment,
  isPlaying,
  node,
  onGlobalSettingsChange,
  onSelectedSegmentContentChange,
  taskSegments,
  onTrackSegmentsContentChange,
  onSelectedSegmentDurationChange,
}: Readonly<PreviewAreaProps>) {
  const [firstVideoMetadata, setFirstVideoMetadata] = useState<MultiTrackVideoMetadata | null>(null)
  const [resolutionInput, setResolutionInput] = useState(() => collectMultiTrackPreviewResolutionInput(node))
  const videoSegments = useMemo(() => (
    data.tracks
      .filter((track) => track.type === 'video')
      .flatMap((track) => track.segments)
  ), [data.tracks])
  const firstVideoUrl = useMemo(() => {
    const firstVideoSegment = videoSegments.find((segment) => segment.content.media_type === 'video')
    if (!firstVideoSegment) return null
    return mediaContentToViewUrl({
      source_type: firstVideoSegment.content.source_type ?? 'input',
      file_path: firstVideoSegment.content.file_path,
      local_path: firstVideoSegment.content.local_path,
      url: firstVideoSegment.content.url,
      slot_name: firstVideoSegment.content.slot_name,
    })
  }, [videoSegments])
  const activeVideo = selectedSegment?.trackType === 'task'
    ? null
    : getActivePreviewVideoSegment(data, currentTime, selectedSegment?.trackType === 'video' ? selectedSegment.segment.id : null)
  const resolution = parseMultiTrackPreviewResolution(resolutionInput, firstVideoMetadata)
  const muted = selectedSegment?.trackType === 'video'
    ? data.muted === true || selectedSegment.segment.content.volume === 0
    : data.muted === true
  const volume = selectedSegment?.trackType === 'video'
    ? selectedSegment.segment.content.volume ?? data.volume ?? 1
    : data.volume ?? 1
  const selectedVideoDuration = selectedSegment?.trackType === 'video'
    ? frameToSeconds(segmentDuration(selectedSegment.segment), data.frame_rate)
    : null

  useEffect(() => {
    if (!firstVideoUrl) {
      setFirstVideoMetadata(null)
      return
    }

    let cancelled = false
    loadBrowserVideoMetadata(firstVideoUrl)
      .then((metadata) => {
        if (!cancelled) setFirstVideoMetadata({ width: metadata.width, height: metadata.height })
      })
      .catch((error: unknown) => {
        console.error('[PreviewArea] failed to read first video metadata:', error)
        if (!cancelled) setFirstVideoMetadata(null)
      })

    return () => {
      cancelled = true
    }
  }, [firstVideoUrl])

  useEffect(() => {
    let currentSignature = resolutionInputSignature(collectMultiTrackPreviewResolutionInput(node))
    setResolutionInput((current) => (
      resolutionInputSignature(current) === currentSignature
        ? current
        : collectMultiTrackPreviewResolutionInput(node)
    ))

    const timer = window.setInterval(() => {
      const nextInput = collectMultiTrackPreviewResolutionInput(node)
      const nextSignature = resolutionInputSignature(nextInput)
      if (nextSignature === currentSignature) return
      currentSignature = nextSignature
      setResolutionInput(nextInput)
    }, RESOLUTION_POLL_INTERVAL_MS)

    return () => window.clearInterval(timer)
  }, [node])

  if (selectedSegment?.trackType === 'task') {
    return (
      <div
        className="flex min-h-24 flex-1 overflow-hidden rounded-sm bg-background"
        onClick={(event) => event.stopPropagation()}
      >
        <TaskSegmentEditor
          segment={selectedSegment.segment}
          trackSegments={taskSegments}
          videoSegments={videoSegments}
          onContentChange={onSelectedSegmentContentChange}
          onTrackSegmentsContentChange={onTrackSegmentsContentChange}
        />
      </div>
    )
  }

  return (
    <div
      className="relative flex min-h-24 flex-1 items-center justify-center overflow-hidden rounded-sm bg-black text-xs text-muted-foreground"
      onClick={(event) => event.stopPropagation()}
    >
      <div className="flex h-full min-h-24 items-center justify-center">
        <VideoPreview
          activeVideo={activeVideo}
          resolution={resolution}
          isPlaying={isPlaying}
          muted={muted}
          volume={volume}
        />
      </div>
      <PreviewFloatingToolbar
        globalMuted={data.muted === true}
        globalVolume={data.volume ?? 1}
        frameRate={data.frame_rate}
        selectedVideoVolume={selectedSegment?.trackType === 'video' ? selectedSegment.segment.content.volume ?? data.volume ?? 1 : null}
        selectedVideoDuration={selectedVideoDuration}
        onGlobalSettingsChange={onGlobalSettingsChange}
        onSelectedSegmentContentChange={onSelectedSegmentContentChange}
        onSelectedSegmentDurationChange={onSelectedSegmentDurationChange}
      />
    </div>
  )
}
