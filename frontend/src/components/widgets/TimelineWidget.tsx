import { useRef, useState, useEffect } from 'react'
import type { WheelEvent } from 'react'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import type { TimelineData, Track, Segment, TimeDisplayFormat, MaintainSegment, AudioSegment } from '@/types/timeline'
import { createDefaultTimelineData, scaleImageItemsToDuration } from '@/lib/timeline-utils'
import { audioContentToViewUrl } from '@/lib/media-url'
import { Toolbar } from './timeline/Toolbar'
import { TimelineRuler } from './timeline/TimelineRuler'
import { MaintainTrack } from './timeline/MaintainTrack'
import { AudioTrack } from './timeline/AudioTrack'
import { EditPanel } from './timeline/EditPanel'
import { TooltipProvider } from '@/components/ui/tooltip'
import { LocaleContext } from '@/lib/i18n'
import { useCanvasScale } from '@/hooks/use-canvas-scale'
import { useElementWidth } from '@/hooks/use-element-width'
import { useLatestRef } from '@/hooks/use-latest-ref'
import { useNodeInputConnectionDisabled } from '@/hooks/use-node-input-connection-disabled'

function ensureDefaults(raw: unknown): TimelineData {
  if (raw && typeof raw === 'object' && !Array.isArray(raw)) {
    const d = raw as Partial<TimelineData>
    if (Array.isArray(d.tracks) && typeof d.total_length === 'number' && typeof d.frame_rate === 'number') {
      return raw as TimelineData
    }
  }
  return createDefaultTimelineData()
}

/** Build a browser-playable URL from an audio segment's content. */
function audioUrl(seg: AudioSegment): string | null {
  return audioContentToViewUrl(seg.content)
}

/** Track last synced frame to avoid unnecessary seeks */
const lastSyncedFrame = { current: -1 }

/** Sync a single audio element to the current playhead position. */
function syncOneAudio(
  audio: HTMLAudioElement,
  seg: AudioSegment,
  frame: number,
  frameRate: number,
  playing: boolean,
) {
  const inRange = frame >= seg.start_frame && frame <= seg.end_frame
  if (!inRange) {
    if (!audio.paused) audio.pause()
    return
  }

  // Seek audio to the correct position when frame changes
  if (frame !== lastSyncedFrame.current) {
    // Use origin_start_frame so trimming the left edge plays from the right offset in the file
    const originStart = seg.origin_start_frame ?? seg.start_frame
    const audioFrame = frame - originStart
    const audioTime = audioFrame / frameRate
    // Only seek if the difference is significant (> 100ms) to avoid jitter
    if (Math.abs(audio.currentTime - audioTime) > 0.1) {
      audio.currentTime = audioTime
    }
    lastSyncedFrame.current = frame
  }

  // Handle play/pause state
  if (playing) {
    if (audio.paused && audio.readyState >= 2) {
      audio.play().catch(() => {})
    }
  } else {
    if (!audio.paused) {
      audio.pause()
    }
  }
}

export function TimelineWidget({ value, onChange, app, node, widget }: Readonly<ReactWidgetProps<TimelineData>>) {
  const data = ensureDefaults(value)
  const [displayFormat, setDisplayFormat] = useState<TimeDisplayFormat>('frames')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const scrollContainerRef = useRef<HTMLDivElement>(null)
  const contentWidth = useElementWidth(scrollContainerRef)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playheadFrame, setPlayheadFrame] = useState(0)
  const [showSeekLabel, setShowSeekLabel] = useState(false)
  const seekLabelTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [zoom, setZoom] = useState(1)
  const isNodeV2 = app?.ui?.settings?.settingsValues?.['Comfy.VueNodes.Enabled']
  const locale = app?.ui?.settings?.settingsValues?.['Comfy.Locale']
  const canvasScale = useCanvasScale(app)
  useNodeInputConnectionDisabled(node, widget, 'prompt_override')
  // Refs for playback loop (avoid stale closures)
  const playbackRef = useRef<{
    rafId: number | null
    lastTimestamp: number | null
    frame: number
    audioEls: Map<string, HTMLAudioElement>
  }>({ rafId: null, lastTimestamp: null, frame: 0, audioEls: new Map() })

  const scaledWidth = Math.max(contentWidth, 1) * zoom

  function handleRulerWheel(e: WheelEvent<HTMLDivElement>) {
    const el = scrollContainerRef.current
    if (!el || el.scrollWidth <= el.clientWidth) return

    e.preventDefault()
    e.stopPropagation()

    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? el.clientWidth : 1
    const primaryDelta = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY
    el.scrollLeft = Math.max(
      0,
      Math.min(el.scrollWidth - el.clientWidth, el.scrollLeft + primaryDelta * unit),
    )
  }

  // Always-current ref for data so the RAF loop never has a stale closure
  const dataRef = useLatestRef(data)

  // Keep playback ref in sync with react state
  useEffect(() => {
    playbackRef.current.frame = playheadFrame
  }, [playheadFrame])

  /**
   * Ensure audio HTMLElements exist for every audio segment.
   * Elements are created once and reused; removed segments are cleaned up.
   * Safe to call any time — cheap when nothing changed.
   */
  function ensureAudioEls() {
    const { tracks } = dataRef.current
    const ref = playbackRef.current
    const allSegs = tracks
      .filter((t) => t.type === 'audio')
      .flatMap((t) => t.segments as AudioSegment[])

    for (const seg of allSegs) {
      if (!ref.audioEls.has(seg.id)) {
        const src = audioUrl(seg)
        if (!src) continue
        const audio = new Audio(src)
        audio.preload = 'auto'
        // Ensure playbackRate is explicitly set to 1
        audio.playbackRate = 1
        ref.audioEls.set(seg.id, audio)
      } else {
        // Re-ensure playbackRate is 1 on existing elements
        const existingAudio = ref.audioEls.get(seg.id)!
        if (existingAudio.playbackRate === 1) continue
        existingAudio.playbackRate = 1
      }
    }
    // Clean up removed segments
    for (const [id, el] of ref.audioEls.entries()) {
      if (!allSegs.some((s) => s.id === id)) {
        el.pause()
        ref.audioEls.delete(id)
      }
    }
  }

  /** Sync play/pause/seek for all audio elements at the given frame position. */
  function syncAudio(frame: number, playing: boolean) {
    const { frame_rate, tracks } = dataRef.current
    const allSegs = tracks
      .filter((t) => t.type === 'audio')
      .flatMap((t) => t.segments as AudioSegment[])

    for (const [id, audio] of playbackRef.current.audioEls.entries()) {
      const seg = allSegs.find((s) => s.id === id)
      if (seg) {
        syncOneAudio(audio, seg, frame, frame_rate, playing)
      } else if (!audio.paused) {
        audio.pause()
      }
    }
  }

  // Re-create audio elements whenever tracks change (segments added/removed)
  useEffect(() => {
    ensureAudioEls()
  }, [data.tracks]) // eslint-disable-line react-hooks/exhaustive-deps

  // Playback animation loop — reads from dataRef to avoid stale closures
  useEffect(() => {
    const ref = playbackRef.current
    if (!isPlaying) {
      if (ref.rafId !== null) {
        cancelAnimationFrame(ref.rafId)
        ref.rafId = null
        lastSyncedFrame.current = -1
      }
      syncAudio(ref.frame, false)
      return
    }

    // Ensure audio is ready before starting the loop
    ensureAudioEls()

    const startTimeRef = { current: 0 }
    const startFrameRef = { current: 0 }

    function tick(ts: number) {
      const elapsed = (ts - startTimeRef.current) / 1000

      const { frame_rate, total_length } = dataRef.current
      const nextFrame = startFrameRef.current + elapsed * frame_rate
      const clamped = Math.min(total_length - 1, nextFrame)
      ref.frame = clamped
      setPlayheadFrame(Math.round(clamped))
      syncAudio(clamped, true)

      if (clamped >= total_length - 1) {
        setIsPlaying(false)
        setShowSeekLabel(true)
        if (seekLabelTimeoutRef.current) clearTimeout(seekLabelTimeoutRef.current)
        seekLabelTimeoutRef.current = setTimeout(() => setShowSeekLabel(false), 1500)
        return
      }
      ref.rafId = requestAnimationFrame(tick)
    }

    startTimeRef.current = performance.now()
    startFrameRef.current = ref.frame
    ref.rafId = requestAnimationFrame(tick)
    return () => {
      if (ref.rafId !== null) cancelAnimationFrame(ref.rafId)
    }
  }, [isPlaying]) // eslint-disable-line react-hooks/exhaustive-deps

  // Stop all audio on unmount
  useEffect(() => {
    return () => {
      for (const el of playbackRef.current.audioEls.values()) el.pause()
    }
  }, [])

  // Auto-scroll to keep playhead in view during playback
  useEffect(() => {
    if (!isPlaying || !scrollContainerRef.current) return
    const container = scrollContainerRef.current
    const playheadX = (playheadFrame / Math.max(data.total_length - 1, 1)) * scaledWidth
    const { scrollLeft, clientWidth } = container
    const margin = clientWidth * 0.15
    if (playheadX < scrollLeft + margin) {
      container.scrollLeft = Math.max(0, playheadX - margin)
    } else if (playheadX > scrollLeft + clientWidth - margin) {
      container.scrollLeft = playheadX - clientWidth + margin
    }
  }, [playheadFrame, isPlaying, scaledWidth, data.total_length])

  function update(patch: Partial<TimelineData>) {
    onChange({ ...data, ...patch })
  }

  function updateTrack(trackId: string, patch: Partial<Track>) {
    update({ tracks: data.tracks.map((t) => (t.id === trackId ? { ...t, ...patch } : t)) })
  }

  function updateSegments(trackId: string, segments: Segment[]) {
    update({ tracks: data.tracks.map((t) => (t.id === trackId ? { ...t, segments } : t)) })
  }

  function handleContextMenu(e: React.MouseEvent) {
    if (isNodeV2) {
      e.stopPropagation()
    }
  }

  function handlePlayPause() {
    if (!isPlaying && playheadFrame >= data.total_length - 1) {
      setPlayheadFrame(0)
      playbackRef.current.frame = 0
    }
    const wasPlaying = isPlaying
    setIsPlaying((v) => !v)

    // When stopping, show seek label for 1.5s
    if (wasPlaying) {
      setShowSeekLabel(true)
      if (seekLabelTimeoutRef.current) clearTimeout(seekLabelTimeoutRef.current)
      seekLabelTimeoutRef.current = setTimeout(() => setShowSeekLabel(false), 1500)
    } else {
      setShowSeekLabel(false)
    }
  }

  function handleSeek(frame: number) {
    setPlayheadFrame(frame)
    playbackRef.current.frame = frame
    lastSyncedFrame.current = frame
    syncAudio(frame, isPlaying)

    // Show seek label, hide after 1.5s unless playing
    setShowSeekLabel(true)
    if (!isPlaying) {
      if (seekLabelTimeoutRef.current) clearTimeout(seekLabelTimeoutRef.current)
      seekLabelTimeoutRef.current = setTimeout(() => setShowSeekLabel(false), 1500)
    }
  }

  function handleDeleteSelected() {
    if (!selectedId) return

    const updated = data.tracks.map((t) => {
      if (t.type !== 'maintain') {
        return { ...t, segments: t.segments.filter((s) => s.id !== selectedId) }
      }

      const segments = t.segments as MaintainSegment[]
      const idx = segments.findIndex((s) => s.id === selectedId)
      if (idx === -1) return t

      // Allow deleting the last segment (empty track allowed)
      if (segments.length === 1) {
        return { ...t, segments: [] }
      }

      const removed = segments[idx]
      const span = removed.end_frame - removed.start_frame + 1

      const newSegments = segments.map((s, i) => {
        if (i === idx) return null
        const result = { ...s }
        if (i > idx) {
          result.start_frame = s.start_frame - span
          result.end_frame = s.end_frame - span
        }
        return result
      }).filter((s): s is MaintainSegment => s !== null)

      if (idx > 0) {
        newSegments[idx - 1] = { ...newSegments[idx - 1], end_frame: removed.start_frame - 1 }
      }

      return { ...t, segments: newSegments }
    })
    update({ tracks: updated })
    setSelectedId(null)
  }

  // Find selected maintain track and segment for EditPanel
  const maintainTrack = data.tracks.find((t) => t.type === 'maintain')
  const selectedSegment = maintainTrack && selectedId
    ? maintainTrack.segments.find((s) => s.id === selectedId) as MaintainSegment | null ?? null
    : null

  function handleContentChange(patch: { text?: string; images?: unknown[] }) {
    if (!maintainTrack || !selectedId) return
    updateSegments(maintainTrack.id, maintainTrack.segments.map((s) =>
      s.id === selectedId ? { ...s, content: { ...s.content, ...patch } } : s,
    ))
  }

  function handleSegmentDurationChange(segmentId: string, newDuration: number) {
    if (!maintainTrack) return
    const segments = maintainTrack.segments as MaintainSegment[]
    const idx = segments.findIndex((s) => s.id === segmentId)
    if (idx === -1) return

    const seg = segments[idx]
    const oldDuration = seg.end_frame - seg.start_frame + 1
    if (newDuration === oldDuration) return

    const delta = newDuration - oldDuration
    const newEndFrame = seg.end_frame + delta

    // Update this segment and shift all subsequent segments
    const updated = segments.map((s, i) => {
      if (i < idx) return s
      if (i === idx) {
        return {
          ...s,
          end_frame: newEndFrame,
          content: {
            ...s.content,
            images: scaleImageItemsToDuration(s.content.images, oldDuration, newDuration),
          },
        }
      }
      // Shift subsequent segments
      return {
        ...s,
        start_frame: s.start_frame + delta,
        end_frame: s.end_frame + delta,
      }
    })

    // Calculate new total length needed
    const maxEndFrame = updated.reduce((max, s) => Math.max(max, s.end_frame), 0)
    const newTotalLength = Math.max(data.total_length, maxEndFrame + 1)

    // Update segments and total length
    update({
      tracks: data.tracks.map((t) =>
        t.type === 'maintain' ? { ...t, segments: updated } : t,
      ),
      total_length: newTotalLength,
    })
  }

  function handleGlobalClick(e: React.MouseEvent) {
    const target = e.target as HTMLElement
    // Ignore clicks inside EditPanel, segment blocks, or media selector popover
    if (
      target.closest('[data-edit-panel]') ||
      target.closest('[data-segment-block]') ||
      target.closest('[data-add-popover]') ||
      target.closest('[data-media-selector]') ||
      target.closest('[data-audio-popover]') ||
      target.closest('[data-time-toolbar]') ||
      target.closest('[data-radix-popper-content-wrapper]') ||
      target.closest('[data-radix-context-menu-content]')
    ) return
    setSelectedId(null)
  }

  // Clamp the inverse-scale applied to the EditPanel overlay so it stays usable
  // at extreme canvas zoom levels (range: 0.5× – 2.0× of its natural size).
  const EDIT_PANEL_SCALE_MIN = 0.8
  const EDIT_PANEL_SCALE_MAX = 1.5
  const editPanelScale = Math.min(EDIT_PANEL_SCALE_MAX, Math.max(EDIT_PANEL_SCALE_MIN, 1 / canvasScale))
  // DOM width so the panel's visual width always equals the widget's screen width
  const editPanelDomWidth = contentWidth / editPanelScale

  return (
    <LocaleContext.Provider value={locale}>
    <TooltipProvider>
      {/* Root: relative so the EditPanel overlay can position against it; no overflow-hidden here */}
      <div
        className={`relative flex flex-col h-full w-full text-foreground font-sans text-xs select-none${isNodeV2 ? ' nodeNew' : ''}`}
        onContextMenu={handleContextMenu}
        onClick={handleGlobalClick}
      >
        {/* Inner container: clips track content, fills remaining height */}
        <div className="flex flex-col flex-1 min-h-0 rounded overflow-hidden">
          {/* Toolbar */}
          <Toolbar
            totalLength={data.total_length}
            frameRate={data.frame_rate}
            displayFormat={displayFormat}
            onTotalLengthChange={(newTotal: number) => {
              const oldTotal = data.total_length

              // Only trim clips if we're shortening the timeline
              if (newTotal < oldTotal && maintainTrack) {
                const segments = maintainTrack.segments as MaintainSegment[]
                let updated = segments.map((seg) => {
                  // If segment ends beyond new total, truncate it
                  if (seg.end_frame >= newTotal) {
                    const oldDuration = seg.end_frame - seg.start_frame + 1
                    const newDuration = newTotal - seg.start_frame
                    return {
                      ...seg,
                      end_frame: newTotal - 1,
                      content: {
                        ...seg.content,
                        images: scaleImageItemsToDuration(seg.content.images, oldDuration, newDuration),
                      },
                    }
                  }
                  return seg
                })

                // Remove segments that become < 5 frames after truncation
                // Process from the last segment backwards to safely remove
                const minFrames = 5
                for (let i = updated.length - 1; i >= 0; i--) {
                  const seg = updated[i]
                  const segDuration = seg.end_frame - seg.start_frame + 1
                  if (segDuration < minFrames) {
                    // Shift subsequent segments' start positions
                    const removedSpan = segDuration
                    updated = updated.map((s, idx) => {
                      if (idx === i) return null
                      const result = { ...s }
                      if (idx > i) {
                        result.start_frame = Math.max(0, s.start_frame - removedSpan)
                        result.end_frame = Math.max(0, s.end_frame - removedSpan)
                      }
                      return result
                    }).filter((s): s is MaintainSegment => s !== null)

                    // Connect the gap: make previous segment extend to removed segment's start
                    if (i > 0) {
                      updated[i - 1] = { ...updated[i - 1], end_frame: seg.start_frame - 1 }
                    }
                  }
                }

                update({
                  tracks: data.tracks.map((t) =>
                    t.type === 'maintain' ? { ...t, segments: updated } : t,
                  ),
                  total_length: newTotal,
                })
              } else {
                update({ total_length: newTotal })
              }
            }}
            onFrameRateChange={(fps) => update({ frame_rate: fps })}
            onDisplayFormatChange={setDisplayFormat}
            isPlaying={isPlaying}
            onPlayPause={handlePlayPause}
            hasSelection={selectedId !== null}
            canDelete={selectedId !== null && (
              data.tracks.some((t) => t.type !== 'maintain' && t.segments.some((s) => s.id === selectedId)) ||
              (maintainTrack?.segments.length ?? 0) > 0
            )}
            onDeleteSelected={handleDeleteSelected}
            zoom={zoom}
            onZoomChange={setZoom}
            selectedSegment={selectedSegment}
            onSelectedSegmentDurationChange={handleSegmentDurationChange}
          />

          {/* Shared horizontal scroll container for ruler + tracks */}
          <div ref={scrollContainerRef} className="flex-1 min-h-0 overflow-x-hidden overflow-y-auto">
            <div className="h-full flex flex-col" style={{ width: scaledWidth, minWidth: '100%' }}>
              {/* Ruler row — fixed height */}
              <div className="border-t shrink-0">
                <TimelineRuler
                  totalFrames={data.total_length}
                  frameRate={data.frame_rate}
                  displayFormat={displayFormat}
                  width={scaledWidth}
                  canvasScale={canvasScale}
                  playheadFrame={playheadFrame}
                  showLabel={showSeekLabel || isPlaying}
                  onSeek={handleSeek}
                  onWheel={handleRulerWheel}
                />
              </div>

              {/* Track rows — fill remaining height with proportional flex-grow */}
              <div className="relative flex-1 min-h-0 flex flex-col">
                {/* Global playhead line spanning all tracks */}
                {playheadFrame !== undefined && scaledWidth > 0 && (
                  <div
                    className="absolute top-0 h-full w-px bg-red-400 pointer-events-none z-30"
                    style={{ left: (playheadFrame / Math.max(data.total_length - 1, 1)) * scaledWidth }}
                  />
                )}
                {data.tracks.map((track) => {
                  if (track.type === 'maintain') {
                    return (
                      <MaintainTrack
                        key={track.id}
                        track={track}
                        totalFrames={data.total_length}
                        frameRate={data.frame_rate}
                        displayFormat={displayFormat}
                        areaWidth={scaledWidth}
                        canvasScale={canvasScale}
                        selectedId={selectedId}
                        onSelectedIdChange={setSelectedId}
                        onTrackChange={(patch) => updateTrack(track.id, patch)}
                        onSegmentsChange={(segs) => updateSegments(track.id, segs)}
                        onExtendTimeline={(segs, newLength) => {
                          update({
                            tracks: data.tracks.map((t) =>
                              t.type === 'maintain' ? { ...t, segments: segs } : t
                            ),
                            total_length: newLength,
                          })
                        }}
                        grow={2}
                      />
                    )
                  }
                  if (track.type === 'audio') {
                    return (
                      <AudioTrack
                        key={track.id}
                        track={track}
                        totalFrames={data.total_length}
                        frameRate={data.frame_rate}
                        displayFormat={displayFormat}
                        areaWidth={scaledWidth}
                        selectedId={selectedId}
                        onSelectedIdChange={setSelectedId}
                        onTrackChange={(patch) => updateTrack(track.id, patch)}
                        onSegmentsChange={(segs) => updateSegments(track.id, segs)}
                        node={node}
                        app={app}
                        grow={1}
                      />
                    )
                  }
                  return null
                })}
              </div>
            </div>
          </div>
        </div>

        {/* EditPanel overlay — positioned below the widget, counter-scaled so it appears
            at a consistent pixel size regardless of canvas zoom level */}
        {selectedSegment && maintainTrack && (
          <div
            className="absolute left-0 z-50"
            style={{
              top: '100%',
              transformOrigin: 'top left',
              transform: `scale(${editPanelScale})`,
              width: editPanelDomWidth,
            }}
          >
            <EditPanel
              segment={selectedSegment}
              allSegments={maintainTrack.segments as MaintainSegment[]}
              totalFrames={data.total_length}
              frameRate={data.frame_rate}
              displayFormat={displayFormat}
              areaWidth={editPanelDomWidth}
              canvasScale={canvasScale * editPanelScale}
              trackColor={maintainTrack.color}
              onContentChange={handleContentChange}
              onAllSegmentsChange={(segs) => {
                updateSegments(maintainTrack.id, segs)
                if (selectedId && !segs.some((seg) => seg.id === selectedId)) {
                  setSelectedId(null)
                }
              }}
              node={node}
              app={app}
            />
          </div>
        )}
      </div>
    </TooltipProvider>
    </LocaleContext.Provider>
  )
}
