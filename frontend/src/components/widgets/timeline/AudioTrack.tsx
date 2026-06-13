import { useEffect, useRef, useState, useMemo } from 'react'
import { createPortal } from 'react-dom'
import { Music2 } from 'lucide-react'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { Popover, PopoverContent, PopoverAnchor } from '@/components/ui/popover'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import type { MediaTab } from '@/components/widgets/mediaSelector/MediaSelector'
import { SegmentBlock } from './SegmentBlock'
import { TrackRow } from './TrackRow'
import { AudioWaveform } from './AudioWaveform'
import type { Track, AudioSegment, Segment, Marker, TimeDisplayFormat } from '@/types/timeline'
import type { SlotItem } from '@/lib/timeline-utils'
import { useT } from '@/lib/i18n'
import { uuid } from '@/lib/uuid'
import { CUSTOM_NODE_CLASS } from '@/lib/constants'
import { computeSlotItems } from '@/lib/timeline-utils'
import { audioContentToViewUrl, mediaPathToViewUrl } from '@/lib/media-url'

interface AudioTrackProps {
  track: Track
  totalFrames: number
  frameRate: number
  displayFormat: TimeDisplayFormat
  areaWidth: number
  selectedId: string | null
  onSelectedIdChange: (id: string | null) => void
  onTrackChange: (patch: Partial<Track>) => void
  onSegmentsChange: (segments: Segment[]) => void
  node?: any
  app?: any
  /** flex-grow value for proportional height sizing */
  grow?: number
}

/** Map segment source_type to the MediaSelector tab */
function sourceTypeToTab(sourceType: string | undefined): MediaTab {
  if (sourceType === 'input') return 'inputs'
  if (sourceType === 'output') return 'outputs'
  if (sourceType === 'url') return 'url'
  if (sourceType === 'local') return 'local'
  return 'inputs'
}

function frameToX(frame: number, total: number, width: number) {
  return (frame / Math.max(total - 1, 1)) * width
}

export function AudioTrack({
  track,
  totalFrames,
  frameRate,
  displayFormat,
  areaWidth,
  selectedId,
  onSelectedIdChange,
  onTrackChange,
  onSegmentsChange,
  node,
  app,
  grow,
}: Readonly<AudioTrackProps>) {
  const t = useT()
  const containerRef = useRef<HTMLElement>(null)
  const [pendingDropFrame, setPendingDropFrame] = useState<number | null>(null)
  // State for editing a marker label inline
  const [editingMarker, setEditingMarker] = useState<{ segId: string; markerId: string; label: string } | null>(null)
  const [popoverOpen, setPopoverOpen] = useState(false)
  const [popoverDefaultTab, setPopoverDefaultTab] = useState<MediaTab>('inputs')
  const [anchorPos, setAnchorPos] = useState({ x: 0, y: 0 })
  const [editingSegId, setEditingSegId] = useState<string | null>(null)
  const [selectorValue, setSelectorValue] = useState('')
  const [rightClickedId, setRightClickedId] = useState<string | null>(null)
  const [contextMenuOpen, setContextMenuOpen] = useState(false)
  // Keep refs to temporary Audio elements so we can clean up event listeners
  const tempAudioRefs = useRef<HTMLAudioElement[]>([])

  // Recompute slot items when popover opens to get fresh graph data
  const slotItems = useMemo(() => computeSlotItems(node, app, 'audio'), [node, app, popoverOpen])

  // Cleanup temporary Audio elements on unmount
  useEffect(() => {
    return () => {
      tempAudioRefs.current.forEach((audio) => {
        audio.removeEventListener('loadedmetadata', () => {})
        audio.removeEventListener('error', () => {})
        audio.src = ''
      })
      tempAudioRefs.current = []
    }
  }, [])

  // When an overlay is open:
  //   1. Elevate the widget container above the dismiss backdrop (z-9999 > z-9997)
  //      so clicks inside the widget still reach the widget and don't hit the backdrop.
  //   2. The backdrop (rendered in JSX via createPortal) catches all clicks outside
  //      the widget/popup, reliably closing overlays even in ComfyUI's NodeV2 where
  //      the LiteGraph canvas consumes pointer events before Radix's dismiss fires.
  useEffect(() => {
    const widgetEl = containerRef.current?.closest('.comfyui-react-widget') as HTMLElement | null
    if (!widgetEl) return
    const cls = `${CUSTOM_NODE_CLASS}-overlay-active`
    widgetEl.classList.toggle(cls, popoverOpen || contextMenuOpen)
    return () => widgetEl.classList.remove(cls)
  }, [popoverOpen, contextMenuOpen])

  const segments = track.segments as AudioSegment[]

  // Total occupied frames so we know if more audio can be imported
  const lastOccupied = segments.reduce((max, s) => Math.max(max, s.end_frame), -1)
  const canImport = lastOccupied < totalFrames - 1

  function openPopoverAt(x: number, y: number, defaultTab: MediaTab, segId: string | null, currentValue: string) {
    setAnchorPos({ x, y })
    setPopoverDefaultTab(defaultTab)
    setEditingSegId(segId)
    setSelectorValue(currentValue)
    setPopoverOpen(true)
  }

  function openPopover(e: React.MouseEvent, defaultTab: MediaTab, segId: string | null, currentValue: string) {
    const rect = containerRef.current?.getBoundingClientRect()
    const el = containerRef.current
    const zoom = (rect && el) ? rect.width / el.offsetWidth : 1
    openPopoverAt(
      (e.clientX - (rect?.left ?? 0)) / zoom,
      (e.clientY - (rect?.top ?? 0)) / zoom,
      defaultTab,
      segId,
      currentValue,
    )
  }

  function handleSelectorChange(filePath: string, source: 'input' | 'output' | 'local' = 'input') {
    setSelectorValue(filePath)

    // Handle multiple files
    const isMultiFile = filePath.includes('|MULTIPLE|')
    const paths = isMultiFile ? filePath.split('|MULTIPLE|') : [filePath]

    // For editing mode, only use the first file
    if (editingSegId) {
      const path = paths[0]
      const fileName = path.split('/').pop() ?? path
      const isUrl = path.startsWith('http')
      const content = isUrl
        ? { source_type: 'url' as const, url: path, file_name: fileName }
        : source === 'local'
          ? { source_type: 'local' as const, local_path: path, file_name: fileName }
          : { source_type: source, file_path: path, file_name: fileName }
      onSegmentsChange(
        segments.map((s) =>
          s.id === editingSegId
            ? {
                ...s,
                content,
              }
            : s,
        ),
      )
      setPopoverOpen(false)
      return
    }

    // For adding mode - handle multiple files
    async function processAudioFile(
      path: string,
      cursor: number,
      currentSegments: AudioSegment[],
      sourceType: 'input' | 'output' | 'local',
    ): Promise<{ newSeg: AudioSegment | null; nextCursor: number; updatedSegments: AudioSegment[] }> {
      const fileName = path.split('/').pop() ?? path
      const isUrl = path.startsWith('http')
      const content = isUrl
        ? { source_type: 'url' as const, url: path, file_name: fileName }
        : sourceType === 'local'
          ? { source_type: 'local' as const, local_path: path, file_name: fileName }
          : { source_type: sourceType, file_path: path, file_name: fileName }

      // Build source URL
      const src = audioContentToViewUrl(content) ?? path

      const actualDuration = await new Promise<number>((resolve) => {
        const audio = new Audio(src)
        tempAudioRefs.current.push(audio)
        audio.preload = 'metadata'
        audio.addEventListener('loadedmetadata', () => resolve(audio.duration))
        audio.addEventListener('error', () => resolve(5)) // fallback to 5 seconds
      })

      const actualFrames = Math.ceil(actualDuration * frameRate)
      const segEnd = Math.min(totalFrames - 1, cursor + actualFrames - 1)

      // If audio exceeds remaining space, truncate it
      if (cursor >= totalFrames) {
        return { newSeg: null, nextCursor: cursor, updatedSegments: currentSegments }
      }

      const newSeg: AudioSegment = {
        id: uuid(),
        start_frame: cursor,
        end_frame: segEnd,
        origin_start_frame: cursor,
        origin_end_frame: cursor + actualFrames - 1,
        content,
        color: track.color,
        markers: [],
      }

      return {
        newSeg,
        nextCursor: segEnd + 1,
        updatedSegments: [...currentSegments, newSeg],
      }
    }

    async function processAllFiles() {
      let cursor = lastOccupied >= 0 ? lastOccupied + 1 : 0
      let currentSegments = [...segments]

      for (const path of paths) {
        if (cursor >= totalFrames) break

        if (path.startsWith('__slot__:')) {
          const slotName = path.slice('__slot__:'.length)
          const slotItem = slotItems.find((item: SlotItem) => item.value === path)
          const audioName = slotItem?.audio_name
          const url = audioName ? mediaPathToViewUrl(audioName, 'input') : undefined

          const actualDuration = await new Promise<number>((resolve) => {
            if (url) {
              const audio = new Audio(url)
              tempAudioRefs.current.push(audio)
              audio.preload = 'metadata'
              audio.addEventListener('loadedmetadata', () => resolve(audio.duration))
              audio.addEventListener('error', () => resolve(5))
            } else {
              resolve(5)
            }
          })

          const actualFrames = Math.ceil(actualDuration * frameRate)
          const segEnd = Math.min(totalFrames - 1, cursor + actualFrames - 1)

          const newSeg: AudioSegment = {
            id: uuid(),
            start_frame: cursor,
            end_frame: segEnd,
            origin_start_frame: cursor,
            origin_end_frame: cursor + actualFrames - 1,
            content: { source_type: 'slot', slot_name: slotName, file_name: slotName, url },
            color: track.color,
            markers: [],
          }

          currentSegments = [...currentSegments, newSeg]
          cursor = segEnd + 1
        } else {
          const result = await processAudioFile(path, cursor, currentSegments, source)
          if (result.newSeg) {
            cursor = result.nextCursor
            currentSegments = result.updatedSegments
          }
        }
      }

      onSegmentsChange([...currentSegments].sort((a, b) => a.start_frame - b.start_frame))
      if (currentSegments.length > segments.length) {
        setEditingSegId(currentSegments[currentSegments.length - 1].id)
      }
      setPopoverOpen(false)
    }

    processAllFiles()
  }

  function handleDeleteSegment(segId: string) {
    onSegmentsChange(segments.filter((s) => s.id !== segId))
    setPopoverOpen(false)
    if (selectedId === segId) onSelectedIdChange(null)
  }

  function handleSegmentAtX(frame: number): AudioSegment | null {
    return segments.find((s) => frame >= s.start_frame && frame <= s.end_frame) ?? null
  }

  function handleClick(e: React.MouseEvent) {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    const scale = areaWidth / Math.max(totalFrames - 1, 1)
    const frame = Math.round((e.clientX - rect.left) / scale)
    if (!handleSegmentAtX(frame)) {
      onSelectedIdChange(null)
    }
  }

  function handleDragOver(e: React.DragEvent) {
    e.preventDefault()
    const rect = e.currentTarget.getBoundingClientRect()
    const x = e.clientX - rect.left
    const frame = Math.round((x / areaWidth) * (totalFrames - 1))
    setPendingDropFrame(frame)
  }

  async function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const AUDIO_EXTS = new Set(['.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac', '.opus', '.wma'])
    const files = Array.from(e.dataTransfer.files).filter((f) => {
      if (f.type.startsWith('audio/')) return true
      const ext = `.${f.name.split('.').pop()?.toLowerCase() ?? ''}`
      return AUDIO_EXTS.has(ext)
    })
    if (files.length === 0) {
      setPendingDropFrame(null)
      return
    }
    if (!canImport) return
    const updated = [...segments]
    let cursor = pendingDropFrame ?? (lastOccupied >= 0 ? lastOccupied + 1 : 0)
    for (const file of files) {
      if (cursor >= totalFrames) break
      try {
        const form = new FormData()
        form.append('file', file)
        const res = await fetch('/easy-media/upload', { method: 'POST', body: form })
        if (!res.ok) continue
        const { file_name: fileName } = await res.json() as { file_name: string }

        // Detect actual audio duration from the uploaded file
        const src = mediaPathToViewUrl(fileName, 'input')
        const duration = await new Promise<number>((resolve) => {
          const audio = new Audio(src)
          audio.preload = 'metadata'
          tempAudioRefs.current.push(audio)
          audio.addEventListener('loadedmetadata', () => resolve(audio.duration))
          audio.addEventListener('error', () => resolve(5))
        })

        const actualFrames = Math.ceil(duration * frameRate)
        const dropEnd = Math.min(totalFrames - 1, cursor + actualFrames - 1)
        const newSeg: AudioSegment = {
          id: uuid(),
          start_frame: cursor,
          end_frame: dropEnd,
          origin_start_frame: cursor,
          origin_end_frame: cursor + actualFrames - 1,
          content: {
            source_type: 'input',
            file_path: fileName,
            file_name: fileName,
          },
          color: track.color,
          markers: [],
        }
        updated.push(newSeg)
        cursor = newSeg.end_frame + 1
      } catch {
        // skip failed uploads
      }
    }
    onSegmentsChange(updated.toSorted((a, b) => a.start_frame - b.start_frame))
    setPendingDropFrame(null)
  }

  function handleResizeEnd(segmentId: string, edge: 'start' | 'end', deltaFrames: number, minStart = 0) {
    if (track.locked) return
    const seg = segments.find((s) => s.id === segmentId)
    if (!seg) return
    const originStart = seg.origin_start_frame ?? seg.start_frame
    const originEnd = seg.origin_end_frame ?? seg.end_frame
    const others = segments.filter((s) => s.id !== segmentId).toSorted((a, b) => a.start_frame - b.start_frame)
    let newStart = seg.start_frame
    let newEnd = seg.end_frame
    if (edge === 'start') {
      newStart = Math.max(minStart, Math.max(originStart, seg.start_frame + deltaFrames))
      const prev = others.findLast((s) => s.end_frame < seg.end_frame)
      if (prev) newStart = Math.max(newStart, prev.end_frame + 1)
      newStart = Math.min(newStart, seg.end_frame - 1)
    } else {
      newEnd = Math.min(originEnd, seg.end_frame + deltaFrames)
      newEnd = Math.min(newEnd, totalFrames - 1)
      const next = others.find((s) => s.start_frame > seg.start_frame)
      if (next) newEnd = Math.min(newEnd, next.start_frame - 1)
      newEnd = Math.max(newEnd, seg.start_frame + 1)
    }
    onSegmentsChange(segments.map((s) => s.id === segmentId ? { ...s, start_frame: newStart, end_frame: newEnd } : s))
  }

  function handleDragEnd(segmentId: string, deltaFrames: number, origStart: number) {
    if (track.locked) return
    const seg = segments.find((s) => s.id === segmentId)
    if (!seg) return
    const span = seg.end_frame - seg.start_frame
    const newStart = Math.max(0, origStart + deltaFrames)
    const newEnd = Math.min(totalFrames - 1, newStart + span)
    const finalStart = newEnd - span
    const shift = Math.max(0, finalStart) - seg.start_frame

    // Prevent overlap; also shift origin bounds so waveform ratios stay correct
    const sorted = segments
      .map((s) => {
        if (s.id !== segmentId) return s
        return {
          ...s,
          start_frame: Math.max(0, finalStart),
          end_frame: Math.min(totalFrames - 1, newEnd),
          origin_start_frame: (s.origin_start_frame ?? s.start_frame) + shift,
          origin_end_frame: (s.origin_end_frame ?? s.end_frame) + shift,
        }
      })
      .toSorted((a, b) => a.start_frame - b.start_frame)
    onSegmentsChange(sorted)
  }

  function addMarker(segId: string, frame: number) {
    const updated = segments.map((s) => {
      if (s.id !== segId) return s
      const markers: Marker[] = [...(s.markers ?? []), { id: uuid(), frame, label: '' }]
      return { ...s, markers }
    })
    onSegmentsChange(updated)
  }

  function deleteMarker(segId: string, markerId: string) {
    const updated = segments.map((s) => {
      if (s.id !== segId) return s
      return { ...s, markers: (s.markers ?? []).filter((m) => m.id !== markerId) }
    })
    onSegmentsChange(updated)
  }

  function commitMarkerLabel(segId: string, markerId: string, label: string) {
    const updated = segments.map((s) => {
      if (s.id !== segId) return s
      return {
        ...s,
        markers: (s.markers ?? []).map((m) => (m.id === markerId ? { ...m, label } : m)),
      }
    })
    onSegmentsChange(updated)
    setEditingMarker(null)
  }

  const toolSlots: [React.ReactNode, React.ReactNode, React.ReactNode] = [
    <button
      key="marker"
      type="button"
      title={t('audioTrack.addMarker')}
      className="w-full h-full flex items-center justify-center hover:bg-accent"
      onClick={() => {
        if (!segments.length) return
        const seg = segments[0]
        const mid = Math.floor((seg.start_frame + seg.end_frame) / 2)
        addMarker(seg.id, mid)
      }}
    >
      {/* <Bookmark className="w-2.5 h-2.5 text-muted-foreground" /> */}
    </button>,
    undefined,
    undefined,
  ]

  return (
    <TrackRow track={track} onTrackChange={onTrackChange} toolSlots={toolSlots} grow={grow}>
      <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
        <ContextMenu>
          <ContextMenuTrigger className="relative block w-full h-full" asChild>
            <section
              ref={containerRef}
              aria-label={t('audioTrack.dropZone')}
              className="relative w-full h-full cursor-default"
              onDragOver={handleDragOver}
              onDragLeave={() => setPendingDropFrame(null)}
              onDrop={handleDrop}
              onClick={handleClick}
              onContextMenu={() => setContextMenuOpen(true)}
              onDoubleClick={(e) => {
                const target = e.target as HTMLElement
                if (target.closest('[data-segment-block]')) return
                if (!canImport) return
                openPopover(e, 'inputs', null, '')
              }}
            >
              {/* Virtual anchor positioned at click coordinates */}
              <PopoverAnchor asChild>
                <span
                  className="absolute w-0 h-0 pointer-events-none"
                  style={{ left: anchorPos.x, top: anchorPos.y }}
                />
              </PopoverAnchor>
            {segments.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground pointer-events-none select-none opacity-40">
                <div className="flex items-center gap-1">
                  <Music2 className="w-3 h-3" />
                  <span className="text-[11px]">
                    {t('audioTrack.placeholder')}
                  </span>
                </div>
                <span className="text-[10px]">{t('audioTrack.placeholderHint')}</span>
              </div>
            ) : (
              <>
                {segments.map((seg) => (
                  <SegmentBlock
                    key={seg.id}
                    segment={seg}
                    totalFrames={totalFrames}
                    areaWidth={areaWidth}
                    interactive={!track.locked}
                    minStart={0}
                    selected={selectedId === seg.id}
                    onSelect={(s) => {
                      onSelectedIdChange(selectedId === s.id ? null : s.id)
                    }}
                    onDoubleClick={() => {
                      openPopoverAt(
                        frameToX(seg.start_frame, totalFrames, areaWidth),
                        8,
                        sourceTypeToTab(seg.content.source_type),
                        seg.id,
                        seg.content.file_path ?? seg.content.local_path ?? seg.content.url ?? '',
                      )
                    }}
                    onDragEnd={handleDragEnd}
                    onResizeEnd={handleResizeEnd}
                    frameRate={frameRate}
                    displayFormat={displayFormat}
                    backgroundSlot={<AudioWaveform
                      content={seg.content}
                      startRatio={seg.origin_end_frame !== undefined && seg.origin_start_frame !== undefined
                        ? (seg.start_frame - seg.origin_start_frame) / (seg.origin_end_frame - seg.origin_start_frame)
                        : 0}
                      endRatio={seg.origin_end_frame !== undefined && seg.origin_start_frame !== undefined
                        ? (seg.end_frame - seg.origin_start_frame) / (seg.origin_end_frame - seg.origin_start_frame)
                        : 1}
                    />}
                    onContextMenu={(_, s) => {
                      onSelectedIdChange(s.id)
                      setRightClickedId(s.id)
                    }}
                  >
                    <span className="absolute top-0.5 truncate text-[8px]">{seg.content.file_name}</span>

                    {/* Markers */}
                    {(seg.markers ?? []).map((marker) => (
                      <TooltipProvider key={marker.id} delayDuration={200}>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <button
                              type="button"
                              aria-label={marker.label ? t('audioTrack.markerAriaLabel', { label: marker.label }) : t('audioTrack.markerAtFrame', { frame: marker.frame })}
                              className="absolute top-0 w-0.5 h-full bg-yellow-300/80 cursor-pointer hover:bg-yellow-200 z-20"
                              style={{ left: frameToX(marker.frame, totalFrames, areaWidth) }}
                              onClick={(e) => {
                                e.stopPropagation()
                                setEditingMarker({ segId: seg.id, markerId: marker.id, label: marker.label ?? '' })
                              }}
                            />
                          </TooltipTrigger>
                          <TooltipContent side="top">
                            <p>{marker.label || t('audioTrack.frameLabel', { n: marker.frame })}</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ))}
                  </SegmentBlock>
                ))}
              </>
            )}

            {/* Drop indicator */}
            {pendingDropFrame !== null && (
              <div
                className="absolute top-0 w-0.5 h-full bg-foreground/60 pointer-events-none"
                style={{ left: frameToX(pendingDropFrame, totalFrames, areaWidth) }}
              />
            )}
          </section>
        </ContextMenuTrigger>

        <ContextMenuContent onCloseAutoFocus={() => { setRightClickedId(null); setContextMenuOpen(false) }}>
          <ContextMenuItem
            disabled={!canImport}
            onClick={() => setTimeout(() => openPopoverAt(8, 8, 'inputs', null, ''), 0)}
          >
            {t('audioTrack.contextAdd')}
          </ContextMenuItem>
          {rightClickedId && (() => {
            const rightClickedSeg = segments.find((s) => s.id === rightClickedId)
            return (
              <>
                <ContextMenuItem
                  onClick={() => {
                    if (!rightClickedSeg) return
                    setTimeout(() => openPopoverAt(
                      frameToX(rightClickedSeg.start_frame, totalFrames, areaWidth),
                      8,
                      sourceTypeToTab(rightClickedSeg.content.source_type),
                      rightClickedSeg.id,
                      rightClickedSeg.content.file_path ?? rightClickedSeg.content.local_path ?? rightClickedSeg.content.url ?? '',
                    ), 0)
                    setRightClickedId(null)
                  }}
                >
                  {t('audioTrack.contextEdit')}
                </ContextMenuItem>
                <ContextMenuItem onClick={() => { handleDeleteSegment(rightClickedId); setRightClickedId(null) }}>
                  {t('audioTrack.contextDelete')}
                </ContextMenuItem>
              </>
            )
          })()}
        </ContextMenuContent>
        </ContextMenu>

        <PopoverContent
          data-audio-popover=""
          className="p-0 w-auto"
          side="bottom"
          align="start"
          onOpenAutoFocus={(e: Event) => e.preventDefault()}
        >
          <MediaSelector
            value={selectorValue}
            onChange={handleSelectorChange}
            mediaType="audio"
            defaultTab={popoverDefaultTab}
            slotItems={slotItems}
          />
        </PopoverContent>
      </Popover>

      {/* Marker label editor dialog */}
      <Dialog open={!!editingMarker} onOpenChange={(open) => { if (!open) setEditingMarker(null) }}>
        <DialogContent className="max-w-xs p-4 space-y-2">
          <DialogTitle className="text-xs text-muted-foreground font-normal">
            {t('audioTrack.markerLabelInput')}
          </DialogTitle>
          {editingMarker != null && (() => {
            const m = editingMarker
            return (
              <>
                <Input
                  autoFocus
                  className="h-7 text-xs"
                  value={m.label}
                  onChange={(e) => setEditingMarker({ ...m, label: e.target.value })}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') commitMarkerLabel(m.segId, m.markerId, m.label)
                    if (e.key === 'Escape') setEditingMarker(null)
                  }}
                />
                <div className="flex gap-2 justify-end">
                  <Button size="sm" variant="ghost" className="h-6 text-xs" onClick={() => setEditingMarker(null)}>{t('common.cancel')}</Button>
                  <Button size="sm" variant="destructive" className="h-6 text-xs" onClick={() => { deleteMarker(m.segId, m.markerId); setEditingMarker(null) }}>{t('common.delete')}</Button>
                  <Button size="sm" className="h-6 text-xs" onClick={() => commitMarkerLabel(m.segId, m.markerId, m.label)}>{t('common.save')}</Button>
                </div>
              </>
            )
          })()}
        </DialogContent>
      </Dialog>

      {/* Dismiss backdrop: rendered to body via portal so it sits above the LiteGraph
          canvas (z-9997) but below popup content (z-9999) and the elevated widget
          (z-9999). Any click that reaches this backdrop is "outside" — close overlays. */}
      {(popoverOpen || contextMenuOpen) && createPortal(
        <div
          aria-hidden="true"
          style={{ position: 'fixed', inset: 0, zIndex: 9997 }}
          onPointerDown={() => {
            setPopoverOpen(false)
            setContextMenuOpen(false)
          }}
        />,
        document.body,
      )}
    </TrackRow>
  )
}
