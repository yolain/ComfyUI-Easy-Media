import { useRef, useState } from 'react'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'
import { Popover, PopoverContent, PopoverAnchor } from '@/components/ui/popover'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import { getImageSrc } from '../../../lib/image-utils'
import { InsertButton } from './InsertButton'
import { SegmentBlock } from './SegmentBlock'
import { TrackRow, MAINTAIN_TRACK_HEIGHT } from './TrackRow'
import { Plus } from 'lucide-react'
import type {
  Track,
  MaintainSegment,
  ImageItem,
  Segment,
  TimeDisplayFormat,
} from '@/types/timeline'
import { useT } from '@/lib/i18n'
import { uuid } from '@/lib/uuid'
import { scaleImageItemsToDuration } from '@/lib/timeline-utils'

interface MaintainTrackProps {
  track: Track
  totalFrames: number
  frameRate: number
  displayFormat: TimeDisplayFormat
  areaWidth: number
  canvasScale?: number
  selectedId: string | null
  onSelectedIdChange: (id: string | null) => void
  onTrackChange: (patch: Partial<Track>) => void
  onSegmentsChange: (segments: Segment[]) => void
  /** Callback to extend both segments and total length in one update */
  onExtendTimeline?: (segments: Segment[], newTotalLength: number) => void
  /** flex-grow value for proportional height sizing */
  grow?: number
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Calculate segment duration using formula: (fps * multiplier) + 1
 * Multiplier is based on totalFrames to give proportional segment sizes.
 * For 24fps, this gives approximately 0.5s to 5s segments depending on totalFrames.
 */
function calcSegmentDuration(frameRate: number): number {
  // const multiplier = Math.max(1, Math.floor(totalFrames / frameRate / 2))
  return frameRate * 2 + 1
}

/** Evenly distribute all segments across total span. Extra frames go to front segments (前大后小). */
function distributeEvenly(segs: Segment[], totalFrames: number): Segment[] {
  if (segs.length === 0) return segs
  const base = Math.floor(totalFrames / segs.length)
  const remainder = totalFrames % segs.length

  return segs.map((s, i) => {
    const isExtra = i < remainder
    const size = base + (isExtra ? 1 : 0)
    const start_frame = i * base + Math.min(i, remainder)
    const end_frame = i < segs.length - 1 ? start_frame + size - 1 : totalFrames - 1
    return { ...s, start_frame, end_frame }
  })
}

/** Upload a File to the ComfyUI input directory. Returns the stored filename (with subfolder prefix). */
async function uploadImageFile(file: File): Promise<string> {
  // Try the easy-media upload first (consistent with AudioTrack)
  // const form = new FormData()
  // form.append('file', file)
  // const res = await fetch('/easy-media/upload', { method: 'POST', body: form })
  // if (res.ok) {
  //   const json = await res.json() as { file_name: string }
  //   return json.file_name
  // }
  // Fallback to ComfyUI native endpoint
  const form2 = new FormData()
  form2.append('image', file)
  form2.append('type', 'input')
  form2.append('overwrite', 'false')
  const res2 = await fetch('/upload/image', { method: 'POST', body: form2 })
  if (!res2.ok) throw new Error(`Upload failed: ${res2.status}`)
  const json2 = await res2.json() as { name: string; subfolder?: string }
  const sub = json2.subfolder ? `${json2.subfolder}/` : ''
  return `${sub}${json2.name}`
}

/** Returns pixel left and width for a segment in the track area. */
export function segmentRect(start: number, end: number, total: number, areaW: number) {
  const scale = areaW / Math.max(total - 1, 1)
  const left = start * scale
  const right = Math.min((end + 1) * scale, areaW)
  return { left, width: Math.max(right - left, 2) }
}

/** Find the gap containing the given frame. Returns null if frame is on a segment or no valid gap exists. */
function findGapAtFrame(sortedSegs: MaintainSegment[], frame: number, totalFrames: number): { start: number; end: number } | null {
  if (sortedSegs.length === 0) return null

  // Check gap before first segment
  if (frame < sortedSegs[0].start_frame) {
    return { start: 0, end: sortedSegs[0].start_frame - 1 }
  }

  // Check gaps between segments
  for (let i = 0; i < sortedSegs.length - 1; i++) {
    const gapStart = sortedSegs[i].end_frame + 1
    const gapEnd = sortedSegs[i + 1].start_frame - 1
    if (frame >= gapStart && frame <= gapEnd) {
      return { start: gapStart, end: gapEnd }
    }
  }

  // Check gap after last segment
  const lastSeg = sortedSegs[sortedSegs.length - 1]
  if (frame > lastSeg.end_frame && lastSeg.end_frame < totalFrames - 1) {
    return { start: lastSeg.end_frame + 1, end: totalFrames - 1 }
  }

  return null
}

// ── Sub-components ───────────────────────────────────────────────────────────

/** Tiled image background for a maintain segment that has images. */
function SegmentImageBackground({ images }: Readonly<{ images: ImageItem[] }>) {
  const srcs = images.map((img) => getImageSrc(img)).filter((src): src is string => src !== null)
  if (srcs.length === 0) return null
  const repeatedSrcs = Array.from({ length: 24 }, (_, repeatIndex) =>
    srcs.map((src, srcIndex) => ({ src, key: `${repeatIndex}-${srcIndex}` })),
  ).flat()

  return (
    <div className="absolute inset-0 bg-black overflow-hidden">
      <div className="flex h-full w-max">
        {repeatedSrcs.map(({ src, key }) => (
          <img
            key={key}
            src={src}
            alt=""
            className="h-full w-auto max-w-none shrink-0 object-contain"
            draggable={false}
            onError={(e) => {
              ;(e.currentTarget as HTMLImageElement).style.display = 'none'
            }}
          />
        ))}
      </div>
    </div>
  )
}

/** The content displayed inside a maintain segment block. */
function SegmentBlockContent({
  seg,
  index,
}: Readonly<{ seg: MaintainSegment; index: number }>) {
  const t = useT()
  const content = seg.content
  const hasImages = content.images.length > 0
  const hasText = content.text.trim().length > 0

  if (!hasImages) {
    return (
      <div className="flex items-center justify-center w-full h-full px-1">
          {hasText ? (
            <span className="text-[9px] text-foreground leading-tight line-clamp-3 text-center">
              {content.text}
            </span>
          ) : (
            <div className="flex flex-col gap-1">
              <span className="text-[9px] text-foreground leading-tight truncate text-center">
                {t('maintainTrack.segmentLabel', { n: index + 1 })}
              </span>
              <span className="text-[8px] text-muted-foreground leading-tight truncate text-center">
                {t('maintainTrack.segmentPlaceholder')}
              </span>
            </div>
          )}
      </div>
    )
  }

  return (
    <>
      {hasText && (
        <div className="absolute bottom-0 left-0 right-0 px-1 pb-0.5 pointer-events-none z-10  bg-black">
          <span className="block text-[9px] leading-tight truncate text-white drop-shadow">
            {content.text}
          </span>
        </div>
      )}
    </>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export function MaintainTrack({
  track,
  totalFrames,
  frameRate,
  displayFormat,
  areaWidth,
  canvasScale = 1,
  selectedId,
  onSelectedIdChange,
  onTrackChange,
  onSegmentsChange,
  onExtendTimeline,
  grow,
}: Readonly<MaintainTrackProps>) {
  const t = useT()
  const containerRef = useRef<HTMLDivElement>(null)

  const [rightClickedId, setRightClickedId] = useState<string | null>(null)

  // Drag indicator
  const [pendingDropFrame, setPendingDropFrame] = useState<number | null>(null)

  // Media selector popover for clicking empty area
  const [addPopoverOpen, setAddPopoverOpen] = useState(false)
  const [addAnchorPos, setAddAnchorPos] = useState({ x: 0, y: 0 })
  const [selectorValue, setSelectorValue] = useState('')
  // Track which segment was just double-clicked to prevent section handler from adding empty segment
  const lastDoubleClickedSegId = useRef<string | null>(null)

  const segments = track.segments as MaintainSegment[]

  // ── Drag/resize ────────────────────────────────────────────────────────────

  function handleTrackAreaClick(e: React.MouseEvent) {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect) return
    const x = e.clientX - rect.left
    if (segments.length === 0) {
      // Do nothing on click when empty - only double-click or right-click
      return
    }
    if (!getSegmentAtX(x)) {
      onSelectedIdChange(null)
    }
  }

  function handleTrackAreaDoubleClick(e: React.MouseEvent) {
    const rect = containerRef.current?.getBoundingClientRect()
    if (!rect || track.locked) return
    const x = e.clientX - rect.left
    const scale = areaWidth / Math.max(totalFrames - 1, 1)
    const frame = Math.round(x / scale)

    if (segments.length === 0) {
      // No segments - add one covering entire timeline
      onSegmentsChange([{
        id: uuid(),
        start_frame: 0,
        end_frame: totalFrames - 1,
        content: { text: '', images: [], type: 'flf' },
        color: track.color,
      }])
      return
    }

    // Check if clicked on a segment - if so, don't add empty segment
    const clickedSeg = segments.find((s) => frame >= s.start_frame && frame <= s.end_frame)
    if (clickedSeg) return

    // Find the gap containing the clicked frame
    const sortedSegs = [...segments].sort((a, b) => a.start_frame - b.start_frame)
    const gap = findGapAtFrame(sortedSegs, frame, totalFrames)
    if (!gap) return

    onSegmentsChange([
      ...segments,
      {
        id: uuid(),
        start_frame: gap.start,
        end_frame: gap.end,
        content: { text: '', images: [], type: 'flf' },
        color: track.color,
      },
    ])
  }

  function handleDragEnd(segmentId: string, deltaFrames: number, origStart: number) {
    const seg = segments.find((s) => s.id === segmentId)
    if (!seg) return
    const span = seg.end_frame - seg.start_frame
    const newStart = Math.max(0, origStart + deltaFrames)
    const newEnd = Math.min(totalFrames - 1, newStart + span)
    const clampedStart = newEnd - span
    const withDragged = segments.map((s) =>
      s.id === segmentId ? { ...s, start_frame: clampedStart, end_frame: newEnd } : s,
    )
    const sortedByCenter = [...withDragged].sort(
      (a, b) => (a.start_frame + a.end_frame) - (b.start_frame + b.end_frame),
    )
    const originalSlots = [...segments]
      .sort((a, b) => a.start_frame - b.start_frame)
      .map((s) => ({ start_frame: s.start_frame, end_frame: s.end_frame }))
    const result = sortedByCenter.map((s, i) => ({
      ...s,
      start_frame: originalSlots[i].start_frame,
      end_frame: originalSlots[i].end_frame,
    }))
    onSegmentsChange(result)
  }

  function handleResizeEnd(segmentId: string, edge: 'start' | 'end', deltaFrames: number) {
    if (track.locked) return
    const idx = segments.findIndex((s) => s.id === segmentId)
    if (idx === -1) return
    const seg = segments[idx]
    const updated = [...segments]

    if (edge === 'end') {
      handleResizeEndEdge(updated, idx, seg, deltaFrames)
    } else {
      handleResizeStartEdge(updated, idx, seg, deltaFrames)
    }
    onSegmentsChange(updated)
  }

  function handleResizeEndEdge(updated: MaintainSegment[], idx: number, seg: MaintainSegment, delta: number) {
    const next = updated[idx + 1]
    const minEnd = seg.start_frame + 1
    const newEnd = Math.max(minEnd, Math.min(seg.end_frame + delta, totalFrames - 1))
    updated[idx] = resizeSegmentWithScaledImages(seg, seg.start_frame, newEnd)

    // 当有后续片段时，让它们跟随到当前片段后面（无论拉伸还是缩减）
    if (next) {
      const expectedStart = newEnd + 1
      if (next.start_frame !== expectedStart) {
        const shift = expectedStart - next.start_frame
        updated[idx + 1] = { ...next, start_frame: expectedStart, end_frame: next.end_frame + shift }
        for (let i = idx + 2; i < updated.length; i++) {
          updated[i] = { ...updated[i], start_frame: updated[i].start_frame + shift, end_frame: updated[i].end_frame + shift }
        }
      }
    }
  }

  function handleResizeStartEdge(updated: MaintainSegment[], idx: number, seg: MaintainSegment, delta: number) {
    const prev = updated[idx - 1]
    const maxStart = seg.end_frame - 1
    const minStart = prev ? prev.start_frame + 1 : 0
    const newStart = Math.max(minStart, Math.min(maxStart, seg.start_frame + delta))
    updated[idx] = resizeSegmentWithScaledImages(seg, newStart, seg.end_frame)
    if (prev) updated[idx - 1] = resizeSegmentWithScaledImages(prev, prev.start_frame, newStart - 1)

    // 让后续片段跟随到当前片段后面
    for (let i = idx + 1; i < updated.length; i++) {
      const expectedStart = updated[i - 1].end_frame + 1
      if (updated[i].start_frame !== expectedStart) {
        const shift = expectedStart - updated[i].start_frame
        updated[i] = { ...updated[i], start_frame: expectedStart, end_frame: updated[i].end_frame + shift }
      }
    }
  }

  function resizeSegmentWithScaledImages(
    seg: MaintainSegment,
    startFrame: number,
    endFrame: number,
  ): MaintainSegment {
    const oldDuration = seg.end_frame - seg.start_frame + 1
    const newDuration = endFrame - startFrame + 1
    if (oldDuration === newDuration) {
      return { ...seg, start_frame: startFrame, end_frame: endFrame }
    }

    return {
      ...seg,
      start_frame: startFrame,
      end_frame: endFrame,
      content: {
        ...seg.content,
        images: scaleImageItemsToDuration(seg.content.images, oldDuration, newDuration),
      },
    }
  }

  /**
   * Find the first gap in the timeline not covered by any segment.
   * Returns { start, end } or null if the timeline is fully covered.
   */
  function findFirstGap(): { start: number; end: number } | null {
    const sorted = [...segments].sort((a, b) => a.start_frame - b.start_frame)
    // Gap before first segment
    if (sorted.length === 0 || sorted[0].start_frame > 0) {
      return { start: 0, end: sorted.length > 0 ? sorted[0].start_frame - 1 : totalFrames - 1 }
    }
    // Gaps between segments
    for (let i = 0; i < sorted.length - 1; i++) {
      const gapStart = sorted[i].end_frame + 1
      const gapEnd = sorted[i + 1].start_frame - 1
      if (gapStart <= gapEnd) return { start: gapStart, end: gapEnd }
    }
    // Gap after last segment
    const afterLast = sorted.at(-1)!.end_frame + 1
    if (afterLast <= totalFrames - 1) return { start: afterLast, end: totalFrames - 1 }
    return null
  }

  function handleInsertSegment(position: 'left' | 'right') {
    const idx = segments.findIndex((s) => s.id === selectedId)
    if (idx === -1) return

    const seg = segments[idx]
    const segDuration = calcSegmentDuration(frameRate)

    // For 'left': new segment occupies [seg.start_frame, seg.start_frame + segDuration - 1],
    //   the selected segment and all after it shift right by segDuration.
    // For 'right': new segment occupies [seg.end_frame + 1, seg.end_frame + segDuration],
    //   all segments after the selected one shift right by segDuration.
    const insertStart = position === 'left' ? seg.start_frame : seg.end_frame + 1
    const shiftFromIdx = position === 'left' ? idx : idx + 1

    const newSegment: MaintainSegment = {
      id: uuid(),
      start_frame: insertStart,
      end_frame: insertStart + segDuration - 1,
      content: { text: '', images: [], type: 'flf' as const },
      color: track.color,
    }

    // Shift all segments from shiftFromIdx onwards right by segDuration
    const shifted = segments.map((s, i) => {
      if (i < shiftFromIdx) return s
      return { ...s, start_frame: s.start_frame + segDuration, end_frame: s.end_frame + segDuration }
    })

    const newSegments = [
      ...shifted.slice(0, shiftFromIdx),
      newSegment,
      ...shifted.slice(shiftFromIdx),
    ]

    const lastSeg = newSegments.at(-1)!
    const newTotalLength = lastSeg.end_frame + 1

    if (newTotalLength > totalFrames && onExtendTimeline) {
      onExtendTimeline(newSegments, newTotalLength)
    } else {
      onSegmentsChange(newSegments)
    }

    onSelectedIdChange(newSegment.id)
  }

  function cloneSegment(id: string) {
    if (track.locked) return
    const sortedSegments = [...segments].sort((a, b) => a.start_frame - b.start_frame)
    const idx = sortedSegments.findIndex((s) => s.id === id)
    if (idx === -1) return

    const seg = sortedSegments[idx]
    const duration = seg.end_frame - seg.start_frame + 1
    const cloneStart = seg.end_frame + 1
    const clonedSegment: MaintainSegment = {
      ...structuredClone(seg),
      id: uuid(),
      start_frame: cloneStart,
      end_frame: cloneStart + duration - 1,
      color: track.color,
    }

    const subsequentSegments: MaintainSegment[] = []
    let nextStart = clonedSegment.end_frame + 1
    for (const nextSegment of sortedSegments.slice(idx + 1)) {
      const nextDuration = nextSegment.end_frame - nextSegment.start_frame + 1
      subsequentSegments.push({
        ...nextSegment,
        start_frame: nextStart,
        end_frame: nextStart + nextDuration - 1,
      })
      nextStart += nextDuration
    }

    const newSegments = [
      ...sortedSegments.slice(0, idx + 1),
      clonedSegment,
      ...subsequentSegments,
    ]
    const newTotalLength = totalFrames + duration

    if (onExtendTimeline) {
      onExtendTimeline(newSegments, newTotalLength)
    } else {
      onSegmentsChange(newSegments)
    }
    onSelectedIdChange(clonedSegment.id)
  }

  function addSegment() {
    if (segments.length === 0) {
      // Empty track: add segment with full duration
      onSegmentsChange([{
        id: uuid(),
        start_frame: 0,
        end_frame: totalFrames - 1,
        content: { text: '', images: [], type: 'flf' },
        color: track.color,
      }])
      return
    }
    // Prefer filling an existing gap
    const gap = findFirstGap()
    if (gap) {
      onSegmentsChange([
        ...segments,
        {
          id: uuid(),
          start_frame: gap.start,
          end_frame: gap.end,
          content: { text: '', images: [], type: 'flf' },
          color: track.color,
        },
      ].sort((a, b) => a.start_frame - b.start_frame))
      return
    }
    // No gap (full timeline): extend after last segment with calculated duration
    const last = segments.at(-1)!
    const segDuration = calcSegmentDuration(frameRate)
    const newSegStart = last.end_frame + 1
    const newSegEnd = newSegStart + segDuration - 1
    const newSegment = {
      id: uuid(),
      start_frame: newSegStart,
      end_frame: newSegEnd,
      content: { text: '', images: [], type: 'flf' as const },
      color: track.color,
    }

    onSegmentsChange([...segments, newSegment])
    // Extend total length if needed (use combined callback to avoid batching issues)
    if (onExtendTimeline) {
      onExtendTimeline([...segments, newSegment], newSegEnd + 1)
    }
  }

  function deleteSegment(id: string) {
    const idx = segments.findIndex((s) => s.id === id)
    if (idx === -1) return

    // If this is the last segment, just remove it (allow empty track)
    if (segments.length === 1) {
      onSegmentsChange([])
      if (selectedId === id) onSelectedIdChange(null)
      return
    }

    const removed = segments[idx]
    const span = removed.end_frame - removed.start_frame + 1

    // Remove the segment and shift only segments AFTER it left by the removed segment's span
    const updated = segments.map((s, i) => {
      if (i === idx) return null  // will be filtered out
      const result = { ...s }
      // Only shift segments that originally came AFTER the deleted one
      if (i > idx) {
        result.start_frame = s.start_frame - span
        result.end_frame = s.end_frame - span
      }
      return result
    }).filter((s): s is MaintainSegment => s !== null)

    // Extend the previous segment to fill the gap
    if (idx > 0) {
      updated[idx - 1] = { ...updated[idx - 1], end_frame: removed.start_frame - 1 }
    }

    onSegmentsChange(updated)
    if (selectedId === id) onSelectedIdChange(null)
  }

  // ── External drag-and-drop (upload images) ────────────────────────────────

  function getSegmentAtX(x: number): MaintainSegment | null {
    const scale = areaWidth / Math.max(totalFrames - 1, 1)
    const frame = Math.round(x / scale)
    return segments.find((s) => frame >= s.start_frame && frame <= s.end_frame) ?? null
  }

  function handleDragOver(e: React.DragEvent) {
    if (e.dataTransfer.types.includes('Files')) {
      e.preventDefault()
      const rect = e.currentTarget.getBoundingClientRect()
      const x = (e.clientX - rect.left) / canvasScale
      setPendingDropFrame(Math.round((x / areaWidth) * (totalFrames - 1)))
    }
  }

  async function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'])
    const files = Array.from(e.dataTransfer.files).filter((f) => {
      if (f.type.startsWith('image/')) return true
      const ext = `.${f.name.split('.').pop()?.toLowerCase() ?? ''}`
      return IMAGE_EXTS.has(ext)
    })
    if (files.length === 0) { setPendingDropFrame(null); return }

    // Upload all files first
    const paths: string[] = []
    for (const file of files) {
      try {
        const path = await uploadImageFile(file)
        paths.push(path)
      } catch (err) {
        console.error('[MaintainTrack] upload failed:', err)
      }
    }
    if (paths.length === 0) { setPendingDropFrame(null); return }

    // If empty track: create one segment with all images distributed evenly
    if (segments.length === 0) {
      const segDuration = Math.floor(totalFrames / paths.length)
      const remainder = totalFrames % paths.length
      const newSegments: MaintainSegment[] = []
      let currentStart = 0

      for (let i = 0; i < paths.length; i++) {
        const span = segDuration + (i < remainder ? 1 : 0)
        const newItems: ImageItem[] = [{
          source_type: 'input' as const,
          file_path: paths[i],
          file_name: paths[i].split('/').pop() ?? paths[i],
        }]
        newSegments.push({
          id: uuid(),
          start_frame: currentStart,
          end_frame: currentStart + span - 1,
          content: { text: '', images: newItems, type: 'flf' as const },
          color: track.color,
        })
        currentStart += span
      }

      onSegmentsChange(newSegments)
      setPendingDropFrame(null)
      return
    }

    // Check if there's a gap
    const gap = findFirstGap()
    const segDuration = calcSegmentDuration(frameRate)

    if (gap) {
      // Fill gap with first image, extend with remaining (same as addSegment logic)
      const newSegments: MaintainSegment[] = []
      let currentEndFrame = gap.start

      for (const path of paths) {
        if (currentEndFrame > totalFrames - 1) break
        const span = Math.min(segDuration, totalFrames - currentEndFrame)
        newSegments.push({
          id: uuid(),
          start_frame: currentEndFrame,
          end_frame: currentEndFrame + span - 1,
          content: {
            text: '',
            images: [{
              source_type: 'input' as const,
              file_path: path,
              file_name: path.split('/').pop() ?? path,
            }],
            type: 'flf' as const,
          },
          color: track.color,
        })
        currentEndFrame += segDuration
      }

      const updated = [...segments, ...newSegments].sort((a, b) => a.start_frame - b.start_frame)
      onSegmentsChange(updated)

      // Extend total length if needed
      const lastSeg = newSegments[newSegments.length - 1]
      if (lastSeg && onExtendTimeline) {
        onExtendTimeline(updated, lastSeg.end_frame + 1)
      }
      setPendingDropFrame(null)
      return
    }

    // No gap (full timeline): extend after last segment - same logic as addSegment
    const lastSeg = segments[segments.length - 1]
    const newSegments: MaintainSegment[] = []
    let currentEndFrame = lastSeg.end_frame + 1

    for (const path of paths) {
      newSegments.push({
        id: uuid(),
        start_frame: currentEndFrame,
        end_frame: currentEndFrame + segDuration - 1,
        content: {
          text: '',
          images: [{
            source_type: 'input' as const,
            file_path: path,
            file_name: path.split('/').pop() ?? path,
          }],
          type: 'flf' as const,
        },
        color: track.color,
      })
      currentEndFrame += segDuration
    }

    const updated = [...segments, ...newSegments]
    onSegmentsChange(updated)

    // Extend total length using combined callback
    const lastNewSeg = newSegments[newSegments.length - 1]
    if (onExtendTimeline) {
      onExtendTimeline(updated, lastNewSeg.end_frame + 1)
    }
    setPendingDropFrame(null)
  }

  // ── Render ────────────────────────────────────────────────────────────────

  return (
    <TrackRow track={track} onTrackChange={onTrackChange} height={grow === undefined ? MAINTAIN_TRACK_HEIGHT : undefined} grow={grow}>
      <ContextMenu>
        <ContextMenuTrigger className="relative block w-full h-full" asChild>
          <section
            ref={containerRef}
            aria-label={t('maintainTrack.dropZone')}
            className="relative w-full h-full"
            onDragOver={handleDragOver}
            onDragLeave={() => setPendingDropFrame(null)}
            onDrop={handleDrop}
            onClick={handleTrackAreaClick}
            onDoubleClick={handleTrackAreaDoubleClick}
          >
            {segments.map((seg, index) => (
              <SegmentBlock
                key={seg.id}
                segment={seg}
                totalFrames={totalFrames}
                areaWidth={areaWidth}
                interactive={!track.locked}
                hideLeftHandle={false}
                disableLeftResize={index === 0}
                selected={selectedId === seg.id}
                onSelect={(s) => onSelectedIdChange(selectedId === s.id ? null : s.id)}
                onDragEnd={handleDragEnd}
                onResizeEnd={handleResizeEnd}
                frameRate={frameRate}
                displayFormat={displayFormat}
                onContextMenu={(_, s) => {
                  setRightClickedId(s.id)
                  onSelectedIdChange(s.id)
                }}
                onDoubleClick={(s) => {
                  // Set flag to prevent section from adding empty segment
                  lastDoubleClickedSegId.current = s.id
                  // Clear after a short delay
                  setTimeout(() => { lastDoubleClickedSegId.current = null }, 100)
                }}
                backgroundSlot={
                  seg.content.images.length > 0
                    ? <SegmentImageBackground images={seg.content.images} />
                    : undefined
                }
              >
                <SegmentBlockContent seg={seg} index={index} />
              </SegmentBlock>
            ))}

            {/* Insert buttons for selected segment */}
            {selectedId && !track.locked && (() => {
              const selIdx = segments.findIndex((s) => s.id === selectedId)
              if (selIdx === -1) return null
              const selSeg = segments[selIdx]

              const leftBtnX = (selSeg.start_frame / Math.max(totalFrames - 1, 1)) * areaWidth
              const rightBtnX = ((selSeg.end_frame + 1) / Math.max(totalFrames - 1, 1)) * areaWidth

              return (
                <>
                  {/* Left insert button — hidden for the first segment */}
                  <div
                    className="absolute pointer-events-auto"
                    style={{
                      left: selIdx == 0 ? leftBtnX + 10 : leftBtnX,
                      top: -16,
                      transform: 'translate(-50%, 0)',
                      zIndex: 25,
                    }}
                  >
                    <InsertButton
                      position="left"
                      onClick={() => handleInsertSegment('left')}
                    />
                  </div>

                  {/* Right insert button */}
                  <div
                    className="absolute pointer-events-auto"
                    style={{
                      left: selIdx >= segments.length - 1 || segments.length == 1 ? rightBtnX - 10 : rightBtnX,
                      top: -16,
                      transform: 'translate(-50%, 0)',
                      zIndex: 25,
                    }}
                  >
                    <InsertButton
                      position="right"
                      onClick={() => handleInsertSegment('right')}
                    />
                  </div>
                </>
              )
            })()}

            {/* Empty state with its own context menu */}
            {segments.length === 0 ? (
              <ContextMenu>
                <ContextMenuTrigger asChild>
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground select-none opacity-40">
                    <div className="flex items-center gap-1">
                      <Plus className="w-3 h-3" />
                      <span className="text-[11px]">
                        {t('maintainTrack.placeholder')}
                      </span>
                    </div>
                    <span className="text-[10px]">{t('maintainTrack.placeholderHint')}</span>
                  </div>
                </ContextMenuTrigger>
                <ContextMenuContent>
                  <ContextMenuItem onClick={addSegment}>{t('maintainTrack.contextAdd')}</ContextMenuItem>
                  <ContextMenuItem
                    onClick={() => {
                      setAddAnchorPos({ x: 0, y: 60 })
                      setSelectorValue('')
                      setAddPopoverOpen(true)
                    }}
                  >
                    {t('maintainTrack.addMedia')}
                  </ContextMenuItem>
                </ContextMenuContent>
              </ContextMenu>
            ) : (
              /* Placeholder for non-empty state (hidden) */
              <div className="hidden" />
            )}

            {/* Drop indicator */}
            {pendingDropFrame !== null && (
              <div
                className="absolute top-0 w-0.5 h-full bg-foreground/60 pointer-events-none"
                style={{ left: (pendingDropFrame / Math.max(totalFrames - 1, 1)) * areaWidth }}
              />
            )}

          </section>
        </ContextMenuTrigger>

        {/* Context menu */}
        <ContextMenuContent onCloseAutoFocus={() => setRightClickedId(null)}>
          <ContextMenuItem onClick={() => onSegmentsChange(distributeEvenly(segments, totalFrames))}>
            {t('maintainTrack.contextDistribute')}
          </ContextMenuItem>
          <ContextMenuItem onClick={addSegment}>{t('maintainTrack.contextAdd')}</ContextMenuItem>
          {rightClickedId && (
            <>
              <ContextMenuItem
                onClick={() => { cloneSegment(rightClickedId); setRightClickedId(null) }}
              >
                {t('maintainTrack.contextClone')}
              </ContextMenuItem>
              <ContextMenuItem
                onClick={() => { deleteSegment(rightClickedId); setRightClickedId(null) }}
              >
                {t('maintainTrack.contextDelete')}
              </ContextMenuItem>
            </>
          )}
        </ContextMenuContent>
      </ContextMenu>

      {/* Popover for adding segment via click on empty area */}
      <Popover open={addPopoverOpen} onOpenChange={setAddPopoverOpen}>
        <PopoverAnchor asChild>
          <span
            className="absolute w-0 h-0 pointer-events-none"
            style={{ left: addAnchorPos.x, top: addAnchorPos.y }}
          />
        </PopoverAnchor>
        <PopoverContent
          data-add-popover=""
          className="p-0 w-auto"
          side="bottom"
          align="start"
          onOpenAutoFocus={(e) => e.preventDefault()}
        >
          <MediaSelector
            value={selectorValue}
            onChange={(filePath, source) => {
              if (!filePath) {
                setAddPopoverOpen(false)
                return
              }

              const isMultiFile = filePath.includes('|MULTIPLE|')
              const paths = isMultiFile ? filePath.split('|MULTIPLE|') : [filePath]
              const currentSourceType = source ?? 'input'

              // If empty track: create new segments distributed across totalFrames
              if (segments.length === 0) {
                const newSegments: MaintainSegment[] = []
                let currentEndFrame = 0

                for (let i = 0; i < paths.length; i++) {
                  const path = paths[i]
                  const fileName = path.split('/').pop() ?? path
                  const isUrl = path.startsWith('http')

                  // Calculate segment span (evenly distribute remaining frames, 前大后小)
                  const remainingFrames = totalFrames - currentEndFrame
                  const remainingSlots = paths.length - i
                  const baseSpan = Math.floor(remainingFrames / remainingSlots)
                  const remainder = remainingFrames % remainingSlots
                  const span = baseSpan + (i < remainder ? 1 : 0)

                  const seg: MaintainSegment = {
                    id: uuid(),
                    start_frame: currentEndFrame,
                    end_frame: Math.min(currentEndFrame + span - 1, totalFrames - 1),
                    content: {
                      text: '',
                      images: [{
                        source_type: isUrl ? 'url' : currentSourceType,
                        file_path: isUrl ? undefined : path,
                        url: isUrl ? path : undefined,
                        file_name: fileName,
                      }],
                      type: 'flf',
                    },
                    color: track.color,
                  }
                  newSegments.push(seg)
                  currentEndFrame = seg.end_frame + 1
                }

                onSegmentsChange(newSegments)
                setAddPopoverOpen(false)
                return
              }

              // If has segments: check if there's a gap to fill first
              const gap = findFirstGap()
              if (gap) {
                // Fill gap with first image, extend with remaining
                const segDuration = calcSegmentDuration(frameRate)
                const newSegments: MaintainSegment[] = []
                let currentEndFrame = gap.start

                for (let i = 0; i < paths.length; i++) {
                  if (currentEndFrame > totalFrames - 1) break
                  const path = paths[i]
                  const fileName = path.split('/').pop() ?? path
                  const isUrl = path.startsWith('http')

                  const span = Math.min(segDuration, totalFrames - currentEndFrame)
                  newSegments.push({
                    id: uuid(),
                    start_frame: currentEndFrame,
                    end_frame: currentEndFrame + span - 1,
                    content: {
                      text: '',
                      images: [{
                        source_type: isUrl ? 'url' : currentSourceType,
                        file_path: isUrl ? undefined : path,
                        url: isUrl ? path : undefined,
                        file_name: fileName,
                      }],
                      type: 'flf',
                    },
                    color: track.color,
                  })
                  currentEndFrame += segDuration
                }

                onSegmentsChange([...segments, ...newSegments].sort((a, b) => a.start_frame - b.start_frame))
                setAddPopoverOpen(false)
                return
              }

              // No gap (filled): extend after last segment with calculated duration for each image
              const lastSeg = segments[segments.length - 1]
              const segDuration = calcSegmentDuration(frameRate)
              const newSegments: MaintainSegment[] = []
              let currentEndFrame = lastSeg.end_frame + 1

              for (let i = 0; i < paths.length; i++) {
                const path = paths[i]
                const fileName = path.split('/').pop() ?? path
                const isUrl = path.startsWith('http')

                newSegments.push({
                  id: uuid(),
                  start_frame: currentEndFrame,
                  end_frame: currentEndFrame + segDuration - 1,
                  content: {
                    text: '',
                    images: [{
                      source_type: isUrl ? 'url' : currentSourceType,
                      file_path: isUrl ? undefined : path,
                      url: isUrl ? path : undefined,
                      file_name: fileName,
                    }],
                    type: 'flf',
                  },
                  color: track.color,
                })
                currentEndFrame += segDuration
              }

              onSegmentsChange([...segments, ...newSegments])

              // Extend total length if needed (use combined callback to avoid batching issues)
              if (onExtendTimeline) {
                const lastNewSeg = newSegments[newSegments.length - 1]
                if (lastNewSeg) {
                  onExtendTimeline([...segments, ...newSegments], lastNewSeg.end_frame + 1)
                }
              }
              setAddPopoverOpen(false)
            }}
            mediaType="image"
            defaultTab="inputs"
          />
        </PopoverContent>
      </Popover>
    </TrackRow>
  )
}
