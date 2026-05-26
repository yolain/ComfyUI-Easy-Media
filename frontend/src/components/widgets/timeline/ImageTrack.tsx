import { useRef, useState } from 'react'
import { Image, Plus, Trash2 } from 'lucide-react'
import { Popover, PopoverContent, PopoverAnchor } from '@/components/ui/popover'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import type { MediaTab } from '@/components/widgets/mediaSelector/MediaSelector'
import { SegmentBlock } from './SegmentBlock'
import { TrackRow } from './TrackRow'
import type { Track, ImageSegment, Segment } from '@/types/timeline'
import { useT } from '@/lib/i18n'
import { uuid } from '@/lib/uuid'

interface ImageTrackProps {
  track: Track
  totalFrames: number
  areaWidth: number
  selectedId: string | null
  onSelectedIdChange: (id: string | null) => void
  onTrackChange: (patch: Partial<Track>) => void
  onSegmentsChange: (segments: Segment[]) => void
}

// Default image duration ~2 seconds at 24fps
const DEFAULT_IMAGE_SPAN = 48

/** Returns the thumbnail src URL for an image segment, or null if unavailable. */
function getSegmentImageSrc(content: ImageSegment['content']): string | null {
  if (content.url) return content.url
  if (content.file_path) {
    const relPath = content.file_path
    // Handle both forward slash and backslash as path separator
    const lastSlash = Math.max(relPath.lastIndexOf('/'), relPath.lastIndexOf('\\'))
    const filename = lastSlash >= 0 ? relPath.slice(lastSlash + 1) : relPath
    const subfolder = lastSlash >= 0 ? relPath.slice(0, lastSlash) : ''
    const typeParam = content.source_type === 'output' ? 'output' : 'input'
    return `/view?filename=${encodeURIComponent(filename)}&type=${typeParam}&subfolder=${encodeURIComponent(subfolder)}`
  }
  return null
}

function SegmentThumbnail({ content }: Readonly<{ content: ImageSegment['content'] }>) {
  const src = getSegmentImageSrc(content)
  if (!src) return null
  return (
    <div className="flex items-center justify-center w-full h-full">
      <img
        src={src}
        alt={content.file_name}
        className="h-full object-cover"
        onError={(e) => { (e.target as HTMLImageElement).style.display = 'none' }}
      />
    </div>
  )
}

/** Map segment source_type to the MediaSelector tab */
function sourceTypeToTab(sourceType: string | undefined): MediaTab {
  if (sourceType === 'input') return 'inputs'
  if (sourceType === 'url') return 'url'
  if (sourceType === 'local') return 'local'
  return 'inputs'
}

export function ImageTrack({
  track,
  totalFrames,
  areaWidth,
  selectedId,
  onSelectedIdChange,
  onTrackChange,
  onSegmentsChange,
}: Readonly<ImageTrackProps>) {
  const t = useT()
  const containerRef = useRef<HTMLElement>(null)
  const [pendingDropFrame, setPendingDropFrame] = useState<number | null>(null)
  const [popoverOpen, setPopoverOpen] = useState(false)
  const [popoverDefaultTab, setPopoverDefaultTab] = useState<MediaTab>('inputs')
  const [anchorPos, setAnchorPos] = useState({ x: 0, y: 0 })
  /** The segment being edited via the selector, or null when adding a new one */
  const [editingSegId, setEditingSegId] = useState<string | null>(null)
  const [selectorValue, setSelectorValue] = useState('')

  const segments = track.segments as ImageSegment[]
  const lastOccupied = segments.reduce((max, s) => Math.max(max, s.end_frame), -1)
  const canImport = lastOccupied < totalFrames - 1

  function openPopover(e: React.MouseEvent, defaultTab: MediaTab, segId: string | null, currentValue: string) {
    const rect = containerRef.current?.getBoundingClientRect()
    const el = containerRef.current
    const zoom = (rect && el) ? rect.width / el.offsetWidth : 1
    setAnchorPos({
      x: (e.clientX - (rect?.left ?? 0)) / zoom,
      y: (e.clientY - (rect?.top ?? 0)) / zoom,
    })
    setPopoverDefaultTab(defaultTab)
    setEditingSegId(segId)
    setSelectorValue(currentValue)
    setPopoverOpen(true)
  }

  function handleSelectorChange(filePath: string) {
    setSelectorValue(filePath)
    const fileName = filePath.split('/').pop() ?? filePath
    const isUrl = filePath.startsWith('http')

    if (editingSegId) {
      // Update existing segment
      onSegmentsChange(
        segments.map((s) =>
          s.id === editingSegId
            ? {
                ...s,
                content: isUrl
                  ? { source_type: 'url', url: filePath, file_name: fileName }
                  : { source_type: 'input', file_path: filePath, file_name: fileName },
              }
            : s,
        ),
      )
    } else {
      // Add new segment
      const cursor = lastOccupied >= 0 ? lastOccupied + 1 : 0
      if (cursor >= totalFrames) return
      const newSeg: ImageSegment = {
        id: uuid(),
        start_frame: cursor,
        end_frame: Math.min(totalFrames - 1, cursor + DEFAULT_IMAGE_SPAN - 1),
        content: isUrl
          ? { source_type: 'url', url: filePath, file_name: fileName }
          : { source_type: 'input', file_path: filePath, file_name: fileName },
        color: track.color,
      }
      onSegmentsChange([...segments, newSeg].toSorted((a, b) => a.start_frame - b.start_frame))
      setEditingSegId(newSeg.id)
    }
    setPopoverOpen(false)
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
    const files = e.dataTransfer.files
    if (!files || !canImport) return
    const updated = [...segments]
    let cursor = pendingDropFrame ?? (lastOccupied >= 0 ? lastOccupied + 1 : 0)
    for (const file of Array.from(files)) {
      if (cursor >= totalFrames) break
      try {
        const form = new FormData()
        form.append('image', file)
        form.append('type', 'input')
        form.append('overwrite', 'false')
        const res = await fetch('/upload/image', { method: 'POST', body: form })
        if (!res.ok) continue
        const json = await res.json() as { name: string; subfolder?: string }
        const sub = json.subfolder ? `${json.subfolder}/` : ''
        const filePath = `${sub}${json.name}`
        const newSeg: ImageSegment = {
          id: uuid(),
          start_frame: cursor,
          end_frame: Math.min(totalFrames - 1, cursor + DEFAULT_IMAGE_SPAN - 1),
          content: {
            source_type: 'input',
            file_path: filePath,
            file_name: json.name,
          },
          color: track.color,
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

  function handleResizeEnd(segmentId: string, edge: 'start' | 'end', deltaFrames: number) {
    if (track.locked) return
    const seg = segments.find((s) => s.id === segmentId)
    if (!seg) return
    const others = segments.filter((s) => s.id !== segmentId).sort((a, b) => a.start_frame - b.start_frame)
    let newStart = seg.start_frame
    let newEnd = seg.end_frame
    if (edge === 'start') {
      newStart = Math.max(0, seg.start_frame + deltaFrames)
      const prev = others.findLast((s) => s.end_frame < seg.end_frame)
      if (prev) newStart = Math.max(newStart, prev.end_frame + 1)
      newStart = Math.min(newStart, seg.end_frame - 1)
    } else {
      newEnd = Math.min(totalFrames - 1, seg.end_frame + deltaFrames)
      const next = others.find((s) => s.start_frame > seg.start_frame)
      if (next) newEnd = Math.min(newEnd, next.start_frame - 1)
      newEnd = Math.max(newEnd, seg.start_frame + 1)
    }
    onSegmentsChange(segments.map((s) => (s.id === segmentId ? { ...s, start_frame: newStart, end_frame: newEnd } : s)))
  }

  function handleDeleteSegment(segId: string) {
    onSegmentsChange(segments.filter((s) => s.id !== segId))
    setPopoverOpen(false)
    if (selectedId === segId) onSelectedIdChange(null)
  }

  function handleToolAdd() {
    if (!canImport) return
    setAnchorPos({ x: 8, y: 8 })
    setPopoverDefaultTab('inputs')
    setEditingSegId(null)
    setSelectorValue('')
    setPopoverOpen(true)
  }

  const toolSlots: [React.ReactNode, React.ReactNode, React.ReactNode] = [
    undefined,
    <button
      key="add"
      type="button"
      title={t('imageTrack.addSegment')}
      disabled={!canImport}
      className="w-full h-full flex items-center justify-center hover:bg-accent disabled:opacity-25 cursor-pointer"
      onClick={handleToolAdd}
    >
      <Plus className="w-2.5 h-2.5 text-muted-foreground" />
    </button>,
    <button
      key="delete"
      type="button"
      title={t('imageTrack.deleteSegment')}
      disabled={!selectedId}
      className="w-full h-full flex items-center justify-center hover:bg-accent disabled:opacity-25 cursor-pointer"
      onClick={() => { if (selectedId) handleDeleteSegment(selectedId) }}
    >
      <Trash2 className="w-2.5 h-2.5 text-muted-foreground" />
    </button>,
  ]

  function handleDragEnd(segmentId: string, deltaFrames: number, origStart: number) {
    if (track.locked) return
    const seg = segments.find((s) => s.id === segmentId)
    if (!seg) return
    const span = seg.end_frame - seg.start_frame
    const newStart = Math.max(0, origStart + deltaFrames)
    const newEnd = Math.min(totalFrames - 1, newStart + span)
    const finalStart = newEnd - span
    const sorted = segments
      .map((s) =>
        s.id === segmentId
          ? { ...s, start_frame: Math.max(0, finalStart), end_frame: Math.min(totalFrames - 1, newEnd) }
          : s,
      )
      .toSorted((a, b) => a.start_frame - b.start_frame)
    onSegmentsChange(sorted)
  }

  return (
    <TrackRow track={track} onTrackChange={onTrackChange} toolSlots={toolSlots}>
      <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
        <section
          ref={containerRef}
          aria-label={t('imageTrack.dropZone')}
          className="relative w-full h-full"
          onDragOver={handleDragOver}
          onDragLeave={() => setPendingDropFrame(null)}
          onDrop={handleDrop}
          onContextMenu={(e) => {
            e.preventDefault()
            if (!canImport && !editingSegId) return
            openPopover(e, 'inputs', null, '')
          }}
        >
          {/* Virtual anchor positioned at right-click coordinates */}
          <PopoverAnchor asChild>
            <span
              className="absolute w-0 h-0 pointer-events-none"
              style={{ left: anchorPos.x, top: anchorPos.y }}
            />
          </PopoverAnchor>
            {segments.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-muted-foreground pointer-events-none select-none opacity-40">
                <div className="flex items-center gap-1">
                  <Image className="w-3 h-3" />
                  <span className="text-[11px]">{t('imageTrack.placeholder')}</span>
                </div>
                <span className="text-[10px]">{t('imageTrack.placeholderHint')}</span>
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
                    selected={selectedId === seg.id}
                    onSelect={(s) => onSelectedIdChange(selectedId === s.id ? null : s.id)}
                    onDragEnd={handleDragEnd}
                    onResizeEnd={handleResizeEnd}
                    backgroundSlot={<SegmentThumbnail content={seg.content} />}
                    onContextMenu={(e) => {
                      e.preventDefault()
                      e.stopPropagation()
                      onSelectedIdChange(seg.id)
                      openPopover(
                        e,
                        sourceTypeToTab(seg.content.source_type),
                        seg.id,
                        seg.content.file_path ?? seg.content.local_path ?? seg.content.url ?? '',
                      )
                    }}
                  />
                ))}
              </>
            )}

            {pendingDropFrame !== null && (
              <div
                className="absolute top-0 w-0.5 h-full bg-foreground/60 pointer-events-none"
                style={{ left: (pendingDropFrame / Math.max(totalFrames - 1, 1)) * areaWidth }}
              />
            )}
        </section>

        <PopoverContent
          className="p-0 w-auto"
          side="bottom"
          align="start"
          onOpenAutoFocus={(e) => e.preventDefault()}
        >
          {/* {editingSegId && (
            <div className="flex justify-end px-2 pt-1.5">
              <button
                type="button"
                className="text-[11px] text-destructive hover:underline"
                onClick={() => handleDeleteSegment(editingSegId)}
              >
                {t('common.delete')}
              </button>
            </div>
          )} */}
          <MediaSelector
            value={selectorValue}
            onChange={handleSelectorChange}
            mediaType="image"
            defaultTab={popoverDefaultTab}
          />
        </PopoverContent>
      </Popover>
    </TrackRow>
  )
}
