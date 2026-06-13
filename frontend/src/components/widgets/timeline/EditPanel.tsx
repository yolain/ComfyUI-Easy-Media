import React, { useCallback, useEffect, useRef, useState, useMemo } from 'react'
import { Plus } from 'lucide-react'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Popover, PopoverContent, PopoverAnchor } from '@/components/ui/popover'
import {
  ContextMenu,
  ContextMenuContent,
  ContextMenuItem,
  ContextMenuSeparator,
  ContextMenuTrigger,
} from '@/components/ui/context-menu'

import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import type { MediaTab } from '@/components/widgets/mediaSelector/MediaSelector'
import type { MaintainContent, MaintainType, ImageItem, MaintainSegment, Segment, TimeDisplayFormat } from '@/types/timeline'
import { useT } from '@/lib/i18n'
import { uuid } from '@/lib/uuid'
import { TimelineRuler } from './TimelineRuler'
import { imageItemFromPath, imageItemFromUrl, tiledImageBackground } from '@/lib/image-utils'
import { computeSlotItems } from '@/lib/timeline-utils'
import type { SlotItem } from '@/lib/timeline-utils'

interface EditPanelProps {
  segment: MaintainSegment
  allSegments: MaintainSegment[]
  totalFrames: number
  frameRate: number
  displayFormat: TimeDisplayFormat
  areaWidth: number
  canvasScale: number
  trackColor: string
  onContentChange: (patch: Partial<MaintainContent>) => void
  onAllSegmentsChange: (segs: Segment[]) => void
  node?: any
  app?: any
}

// ── Sub-block types ──────────────────────────────────────────────────────────

interface SubBlock {
  /** Stable key for React – index-based */
  id: string
  item: ImageItem
  /** Frame position relative to the segment's start (0-based, 0..span-1) */
  start_frame: number
  end_frame: number
}

// ── Combined prompt editor helpers ──────────────────────────────────────────

function escHtml(str: string): string {
  return String(str)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;')
}

function buildHighlightHtml(raw: string): string {
  const sep = '<span style="color:var(--highlight)" data-pipe="1">|</span>'
  return raw.split(/[|｜]/).map(escHtml).join(sep)
}

const COMBINED_TEXT_STYLE = {
  fontFamily: 'inherit',
  fontSize: '10px',
  lineHeight: '1.4',
  letterSpacing: 0,
  whiteSpace: 'pre-wrap',
  overflowWrap: 'break-word',
  wordBreak: 'break-word',
} satisfies React.CSSProperties

function applyTextsToMaintainSegments(
  parts: string[],
  segments: MaintainSegment[],
  totalFrames: number,
  color: string,
): MaintainSegment[] {
  if (parts.length === 0) return []

  if (parts.length === segments.length) {
    return segments.map((s, i) => ({ ...s, content: { ...s.content, text: parts[i] } }))
  }
  const count = Math.max(1, parts.length)
  const base = Math.floor(totalFrames / count)
  const remainder = totalFrames % count
  return parts.map((text, i) => ({
    id: segments[i]?.id ?? uuid(),
    start_frame: i * base + Math.min(i, remainder),
    end_frame: i < count - 1
      ? i * base + Math.min(i, remainder) + base + (i < remainder ? 1 : 0) - 1
      : totalFrames - 1,
    content: {
      text,
      images: segments[i]?.content.images ?? [],
      type: segments[i]?.content.type ?? 'flf',
    },
    color: segments[i]?.color ?? color,
    markers: segments[i]?.markers,
  }))
}

// ── MaintainCombinedEditor ───────────────────────────────────────────────────

interface MaintainCombinedEditorProps {
  allSegments: MaintainSegment[]
  totalFrames: number
  trackColor: string
  onAllSegmentsChange: (segs: Segment[]) => void
}

function MaintainCombinedEditor({
  allSegments,
  totalFrames,
  trackColor,
  onAllSegmentsChange,
}: Readonly<MaintainCombinedEditorProps>) {
  const t = useT()
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const overlayRef = useRef<HTMLDivElement>(null)
  const composing = useRef(false)
  const segmentsRef = useRef(allSegments)
  segmentsRef.current = allSegments

  const combinedText = allSegments.map((s) => s.content.text).join('|')
  const [text, setText] = useState(() => combinedText)

  useEffect(() => {
    if (composing.current) return
    setText((current) => (current === combinedText ? current : combinedText))
  }, [combinedText])

  function syncScroll() {
    if (textareaRef.current && overlayRef.current) {
      overlayRef.current.scrollTop = textareaRef.current.scrollTop
    }
  }

  const commit = useCallback((raw: string) => {
    const parts = raw === '' ? [] : raw.split(/[|｜]/)
    onAllSegmentsChange(
      applyTextsToMaintainSegments(parts, segmentsRef.current, totalFrames, trackColor),
    )
  }, [totalFrames, trackColor, onAllSegmentsChange])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const raw = e.target.value
    setText(raw)
    if (composing.current) return
    commit(raw)
  }, [commit])

  return (
    <div className="flex flex-col gap-1">
      <div className="relative w-full h-30 overflow-hidden">
        {/* Highlight overlay – mirrors textarea content with styled separators */}
        <div
          ref={overlayRef}
          aria-hidden="true"
          className="absolute inset-0 overflow-y-auto pointer-events-none"
          style={{ ...COMBINED_TEXT_STYLE, padding: 0 }}
          dangerouslySetInnerHTML={{ __html: buildHighlightHtml(text) }}
        />
        {/* Transparent textarea for native input handling */}
        <Textarea
          ref={textareaRef}
          value={text}
          onChange={handleChange}
          onScroll={syncScroll}
          spellCheck={false}
          onCompositionStart={() => { composing.current = true }}
          onCompositionEnd={() => {
            composing.current = false
            const raw = textareaRef.current?.value ?? ''
            setText(raw)
            commit(raw)
          }}
          className="absolute inset-0 w-full h-full resize-none border-0 shadow-none focus-visible:ring-0 p-0 bg-transparent outline-none"
          style={{ ...COMBINED_TEXT_STYLE, color: 'transparent', caretColor: 'var(--foreground)' }}
        />
      </div>
      <p className="text-[9px] text-muted-foreground">{t('maintainTrack.combinedHint')}</p>
    </div>
  )
}

function buildSubBlocks(images: ImageItem[], span: number): SubBlock[] {
  return images.map((img, i) => {
    const perImg = Math.max(1, Math.floor(span / Math.max(images.length, 1)))
    return {
      id: String(i),
      item: img,
      start_frame: img.start_frame ?? i * perImg,
      end_frame: img.end_frame ?? Math.min(span - 1, i * perImg + perImg - 1),
    }
  })
}

function subBlockRect(start: number, end: number, span: number, width: number) {
  const scale = width / Math.max(span - 1, 1)
  const left = start * scale
  const right = Math.min((end + 1) * scale, width)
  return { left, width: Math.max(right - left, 20) }
}

async function uploadImageFile(file: File): Promise<string> {
  const form = new FormData()
  form.append('image', file)
  form.append('type', 'input')
  form.append('overwrite', 'false')
  const res = await fetch('/upload/image', { method: 'POST', body: form })
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`)
  const json = await res.json() as { name: string; subfolder?: string }
  const sub = json.subfolder ? `${json.subfolder}/` : ''
  return `${sub}${json.name}`
}

function formatTime(frame: number, frameRate: number, displayFormat: TimeDisplayFormat): string {
  return displayFormat === 'seconds'
    ? `${(frame / frameRate).toFixed(1)}s`
    : `${frame}f`
}

// ── SubImageBlock ────────────────────────────────────────────────────────────

interface SubImageBlockProps {
  block: SubBlock
  span: number
  areaWidth: number
  frameRate: number
  displayFormat: TimeDisplayFormat
  trackColor: string
  selected?: boolean
  onSelect?: (id: string) => void
  onMoveEnd: (id: string, newStart: number, newEnd: number) => void
  onResizeEnd: (id: string, newStart: number, newEnd: number) => void
  onEdit: () => void
  onRemove: () => void
  onContextMenu?: (e: React.MouseEvent) => void
}

/** Returns a slightly lighter border color from a hex color */
function lightenColor(hex: string, amount: number = 0.3): string {
  if (!hex.startsWith('#') || hex.length < 7) return 'var(--border)'
  const r = Number.parseInt(hex.slice(1, 3), 16)
  const g = Number.parseInt(hex.slice(3, 5), 16)
  const b = Number.parseInt(hex.slice(5, 7), 16)
  const newR = Math.round(r + (255 - r) * amount)
  const newG = Math.round(g + (255 - g) * amount)
  const newB = Math.round(b + (255 - b) * amount)
  return `#${newR.toString(16).padStart(2, '0')}${newG.toString(16).padStart(2, '0')}${newB.toString(16).padStart(2, '0')}`
}

const SubImageBlock = React.forwardRef<HTMLDivElement, SubImageBlockProps>(function SubImageBlock({
  block,
  span,
  areaWidth,
  frameRate,
  displayFormat,
  trackColor,
  selected = false,
  onSelect,
  onMoveEnd,
  onResizeEnd,
  onEdit,
  onRemove,
  onContextMenu,
}: Readonly<SubImageBlockProps>, forwardedRef) {
  const t = useT()
  const [localStart, setLocalStart] = useState(block.start_frame)
  const [localEnd, setLocalEnd] = useState(block.end_frame)
  const [labelVisible, setLabelVisible] = useState(false)
  const [isResizing, setIsResizing] = useState(false)

  const localStartRef = useRef(block.start_frame)
  const localEndRef = useRef(block.end_frame)
  const isDraggingRef = useRef(false)
  const didMoveRef = useRef(false)
  const labelTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!isDraggingRef.current) {
      localStartRef.current = block.start_frame
      localEndRef.current = block.end_frame
      setLocalStart(block.start_frame)
      setLocalEnd(block.end_frame)
    }
  }, [block.start_frame, block.end_frame])

  const { left, width: blockWidth } = subBlockRect(localStart, localEnd, span, areaWidth)
  // Time-position label – shows relative to segment, not global time
  // Time-position label – shows relative to segment, not global time

  function showLabel() {
    setLabelVisible(true)
    if (labelTimerRef.current) clearTimeout(labelTimerRef.current)
  }

  function hideLabelAfterDelay() {
    isDraggingRef.current = false
    if (labelTimerRef.current) clearTimeout(labelTimerRef.current)
    labelTimerRef.current = setTimeout(() => setLabelVisible(false), 2000)
  }

  function handleMoveMouseDown(e: React.MouseEvent) {
    if (e.button !== 0) return
    e.preventDefault()
    isDraggingRef.current = true
    didMoveRef.current = false
    showLabel()
    const startX = e.clientX
    const origStart = block.start_frame
    const origEnd = block.end_frame
    const blockSpan = origEnd - origStart
    const scale = areaWidth / Math.max(span - 1, 1)

    function onMouseMove(ev: MouseEvent) {
      const deltaFrames = Math.round((ev.clientX - startX) / scale)
      if (deltaFrames !== 0) didMoveRef.current = true
      const newStart = Math.max(0, Math.min(span - 1 - blockSpan, origStart + deltaFrames))
      const newEnd = newStart + blockSpan
      localStartRef.current = newStart
      localEndRef.current = newEnd
      setLocalStart(newStart)
      setLocalEnd(newEnd)
    }

    function onMouseUp() {
      hideLabelAfterDelay()
      onMoveEnd(block.id, localStartRef.current, localEndRef.current)
      globalThis.removeEventListener('mousemove', onMouseMove)
      globalThis.removeEventListener('mouseup', onMouseUp)
    }

    globalThis.addEventListener('mousemove', onMouseMove)
    globalThis.addEventListener('mouseup', onMouseUp)
  }

  function makeResizeMouseDown(edge: 'start' | 'end') {
    return (e: React.MouseEvent) => {
      if (e.button !== 0) return
      e.preventDefault()
      e.stopPropagation()
      isDraggingRef.current = true
      setIsResizing(true)
      showLabel()
      const startX = e.clientX
      const origStart = block.start_frame
      const origEnd = block.end_frame
      const scale = areaWidth / Math.max(span - 1, 1)

      function onMouseMove(ev: MouseEvent) {
        const deltaFrames = Math.round((ev.clientX - startX) / scale)
        if (edge === 'start') {
          const newStart = Math.max(0, Math.min(origEnd - 1, origStart + deltaFrames))
          localStartRef.current = newStart
          setLocalStart(newStart)
        } else {
          const newEnd = Math.max(origStart + 1, Math.min(span - 1, origEnd + deltaFrames))
          localEndRef.current = newEnd
          setLocalEnd(newEnd)
        }
      }

      function onMouseUp() {
        setIsResizing(false)
        hideLabelAfterDelay()
        onResizeEnd(block.id, localStartRef.current, localEndRef.current)
        globalThis.removeEventListener('mousemove', onMouseMove)
        globalThis.removeEventListener('mouseup', onMouseUp)
      }

      globalThis.addEventListener('mousemove', onMouseMove)
      globalThis.addEventListener('mouseup', onMouseUp)
    }
  }

  const images = [block.item]
  const borderColor = isResizing ? '#eab308' : selected ? 'var(--foreground)' : lightenColor(trackColor)
  const bgColor =  isResizing ? '#eab308' : selected ? 'var(--foreground)' : 'transparent'

  return (
    <div
      ref={forwardedRef}
      role="button"
      tabIndex={0}
      className="absolute top-0 h-full select-none group"
      style={{
        left,
        width: blockWidth,
        backgroundColor: trackColor,
        opacity: 0.9,
        border: `1px solid ${borderColor}`,
        borderRadius: 3,
        overflow: 'hidden',
        cursor: 'grab',
      }}
      onMouseDown={handleMoveMouseDown}
      onClick={(e) => {
        if (didMoveRef.current) return
        e.stopPropagation()
        onSelect?.(block.id)
      }}
      onDoubleClick={() => {
        onSelect?.(block.id)
        onEdit()
      }}
      onContextMenu={onContextMenu}
      onKeyDown={(e) => { if (e.key === 'Enter') { onSelect?.(block.id); onEdit() } }}
    >
      {/* Tiled image background */}
      {images.length > 0 && (
        <div
          className="absolute inset-0 bg-black overflow-hidden"
          style={tiledImageBackground(images)}
        />
      )}

      {/* Left resize handle */}
      <button
        type="button"
        aria-label="Resize start"
        className="absolute left-0 top-0 h-full w-0.5 cursor-ew-resize z-10 border-0 p-0"
        style={{ background: bgColor }}
        onMouseDown={makeResizeMouseDown('start')}
      />
      {/* Right resize handle */}
      <button
        type="button"
        aria-label="Resize end"
        className="absolute right-0 top-0 h-full w-0.5 cursor-ew-resize z-10  border-0 p-0"
        style={{ background: bgColor }}
        onMouseDown={makeResizeMouseDown('end')}
      />

      {/* Delete button */}
      <button
        type="button"
        className="absolute top-0.5 right-0.5 w-3 h-3 z-20 rounded-sm bg-destructive text-white text-[9px] flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
        onClick={(e) => { e.stopPropagation(); onRemove() }}
        title={t('common.delete')}
      >
        ×
      </button>

      {/* Filename (bottom strip) */}
      <div className="absolute bottom-0 left-0 right-0 text-center text-[8px] bg-black/60 text-white truncate px-1 pointer-events-none">
        {block.item.file_name}
      </div>

      {/* Time-position label – shows during drag/resize, auto-hides after 2s */}
      {labelVisible && (
        <div className="absolute top-0.5 left-1 z-20 pointer-events-none">
          <span className="text-[9px] bg-black/80 text-white px-1 py-px rounded leading-none">
            {formatTime(localStart, frameRate, displayFormat)}
          </span>
        </div>
      )}
    </div>
  )
})

// ── EditPanel ────────────────────────────────────────────────────────────────

export function EditPanel({
  segment,
  allSegments,
  totalFrames,
  frameRate,
  displayFormat,
  areaWidth,
  trackColor,
  canvasScale,
  onContentChange,
  onAllSegmentsChange,
  node,
  app,
}: Readonly<EditPanelProps>) {
  const t = useT()
  const span = segment.end_frame - segment.start_frame + 1
  const [promptMode, setPromptMode] = useState<'combined' | 'individual'>('individual')

  // Recompute slot items when popover opens to get fresh graph data

  // Sub-block state: positions within this segment (0..span-1)
  const [subBlocks, setSubBlocks] = useState<SubBlock[]>(() =>
    buildSubBlocks(segment.content.images, span),
  )
  const imageSyncKey = segment.content.images
    .map((img) => [
      img.source_type,
      img.file_path,
      img.local_path,
      img.url,
      img.slot_name,
      img.start_frame,
      img.end_frame,
    ].join(':'))
    .join('|')

  useEffect(() => {
    setSubBlocks(buildSubBlocks(segment.content.images, span))
  }, [segment.id, imageSyncKey, segment.content.images, span])

  // Image selector popover
  const [popoverOpen, setPopoverOpen] = useState(false)
  const [editingBlockId, setEditingBlockId] = useState<string | null>(null)
  const [selectedBlockId, setSelectedBlockId] = useState<string | null>(null)
  const [selectorValue, setSelectorValue] = useState('')
  const [popoverTab, setPopoverTab] = useState<MediaTab>('inputs')
  const [anchorPos, setAnchorPos] = useState({ x: 8, y: 8 })
  const trackAreaRef = useRef<HTMLDivElement>(null)
  const justOpenedPopoverRef = useRef(false)

  // Recompute slot items when popover opens to get fresh graph data
  const slotItems = useMemo(() => computeSlotItems(node, app, 'image'), [node, app, popoverOpen])

  function openPopover(
    e: React.MouseEvent | null,
    blockId: string | null,
    current: string,
    tab: MediaTab,
  ) {
    if (e) {
      const rect = trackAreaRef.current?.getBoundingClientRect()
      setAnchorPos({
        x: e.clientX - (rect?.left ?? 0),
        y: e.clientY - (rect?.top ?? 0),
      })
    }
    justOpenedPopoverRef.current = true
    setEditingBlockId(blockId)
    setSelectorValue(current)
    setPopoverTab(tab)
    setPopoverOpen(true)
  }

  function handleSelectorChange(value: string, source?: 'input' | 'output' | 'local') {
    const isSlot = value.startsWith('__slot__:')
    const isUrl = !isSlot && /^https?:\/\//i.test(value)
    const slotName = isSlot ? value.slice('__slot__:'.length) : undefined

    // Get img src from slotItems if available
    const slotItem = slotItems.find((item: SlotItem) => item.value === value)
    const imgSrc = slotItem?.img

    let newItem: ImageItem
    if (isSlot) {
      newItem = {
        source_type: 'slot',
        slot_name: slotName,
        file_name: slotName ?? value,
        url: imgSrc,
      }
    } else if (isUrl) {
      newItem = imageItemFromUrl(value)
    } else {
      newItem = imageItemFromPath(value, source)
    }

    if (editingBlockId === null) {
      // Add new block
      const lastEnd = subBlocks.reduce((max, b) => Math.max(max, b.end_frame), -1)
      let newStart = lastEnd >= 0 ? lastEnd + 1 : 0
      let updatedBlocks = subBlocks
      if (newStart >= span && subBlocks.length > 0) {
        // Timeline is full — steal the second half of the last block
      const lastBlock = subBlocks.at(-1)!
        const mid = Math.floor((lastBlock.start_frame + lastBlock.end_frame) / 2)
        updatedBlocks = subBlocks.map((b, i) =>
          i === subBlocks.length - 1 ? { ...b, end_frame: mid } : b,
        )
        newStart = mid + 1
      }
      const defaultSpan = updatedBlocks.length === 0
        ? Math.max(1, Math.floor(span / 2))
        : Math.max(1, Math.floor(span / (updatedBlocks.length + 1)))
      const newEnd = Math.min(span - 1, newStart + defaultSpan - 1)
      const newBlock: SubBlock = {
        id: String(updatedBlocks.length),
        item: { ...newItem, start_frame: newStart, end_frame: newEnd },
        start_frame: newStart,
        end_frame: newEnd,
      }
      const updated = [...updatedBlocks, newBlock]
      setSubBlocks(updated)
      onContentChange({ images: updated.map((b) => ({ ...b.item, start_frame: b.start_frame, end_frame: b.end_frame })) })
    } else {
      // Update existing block's image
      const updated = subBlocks.map((b) =>
        b.id === editingBlockId
          ? { ...b, item: { ...newItem, start_frame: b.start_frame, end_frame: b.end_frame } }
          : b,
      )
      setSubBlocks(updated)
      onContentChange({ images: updated.map((b) => ({ ...b.item, start_frame: b.start_frame, end_frame: b.end_frame })) })
    }
    setPopoverOpen(false)
  }

  function updateSubBlockPosition(id: string, newStart: number, newEnd: number) {
    const updated = subBlocks.map((b) =>
      b.id === id ? { ...b, start_frame: newStart, end_frame: newEnd } : b,
    )
    setSubBlocks(updated)
    onContentChange({ images: updated.map((b) => ({ ...b.item, start_frame: b.start_frame, end_frame: b.end_frame })) })
  }

  function handleRemove(id: string) {
    const updated = subBlocks.filter((b) => b.id !== id)
    setSubBlocks(updated)
    onContentChange({ images: updated.map((b) => ({ ...b.item, start_frame: b.start_frame, end_frame: b.end_frame })) })
  }

  function distributeEvenlySubBlocks() {
    if (subBlocks.length === 0) return
    const base = Math.floor(span / subBlocks.length)
    const remainder = span % subBlocks.length
    const updated = subBlocks.map((b, i) => {
      const extra = i < remainder ? 1 : 0
      const start = i * base + Math.min(i, remainder)
      const end = i === subBlocks.length - 1 ? span - 1 : start + base + extra - 1
      return { ...b, start_frame: start, end_frame: end }
    })
    setSubBlocks(updated)
    onContentChange({ images: updated.map((b) => ({ ...b.item, start_frame: b.start_frame, end_frame: b.end_frame })) })
  }

  function handleSubTrackDragOver(e: React.DragEvent<HTMLDivElement>) {
    if (!e.dataTransfer.types.includes('Files')) return
    e.preventDefault()
    e.dataTransfer.dropEffect = 'copy'
  }

  async function handleSubTrackDrop(e: React.DragEvent<HTMLDivElement>) {
    if (!e.dataTransfer.types.includes('Files')) return
    e.preventDefault()

    const imageExts = new Set(['.png', '.jpg', '.jpeg', '.webp', '.gif', '.bmp', '.tiff', '.tif'])
    const files = Array.from(e.dataTransfer.files).filter((file) => {
      if (file.type.startsWith('image/')) return true
      const ext = `.${file.name.split('.').pop()?.toLowerCase() ?? ''}`
      return imageExts.has(ext)
    })
    if (files.length === 0) return

    const rect = e.currentTarget.getBoundingClientRect()
    const clientX = Number.isFinite(e.clientX) ? e.clientX : rect.left + areaWidth / 2
    const x = (clientX - rect.left) / canvasScale
    const dropFrame = Math.max(0, Math.min(span - 1, Math.round((x / areaWidth) * (span - 1))))
    const defaultSpan = Math.max(1, Math.floor(span / Math.max(subBlocks.length + files.length, 1)))

    const uploaded: string[] = []
    for (const file of files) {
      try {
        uploaded.push(await uploadImageFile(file))
      } catch (err) {
        console.error('[EditPanel] upload failed:', err)
      }
    }
    if (uploaded.length === 0) return

    let cursor = dropFrame
    const newBlocks: SubBlock[] = []
    for (const path of uploaded) {
      if (cursor > span - 1) break
      const end = Math.min(span - 1, cursor + defaultSpan - 1)
      const item: ImageItem = {
        source_type: 'input',
        file_path: path,
        file_name: path.split('/').pop() ?? path,
        start_frame: cursor,
        end_frame: end,
      }
      newBlocks.push({
        id: String(subBlocks.length + newBlocks.length),
        item,
        start_frame: cursor,
        end_frame: end,
      })
      cursor = end + 1
    }

    if (newBlocks.length === 0) return
    const updated = [...subBlocks, ...newBlocks]
    setSubBlocks(updated)
    onContentChange({ images: updated.map((b) => ({ ...b.item, start_frame: b.start_frame, end_frame: b.end_frame })) })
  }

  return (
    <div
      data-edit-panel=""
      className="bg-card border border-border rounded shadow-lg overflow-hidden"
      style={{ width: areaWidth }}
    >
      {promptMode === 'combined' ? (
        <div className="p-2 border-border">
          <MaintainCombinedEditor
            key={`combined-${segment.id}-${promptMode}`}
            allSegments={allSegments}
            totalFrames={totalFrames}
            trackColor={trackColor}
            onAllSegmentsChange={onAllSegmentsChange}
          />
        </div>
      ) : (
        <>
        {/* Prompt area */}
        <div className="p-2 border-b border-border">
          <Textarea
            className="w-full h-16 resize-none text-[10px] border-0 shadow-none focus-visible:ring-0 p-0 bg-transparent"
            placeholder={t('maintainTrack.promptPlaceholder')}
            value={segment.content.text}
            onChange={(e) => onContentChange({ text: e.target.value })}
          />
        </div>

      {/* Sub-track ruler */}
      <TimelineRuler
        totalFrames={span}
        frameRate={frameRate}
        displayFormat={displayFormat}
        width={areaWidth}
        canvasScale={canvasScale}
        showLabel={false}
      />

      {/* Sub-track area */}
      <div
        ref={trackAreaRef}
        className="relative w-full select-none bg-muted/30"
        style={{ height: 48 }}
        onDragOver={handleSubTrackDragOver}
        onDrop={handleSubTrackDrop}
      >
        {subBlocks.length === 0 ? (
          <ContextMenu>
            <ContextMenuTrigger asChild>
              <button
                type="button"
                className="absolute inset-0 flex items-center justify-center gap-1 text-[10px] text-muted-foreground hover:text-foreground cursor-pointer"
                onClick={(e) => openPopover(e, null, '', 'inputs')}
              >
                <Plus className="w-2.5 h-2.5" />
                <span>{t('maintainTrack.addImage')}</span>
              </button>
            </ContextMenuTrigger>
            <ContextMenuContent>
              <ContextMenuItem onClick={() => openPopover(null, null, '', 'inputs')}>
                {t('imageTrack.contextAdd')}
              </ContextMenuItem>
            </ContextMenuContent>
          </ContextMenu>
        ) : (
          <>
            {subBlocks.map((block) => {
              return (
                <ContextMenu key={block.id}>
                  <ContextMenuTrigger asChild>
                    <SubImageBlock
                      block={block}
                      span={span}
                      areaWidth={areaWidth}
                      frameRate={frameRate}
                      displayFormat={displayFormat}
                      trackColor={trackColor}
                      selected={selectedBlockId === block.id}
                      onSelect={setSelectedBlockId}
                      onMoveEnd={updateSubBlockPosition}
                      onResizeEnd={updateSubBlockPosition}
                      onEdit={() =>
                        openPopover(
                          null,
                          block.id,
                          block.item.file_path ?? block.item.url ?? '',
                          block.item.source_type === 'url' ? 'url' : 'inputs',
                        )
                      }
                      onRemove={() => handleRemove(block.id)}
                    />
                  </ContextMenuTrigger>
                  <ContextMenuContent>
                    <ContextMenuItem
                      onClick={() => {
                        setSelectedBlockId(block.id)
                        openPopover(
                          null,
                          block.id,
                          block.item.file_path ?? block.item.url ?? '',
                          block.item.source_type === 'url' ? 'url' : 'inputs',
                        )
                      }}
                    >
                      {t('imageTrack.contextReselect')}
                    </ContextMenuItem>
                    <ContextMenuItem onClick={() => openPopover(null, null, '', 'inputs')}>
                      {t('imageTrack.contextAdd')}
                    </ContextMenuItem>
                    <ContextMenuItem
                      disabled={subBlocks.length < 1}
                      onClick={distributeEvenlySubBlocks}
                    >
                      {t('imageTrack.contextDistribute')}
                    </ContextMenuItem>
                    <ContextMenuSeparator />
                    <ContextMenuItem
                      className="text-destructive focus:text-destructive"
                      onClick={() => handleRemove(block.id)}
                    >
                      {t('imageTrack.contextDelete')}
                    </ContextMenuItem>
                  </ContextMenuContent>
                </ContextMenu>
              )
            })}

            {/* Add button */}
            <button
              type="button"
              className="absolute right-1 top-1/2 w-5 h-5 flex items-center justify-center rounded border border-dashed border-border bg-card transition-colors text-muted-foreground z-30 cursor-pointer"
              onClick={(e) => { e.stopPropagation(); openPopover(e, null, '', 'inputs') }}
              title={t('maintainTrack.addImage')}
            >
              <Plus className="w-3 h-3" />
            </button>
          </>
        )}

        {/* Popover for image selector */}
        <Popover open={popoverOpen} onOpenChange={(open) => { if (!open) setPopoverOpen(false) }}>
          <PopoverAnchor asChild>
            <span
              className="absolute w-0 h-0 pointer-events-none"
              style={{ left: anchorPos.x, top: anchorPos.y }}
            />
          </PopoverAnchor>
          <PopoverContent
            className="p-0 w-auto"
            side="bottom"
            align="start"
            onOpenAutoFocus={(e) => e.preventDefault()}
            onInteractOutside={(e) => {
              if (justOpenedPopoverRef.current) {
                justOpenedPopoverRef.current = false
                e.preventDefault()
              }
            }}
          >
            <MediaSelector
              value={selectorValue}
              onChange={handleSelectorChange}
              mediaType="image"
              defaultTab={popoverTab}
              slotItems={slotItems}
            />
          </PopoverContent>
        </Popover>
      </div>
      </>)}

      {/* Bottom toolbar area */}
      <div
        className="flex items-center space-between gap-2 p-2 border-t"
      >
        {/* Left — prompt mode tabs */}
        <Tabs
          value={promptMode}
          onValueChange={(v) => setPromptMode(v as 'combined' | 'individual')}
        >
          <TabsList className="h-6">
            <TabsTrigger value="individual" className="text-[10px] h-5 px-2">
              {t('maintainTrack.tabIndividual')}
            </TabsTrigger>
            <TabsTrigger value="combined" className="text-[10px] h-5 px-2">
              {t('maintainTrack.tabCombined')}
            </TabsTrigger>
          </TabsList>
        </Tabs>
        {/* Right — maintain track type */}
        {
          promptMode === 'combined' ? (<></>) : (
            <div className="flex items-center gap-2 ml-auto">
              <Select
                value={segment.content.type}
                onValueChange={(v) => onContentChange({ type: v as MaintainType })}
              >
                <SelectTrigger className="h-6 w-28 text-[10px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="flf">{t('maintainTrack.flf')}</SelectItem>
                  <SelectItem value="fmlf">{t('maintainTrack.fmlf')}</SelectItem>
                  <SelectItem value="ref">{t('maintainTrack.ref')}</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )
        }
      </div>

    </div>
  )
}
