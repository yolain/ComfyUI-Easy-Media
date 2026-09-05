import { useEffect, useMemo, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { AudioLines, Clapperboard, ImageIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { CUSTOM_NODE_CLASS } from '@/lib/constants'
import { useT } from '@/lib/i18n'
import { cn } from '@/lib/utils'

export type PromptReferenceType = 'image' | 'audio' | 'video'

export interface PromptReferenceResource {
  id: string
  type: PromptReferenceType
  index: number
  label: string
  detail: string
  token: string
  color: string
  thumbnailUrl?: string
}

interface PromptContentEditorProps {
  value: string
  onChange: (value: string) => void
  ariaLabel: string
  placeholder: string
  resources?: PromptReferenceResource[]
  mentionsEnabled?: boolean
  interactivePlaceholder?: boolean
  highlightSystemVariables?: boolean
  highlightPromptSemantics?: boolean
  highlightPipes?: boolean
  className?: string
  testId?: string
}

interface MentionState {
  start: number
  end: number
  query: string
  left: number
  top: number
  activeIndex: number
}

const H3_LANGUAGE_LABELS = new Set([
  'arabic',
  'chinese',
  'english',
  'french',
  'german',
  'hindi',
  'indonesian',
  'italian',
  'japanese',
  'korean',
  'malay',
  'portuguese',
  'russian',
  'spanish',
  'thai',
  'turkish',
  'vietnamese',
])
const TOKEN_PATTERN = /<[^<>\n]*>|@(?:图片|音频|视频|Picture|Image|Audio|Video)\s*\d+|\{[^{}\n]*\}|\[[^\[\]\n]*\]|\(S\d+(?:\s*,\s*S?\d+)*\)|[|｜]/gi
const REFERENCE_CHIP_CLASS = 'prompt-reference-chip inline-flex h-[1.6em] max-w-full items-center gap-1 rounded-md border border-border bg-background px-1.5 py-0.5 mx-1 align-middle font-semibold leading-none shadow-sm'

function isH3PromptSemantic(token: string): boolean {
  if (/^<Subject\s+\d+>$/i.test(token)) return true
  if (/^<\/?d>$/i.test(token) || /^<(?:scenetrans|cutoff)>$/i.test(token)) return true
  if (/^\[Shot\s+\d+\]$/i.test(token) || /^\(S\d+(?:\s*,\s*S?\d+)*\)$/i.test(token)) return true
  const language = token.match(/^\[([A-Za-z]+)\]$/)?.[1].toLowerCase()
  return language !== undefined && H3_LANGUAGE_LABELS.has(language)
}

function isH3ThemeSemantic(token: string): boolean {
  return /^\[Shot\s+\d+\]$/i.test(token)
}

function normalizedReference(token: string): { type: PromptReferenceType; index: number } | null {
  const match = token.match(/^@?(图片|音频|视频|Picture|Image|Audio|Video)\s*(\d+)>?$/i)
    ?? token.match(/^<(Picture|Audio|Video)\s+(\d+)>$/i)
  if (!match) return null
  const rawType = match[1].toLowerCase()
  const type: PromptReferenceType = rawType === '图片' || rawType === 'picture' || rawType === 'image'
    ? 'image'
    : rawType === '音频' || rawType === 'audio'
      ? 'audio'
      : 'video'
  return { type, index: Number(match[2]) }
}

function serializePromptContent(container: ParentNode): string {
  let value = ''
  const append = (text: string) => {
    value += text.replaceAll('\u200B', '')
  }
  const visit = (node: Node) => {
    if (node.nodeType === Node.TEXT_NODE) {
      append(node.textContent ?? '')
      return
    }
    if (!(node instanceof HTMLElement)) return
    if (node.dataset.promptReferenceToken !== undefined) {
      append(node.dataset.promptReferenceToken)
      return
    }
    if (node.tagName === 'BR') {
      append('\n')
      return
    }
    const block = node.tagName === 'DIV' || node.tagName === 'P'
    if (block && value && !value.endsWith('\n')) append('\n')
    node.childNodes.forEach(visit)
  }
  container.childNodes.forEach(visit)
  return value
}

function serializeEditor(editor: HTMLElement): string {
  return serializePromptContent(editor)
}

function normalizeSerializedEditorValue(value: string): string {
  return /^\n*$/.test(value) ? '' : value
}

function caretOffset(editor: HTMLElement): number {
  const selection = window.getSelection()
  if (!selection?.rangeCount) return serializeEditor(editor).length
  const range = selection.getRangeAt(0)
  if (!editor.contains(range.endContainer)) return serializeEditor(editor).length
  const before = range.cloneRange()
  before.selectNodeContents(editor)
  before.setEnd(range.endContainer, range.endOffset)
  return serializePromptContent(before.cloneContents()).length
}

function valueDecorationKey(
  value: string,
  highlightSystemVariables: boolean,
  highlightPromptSemantics: boolean,
  highlightPipes: boolean,
): string {
  const decorations: string[] = []
  TOKEN_PATTERN.lastIndex = 0
  for (const match of value.matchAll(TOKEN_PATTERN)) {
    const token = match[0]
    const reference = normalizedReference(token)
    if (reference && !highlightSystemVariables) {
      decorations.push(`reference:${reference.type}:${reference.index}`)
    } else if (highlightPromptSemantics && isH3PromptSemantic(token)) {
      decorations.push(`semantic:${token.toLowerCase()}`)
    } else if (highlightSystemVariables && (token.startsWith('{') || token.startsWith('<'))) {
      decorations.push('system')
    } else if (highlightPipes && (token === '|' || token === '｜')) {
      decorations.push('pipe')
    }
  }
  return decorations.join('|')
}

function editorDecorationKey(editor: HTMLElement): string {
  return Array.from(editor.querySelectorAll<HTMLElement>(
    '[data-prompt-reference-token], [data-prompt-semantic-token], [data-system-prompt-variable], [data-pipe]',
  )).map((element) => {
    const token = element.dataset.promptReferenceToken
    if (token !== undefined) {
      const reference = normalizedReference(token)
      return reference ? `reference:${reference.type}:${reference.index}` : 'reference:unknown'
    }
    if (element.dataset.systemPromptVariable !== undefined) return 'system'
    if (element.dataset.promptSemanticToken !== undefined) {
      return `semantic:${element.dataset.promptSemanticToken.toLowerCase()}`
    }
    return 'pipe'
  }).join('|')
}

function setCaretOffset(editor: HTMLElement, targetOffset: number) {
  const selection = window.getSelection()
  if (!selection) return
  let offset = Math.max(0, targetOffset)
  for (const node of editor.childNodes) {
    const element = node instanceof HTMLElement ? node : null
    const referenceToken = element?.dataset.promptReferenceToken
    const length = referenceToken !== undefined
      ? referenceToken.length
      : element?.tagName === 'BR'
        ? 1
        : node.textContent?.replaceAll('\u200B', '').length ?? 0
    if (offset <= length) {
      const range = document.createRange()
      if (referenceToken !== undefined) {
        offset === 0 ? range.setStartBefore(node) : range.setStartAfter(node)
      } else if (element?.tagName === 'BR') {
        offset === 0 ? range.setStartBefore(node) : range.setStartAfter(node)
      } else if (node.nodeType === Node.TEXT_NODE) {
        range.setStart(node, Math.min(offset, node.textContent?.length ?? 0))
      } else {
        const textNode = document.createTreeWalker(node, NodeFilter.SHOW_TEXT).nextNode()
        if (textNode) range.setStart(textNode, Math.min(offset, textNode.textContent?.length ?? 0))
        else range.setStartAfter(node)
      }
      range.collapse(true)
      selection.removeAllRanges()
      selection.addRange(range)
      return
    }
    offset -= length
  }
  const range = document.createRange()
  range.selectNodeContents(editor)
  range.collapse(false)
  selection.removeAllRanges()
  selection.addRange(range)
}

function referenceDeletionRange(
  value: string,
  offset: number,
  key: 'Backspace' | 'Delete',
): { start: number; end: number } | null {
  TOKEN_PATTERN.lastIndex = 0
  for (const match of value.matchAll(TOKEN_PATTERN)) {
    if (!normalizedReference(match[0])) continue
    const start = match.index ?? 0
    const end = start + match[0].length
    if (key === 'Backspace' && offset > start && offset <= end) return { start, end }
    if (key === 'Delete' && offset >= start && offset < end) return { start, end }
  }
  return null
}

function appendText(container: HTMLElement | DocumentFragment, text: string) {
  text.split('\n').forEach((part, index) => {
    if (index > 0) container.append(document.createElement('br'))
    if (part) container.append(document.createTextNode(part))
  })
}

function appendDialogueText(container: HTMLElement, text: string) {
  text.split('\n').forEach((part, index) => {
    if (index > 0) container.append(document.createElement('br'))
    if (!part) return
    const dialogue = document.createElement('span')
    dialogue.dataset.promptDialogueContent = 'true'
    dialogue.className = 'text-highlight'
    dialogue.textContent = part
    container.append(dialogue)
  })
}

function insertPlainTextAtSelection(editor: HTMLElement, text: string): boolean {
  const selection = window.getSelection()
  if (!selection?.rangeCount) return false
  const range = selection.getRangeAt(0)
  if (!editor.contains(range.commonAncestorContainer)) return false
  range.deleteContents()
  const fragment = document.createDocumentFragment()
  appendText(fragment, text)
  const caretMarker = document.createTextNode('\u200B')
  fragment.append(caretMarker)
  range.insertNode(fragment)
  const caret = document.createRange()
  caret.setStart(caretMarker, caretMarker.textContent?.length ?? 0)
  caret.collapse(true)
  selection.removeAllRanges()
  selection.addRange(caret)
  return true
}

function selectedPromptText(editor: HTMLElement): string | null {
  const selection = window.getSelection()
  if (!selection?.rangeCount || selection.isCollapsed) return null
  const range = selection.getRangeAt(0)
  if (!editor.contains(range.commonAncestorContainer)) return null
  return serializePromptContent(range.cloneContents())
}

function appendReferenceChip(
  container: HTMLElement,
  token: string,
  resource: PromptReferenceResource | undefined,
) {
  const parsed = normalizedReference(token)
  const chip = document.createElement('span')
  chip.contentEditable = 'false'
  chip.dataset.promptReferenceToken = token
  chip.dataset.promptReferenceType = parsed?.type ?? ''
  chip.className = REFERENCE_CHIP_CLASS
  chip.style.color = resource?.color || 'var(--muted-foreground)'

  if (resource?.type === 'image' && resource.thumbnailUrl) {
    const image = document.createElement('img')
    image.src = resource.thumbnailUrl
    image.alt = ''
    image.draggable = false
    image.className = 'h-4 w-4 shrink-0 rounded-sm object-cover align-middle'
    chip.append(image)
  } else {
    const icon = document.createElement('span')
    icon.className = 'prompt-reference-icon inline-flex h-4 w-4 shrink-0 items-center justify-center'
    icon.dataset.referenceIcon = resource?.type ?? parsed?.type ?? 'image'
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    svg.setAttribute('viewBox', '0 0 24 24')
    svg.setAttribute('fill', 'none')
    svg.setAttribute('stroke', 'currentColor')
    svg.setAttribute('stroke-width', '2')
    svg.setAttribute('stroke-linecap', 'round')
    svg.setAttribute('stroke-linejoin', 'round')
    svg.setAttribute('aria-hidden', 'true')
    svg.classList.add('lucide', 'h-4', 'w-4')
    const type = resource?.type ?? parsed?.type ?? 'image'
    const paths = type === 'audio'
      ? ['M2 10v3', 'M6 6v11', 'M10 3v18', 'M14 8v7', 'M18 5v13', 'M22 10v3']
      : type === 'video'
        ? ['m16 13 5.223-3.482a.5.5 0 0 1 .777.416v8.132a.5.5 0 0 1-.777.416L16 15', 'M2 6h14v12H2z', 'm2 6 3-4', 'm7 6 3-4', 'm12 6 3-4']
        : ['M14.5 4h-9A2.5 2.5 0 0 0 3 6.5v11A2.5 2.5 0 0 0 5.5 20h13a2.5 2.5 0 0 0 2.5-2.5v-8', 'm3 16 5-5c.928-.893 2.072-.893 3 0l4 4', 'm14 14 1-1c.928-.893 2.072-.893 3 0l3 3', 'M14 7h.01']
    paths.forEach((pathData) => {
      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path')
      path.setAttribute('d', pathData)
      svg.append(path)
    })
    icon.append(svg)
    chip.append(icon)
  }

  const label = document.createElement('span')
  label.className = 'inline-flex h-full items-center leading-none'
  label.textContent = token
  chip.append(label)
  chip.title = resource?.detail ?? token
  container.append(chip)
}

function renderValue(
  editor: HTMLElement,
  value: string,
  resources: PromptReferenceResource[],
  highlightSystemVariables: boolean,
  highlightPromptSemantics: boolean,
  highlightPipes: boolean,
) {
  editor.textContent = ''
  let cursor = 0
  let inDialogue = false
  TOKEN_PATTERN.lastIndex = 0
  for (const match of value.matchAll(TOKEN_PATTERN)) {
    const index = match.index ?? 0
    const precedingText = value.slice(cursor, index)
    if (inDialogue) appendDialogueText(editor, precedingText)
    else appendText(editor, precedingText)
    const token = match[0]
    const opensDialogue = /^<d>$/i.test(token)
    const closesDialogue = /^<\/d>$/i.test(token)
    if (closesDialogue) inDialogue = false
    const reference = normalizedReference(token)
    if (reference && !highlightSystemVariables) {
      const resource = resources.find((item) => item.type === reference.type && item.index === reference.index)
      appendReferenceChip(editor, token, resource)
    } else if (highlightPromptSemantics && isH3PromptSemantic(token)) {
      const semantic = document.createElement('span')
      semantic.dataset.promptSemanticToken = token
      semantic.className = isH3ThemeSemantic(token) ? 'text-highlight' : 'text-prompt-semantic'
      semantic.textContent = token
      editor.append(semantic)
    } else if (highlightSystemVariables && (token.startsWith('{') || token.startsWith('<'))) {
      const variable = document.createElement('span')
      variable.dataset.systemPromptVariable = 'true'
      variable.className = 'text-highlight'
      variable.textContent = token
      editor.append(variable)
    } else if (highlightPipes && (token === '|' || token === '｜')) {
      const pipe = document.createElement('span')
      pipe.dataset.pipe = 'true'
      pipe.className = 'text-highlight'
      pipe.textContent = '|'
      editor.append(pipe)
    } else if (inDialogue) {
      appendDialogueText(editor, token)
    } else {
      appendText(editor, token)
    }
    if (highlightPromptSemantics && opensDialogue) inDialogue = true
    cursor = index + token.length
  }
  const trailingText = value.slice(cursor)
  if (inDialogue) appendDialogueText(editor, trailingText)
  else appendText(editor, trailingText)
}

function ReferenceIcon({ type }: Readonly<{ type: PromptReferenceType }>) {
  if (type === 'audio') return <AudioLines className="h-4 w-4" />
  if (type === 'video') return <Clapperboard className="h-4 w-4" />
  return <ImageIcon className="h-4 w-4" />
}

export function PromptContentEditor({
  value,
  onChange,
  ariaLabel,
  placeholder,
  resources = [],
  mentionsEnabled = false,
  interactivePlaceholder = false,
  highlightSystemVariables = false,
  highlightPromptSemantics = false,
  highlightPipes = false,
  className,
  testId,
}: Readonly<PromptContentEditorProps>) {
  const t = useT()
  const wrapperRef = useRef<HTMLDivElement>(null)
  const editorRef = useRef<HTMLDivElement>(null)
  const composingRef = useRef(false)
  const lastRenderKeyRef = useRef('')
  const [mention, setMention] = useState<MentionState | null>(null)
  const [editorEmpty, setEditorEmpty] = useState(value.length === 0)
  const filteredResources = useMemo(() => {
    if (!mention) return resources
    const query = mention.query.toLocaleLowerCase()
    return resources.filter((resource) => (
      !query || `${resource.label} ${resource.detail}`.toLocaleLowerCase().includes(query)
    ))
  }, [mention, resources])
  const renderKey = useMemo(() => JSON.stringify({
    value,
    highlightSystemVariables,
    highlightPromptSemantics,
    highlightPipes,
    resources: resources.map((resource) => [
      resource.id,
      resource.index,
      resource.label,
      resource.detail,
      resource.token,
      resource.color,
      resource.thumbnailUrl,
    ]),
  }), [highlightPipes, highlightPromptSemantics, highlightSystemVariables, resources, value])

  useEffect(() => {
    setEditorEmpty(value.length === 0)
    const editor = editorRef.current
    if (!editor || lastRenderKeyRef.current === renderKey) return
    const focused = document.activeElement === editor
    const offset = focused ? caretOffset(editor) : 0
    renderValue(editor, value, resources, highlightSystemVariables, highlightPromptSemantics, highlightPipes)
    if (focused) setCaretOffset(editor, Math.min(offset, value.length))
    lastRenderKeyRef.current = renderKey
  }, [highlightPipes, highlightPromptSemantics, highlightSystemVariables, renderKey, resources, value])

  function syncMention(valueText: string, offset: number) {
    if (!mentionsEnabled) {
      setMention(null)
      return
    }
    const beforeCaret = valueText.slice(0, offset)
    const match = beforeCaret.match(/@([^\s@|<>]*)$/)
    if (!match) {
      setMention(null)
      return
    }
    if (normalizedReference(match[0])) {
      setMention(null)
      return
    }
    const wrapper = wrapperRef.current
    const selection = window.getSelection()
    const range = selection?.rangeCount ? selection.getRangeAt(0) : null
    const caretRect = range?.getBoundingClientRect?.()
    const wrapperRect = wrapper?.getBoundingClientRect()
    const caretLeft = caretRect?.left ?? wrapperRect?.left ?? 0
    const caretTop = caretRect?.top ?? wrapperRect?.top ?? 0
    const caretBottom = caretRect?.bottom ?? wrapperRect?.bottom ?? 0
    const left = Math.min(Math.max(8, caretLeft), Math.max(8, window.innerWidth - 264))
    const top = caretBottom + 214 <= window.innerHeight
      ? caretBottom + 6
      : Math.max(8, caretTop - 214)
    const start = offset - match[0].length
    setMention((current) => ({
      start,
      end: offset,
      query: match[1],
      left,
      top,
      activeIndex: current?.start === start ? current.activeIndex : 0,
    }))
  }

  function openPlaceholderMention(anchor: HTMLElement) {
    if (!mentionsEnabled || !editorEmpty) return
    const rect = anchor.getBoundingClientRect()
    const left = Math.min(Math.max(8, rect.left), Math.max(8, window.innerWidth - 264))
    const top = rect.bottom + 214 <= window.innerHeight
      ? rect.bottom + 6
      : Math.max(8, rect.top - 214)
    setMention({ start: 0, end: 0, query: '', left, top, activeIndex: 0 })
  }

  function commitEditorChange() {
    const editor = editorRef.current
    if (!editor) return
    const nextValue = normalizeSerializedEditorValue(serializeEditor(editor))
    const offset = nextValue.length === 0 ? 0 : caretOffset(editor)
    const needsDecorationRender = nextValue.length === 0 || editorDecorationKey(editor) !== valueDecorationKey(
      nextValue,
      highlightSystemVariables,
      highlightPromptSemantics,
      highlightPipes,
    )
    if (needsDecorationRender) {
      const scrollTop = editor.scrollTop
      const scrollLeft = editor.scrollLeft
      renderValue(editor, nextValue, resources, highlightSystemVariables, highlightPromptSemantics, highlightPipes)
      setCaretOffset(editor, offset)
      editor.scrollTop = scrollTop
      editor.scrollLeft = scrollLeft
    }
    setEditorEmpty(nextValue.length === 0)
    lastRenderKeyRef.current = JSON.stringify({
      value: nextValue,
      highlightSystemVariables,
      highlightPromptSemantics,
      highlightPipes,
      resources: resources.map((resource) => [
        resource.id,
        resource.index,
        resource.label,
        resource.detail,
        resource.token,
        resource.color,
        resource.thumbnailUrl,
      ]),
    })
    onChange(nextValue)
    syncMention(nextValue, offset)
  }

  function chooseResource(resource: PromptReferenceResource) {
    const editor = editorRef.current
    if (!editor || !mention) return
    const currentValue = serializeEditor(editor)
    const nextValue = `${currentValue.slice(0, mention.start)}${resource.token}${currentValue.slice(mention.end)}`
    renderValue(editor, nextValue, resources, highlightSystemVariables, highlightPromptSemantics, highlightPipes)
    setEditorEmpty(false)
    setCaretOffset(editor, mention.start + resource.token.length)
    lastRenderKeyRef.current = ''
    onChange(nextValue)
    setMention(null)
    editor.focus()
  }

  function deleteAdjacentReference(key: 'Backspace' | 'Delete'): boolean {
    const editor = editorRef.current
    if (!editor) return false
    const currentValue = serializeEditor(editor)
    const deletion = referenceDeletionRange(currentValue, caretOffset(editor), key)
    if (!deletion) return false
    const nextValue = `${currentValue.slice(0, deletion.start)}${currentValue.slice(deletion.end)}`
    renderValue(editor, nextValue, resources, highlightSystemVariables, highlightPromptSemantics, highlightPipes)
    setEditorEmpty(nextValue.length === 0)
    setCaretOffset(editor, deletion.start)
    lastRenderKeyRef.current = ''
    setMention(null)
    onChange(nextValue)
    return true
  }

  return (
    <div
      ref={wrapperRef}
      className="relative h-full min-h-0 max-h-full overflow-hidden [contain:size_layout_paint]"
    >
      <div
        ref={editorRef}
        role="textbox"
        aria-label={ariaLabel}
        aria-multiline="true"
        data-testid={testId}
        data-placeholder={placeholder}
        contentEditable
        suppressContentEditableWarning
        className={cn(
          'prompt-content-editor h-full min-h-0 max-h-full overflow-auto whitespace-pre-wrap wrap-break-word rounded-md px-3 py-2 text-[10px] leading-normal text-foreground caret-foreground outline-none focus-visible:ring-1 focus-visible:ring-ring',
          !interactivePlaceholder && 'empty:before:pointer-events-none empty:before:text-muted-foreground empty:before:content-[attr(data-placeholder)]',
          className,
        )}
        onInput={() => {
          if (!composingRef.current) commitEditorChange()
        }}
        onCompositionStart={() => {
          composingRef.current = true
        }}
        onCompositionEnd={() => {
          composingRef.current = false
          commitEditorChange()
        }}
        onPaste={(event) => {
          event.preventDefault()
          event.stopPropagation()
          event.nativeEvent.stopImmediatePropagation?.()
          const editor = editorRef.current
          const text = event.clipboardData.getData('text/plain')
          if (editor && insertPlainTextAtSelection(editor, text)) commitEditorChange()
        }}
        onCopy={(event) => {
          event.stopPropagation()
          const editor = editorRef.current
          const text = editor ? selectedPromptText(editor) : null
          if (text === null) return
          event.preventDefault()
          event.clipboardData.setData('text/plain', text)
        }}
        onKeyDown={(event) => {
          if ((event.key === 'Backspace' || event.key === 'Delete') && deleteAdjacentReference(event.key)) {
            event.preventDefault()
            return
          }
          if (mention) {
            if (event.key === 'Escape') {
              event.preventDefault()
              setMention(null)
            } else if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
              event.preventDefault()
              setMention((current) => current && filteredResources.length > 0 ? {
                ...current,
                activeIndex: (
                  current.activeIndex + (event.key === 'ArrowDown' ? 1 : -1) + filteredResources.length
                ) % filteredResources.length,
              } : current)
            } else if (event.key === 'Enter' && filteredResources.length > 0) {
              event.preventDefault()
              chooseResource(filteredResources[Math.min(mention.activeIndex, filteredResources.length - 1)])
            }
          }
          if ((event.ctrlKey || event.metaKey) && ['a', 'c', 'v', 'x'].includes(event.key.toLowerCase())) {
            event.stopPropagation()
          }
        }}
        onClick={() => {
          setMention(null)
        }}
        onBlur={(event) => {
          if (wrapperRef.current?.contains(event.relatedTarget as Node | null)) return
          setMention(null)
        }}
      />

      {interactivePlaceholder && editorEmpty && (() => {
        const triggerIndex = placeholder.indexOf('@')
        const beforeTrigger = triggerIndex >= 0 ? placeholder.slice(0, triggerIndex) : placeholder
        const afterTrigger = triggerIndex >= 0 ? placeholder.slice(triggerIndex + 1) : ''
        return (
          <div className="pointer-events-none absolute inset-0 overflow-hidden whitespace-pre-wrap wrap-break-word px-3 py-2 text-[10px] leading-normal text-muted-foreground">
            {beforeTrigger}
            {triggerIndex >= 0 && (
              <Button
                type="button"
                variant="ghost"
                aria-label={t('multitrack.openReferenceResources')}
                className={cn(REFERENCE_CHIP_CLASS, 'pointer-events-auto min-w-0 text-highlight text-[10px]')}
                onPointerDown={(event) => event.preventDefault()}
                onClick={(event) => openPlaceholderMention(event.currentTarget)}
              >
                @
              </Button>
            )}
            {afterTrigger}
          </div>
        )
      })()}

      {mention && createPortal((
        <div className={CUSTOM_NODE_CLASS}>
          <div
            role="listbox"
            aria-label={t('multitrack.referenceResources')}
            className={cn(
              'fixed z-50 flex w-64 flex-col overflow-y-auto rounded-md border border-border bg-popover p-1 text-popover-foreground shadow-md',
              filteredResources.length === 0 ? 'h-64' : 'max-h-52',
            )}
            style={{ left: mention.left, top: mention.top }}
          >
            <div className="shrink-0 px-2 py-1.5 text-[10px] font-semibold text-muted-foreground">
              {t('multitrack.referenceResources')}
            </div>
            {filteredResources.length === 0 ? (
              <div
                data-testid="reference-resources-empty"
                className="flex min-h-0 flex-1 flex-col items-center justify-center gap-4 px-5 pb-7 text-center text-muted-foreground"
              >
                <div aria-hidden="true" className="relative h-14 w-20 opacity-60">
                  <span className="absolute left-8 -top-2 flex h-10 w-10 -rotate-12 items-center justify-center rounded-md border border-border bg-muted shadow-sm">
                    <ImageIcon className="h-5 w-5" />
                  </span>
                  <span className="absolute -left-2 top-1 flex h-10 w-10 rotate-3 items-center justify-center rounded-md border border-border bg-muted shadow-sm">
                    <AudioLines className="h-5 w-5" />
                  </span>
                  <span className="absolute left-6 top-4 flex h-10 w-10 rotate-6 items-center justify-center rounded-md border border-border bg-muted shadow-sm">
                    <Clapperboard className="h-5 w-5" />
                  </span>
                </div>
                <span className=" text-[10px] leading-relaxed">
                  {t('multitrack.referenceResourcesEmpty')}
                </span>
              </div>
            ) : filteredResources.map((resource, index) => (
              <Button
                key={resource.id}
                type="button"
                variant="ghost"
                role="option"
                aria-selected={index === mention.activeIndex}
                className={cn(
                  'h-auto w-full justify-start gap-2 px-2 py-1.5 text-left',
                  index === mention.activeIndex && 'bg-accent text-accent-foreground',
                )}
                onPointerDown={(event) => event.preventDefault()}
                onPointerMove={() => setMention((current) => current ? { ...current, activeIndex: index } : current)}
                onClick={() => chooseResource(resource)}
              >
                {resource.type === 'image' && resource.thumbnailUrl ? (
                  <img
                    src={resource.thumbnailUrl}
                    alt=""
                    className="h-8 w-8 shrink-0 rounded-sm object-cover"
                  />
                ) : (
                  <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-sm bg-muted">
                    <ReferenceIcon type={resource.type} />
                  </span>
                )}
                <span className="min-w-0">
                  <span className="block truncate text-[11px] font-semibold" style={{ color: resource.color }}>
                    {resource.label}
                  </span>
                  <span className="block truncate text-[9px] text-muted-foreground">{resource.detail}</span>
                </span>
              </Button>
            ))}
          </div>
        </div>
      ), document.body)}
    </div>
  )
}
