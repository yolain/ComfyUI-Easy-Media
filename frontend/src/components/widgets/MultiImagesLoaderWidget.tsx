import { useRef, useState } from 'react'
import { ArrowLeft, CloudUpload, Eye, Plus, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Popover, PopoverAnchor, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { MediaSelector, type MediaTab } from '@/components/widgets/mediaSelector/MediaSelector'
import { useCanvasScale } from '@/hooks/use-canvas-scale'
import { useExpandedImagePreview } from '@/hooks/use-expanded-image-preview'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import { LocaleContext, useT } from '@/lib/i18n'
import { mediaContentToViewUrl } from '@/lib/media-url'
import { createTaskImage, splitSelectedTaskMedia, uploadTaskImageFile } from '@/lib/task-image-utils'
import { invalidateMediaListCache } from '@/stores/media-list-store'
import type { MultiTrackSourceType, MultiTrackTaskImage } from '@/types/multitrack'

export interface ImageData { images: MultiTrackTaskImage[] }

const MAX_IMAGES = 25

function selectorTab(image?: MultiTrackTaskImage): MediaTab {
  if (image?.source_type === 'output') return 'outputs'
  if (image?.source_type === 'local') return 'local'
  if (image?.source_type === 'url') return 'url'
  return 'inputs'
}

function sourceType(path: string, source?: 'input' | 'output' | 'local'): MultiTrackSourceType {
  if (source) return source
  return /^https?:\/\//i.test(path) ? 'url' : 'input'
}

export function MultiImagesLoaderWidget({ value, onChange, app }: Readonly<ReactWidgetProps<ImageData>>) {
  const locale = app?.ui?.settings?.settingsValues?.['Comfy.Locale']
  const data: ImageData = value && Array.isArray(value.images) ? value : { images: [] }
  const images = data.images.slice(0, MAX_IMAGES)
  const [selectorOpen, setSelectorOpen] = useState(false)
  const [replaceId, setReplaceId] = useState<string | null>(null)
  const [preview, setPreview] = useState<MultiTrackTaskImage | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const dragId = useRef<string | null>(null)
  const scale = useCanvasScale(app)
  const isNodeV2 = Boolean(app?.ui?.settings?.settingsValues?.['Comfy.VueNodes.Enabled'])
  const activeImage = images.find((image) => image.id === replaceId)
  const columns = Math.min(5, Math.max(1, Math.ceil(Math.sqrt(images.length + (images.length < MAX_IMAGES ? 1 : 0)))))
  const controlSize = Math.min(32, Math.max(22, 26 / Math.max(scale, 0.5)))

  function commit(next: MultiTrackTaskImage[]) {
    onChange({ images: next.slice(0, MAX_IMAGES) })
  }

  function select(path: string, source?: 'input' | 'output' | 'local') {
    const selected = splitSelectedTaskMedia(path).slice(0, replaceId ? 1 : MAX_IMAGES - images.length)
    if (!selected.length) return
    const incoming = selected.map((item) => createTaskImage(item, sourceType(item, source)))
    commit(replaceId
      ? images.map((image) => image.id === replaceId ? { ...incoming[0], id: replaceId } : image)
      : [...images, ...incoming])
    setReplaceId(null)
    setSelectorOpen(false)
  }

  async function dropFiles(files: FileList | null) {
    if (!files?.length) return
    const selected = Array.from(files).filter((file) => file.type.startsWith('image/')).slice(0, MAX_IMAGES - images.length)
    const results = await Promise.allSettled(selected.map(uploadTaskImageFile))
    const uploaded: MultiTrackTaskImage[] = []
    for (const result of results) {
      if (result.status === 'fulfilled') uploaded.push(result.value)
      else console.error('[MultiImagesLoaderWidget] image upload failed:', result.reason)
    }
    if (uploaded.length) {
      invalidateMediaListCache('inputs')
      commit([...images, ...uploaded])
    }
  }

  return (
    <LocaleContext.Provider value={locale}>
      <MultiImagesLoaderContent
        images={images}
        columns={columns}
        controlSize={controlSize}
        isNodeV2={isNodeV2}
        selectorOpen={selectorOpen}
        setSelectorOpen={setSelectorOpen}
        replaceId={replaceId}
        setReplaceId={setReplaceId}
        activeImage={activeImage}
        preview={preview}
        setPreview={setPreview}
        dragOver={dragOver}
        setDragOver={setDragOver}
        dragId={dragId}
        commit={commit}
        select={select}
        dropFiles={dropFiles}
      />
    </LocaleContext.Provider>
  )
}

interface ContentProps {
  images: MultiTrackTaskImage[]
  columns: number
  controlSize: number
  isNodeV2: boolean
  selectorOpen: boolean
  setSelectorOpen: (open: boolean) => void
  replaceId: string | null
  setReplaceId: (id: string | null) => void
  activeImage?: MultiTrackTaskImage
  preview: MultiTrackTaskImage | null
  setPreview: (image: MultiTrackTaskImage | null) => void
  dragOver: boolean
  setDragOver: (over: boolean) => void
  dragId: React.RefObject<string | null>
  commit: (images: MultiTrackTaskImage[]) => void
  select: (path: string, source?: 'input' | 'output' | 'local') => void
  dropFiles: (files: FileList | null) => Promise<void>
}

function MultiImagesLoaderContent(props: Readonly<ContentProps>) {
  const t = useT()
  const { images, columns, controlSize } = props
  const previewUrl = props.preview && mediaContentToViewUrl({ ...props.preview, source_type: props.preview.source_type ?? 'input' })
  const preview = useExpandedImagePreview(props.preview?.id ?? null, () => props.setPreview(null))

  return (
    <div className={`relative flex h-full min-h-0 w-full flex-col overflow-hidden rounded-md bg-muted/20 text-foreground ${props.isNodeV2 ? 'nodeNew' : ''}`}
      onDragOver={(event) => { event.preventDefault(); props.setDragOver(true) }}
      onDragLeave={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node)) props.setDragOver(false) }}
      onDrop={(event) => {
        event.preventDefault()
        props.setDragOver(false)
        void props.dropFiles(event.dataTransfer.files)
      }}
    >
      <Popover open={props.selectorOpen} onOpenChange={(open) => {
        props.setSelectorOpen(open)
        if (!open) props.setReplaceId(null)
      }}>
        {images.length === 0 ? (
          <div className={`task-image-picker-empty relative flex h-full min-h-0 w-full flex-1 flex-col items-center justify-center gap-2 rounded-md px-4 py-2 text-foreground shadow-sm ${props.dragOver ? 'border border-primary bg-accent/20' : 'bg-muted/20'}`}>
            <CloudUpload className="size-12 shrink-0" />
            <span className="mt-1 text-base font-semibold">{t('multitrack.selectImage')}</span>
            <span className="max-w-full whitespace-normal text-center text-sm text-muted-foreground">{t('multitrack.imageDropHint')}</span>
            <PopoverTrigger asChild>
              <Button type="button" variant="ghost" className="absolute inset-0 h-full w-full cursor-pointer rounded-md p-0 hover:bg-accent/20 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
                aria-label={t('multitrack.selectImage')}
                onClick={() => props.setReplaceId(null)} />
            </PopoverTrigger>
          </div>
        ) : (
          <PopoverAnchor asChild>
          <div className={`task-image-grid grid h-full min-h-0 w-full flex-1 auto-rows-max content-start gap-2 overflow-y-auto rounded-md p-3 transition-colors ${props.dragOver ? 'border border-primary bg-accent/20' : 'bg-muted/20'}`}
            style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
            {images.map((image, index) => {
              const url = mediaContentToViewUrl({ ...image, source_type: image.source_type ?? 'input' })
              const label = image.file_name ?? image.file_path ?? image.local_path ?? image.url ?? String(index + 1)
              return (
                <div key={image.id} className="group relative aspect-square min-w-0 w-full self-start overflow-hidden rounded-md border border-border bg-black"
                  draggable
                  onDragStart={() => { props.dragId.current = image.id }}
                  onDragEnd={() => { props.dragId.current = null }}
                  onDrop={(event) => {
                    if (event.dataTransfer.files.length > 0 || !props.dragId.current) return
                    event.preventDefault()
                    event.stopPropagation()
                    props.setDragOver(false)
                    const sourceId = props.dragId.current
                    if (sourceId === image.id) return
                    const next = [...images]
                    const from = next.findIndex((item) => item.id === sourceId)
                    if (from < 0) return
                    const target = next.findIndex((item) => item.id === image.id)
                    const [moved] = next.splice(from, 1)
                    next.splice(target, 0, moved)
                    props.commit(next)
                  }}>
                  <Button variant="ghost" className="absolute inset-0 h-full w-full rounded-none p-0" aria-label={t('multiImagesLoader.replace', { n: index + 1 })}
                    onClick={() => { props.setReplaceId(image.id); props.setSelectorOpen(true) }}>
                    {url ? <img src={url} alt={label} className="h-full w-full object-contain" draggable={false} /> : <span className="px-1 text-xs text-image-tile-foreground">{label}</span>}
                  </Button>
                  <span className="pointer-events-none absolute bottom-0 left-0 bg-black/50 px-1.5 text-[10px] text-image-tile-foreground">{index + 1}</span>
                  <div className="absolute right-1 top-1 flex gap-1 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100">
                    <Button variant="secondary" size="icon" style={{ width: controlSize, height: controlSize }} aria-label={t('multiImagesLoader.preview')}
                      onClick={() => props.setPreview(image)}><Eye /></Button>
                    <Button variant="destructive" size="icon" style={{ width: controlSize, height: controlSize }} aria-label={t('multiImagesLoader.delete')}
                      onClick={() => props.commit(images.filter((item) => item.id !== image.id))}><Trash2 /></Button>
                  </div>
                </div>
              )
            })}
            {images.length < MAX_IMAGES && <Button variant="outline" className="task-image-grid-add aspect-square h-auto w-full self-start border-dashed text-muted-foreground" aria-label={t('multiImagesLoader.add')}
              onClick={() => { props.setReplaceId(null); props.setSelectorOpen(true) }}><Plus /></Button>}
          </div>
          </PopoverAnchor>
        )}
        <PopoverContent className="w-auto p-0" align="start">
          <MediaSelector key={props.replaceId ?? 'add'} value={props.activeImage?.file_path ?? props.activeImage?.local_path ?? props.activeImage?.url ?? ''}
            mediaType="image" defaultTab={selectorTab(props.activeImage)} allowMultipleSelection={!props.replaceId}
            maxSelectionCount={props.replaceId ? 1 : MAX_IMAGES - images.length} onChange={props.select} />
        </PopoverContent>
      </Popover>
      {images.length > 0 && (
        <div className="flex h-9 shrink-0 items-center justify-between gap-2 border-t border-border bg-background/60 px-3 text-xs">
          <span className="text-muted-foreground">{t('multiImagesLoader.imageCount', { n: images.length })}</span>
          <Button type="button" variant="ghost" size="sm" className="h-7 gap-1 px-2 text-xs text-destructive hover:text-destructive"
            onClick={() => {
              props.setSelectorOpen(false)
              props.setReplaceId(null)
              props.setPreview(null)
              props.commit([])
            }}>
            <Trash2 className="size-3.5" />
            {t('multiImagesLoader.clearAll')}
          </Button>
        </div>
      )}
      {previewUrl && (
        <div data-testid="multi-images-expanded-preview" data-zoom={preview.zoom.toFixed(2)}
          className="absolute inset-0 z-30 flex items-center justify-center overflow-hidden bg-black p-3 pt-12"
          onWheel={preview.onWheel}>
          <Button type="button" variant="secondary" className="absolute left-2 top-2 z-20 h-8 cursor-pointer gap-1.5 px-2.5 text-xs"
            aria-label={t('multitrack.backToPreview')} onClick={() => props.setPreview(null)}>
            <ArrowLeft className="h-4 w-4" />
            <span>{t('multitrack.backToPreview')}</span>
          </Button>
          <span className="pointer-events-none absolute right-2 top-2 z-20 rounded bg-background/70 px-2 py-1 text-[10px] tabular-nums text-foreground">
            {Math.round(preview.zoom * 100)}%
          </span>
          <img ref={preview.imageRef} src={previewUrl}
            alt={props.preview?.file_name ?? props.preview?.file_path ?? props.preview?.url ?? ''}
            className="max-h-full max-w-full object-contain transition-transform duration-100 ease-out"
            style={{ transform: `scale(${preview.zoom})`, transformOrigin: `${preview.origin.x}% ${preview.origin.y}%` }}
            draggable={false} />
        </div>
      )}
    </div>
  )
}
