import { useRef, useState } from 'react'
import { FolderOpen, RotateCcw, Braces, Download, Cable } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ColorPickerPopover } from '@/components/ui/color-picker'
import { Slider } from '@/components/ui/slider'
import { Textarea } from '@/components/ui/textarea'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from '@/components/ui/dialog'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { MediaSelector } from '@/components/widgets/mediaSelector/MediaSelector'
import { CompareVideo } from '@/components/widgets/compareVideoWidget'
import { LocaleContext, useT } from '@/lib/i18n'
import { BEAUTY_GROUPS, NONNEGATIVE_BEAUTY_PARAMETERS, DEFAULT_BEAUTY_PRESET, normalizeBeautyPreset, type BeautyPreset } from '@/lib/face-beauty'
import { mediaPathToViewUrl } from '@/lib/media-url'
import { useVerticalWheelScroll } from '@/hooks/use-vertical-wheel-scroll'
import { useFaceBeautyPreview } from '@/hooks/use-face-beauty-preview'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
export function FaceBeautyWidget(props: ReactWidgetProps<BeautyPreset>) {
  return <LocaleContext.Provider value={props.app?.ui?.settings?.settingsValues?.['Comfy.Locale']}><TooltipProvider delayDuration={250}><FaceBeautyEditor {...props} /></TooltipProvider></LocaleContext.Provider>
}
function FaceBeautyEditor({ value, onChange, node, app }: ReactWidgetProps<BeautyPreset>) {
  const t = useT()
  const slidersRef = useRef<HTMLDivElement>(null)
  useVerticalWheelScroll(slidersRef)
  let preset = DEFAULT_BEAUTY_PRESET
  let presetError = ''
  try { preset = normalizeBeautyPreset(value) }
  catch (cause) { presetError = cause instanceof Error ? cause.message : String(cause) }
  const { preview, error, loading, canvasRef, connected, hasImageConnection, downloadModel } = useFaceBeautyPreview(node, preset)
  const [activeGroup, setActiveGroup] = useState<typeof BEAUTY_GROUPS[number]['id']>('face')
  const group = BEAUTY_GROUPS.find(item => item.id === activeGroup) ?? BEAUTY_GROUPS[0]
  const [pickerOpen, setPickerOpen] = useState(false)
  const [jsonOpen, setJsonOpen] = useState(false)
  const [json, setJson] = useState('')
  const [jsonError, setJsonError] = useState('')
  const sourceUrl = preview ? mediaPathToViewUrl(preview.filename, preview.type) : ''
  return <div className="flex h-full min-h-0 w-full overflow-hidden rounded-md border border-border bg-background text-foreground"
    onPointerDown={event => event.stopPropagation()} onWheel={event => event.stopPropagation()}>
    <div className="relative min-h-0 min-w-0 basis-2/3 overflow-hidden">
      {preview ? <CompareVideo app={app} sourceUrl={sourceUrl} outputUrl={sourceUrl} mediaType="image"
        outputContent={<canvas ref={canvasRef} aria-label={t('faceBeauty.preview')} className="h-full w-full object-contain" />} />
        : <div className="flex h-full items-center justify-center p-6 text-center text-sm text-muted-foreground">{t(connected ? 'faceBeauty.runInput' : 'faceBeauty.empty')}</div>}
      {loading && <div role="status" className="absolute bottom-3 left-3 rounded bg-background/90 p-2 text-xs">{t('faceBeauty.loading')}</div>}
    </div>
    <div data-face-beauty-controls className="flex min-h-0 min-w-0 basis-1/3 flex-col overflow-hidden border-l border-border">
      <div data-face-beauty-toolbar className="flex shrink-0 flex-wrap items-center gap-1 border-b border-border bg-background p-3">
        <Popover open={pickerOpen} onOpenChange={setPickerOpen}>
          <Tooltip><TooltipTrigger asChild><PopoverTrigger asChild><Button size="sm" variant="secondary" className="h-6 gap-1 px-2"><FolderOpen className="size-4" />{t('faceBeauty.select')}</Button></PopoverTrigger></TooltipTrigger><TooltipContent>{t('faceBeauty.select')}</TooltipContent></Tooltip>
          <PopoverContent className="w-auto p-0" align="end">
            <MediaSelector value={preset.source?.file_path ?? ''} mediaType="image" defaultTab={preset.source?.source_type === 'output' ? 'outputs' : 'inputs'} onChange={(filePath, sourceType = 'input') => {
                if (sourceType === 'local' || /^https?:/i.test(filePath)) { setJsonError(t('faceBeauty.localOnly')); return }
                onChange({ ...preset, source_mode: 'picker', source: { source_type: sourceType, file_path: filePath } }); setPickerOpen(false)
              }}
              onSourceChange={({ filePath, sourceType }) => {
                if (sourceType === 'local' || /^https?:/i.test(filePath)) { setJsonError(t('faceBeauty.localOnly')); return }
                onChange({ ...preset, source_mode: 'picker', source: { source_type: sourceType, file_path: filePath } }); setPickerOpen(false)
              }} />
          </PopoverContent>
        </Popover>
        {hasImageConnection && !connected && <Tooltip><TooltipTrigger asChild><Button size="icon" variant="ghost" className="h-6 w-6" aria-label={t('faceBeauty.useInput')} onClick={() => onChange({ ...preset, source_mode: 'image' })}><Cable className="size-3" /></Button></TooltipTrigger><TooltipContent>{t('faceBeauty.useInput')}</TooltipContent></Tooltip>}
        <Tooltip><TooltipTrigger asChild><Button size="icon" variant="ghost" className="h-6 w-6" aria-label={t('faceBeauty.downloadModel')} disabled={loading} onClick={() => void downloadModel()}><Download className="size-3" /></Button></TooltipTrigger><TooltipContent>{t('faceBeauty.downloadModel')}</TooltipContent></Tooltip>
        <Tooltip><TooltipTrigger asChild><Button size="icon" variant="ghost" className="h-6 w-6" aria-label={t('faceBeauty.reset')} onClick={() => onChange({ ...preset, parameters: { ...DEFAULT_BEAUTY_PRESET.parameters }, colors: { ...DEFAULT_BEAUTY_PRESET.colors } })}><RotateCcw className="size-3" /></Button></TooltipTrigger><TooltipContent>{t('faceBeauty.reset')}</TooltipContent></Tooltip>
        <Tooltip><TooltipTrigger asChild><Button size="icon" variant="ghost" className="h-6 w-6" aria-label={t('faceBeauty.json')} onClick={() => { setJson(JSON.stringify(preset, null, 2)); setJsonError(''); setJsonOpen(true) }}><Braces className="size-3" /></Button></TooltipTrigger><TooltipContent>{t('faceBeauty.json')}</TooltipContent></Tooltip>
      </div>
      <div data-face-beauty-categories role="group" aria-label={t('faceBeauty.categories')}
        className="flex shrink-0 flex-wrap items-center gap-1 border-b border-border px-2 py-1.5">
        {BEAUTY_GROUPS.map(item => <Button key={item.id} type="button" size="sm"
          variant={activeGroup === item.id ? 'secondary' : 'ghost'} aria-pressed={activeGroup === item.id}
          className="h-6 shrink-0 rounded-full px-2 py-0 text-xs" onClick={() => {
            setActiveGroup(item.id)
            if (slidersRef.current) slidersRef.current.scrollTop = 0
          }}>{t(`faceBeauty.${item.label}`)}</Button>)}
      </div>
      <div data-face-beauty-status className="shrink-0 px-3 pt-3">
      {(error || presetError || jsonError) && <p role="alert" className="mb-3 break-words text-xs text-destructive">{error || presetError || jsonError}</p>}
      {preview && !preview.face_found && <p role="status" className="mb-3 text-xs text-muted-foreground">{t('faceBeauty.noFace')}</p>}
      </div>
      <div ref={slidersRef} data-face-beauty-sliders data-capture-wheel="true" className="min-h-0 flex-1 overflow-y-auto overscroll-contain px-3 pb-3">
      <h3 className="mb-3 text-sm font-medium">{t(`faceBeauty.${group.label}`)}</h3>
      {group.parameters.map(key => <div key={key} className="mb-5">
        <div className="mb-2 flex justify-between gap-2 text-xs"><span>{t(`faceBeauty.${key}`)}</span><span className="text-muted-foreground tabular-nums">{Math.round(preset.parameters[key] * 100)}</span></div>
        <Slider aria-label={t(`faceBeauty.${key}`)} value={[preset.parameters[key]]} min={NONNEGATIVE_BEAUTY_PARAMETERS.has(key) ? 0 : -1} max={1} step={0.01}
          disabled={Boolean(presetError)} onValueChange={([next]) => onChange({ ...preset, parameters: { ...preset.parameters, [key]: next ?? 0 } })} />
        {(key === 'lip_tint' || key === 'blush') && <div className="mt-2"><ColorPickerPopover
          value={preset.colors[key === 'lip_tint' ? 'lip' : 'blush']} hideEyedropper
          triggerAriaLabel={t(key === 'lip_tint' ? 'faceBeauty.lipColor' : 'faceBeauty.blushColor')}
          triggerLabel={t(key === 'lip_tint' ? 'faceBeauty.lipColor' : 'faceBeauty.blushColor')}
          onValueChange={(_color, parsed) => onChange({ ...preset, colors: { ...preset.colors, [key === 'lip_tint' ? 'lip' : 'blush']: parsed.hex.slice(0, 7) } })}
        /></div>}
      </div>)}
      </div>
    </div>
    <Dialog open={jsonOpen} onOpenChange={setJsonOpen}><DialogContent><DialogHeader><DialogTitle>{t('faceBeauty.json')}</DialogTitle></DialogHeader>
      <Textarea className="h-72 font-mono text-xs" aria-label={t('faceBeauty.json')} value={json} onChange={event => setJson(event.target.value)} />
      <p className="text-xs text-muted-foreground">{t('faceBeauty.jsonHint')}</p>
      {jsonError && <p role="alert" className="text-xs text-destructive">{jsonError}</p>}
      <DialogFooter><Button onClick={() => {
        try { onChange(normalizeBeautyPreset(JSON.parse(json))); setJsonOpen(false); setJsonError('') }
        catch (cause) { setJsonError(cause instanceof Error ? cause.message : String(cause)) }
      }}>{t('faceBeauty.apply')}</Button></DialogFooter>
    </DialogContent></Dialog>
  </div>
}
