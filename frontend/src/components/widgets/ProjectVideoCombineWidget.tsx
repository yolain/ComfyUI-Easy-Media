import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Loader2, Maximize2, Minimize2, Pause, Play, RefreshCw, ZoomOut } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useCanvasScale } from '@/hooks/use-canvas-scale'
import { useElementWidth } from '@/hooks/use-element-width'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import { LocaleContext, translate } from '@/lib/i18n'
import { addMediaRevision, mediaContentToViewUrl } from '@/lib/media-url'
import { formatMultiTrackTime } from '@/lib/multitrack-utils'
import { MultiTrackRuler, MULTITRACK_LEFT_GUTTER, MULTITRACK_RIGHT_RESERVE } from './multitrack/MultiTrackRuler'
import { DEFAULT_PROJECT_DATA, type ProjectClip, type ProjectData } from '@/types/project'

const MIN_CLIP_FRAMES = 1
const ICON_BUTTON_CLASS = 'h-6 w-6 shrink-0 [&_svg]:size-3.5'

interface GraphLink {
  origin_id?: number | string
  target_id?: number | string
}

interface ProjectNode {
  id?: number | string
  graph?: { links?: Record<string, GraphLink> | GraphLink[] }
}

function ensureProjectData(value: unknown): ProjectData {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return DEFAULT_PROJECT_DATA
  const data = value as Partial<ProjectData>
  return {
    project_name: typeof data.project_name === 'string' ? data.project_name : 'default',
    width: Number.isFinite(data.width) ? Math.max(0, Number(data.width)) : 0,
    height: Number.isFinite(data.height) ? Math.max(0, Number(data.height)) : 0,
    frame_rate: Number.isFinite(data.frame_rate) && Number(data.frame_rate) > 0 ? Number(data.frame_rate) : 24,
    clips: Array.isArray(data.clips) ? data.clips : [],
    auto_combine: data.auto_combine !== false,
    updated_at: data.updated_at,
  }
}

function graphLinks(node: ProjectNode): GraphLink[] {
  const links = node.graph?.links
  if (!links) return []
  return (Array.isArray(links) ? links : Object.values(links)).filter(Boolean)
}

function downstreamNodeIds(node: ProjectNode): Set<string> {
  const rootId = String(node.id ?? '')
  const result = new Set<string>(rootId ? [rootId] : [])
  const pending = rootId ? [rootId] : []
  const links = graphLinks(node)
  while (pending.length > 0) {
    const originId = pending.shift()!
    for (const link of links) {
      if (String(link.origin_id ?? '') !== originId || link.target_id == null) continue
      const targetId = String(link.target_id)
      if (result.has(targetId)) continue
      result.add(targetId)
      pending.push(targetId)
    }
  }
  return result
}

function clipDuration(clip: ProjectClip): number {
  return Math.max(MIN_CLIP_FRAMES, clip.source_end_frame - clip.source_start_frame)
}

function totalFrames(clips: ProjectClip[]): number {
  return clips.filter((clip) => clip.enabled !== false).reduce((total, clip) => total + clipDuration(clip), 0)
}

function clipAtFrame(clips: ProjectClip[], frame: number): { clip: ProjectClip; start: number } | null {
  let cursor = 0
  for (const clip of clips) {
    if (clip.enabled === false) continue
    const end = cursor + clipDuration(clip)
    if (frame < end) return { clip, start: cursor }
    cursor = end
  }
  return null
}

function clipStartFrame(clips: ProjectClip[], targetId: string): number {
  let cursor = 0
  for (const clip of clips) {
    if (clip.enabled === false) continue
    if (clip.id === targetId) return cursor
    cursor += clipDuration(clip)
  }
  return 0
}

function errorMessage(payload: unknown, fallback: string): string {
  if (payload && typeof payload === 'object' && 'error' in payload && typeof payload.error === 'string') {
    return payload.error
  }
  return fallback
}

function projectClipUrl(clip: ProjectClip): string | null {
  const url = mediaContentToViewUrl({ source_type: 'output', file_path: clip.file_path })
  return url ? addMediaRevision(url, clip.media_revision) : null
}

export function ProjectVideoCombineWidget({ value, onChange, app, node }: Readonly<ReactWidgetProps<ProjectData>>) {
  const data = ensureProjectData(value)
  const locale = app?.ui?.settings?.settingsValues?.['Comfy.Locale']
  const t = useCallback(
    (path: string, params?: Record<string, string | number>) => translate(locale, path, params),
    [locale],
  )
  const [currentFrame, setCurrentFrame] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [timelineCollapsed, setTimelineCollapsed] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isCombining, setIsCombining] = useState(false)
  const [projects, setProjects] = useState<string[]>(['default'])
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const timelineContainerRef = useRef<HTMLDivElement>(null)
  const refreshRequestRef = useRef(0)
  const canvasScale = useCanvasScale(app)
  const timelineWidth = Math.max(1, useElementWidth(timelineContainerRef))
  const scaledTimelineWidth = timelineWidth * zoom
  const total = totalFrames(data.clips)
  const active = clipAtFrame(data.clips, Math.min(currentFrame, Math.max(0, total - 1)))
  const activeUrl = active ? projectClipUrl(active.clip) : null
  const activeIndex = active ? data.clips.findIndex((clip) => clip.id === active.clip.id) : -1
  const nextClip = activeIndex >= 0 ? data.clips.slice(activeIndex + 1).find((clip) => clip.enabled !== false) : undefined
  const nextUrl = nextClip ? projectClipUrl(nextClip) : null

  const refreshProject = useCallback(async (projectName: string, showError = true) => {
    if (!projectName) return
    const requestId = ++refreshRequestRef.current
    setIsRefreshing(true)
    const switchingProject = projectName !== data.project_name
    if (switchingProject) {
      setIsPlaying(false)
      setCurrentFrame(0)
      setSelectedClipId(null)
      onChange({ ...DEFAULT_PROJECT_DATA, project_name: projectName, auto_combine: data.auto_combine })
    }
    try {
      const response = await app.api.fetchApi(`/easy-media/project?project_name=${encodeURIComponent(projectName)}`)
      const payload: unknown = await response.json()
      if (!response.ok) throw new Error(errorMessage(payload, t('projectVideoCombine.refreshFailed')))
      if (requestId !== refreshRequestRef.current) return
      onChange({ ...ensureProjectData(payload), auto_combine: data.auto_combine })
    } catch (error) {
      if (showError && requestId === refreshRequestRef.current) {
        app.extensionManager.toast.add({
          severity: 'error',
          summary: t('projectVideoCombine.refreshFailed'),
          detail: error instanceof Error ? error.message : String(error),
          life: 5000,
        })
      }
    } finally {
      if (requestId === refreshRequestRef.current) setIsRefreshing(false)
    }
  }, [app, data.auto_combine, data.project_name, onChange, t])

  const loadProjects = useCallback(async () => {
    try {
      const response = await app.api.fetchApi('/easy-media/projects')
      const payload: unknown = await response.json()
      if (!response.ok) throw new Error(errorMessage(payload, t('projectVideoCombine.projectListFailed')))
      const names = payload && typeof payload === 'object' && 'projects' in payload && Array.isArray(payload.projects)
        ? payload.projects.filter((name): name is string => typeof name === 'string' && name.length > 0)
        : []
      setProjects(names.includes('default') ? names : ['default', ...names])
    } catch (error) {
      app.extensionManager.toast.add({
        severity: 'error',
        summary: t('projectVideoCombine.projectListFailed'),
        detail: error instanceof Error ? error.message : String(error),
        life: 5000,
      })
    }
  }, [app, t])

  useEffect(() => {
    if (data.clips.length === 0) void refreshProject(data.project_name || 'default', false)
  }, [])

  useEffect(() => {
    const handleSuccess = () => void refreshProject(data.project_name || 'default', false)
    app.api.addEventListener('execution_success', handleSuccess)
    return () => app.api.removeEventListener('execution_success', handleSuccess)
  }, [app.api, data.project_name, refreshProject])

  useEffect(() => {
    const handleSelected = (event: CustomEvent<unknown>) => {
      const payload = event.detail
      if (!payload || typeof payload !== 'object') return
      const nodeId = 'node_id' in payload ? String(payload.node_id) : ''
      if (nodeId && nodeId !== String((node as ProjectNode).id ?? '')) return
      const projectName = 'project_name' in payload ? String(payload.project_name) : ''
      if (projectName) void refreshProject(projectName, false)
    }
    app.api.addCustomEventListener('easy-media.project.selected', handleSelected)
    return () => app.api.removeCustomEventListener('easy-media.project.selected', handleSelected)
  }, [app.api, node, refreshProject])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !active) return
    const localSeconds = (active.clip.source_start_frame + Math.max(0, currentFrame - active.start)) / data.frame_rate
    if (Math.abs(video.currentTime - localSeconds) > 0.08) video.currentTime = localSeconds
    if (isPlaying) void video.play().catch(() => setIsPlaying(false))
    else video.pause()
  }, [active?.clip.id, activeUrl, data.frame_rate, isPlaying])

  function seek(frame: number) {
    const nextFrame = Math.max(0, Math.min(total, Math.round(frame)))
    setCurrentFrame(nextFrame)
    const nextActive = clipAtFrame(data.clips, Math.min(nextFrame, Math.max(0, total - 1)))
    setSelectedClipId(nextActive?.clip.id ?? null)
  }

  function handleVideoTimeUpdate() {
    const video = videoRef.current
    if (!video || !active || !isPlaying) return
    const localFrame = Math.round(video.currentTime * data.frame_rate) - active.clip.source_start_frame
    setCurrentFrame(Math.min(total, active.start + Math.max(0, localFrame)))
  }

  async function combineProject() {
    const projectName = data.project_name || 'default'
    try {
      const queue = await app.api.getQueue()
      if (queue.Running.length > 0 || queue.Pending.length > 0) {
        throw new Error(t('projectVideoCombine.queueBusy'))
      }
      setIsCombining(true)
      const prompt = await app.graphToPrompt()
      const keepIds = downstreamNodeIds(node as ProjectNode)
      const allOutput = prompt.output as Record<string, { inputs?: Record<string, unknown> }>
      const allNodeIds = new Set(Object.keys(allOutput))
      const filteredOutput = Object.fromEntries(
        Object.entries(allOutput).filter(([nodeId]) => keepIds.has(nodeId)).map(([nodeId, apiNode]) => {
          const inputs = { ...(apiNode.inputs ?? {}) }
          for (const [inputName, inputValue] of Object.entries(inputs)) {
            if (Array.isArray(inputValue) && inputValue.length === 2 && allNodeIds.has(String(inputValue[0])) && !keepIds.has(String(inputValue[0]))) {
              delete inputs[inputName]
            }
          }
          if (nodeId === String((node as ProjectNode).id ?? '')) {
            inputs.project_name = projectName
            inputs.project_data = JSON.stringify({ ...data, auto_combine: true })
          }
          return [nodeId, { ...apiNode, inputs }]
        }),
      )
      if (Object.keys(filteredOutput).length < 2) throw new Error(t('projectVideoCombine.connectSaveNode'))
      await app.api.queuePrompt(0, { ...prompt, output: filteredOutput as typeof prompt.output })
      app.extensionManager.toast.add({
        severity: 'info',
        summary: t('projectVideoCombine.combineQueued'),
        detail: t('projectVideoCombine.combineQueuedDetail', { project: projectName }),
        life: 4000,
      })
    } catch (error) {
      app.extensionManager.toast.add({
        severity: 'error',
        summary: t('projectVideoCombine.combineFailed'),
        detail: error instanceof Error ? error.message : String(error),
        life: 6000,
      })
    } finally {
      setIsCombining(false)
    }
  }

  function tooltip(label: string, child: ReactNode) {
    return <Tooltip><TooltipTrigger asChild><span className="inline-flex shrink-0">{child}</span></TooltipTrigger><TooltipContent>{label}</TooltipContent></Tooltip>
  }

  const playableWidth = Math.max(1, scaledTimelineWidth - MULTITRACK_LEFT_GUTTER - MULTITRACK_RIGHT_RESERVE)
  const playheadLeft = MULTITRACK_LEFT_GUTTER + (Math.min(currentFrame, total) / Math.max(total, 1)) * playableWidth

  return (
    <LocaleContext.Provider value={locale}>
      <TooltipProvider>
        <div className="flex h-full min-h-[520px] w-full flex-col overflow-hidden rounded-md border border-border bg-background text-foreground">
          <div className="flex h-10 shrink-0 items-center justify-between border-b border-border px-2">
            <Select value={data.project_name || 'default'} onOpenChange={(open) => { if (open) void loadProjects() }} onValueChange={(name) => void refreshProject(name)}>
              <SelectTrigger className="h-7 w-32 text-xs" aria-label={t('projectVideoCombine.selectProject')}><SelectValue /></SelectTrigger>
              <SelectContent>{projects.map((name) => <SelectItem key={name} value={name}>{name}</SelectItem>)}</SelectContent>
            </Select>
            <div className="flex items-center gap-2">
              {tooltip(t('projectVideoCombine.refresh'), (
                <Button type="button" variant="ghost" size="icon" className="h-7 w-7" disabled={isRefreshing} onClick={() => void refreshProject(data.project_name || 'default')}>
                  {isRefreshing ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
                </Button>
              ))}
              <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Checkbox checked={data.auto_combine} onCheckedChange={(checked) => onChange({ ...data, auto_combine: checked === true })} />
                {t('projectVideoCombine.autoCombine')}
              </label>
              <Button type="button" size="sm" className="h-7 gap-1.5 text-xs" disabled={isCombining || data.clips.length === 0} onClick={() => void combineProject()}>
                {isCombining ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
                {t('projectVideoCombine.combine')}
              </Button>
            </div>
          </div>

          <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-black p-3">
            {activeUrl ? (
              <video ref={videoRef} key={activeUrl} src={activeUrl} preload="auto" className="h-full max-h-full w-full object-contain" onTimeUpdate={handleVideoTimeUpdate} onEnded={() => seek(Math.min(total, (active?.start ?? 0) + clipDuration(active!.clip)))} />
            ) : <div className="text-xs text-muted-foreground">{t('projectVideoCombine.emptyPreview')}</div>}
            {nextUrl && <video key={nextUrl} src={nextUrl} preload="auto" className="hidden" muted aria-hidden />}
          </div>

          <div className="grid h-9 shrink-0 grid-cols-[1fr_auto_1fr] items-center border-y border-border px-1 text-[10px]">
            <div />
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="w-16 text-right tabular-nums text-gradient">{formatMultiTrackTime(currentFrame, { frameRate: data.frame_rate, showFrames: true })}</span>
              <Button type="button" variant="secondary" size="icon" className={`${ICON_BUTTON_CLASS} rounded-full`} disabled={total === 0} aria-label={isPlaying ? t('multitrack.pause') : t('multitrack.play')} onClick={() => setIsPlaying((playing) => !playing)}>{isPlaying ? <Pause /> : <Play />}</Button>
              <span className="w-16 tabular-nums">{formatMultiTrackTime(total, { frameRate: data.frame_rate, showFrames: true })}</span>
            </div>
            <div className="ml-auto flex items-center gap-1">
              {tooltip(timelineCollapsed ? t('multitrack.showTimeline') : t('multitrack.hideTimeline'), <Button type="button" variant="ghost" size="icon" className={ICON_BUTTON_CLASS} onClick={() => setTimelineCollapsed((collapsed) => !collapsed)}>{timelineCollapsed ? <Maximize2 /> : <Minimize2 />}</Button>)}
              <ZoomOut className="size-3.5 text-muted-foreground" />
              <Slider min={1} max={6} step={0.25} value={[zoom]} onValueChange={([next]) => setZoom(next)} className="h-3 w-12" aria-label={t('multitrack.timelineZoom')} />
            </div>
          </div>

          {!timelineCollapsed && (
            <div ref={timelineContainerRef} className="no-scrollbar h-[104px] shrink-0 overflow-x-auto overflow-y-hidden bg-muted/20">
              <div style={{ width: scaledTimelineWidth, minWidth: '100%' }}>
                <MultiTrackRuler totalLength={total} frameRate={data.frame_rate} width={scaledTimelineWidth} canvasScale={canvasScale} currentTime={currentFrame} taskMarkers={[]} selectedTaskMarkerId={null} onSeek={seek} onSelectTaskMarker={() => {}} onMoveTaskMarker={() => {}} onDeleteTaskMarker={() => {}} />
                <div className="relative h-20" style={{ width: scaledTimelineWidth }}>
                  <div className="absolute top-2 flex h-14 overflow-hidden rounded-sm border border-border bg-muted/40" style={{ left: MULTITRACK_LEFT_GUTTER, width: playableWidth }}>
                    {data.clips.filter((clip) => clip.enabled !== false).map((clip) => {
                      const selected = selectedClipId === clip.id
                      return (
                        <Button key={clip.id} type="button" variant="ghost" className={`h-full min-w-0 shrink-0 rounded-none border-r border-border px-2 ${selected ? 'bg-primary/20 ring-1 ring-inset ring-primary' : 'bg-secondary/40 hover:bg-secondary/60'}`} style={{ width: `${(clipDuration(clip) / Math.max(total, 1)) * 100}%` }} onClick={() => { setSelectedClipId(clip.id); setCurrentFrame(clipStartFrame(data.clips, clip.id)) }}>
                          <span className="truncate text-[11px] font-medium">{t('projectVideoCombine.clipLabel', { index: clip.index + 1 })}</span>
                        </Button>
                      )
                    })}
                  </div>
                  <div className="pointer-events-none absolute top-0 h-20 w-px bg-destructive" style={{ left: playheadLeft }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </TooltipProvider>
    </LocaleContext.Provider>
  )
}
