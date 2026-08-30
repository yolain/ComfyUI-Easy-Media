import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react'
import { Check, ChevronDown, Loader2, Maximize2, Minimize2, Pause, Play, RefreshCw, Trash2, Volume2, VolumeX, ZoomOut } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useCanvasScale } from '@/hooks/use-canvas-scale'
import { useElementWidth } from '@/hooks/use-element-width'
import { usePauseMediaOnPageExit } from '@/hooks/use-pause-media-on-page-exit'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import { LocaleContext, translate } from '@/lib/i18n'
import { addMediaRevision, mediaContentToViewUrl } from '@/lib/media-url'
import { formatMultiTrackTime } from '@/lib/multitrack-utils'
import { cn } from '@/lib/utils'
import { CompareVideo } from '@/components/widgets/compareVideoWidget'
import { MultiTrackRuler, MULTITRACK_LEFT_GUTTER, MULTITRACK_RIGHT_RESERVE } from './multitrack/MultiTrackRuler'
import { DEFAULT_PROJECT_DATA, type ProjectClip, type ProjectData, type ProjectVideoFile } from '@/types/project'

const MIN_CLIP_FRAMES = 1
const PLAYBACK_UI_UPDATE_INTERVAL_MS = 100
const ICON_BUTTON_CLASS = 'h-6 w-6 shrink-0 [&_svg]:size-3.5'

interface GraphLink {
  origin_id?: number | string
  target_id?: number | string
}

interface ProjectNode {
  id?: number | string
  graph?: {
    links?: Record<string, GraphLink> | GraphLink[] | Map<unknown, GraphLink>
    _nodes?: Array<{ id?: number | string }>
  }
}

const DROP_PROMPT_INPUT = Symbol('drop-prompt-input')

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
  if (Array.isArray(links)) return links.filter(Boolean)
  if (links instanceof Map) return [...links.values()].filter(Boolean)
  return Object.values(links).filter(Boolean)
}

function promptGraphNodeIds(
  node: ProjectNode,
  workflow: unknown,
  output: Record<string, unknown>,
): Set<string> {
  const ids = new Set(Object.keys(output))
  for (const graphNode of node.graph?._nodes ?? []) {
    if (graphNode.id != null) ids.add(String(graphNode.id))
  }
  if (workflow && typeof workflow === 'object' && 'nodes' in workflow && Array.isArray(workflow.nodes)) {
    for (const workflowNode of workflow.nodes) {
      if (workflowNode && typeof workflowNode === 'object' && 'id' in workflowNode && workflowNode.id != null) {
        ids.add(String(workflowNode.id))
      }
    }
  }
  return ids
}

function pruneExternalPromptConnections(
  value: unknown,
  keepIds: Set<string>,
  graphNodeIds: Set<string>,
): unknown | typeof DROP_PROMPT_INPUT {
  if (
    Array.isArray(value)
    && value.length === 2
    && (typeof value[0] === 'string' || typeof value[0] === 'number')
    && typeof value[1] === 'number'
    && graphNodeIds.has(String(value[0]))
  ) {
    return keepIds.has(String(value[0])) ? value : DROP_PROMPT_INPUT
  }
  if (Array.isArray(value)) {
    return value.flatMap((item) => {
      const pruned = pruneExternalPromptConnections(item, keepIds, graphNodeIds)
      return pruned === DROP_PROMPT_INPUT ? [] : [pruned]
    })
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).flatMap(([key, item]) => {
        const pruned = pruneExternalPromptConnections(item, keepIds, graphNodeIds)
        return pruned === DROP_PROMPT_INPUT ? [] : [[key, pruned]]
      }),
    )
  }
  return value
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

function projectVideoFileUrl(file: ProjectVideoFile): string | null {
  const url = mediaContentToViewUrl({ source_type: 'output', file_path: file.file_path })
  return url ? addMediaRevision(url, file.media_revision) : null
}

function projectVideoFiles(clip: ProjectClip): ProjectVideoFile[] {
  const files = clip.video_files?.length
    ? clip.video_files
    : [{
        file_path: clip.file_path,
        file_name: clip.file_name,
        media_revision: clip.media_revision,
        source_frame_count: clip.source_frame_count,
      }]
  return files.some((file) => file.file_path === clip.file_path)
    ? files
    : [{
        file_path: clip.file_path,
        file_name: clip.file_name,
        media_revision: clip.media_revision,
        source_frame_count: clip.source_frame_count,
      }, ...files]
}

function selectProjectVideoFile(clip: ProjectClip, file: ProjectVideoFile): ProjectClip {
  const sourceFrameCount = Math.max(MIN_CLIP_FRAMES, file.source_frame_count)
  const sourceStartFrame = Math.min(clip.source_start_frame, sourceFrameCount - 1)
  return {
    ...clip,
    file_path: file.file_path,
    file_name: file.file_name,
    media_revision: file.media_revision,
    source_start_frame: sourceStartFrame,
    source_end_frame: Math.max(sourceStartFrame + 1, Math.min(clip.source_end_frame, sourceFrameCount)),
    source_frame_count: sourceFrameCount,
  }
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
  const [muted, setMuted] = useState(false)
  const [zoom, setZoom] = useState(1)
  const [timelineCollapsed, setTimelineCollapsed] = useState(false)
  const [isRefreshing, setIsRefreshing] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)
  const [deletingFilePath, setDeletingFilePath] = useState<string | null>(null)
  const [isCombining, setIsCombining] = useState(false)
  const [projects, setProjects] = useState<string[]>(['default'])
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null)
  const [previewFilePaths, setPreviewFilePaths] = useState<Record<string, string[]>>({})
  const [playbackNonce, setPlaybackNonce] = useState(0)
  const [previewPauseNonce, setPreviewPauseNonce] = useState(0)
  const videoRefs = useRef(new Map<string, HTMLVideoElement>())
  const currentFrameRef = useRef(0)
  const playbackStartFrameRef = useRef(0)
  const playbackStartedAtRef = useRef(0)
  const playbackUiUpdatedAtRef = useRef(0)
  const playbackRafRef = useRef<number | null>(null)
  const timelineContainerRef = useRef<HTMLDivElement>(null)
  const refreshRequestRef = useRef(0)
  const deletingFileRef = useRef(false)
  const latestDataRef = useRef(data)
  latestDataRef.current = data
  const canvasScale = useCanvasScale(app)
  const timelineWidth = Math.max(1, useElementWidth(timelineContainerRef))
  const scaledTimelineWidth = timelineWidth * zoom
  const total = totalFrames(data.clips)
  const active = clipAtFrame(data.clips, Math.min(currentFrame, Math.max(0, total - 1)))
  const activeUrl = active ? projectClipUrl(active.clip) : null
  const enabledClips = data.clips.filter((clip) => clip.enabled !== false)
  const activeIndex = active ? enabledClips.findIndex((clip) => clip.id === active.clip.id) : -1
  const nextClip = activeIndex >= 0 && enabledClips.length > 1
    ? enabledClips[(activeIndex + 1) % enabledClips.length]
    : undefined
  const nextUrl = nextClip ? projectClipUrl(nextClip) : null
  const projectOptions = projects.includes(data.project_name) || !data.project_name
    ? projects
    : [...projects, data.project_name]

  function selectedFilePaths(clip: ProjectClip): string[] {
    const availablePaths = new Set(projectVideoFiles(clip).map((file) => file.file_path))
    if (availablePaths.size === 1) return [...availablePaths]
    const storedPaths = [...new Set(previewFilePaths[clip.id] ?? [])]
      .filter((path) => availablePaths.has(path)).slice(0, 2)
    return storedPaths.includes(clip.file_path) ? storedPaths : [clip.file_path]
  }

  const activeSelectedFiles = active
    ? selectedFilePaths(active.clip)
        .map((filePath) => projectVideoFiles(active.clip).find((file) => file.file_path === filePath))
        .filter((file): file is ProjectVideoFile => Boolean(file))
    : []
  const compareUrls = activeSelectedFiles.length === 2
    ? activeSelectedFiles.map(projectVideoFileUrl)
    : []
  const isComparing = compareUrls.length === 2 && compareUrls.every(Boolean)

  function selectClip(clip: ProjectClip) {
    seek(clipStartFrame(data.clips, clip.id))
  }

  function selectClipFile(clip: ProjectClip, filePath: string) {
    const file = clip.video_files?.find((candidate) => candidate.file_path === filePath)
    if (!file) return
    onChange({
      ...data,
      clips: data.clips.map((item) => item.id === clip.id ? selectProjectVideoFile(item, file) : item),
    })
  }

  function toggleClipFile(clip: ProjectClip, filePath: string) {
    const currentPaths = selectedFilePaths(clip)
    const selected = currentPaths.includes(filePath)
    if (selected && currentPaths.length === 1) return
    if (!selected && currentPaths.length >= 2) return

    const nextPaths = selected
      ? currentPaths.filter((path) => path !== filePath)
      : [...currentPaths, filePath]
    setPreviewFilePaths((current) => ({ ...current, [clip.id]: nextPaths }))
    setIsPlaying(false)
    selectClip(clip)
    if (nextPaths[0] !== clip.file_path) selectClipFile(clip, nextPaths[0])
  }

  const refreshProject = useCallback(async (projectName: string, showError = true) => {
    if (!projectName || deletingFileRef.current) return
    const requestId = ++refreshRequestRef.current
    setIsRefreshing(true)
    const switchingProject = projectName !== data.project_name
    if (switchingProject) {
      setIsPlaying(false)
      setCurrentFrame(0)
      setSelectedClipId(null)
      setPreviewFilePaths({})
      onChange({ ...DEFAULT_PROJECT_DATA, project_name: projectName, auto_combine: data.auto_combine })
    }
    try {
      const response = await app.api.fetchApi(`/easy-media/project?project_name=${encodeURIComponent(projectName)}`)
      const payload: unknown = await response.json()
      if (!response.ok && response.status === 404 && projectName === 'default') {
        if (requestId !== refreshRequestRef.current) return
        onChange({ ...DEFAULT_PROJECT_DATA, project_name: 'default', auto_combine: data.auto_combine })
        return
      }
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

  async function deleteClipFile(clip: ProjectClip, file: ProjectVideoFile) {
    if (deletingFileRef.current) return
    deletingFileRef.current = true
    setDeletingFilePath(file.file_path)
    const projectName = data.project_name || 'default'
    try {
      const confirmed = await app.extensionManager.dialog.confirm({
        title: t('projectVideoCombine.deleteClipFileTitle'),
        message: t('projectVideoCombine.deleteClipFileConfirm', { file: file.file_name }),
      })
      if (!confirmed) return
      ++refreshRequestRef.current
      setIsRefreshing(false)
      const response = await app.api.fetchApi('/easy-media/project/video', {
        method: 'DELETE',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_name: projectName, segment_index: clip.index, file_path: file.file_path }),
      })
      let payload: unknown
      try {
        payload = await response.json()
      } catch (error) {
        if (!(error instanceof SyntaxError)) throw error
        if (response.status === 404 || response.status === 405) {
          throw new Error(t('projectVideoCombine.deleteClipFileUnavailable'))
        }
        throw new Error(t('projectVideoCombine.deleteClipFileInvalidResponse', { status: response.status }))
      }
      if (!response.ok) throw new Error(errorMessage(payload, t('projectVideoCombine.deleteClipFileFailed')))
      if (
        !payload || typeof payload !== 'object'
        || !('project_name' in payload) || payload.project_name !== projectName
        || !('clips' in payload) || !Array.isArray(payload.clips)
      ) {
        throw new Error(t('projectVideoCombine.deleteClipFileInvalidResponse', { status: response.status }))
      }
      const current = latestDataRef.current
      if ((current.project_name || 'default') !== projectName) return
      const fresh = ensureProjectData(payload)
      const replacement = fresh.clips.find((item) => item.id === clip.id)
      const nextData = {
        ...current,
        updated_at: fresh.updated_at,
        clips: current.clips.flatMap((item) => {
          if (item.id !== clip.id) return [item]
          if (!replacement) return []
          const files = projectVideoFiles(replacement)
          const preferredPaths = [item.file_path, ...(previewFilePaths[item.id] ?? [])]
          const selectedFile = preferredPaths
            .map((path) => files.find((candidate) => candidate.file_path === path))
            .find((candidate) => candidate !== undefined)
            ?? files.find((candidate) => candidate.file_path === replacement.file_path)!
          return [selectProjectVideoFile({ ...item, video_files: files }, selectedFile)]
        }),
      }
      setPreviewFilePaths((currentPaths) => {
        const nextPaths = { ...currentPaths }
        const remainingClip = nextData.clips.find((item) => item.id === clip.id)
        if (!remainingClip) {
          delete nextPaths[clip.id]
          return nextPaths
        }
        const availablePaths = new Set(projectVideoFiles(remainingClip).map((item) => item.file_path))
        const selectedPaths = [...new Set(currentPaths[clip.id] ?? [])]
          .filter((path) => availablePaths.has(path))
        // A comparison requires two surviving selections. Otherwise explicitly
        // reset to the saved primary video instead of retaining comparison state.
        nextPaths[clip.id] = availablePaths.size > 1 && selectedPaths.length === 2
          && selectedPaths.includes(remainingClip.file_path)
          ? selectedPaths
          : [remainingClip.file_path]
        return nextPaths
      })
      setIsPlaying(false)
      setPreviewPauseNonce((nonce) => nonce + 1)
      setPlaybackNonce((nonce) => nonce + 1)
      const nextFrame = Math.min(currentFrameRef.current, Math.max(0, totalFrames(nextData.clips) - 1))
      currentFrameRef.current = nextFrame
      setCurrentFrame(nextFrame)
      setSelectedClipId(clipAtFrame(nextData.clips, nextFrame)?.clip.id ?? null)
      onChange(nextData)
    } catch (error) {
      app.extensionManager.toast.add({
        severity: 'error',
        summary: t('projectVideoCombine.deleteClipFileFailed'),
        detail: error instanceof Error ? error.message : String(error),
        life: 5000,
      })
    } finally {
      deletingFileRef.current = false
      setDeletingFilePath(null)
    }
  }

  async function deleteProject() {
    const projectName = data.project_name || 'default'
    let confirmed = false
    try {
      confirmed = await app.extensionManager.dialog.confirm({
        title: t('projectVideoCombine.deleteProjectTitle'),
        message: t(projectName === 'default'
          ? 'projectVideoCombine.deleteDefaultProjectConfirm'
          : 'projectVideoCombine.deleteProjectConfirm', { project: projectName }),
      })
    } catch (error) {
      app.extensionManager.toast.add({
        severity: 'error',
        summary: t('projectVideoCombine.deleteProjectFailed'),
        detail: error instanceof Error ? error.message : String(error),
        life: 5000,
      })
      return
    }
    if (!confirmed) return

    setIsDeleting(true)
    setIsRefreshing(false)
    ++refreshRequestRef.current
    try {
      const response = await app.api.fetchApi(
        `/easy-media/project?project_name=${encodeURIComponent(projectName)}`,
        { method: 'DELETE' },
      )
      const payload: unknown = await response.json()
      if (!response.ok) throw new Error(errorMessage(payload, t('projectVideoCombine.deleteProjectFailed')))
      setIsPlaying(false)
      setCurrentFrame(0)
      setSelectedClipId(null)
      setPreviewFilePaths({})
      setProjects((names) => projectName === 'default'
        ? names
        : names.filter((name) => name !== projectName))
      onChange({ ...DEFAULT_PROJECT_DATA, project_name: 'default', auto_combine: data.auto_combine })
      app.extensionManager.toast.add({
        severity: 'success',
        summary: t('projectVideoCombine.deleteProjectSuccess'),
        detail: t('projectVideoCombine.deleteProjectSuccessDetail', { project: projectName }),
        life: 4000,
      })
    } catch (error) {
      app.extensionManager.toast.add({
        severity: 'error',
        summary: t('projectVideoCombine.deleteProjectFailed'),
        detail: error instanceof Error ? error.message : String(error),
        life: 5000,
      })
    } finally {
      setIsDeleting(false)
    }
  }

  useEffect(() => {
    if (data.clips.length === 0) void refreshProject(data.project_name || 'default', false)
  }, [])

  useEffect(() => {
    const handleSuccess = () => void refreshProject(data.project_name || 'default', false)
    app.api.addEventListener('execution_success', handleSuccess)
    return () => app.api.removeEventListener('execution_success', handleSuccess)
  }, [app.api, data.project_name, refreshProject])

  useEffect(() => {
    const handleProjectRefresh = (event: CustomEvent<unknown>) => {
      const payload = event.detail
      const projectName = payload && typeof payload === 'object' && 'project_name' in payload
        ? String(payload.project_name)
        : ''
      const currentProjectName = data.project_name || 'default'
      void refreshProject(projectName || currentProjectName, false)
    }
    app.api.addCustomEventListener('easy_multitrack_project_refresh', handleProjectRefresh)
    return () => app.api.removeCustomEventListener('easy_multitrack_project_refresh', handleProjectRefresh)
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
    const syncNode = node as ProjectNode & { __easyMediaSyncPlay?: (startAt: number, muted?: boolean) => void }
    syncNode.__easyMediaSyncPlay = (_startAt, syncMuted) => {
      if (typeof syncMuted === 'boolean') setMuted(syncMuted)
      currentFrameRef.current = 0
      setCurrentFrame(0)
      setSelectedClipId(clipAtFrame(data.clips, 0)?.clip.id ?? null)
      setPlaybackNonce((nonce) => nonce + 1)
      setIsPlaying(true)
    }
    return () => {
      delete syncNode.__easyMediaSyncPlay
    }
  }, [data.clips, node])

  useEffect(() => {
    currentFrameRef.current = currentFrame
  }, [currentFrame])

  const pausePreviewMedia = useCallback(() => {
    if (playbackRafRef.current !== null) {
      cancelAnimationFrame(playbackRafRef.current)
      playbackRafRef.current = null
    }
    for (const video of videoRefs.current.values()) video.pause()
  }, [])

  usePauseMediaOnPageExit(() => {
    pausePreviewMedia()
    setCurrentFrame(currentFrameRef.current)
    setIsPlaying(false)
    setPreviewPauseNonce((nonce) => nonce + 1)
  })

  useEffect(() => () => pausePreviewMedia(), [pausePreviewMedia])

  useEffect(() => {
    const workflowStore = app.extensionManager.workflow
    let activeWorkflowKey = workflowStore.activeWorkflow?.key ?? null
    return workflowStore.$subscribe((_mutation: unknown, state: { activeWorkflow?: { key: string } | null }) => {
      const nextWorkflowKey = state.activeWorkflow?.key ?? null
      if (
        activeWorkflowKey !== null
        && nextWorkflowKey !== null
        && nextWorkflowKey !== activeWorkflowKey
      ) {
        pausePreviewMedia()
        setCurrentFrame(currentFrameRef.current)
        setIsPlaying(false)
        setPreviewPauseNonce((nonce) => nonce + 1)
      }
      activeWorkflowKey = nextWorkflowKey
    })
  }, [app.extensionManager.workflow, pausePreviewMedia])

  useEffect(() => {
    if (!isPlaying || total <= 0) {
      if (playbackRafRef.current !== null) cancelAnimationFrame(playbackRafRef.current)
      playbackRafRef.current = null
      return
    }

    playbackStartedAtRef.current = performance.now()
    playbackUiUpdatedAtRef.current = playbackStartedAtRef.current
    playbackStartFrameRef.current = currentFrameRef.current >= total ? 0 : currentFrameRef.current

    function tick(now: number) {
      const elapsedFrames = Math.floor(((now - playbackStartedAtRef.current) / 1000) * data.frame_rate)
      const nextFrame = playbackStartFrameRef.current + elapsedFrames
      if (nextFrame >= total) {
        currentFrameRef.current = 0
        playbackStartFrameRef.current = 0
        playbackStartedAtRef.current = now
        setCurrentFrame(0)
        setSelectedClipId(clipAtFrame(data.clips, 0)?.clip.id ?? null)
        setPlaybackNonce((nonce) => nonce + 1)
      } else if (nextFrame !== currentFrameRef.current) {
        currentFrameRef.current = nextFrame
        if (now - playbackUiUpdatedAtRef.current >= PLAYBACK_UI_UPDATE_INTERVAL_MS) {
          playbackUiUpdatedAtRef.current = now
          setCurrentFrame(nextFrame)
        }
      }
      playbackRafRef.current = requestAnimationFrame(tick)
    }

    playbackRafRef.current = requestAnimationFrame(tick)
    return () => {
      if (playbackRafRef.current !== null) cancelAnimationFrame(playbackRafRef.current)
      playbackRafRef.current = null
    }
  }, [data.clips, data.frame_rate, isPlaying, total])

  useEffect(() => {
    const video = active ? videoRefs.current.get(active.clip.id) : undefined
    if (!video || !active) return
    const localSeconds = (
      active.clip.source_start_frame + Math.max(0, currentFrameRef.current - active.start)
    ) / data.frame_rate
    try {
      if (Math.abs(video.currentTime - localSeconds) > 0.001) video.currentTime = localSeconds
    } catch (error) {
      console.error('[ProjectVideoCombineWidget] failed to seek preview video:', error)
    }
  }, [active?.clip.id, activeUrl, data.frame_rate, isComparing, playbackNonce])

  useEffect(() => {
    if (isPlaying) return
    const video = active ? videoRefs.current.get(active.clip.id) : undefined
    if (!video || !active) return
    const localSeconds = (active.clip.source_start_frame + Math.max(0, currentFrame - active.start)) / data.frame_rate
    try {
      if (Math.abs(video.currentTime - localSeconds) > 0.001) video.currentTime = localSeconds
    } catch (error) {
      console.error('[ProjectVideoCombineWidget] failed to seek paused preview video:', error)
    }
  }, [active?.clip.id, activeUrl, currentFrame, data.frame_rate, isComparing, isPlaying])

  useEffect(() => {
    const video = nextClip ? videoRefs.current.get(nextClip.id) : undefined
    if (!video || !nextClip) return
    const preloadTime = nextClip.source_start_frame / data.frame_rate
    try {
      if (Math.abs(video.currentTime - preloadTime) > 0.001) video.currentTime = preloadTime
    } catch (error) {
      console.error('[ProjectVideoCombineWidget] failed to preload the next preview video:', error)
    }
  }, [data.frame_rate, nextClip?.id, nextUrl])

  useEffect(() => {
    const video = active ? videoRefs.current.get(active.clip.id) : undefined
    if (!video || !active) return
    for (const [clipId, candidate] of videoRefs.current) {
      if (clipId !== active.clip.id) candidate.pause()
    }
    if (isPlaying) {
      void video.play().catch((error: unknown) => {
        console.error('[ProjectVideoCombineWidget] failed to play preview video:', error)
        setIsPlaying(false)
      })
    } else {
      video.pause()
    }
  }, [active?.clip.id, activeUrl, isComparing, isPlaying, playbackNonce])

  function seek(frame: number) {
    const nextFrame = Math.max(0, Math.min(total, Math.round(frame)))
    currentFrameRef.current = nextFrame
    if (isPlaying) {
      playbackStartFrameRef.current = nextFrame >= total ? 0 : nextFrame
      playbackStartedAtRef.current = performance.now()
    }
    setCurrentFrame(nextFrame)
    const nextActive = clipAtFrame(data.clips, Math.min(nextFrame, Math.max(0, total - 1)))
    setSelectedClipId(nextActive?.clip.id ?? null)
  }

  function togglePlayback() {
    setIsPlaying((playing) => {
      if (playing) setCurrentFrame(currentFrameRef.current)
      return !playing
    })
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
      const graphNodeIds = promptGraphNodeIds(node as ProjectNode, prompt.workflow, allOutput)
      const filteredOutput = Object.fromEntries(
        Object.entries(allOutput).filter(([nodeId]) => keepIds.has(nodeId)).map(([nodeId, apiNode]) => {
          const prunedInputs = pruneExternalPromptConnections(apiNode.inputs ?? {}, keepIds, graphNodeIds)
          const inputs = prunedInputs === DROP_PROMPT_INPUT
            ? {}
            : prunedInputs as Record<string, unknown>
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
            <div className="flex items-center gap-1">
              <Select disabled={deletingFilePath !== null} value={data.project_name || 'default'} onOpenChange={(open) => { if (open) void loadProjects() }} onValueChange={(name) => void refreshProject(name)}>
                <SelectTrigger className="h-7 w-32 text-xs" aria-label={t('projectVideoCombine.selectProject')}><SelectValue /></SelectTrigger>
                <SelectContent>{projectOptions.map((name) => <SelectItem key={name} value={name}>{name}</SelectItem>)}</SelectContent>
              </Select>
              {tooltip(t('projectVideoCombine.deleteProject'), (
                <Button type="button" variant="ghost" size="icon" className="h-7 w-7 text-destructive hover:text-destructive" disabled={isDeleting || isRefreshing || deletingFilePath !== null} aria-label={t('projectVideoCombine.deleteProject')} onClick={() => void deleteProject()}>
                  {isDeleting ? <Loader2 className="size-3.5 animate-spin" /> : <Trash2 className="size-3.5" />}
                </Button>
              ))}
            </div>
            <div className="flex items-center gap-2">
              {tooltip(t('projectVideoCombine.refresh'), (
                <Button type="button" variant="ghost" size="icon" className="h-7 w-7" disabled={isRefreshing || deletingFilePath !== null} aria-label={t('projectVideoCombine.refresh')} onClick={() => void refreshProject(data.project_name || 'default')}>
                  {isRefreshing ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
                </Button>
              ))}
              <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <Checkbox checked={data.auto_combine} onCheckedChange={(checked) => onChange({ ...data, auto_combine: checked === true })} />
                {t('projectVideoCombine.autoCombine')}
              </label>
              <Button type="button" size="sm" className="h-7 gap-1.5 text-xs" disabled={isCombining || deletingFilePath !== null || data.clips.length === 0} onClick={() => void combineProject()}>
                {isCombining ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
                {t('projectVideoCombine.combine')}
              </Button>
            </div>
          </div>

          <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-black p-3">
            {isComparing ? (
              <CompareVideo
                app={app}
                sourceUrl={compareUrls[0]!}
                outputUrl={compareUrls[1]!}
                pausePlaybackNonce={previewPauseNonce}

              />
            ) : activeUrl ? (
              [
                { clip: active!.clip, url: activeUrl, visible: true },
                ...(nextClip && nextUrl ? [{ clip: nextClip, url: nextUrl, visible: false }] : []),
              ].map(({ clip, url, visible }) => (
                <video
                  ref={(element) => {
                    if (element) videoRefs.current.set(clip.id, element)
                    else videoRefs.current.delete(clip.id)
                  }}
                  key={clip.id}
                  src={url}
                  preload="auto"
                  playsInline
                  muted={!visible || muted}
                  aria-hidden={!visible}
                  className={visible
                    ? 'h-full max-h-full w-full object-contain'
                    : 'pointer-events-none absolute left-0 top-0 h-px w-px opacity-0'}
                />
              ))
            ) : <div className="text-xs text-muted-foreground">{t('projectVideoCombine.emptyPreview')}</div>}
          </div>

          <div className="grid h-9 shrink-0 grid-cols-[1fr_auto_1fr] items-center border-y border-border px-1 text-[10px]">
            <div className="flex items-center">
              {tooltip(muted ? t('projectVideoCombine.unmutePreview') : t('projectVideoCombine.mutePreview'), (
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className={ICON_BUTTON_CLASS}
                  aria-label={muted ? t('projectVideoCombine.unmutePreview') : t('projectVideoCombine.mutePreview')}
                  aria-pressed={muted}
                  onClick={() => setMuted((current) => !current)}
                >
                  {muted ? <VolumeX /> : <Volume2 />}
                </Button>
              ))}
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="w-16 text-right tabular-nums text-gradient">{formatMultiTrackTime(currentFrame, { frameRate: data.frame_rate, showFrames: true })}</span>
              <Button type="button" variant="secondary" size="icon" className={`${ICON_BUTTON_CLASS} rounded-full`} disabled={total === 0 || isComparing} aria-label={isPlaying ? t('multitrack.pause') : t('multitrack.play')} onClick={togglePlayback}>{isPlaying ? <Pause /> : <Play />}</Button>
              <span className="w-16 tabular-nums">{formatMultiTrackTime(total, { frameRate: data.frame_rate, showFrames: true })}</span>
            </div>
            <div className="ml-auto flex items-center gap-1">
              {tooltip(timelineCollapsed ? t('multitrack.showTimeline') : t('multitrack.hideTimeline'), <Button type="button" variant="ghost" size="icon" className={ICON_BUTTON_CLASS} aria-label={timelineCollapsed ? t('multitrack.showTimeline') : t('multitrack.hideTimeline')} onClick={() => setTimelineCollapsed((collapsed) => !collapsed)}>{timelineCollapsed ? <Maximize2 /> : <Minimize2 />}</Button>)}
              <ZoomOut className="size-3.5 text-muted-foreground" />
              <Slider min={1} max={6} step={0.25} value={[zoom]} onValueChange={([next]) => setZoom(next)} className="h-3 w-12" aria-label={t('multitrack.timelineZoom')} />
            </div>
          </div>

          <div
            ref={timelineContainerRef}
            className={`no-scrollbar h-[104px] shrink-0 overflow-x-auto overflow-y-hidden bg-muted/20 ${timelineCollapsed ? 'hidden' : ''}`}
            aria-hidden={timelineCollapsed}
          >
            <div style={{ width: scaledTimelineWidth, minWidth: '100%' }}>
              <MultiTrackRuler totalLength={total} frameRate={data.frame_rate} width={scaledTimelineWidth} canvasScale={canvasScale} currentTime={currentFrame} taskMarkers={[]} selectedTaskMarkerId={null} onSeek={seek} onSelectTaskMarker={() => {}} onMoveTaskMarker={() => {}} onDeleteTaskMarker={() => {}} />
              <div className="relative h-20" style={{ width: scaledTimelineWidth }}>
                <div className="absolute top-2 flex h-14 overflow-hidden rounded-sm border border-border bg-muted/40" style={{ left: MULTITRACK_LEFT_GUTTER, width: playableWidth }}>
                  {data.clips.filter((clip) => clip.enabled !== false).map((clip) => {
                      const selected = selectedClipId === clip.id
                      const blockClassName = `h-full min-w-0 shrink-0 rounded-none border-r border-border px-2 ${selected ? 'bg-primary/20 ring-1 ring-inset ring-primary' : 'bg-secondary/40 hover:bg-secondary/60'}`
                      const blockStyle = { width: `${(clipDuration(clip) / Math.max(total, 1)) * 100}%` }
                      const label = (
                        <div className="flex min-w-0 flex-col items-center leading-tight">
                          <span className="truncate text-[11px] font-medium">{t('projectVideoCombine.clipLabel', { number: clip.index + 1 })}</span>
                          <span className="truncate text-[9px] text-muted-foreground">
                            {t('projectVideoCombine.clipContinuity', {
                              mode: t(clip.continuity_mode === 'context'
                                ? 'projectVideoCombine.continuityContext'
                                : 'projectVideoCombine.continuityShot'),
                            })}
                          </span>
                        </div>
                      )
                      const selectedPaths = selectedFilePaths(clip)
                      return (
                        <Popover key={clip.id} onOpenChange={(open) => { if (open) selectClip(clip) }}>
                          <PopoverTrigger asChild>
                            <Button type="button" variant="ghost" className={`${blockClassName} relative justify-center`} style={blockStyle} aria-label={t('projectVideoCombine.selectClipFiles', { number: clip.index + 1 })}>
                              {label}
                              <span className="absolute right-2 flex items-center gap-0.5 text-[9px] text-muted-foreground">
                                {selectedPaths.length > 1 ? selectedPaths.length : null}
                                <ChevronDown className="size-3" />
                              </span>
                            </Button>
                          </PopoverTrigger>
                          <PopoverContent className="w-64 p-1" align="center">
                            <div className="px-2 py-1 text-[10px] text-muted-foreground">{t('projectVideoCombine.selectClipFilesHint')}</div>
                            {projectVideoFiles(clip).map((file) => {
                              const checked = selectedPaths.includes(file.file_path)
                              const disabled = deletingFilePath !== null || (!checked && selectedPaths.length >= 2)
                              return (
                                <div key={file.file_path} className="flex items-center gap-1">
                                  <Button
                                    type="button"
                                    variant="ghost"
                                    role="checkbox"
                                    aria-checked={checked}
                                    disabled={disabled}
                                    className={cn('h-8 min-w-0 flex-1 justify-start gap-2 px-2 text-xs', disabled && 'opacity-50')}
                                    onClick={() => toggleClipFile(clip, file.file_path)}
                                  >
                                    <span className={cn('flex size-4 shrink-0 items-center justify-center rounded-sm border border-border', checked && 'border-primary bg-primary text-primary-foreground')}>
                                      {checked ? <Check className="size-3" /> : null}
                                    </span>
                                    <span className="truncate">{file.file_name}</span>
                                  </Button>
                                  <Button
                                    type="button"
                                    variant="ghost"
                                    size="icon"
                                    className="size-7 shrink-0 text-destructive hover:text-destructive"
                                    disabled={deletingFilePath !== null || isDeleting || isRefreshing}
                                    aria-label={t('projectVideoCombine.deleteClipFile', { file: file.file_name })}
                                    title={t('projectVideoCombine.deleteClipFile', { file: file.file_name })}
                                    onPointerDown={(event) => event.stopPropagation()}
                                    onKeyDown={(event) => event.stopPropagation()}
                                    onClick={(event) => {
                                      event.preventDefault()
                                      event.stopPropagation()
                                      void deleteClipFile(clip, file)
                                    }}
                                  >
                                    {deletingFilePath === file.file_path ? <Loader2 className="size-3.5 animate-spin" /> : <Trash2 className="size-3.5" />}
                                  </Button>
                                </div>
                              )
                            })}
                          </PopoverContent>
                        </Popover>
                      )
                  })}
                </div>
                <div className="pointer-events-none absolute top-0 h-20 w-px bg-destructive" style={{ left: playheadLeft }} />
              </div>
            </div>
          </div>
        </div>
      </TooltipProvider>
    </LocaleContext.Provider>
  )
}
