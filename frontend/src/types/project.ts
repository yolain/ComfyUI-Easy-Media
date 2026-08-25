export type ProjectContinuityMode = 'shot' | 'context'

export interface ProjectClip {
  id: string
  index: number
  file_path: string
  file_name: string
  media_revision?: string
  source_start_frame: number
  source_end_frame: number
  source_frame_count: number
  continuity_mode: ProjectContinuityMode
  enabled: boolean
}

export interface ProjectData {
  project_name: string
  width: number
  height: number
  frame_rate: number
  clips: ProjectClip[]
  auto_combine: boolean
  updated_at?: number
}

export const DEFAULT_PROJECT_DATA: ProjectData = {
  project_name: 'default',
  width: 0,
  height: 0,
  frame_rate: 24,
  clips: [],
  auto_combine: true,
}
