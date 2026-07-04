export type MultiTrackType = 'task' | 'video' | 'audio' | 'subtitle'

export type MultiTrackTaskMode = 'default' | 'ref' | 'edit'

export type MultiTrackMediaType = 'image' | 'audio' | 'video' | 'subtitle' | 'none'

export type MultiTrackSourceType = 'preset' | 'input' | 'output' | 'local' | 'url' | 'slot'

export interface MultiTrackPanoramaView {
  version: 1
  projection: 'equirectangular'
  yaw: number
  pitch: number
  hfov: number
  aspect_ratio: number
}

export interface MultiTrackTaskImage {
  id: string
  source_type?: MultiTrackSourceType
  file_path?: string
  local_path?: string
  url?: string
  slot_name?: string
  file_name?: string
  panorama_view?: MultiTrackPanoramaView
}

export interface MultiTrackSegmentContent {
  media_type: MultiTrackMediaType
  source_type?: MultiTrackSourceType
  file_path?: string
  local_path?: string
  url?: string
  slot_name?: string
  file_name?: string
  text?: string
  user_prompt?: string
  system_prompt?: string
  task_mode?: MultiTrackTaskMode
  images?: MultiTrackTaskImage[]
  duration?: number
  volume_db?: number
  muted?: boolean
  speed?: number
  media_index?: number
  subtitle_style?: MultiTrackSubtitleStyle
}

export interface MultiTrackSubtitleStyle {
  font_size: number
  color: string
  outline_color?: string
  background_color: string
  x: number
  y: number
  width: number
}

export interface MultiTrackSegment {
  id: string
  start_frame: number
  end_frame: number
  origin_start_frame?: number
  content: MultiTrackSegmentContent
  color: string
}

export interface MultiTrack {
  id: string
  name: string
  type: MultiTrackType
  task_mode?: MultiTrackTaskMode
  color: string
  muted: boolean
  solo?: boolean
  volume_db?: number
  locked: boolean
  media_index?: number
  segments: MultiTrackSegment[]
}

export interface TrackData {
  tracks: MultiTrack[]
  total_length: number
  frame_rate: number
  muted?: boolean
  volume_db?: number
}

export interface TracksInfoMediaItem {
  index: number
  track_id: string
  segment_id: string
  source_type?: MultiTrackSourceType
  file_path?: string
  local_path?: string
  url?: string
  slot_name?: string
  file_name?: string
  duration?: number
  panorama_view?: MultiTrackPanoramaView
}

export interface TracksInfo {
  total_length: number
  frame_rate: number
  muted?: boolean
  volume_db?: number
  width: number
  height: number
  tracks: MultiTrack[]
  media: {
    images: TracksInfoMediaItem[]
    audio: TracksInfoMediaItem[]
    video: TracksInfoMediaItem[]
  }
}
