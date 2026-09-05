export type MultiTrackType = 'task' | 'video' | 'audio' | 'subtitle'

export type MultiTrackTaskMode = 'default' | 'l2v' | 'ref' | 'edit'

export type MultiTrackContinuityMode = 'shot' | 'context' | 'context_swap'

export type MultiTrackRefImageSize = 'match' | 'max'

export type MultiTrackUserPromptVariant = 'a' | 'b'

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
  shared_reference?: boolean
  shared_reference_copy?: boolean
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
  user_prompt_b?: string
  user_prompt_variant?: MultiTrackUserPromptVariant
  system_prompt?: string
  task_mode?: MultiTrackTaskMode
  continuity_mode?: MultiTrackContinuityMode
  ref_image_size?: MultiTrackRefImageSize
  images?: MultiTrackTaskImage[]
  duration?: number
  volume_db?: number
  muted?: boolean
  speed?: number
  media_index?: number
  shared_reference?: boolean
  shared_media_index?: number
  /** @deprecated Migrated to shared_reference when legacy workflows are loaded. */
  speaker_reference?: boolean
  subtitle_style?: MultiTrackSubtitleStyle
  subtitle_speech?: MultiTrackSubtitleSpeechSettings
}

export interface MultiTrackSubtitleSpeechSettings {
  model: 'VoxCPM2'
  prompt: string
  cfg: number
  steps: number
  referenceAudio: string
  referenceAudioSourceType?: Extract<MultiTrackSourceType, 'input' | 'output' | 'local'>
}

export interface MultiTrackSubtitleStyle {
  font_size: number
  color: string
  outline_color?: string
  background_color: string
  background_opacity: number
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

export interface MultiTrackTaskMarker {
  id: string
  frame: number
}

export interface MultiTrack {
  id: string
  name: string
  type: MultiTrackType
  task_mode?: MultiTrackTaskMode
  color: string
  muted: boolean
  solo?: boolean
  visible?: boolean
  volume_db?: number
  locked: boolean
  audio_locked?: boolean
  media_index?: number
  segments: MultiTrackSegment[]
}

export interface TrackData {
  tracks: MultiTrack[]
  total_length: number
  frame_rate: number
  task_markers?: MultiTrackTaskMarker[]
  task_overview?: boolean
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
  task_markers?: MultiTrackTaskMarker[]
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
