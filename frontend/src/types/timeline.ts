export type MediaSourceType = 'input' | 'local' | 'url' | 'slot' | 'output'

export interface Marker {
  id: string
  frame: number
  label?: string
}

export interface Segment {
  id: string
  start_frame: number
  end_frame: number
  content: SegmentContent
  color: string
  markers?: Marker[]
}

// ---- Prompt ----
export interface PromptContent {
  text: string
}

export interface PromptSegment extends Segment {
  content: PromptContent
}

// ---- Audio ----
export interface AudioContent {
  source_type: MediaSourceType
  file_path?: string
  local_path?: string
  url?: string
  slot_name?: string
  file_name: string
  duration?: number
}

export interface AudioSegment extends Segment {
  content: AudioContent
  /** Original placement bounds; trim handles cannot exceed these limits */
  origin_start_frame?: number
  origin_end_frame?: number
}

// ---- Image ----
export interface ImageContent {
  source_type: MediaSourceType
  file_path?: string
  local_path?: string
  url?: string
  slot_name?: string
  file_name: string
}

export interface ImageSegment extends Segment {
  content: ImageContent
}

// ---- Video ----
export interface VideoContent {
  source_type: MediaSourceType
  file_path?: string
  local_path?: string
  url?: string
  file_name: string
  duration?: number
}

export interface VideoSegment extends Segment {
  content: VideoContent
}

// ---- Maintain ----
/** A single image entry in a MaintainSegment's images array */
export interface ImageItem {
  source_type: MediaSourceType
  file_path?: string
  local_path?: string
  url?: string
  slot_name?: string
  file_name: string
  /** Optional position within the parent segment's frame span (0-based, relative to segment start) */
  start_frame?: number
  end_frame?: number
}

/** Generation type for a maintain segment */
export type MaintainType = 'flf' | 'fmlf' | 'ref'

export interface MaintainContent {
  text: string
  images: ImageItem[]
  type: MaintainType
}

export interface MaintainSegment extends Segment {
  content: MaintainContent
}

export type SegmentContent = PromptContent | AudioContent | ImageContent | VideoContent | MaintainContent

export type TrackType = 'prompt' | 'audio' | 'image' | 'video' | 'maintain'

export interface Track {
  id: string
  name: string
  type: TrackType
  color: string
  muted: boolean
  locked: boolean
  segments: Segment[]
}

export interface TimelineData {
  tracks: Track[]
  total_length: number  // in frames
  frame_rate: number
}

export type TimeDisplayFormat = 'frames' | 'seconds'
