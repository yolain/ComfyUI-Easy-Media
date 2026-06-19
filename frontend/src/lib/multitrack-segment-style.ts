import type { MultiTrackType } from '@/types/multitrack'

export interface SegmentTrackPresentation {
  backgroundColor: string
  backgroundColorStrong: string
  borderColor: string
  textColor: string
  textClassName: string
  showThumbnail: boolean
  showWaveform: boolean
  waveformSource: 'decoded-media' | null
}

const VIDEO_PRESENTATION: SegmentTrackPresentation = {
  backgroundColor: 'var(--multitrack-video-bg)',
  backgroundColorStrong: 'var(--multitrack-video-bg-strong)',
  borderColor: 'var(--multitrack-video-border)',
  textColor: 'var(--multitrack-video-text)',
  textClassName: 'text-[10px]',
  showThumbnail: true,
  showWaveform: true,
  waveformSource: 'decoded-media',
}

const AUDIO_PRESENTATION: SegmentTrackPresentation = {
  backgroundColor: 'var(--multitrack-audio-bg)',
  backgroundColorStrong: 'var(--multitrack-audio-bg-strong)',
  borderColor: 'var(--multitrack-audio-border)',
  textColor: 'var(--multitrack-audio-text)',
  textClassName: 'text-[10px]',
  showThumbnail: false,
  showWaveform: true,
  waveformSource: 'decoded-media',
}

const TASK_PRESENTATION: SegmentTrackPresentation = {
  backgroundColor: 'var(--multitrack-task-bg)',
  backgroundColorStrong: 'var(--multitrack-task-bg-strong)',
  borderColor: 'var(--multitrack-task-border)',
  textColor: 'var(--multitrack-task-text)',
  textClassName: 'text-[8px]',
  showThumbnail: false,
  showWaveform: false,
  waveformSource: null,
}

export function getSegmentTrackPresentation(trackType: MultiTrackType): SegmentTrackPresentation {
  if (trackType === 'video') return VIDEO_PRESENTATION
  if (trackType === 'audio') return AUDIO_PRESENTATION
  return TASK_PRESENTATION
}
