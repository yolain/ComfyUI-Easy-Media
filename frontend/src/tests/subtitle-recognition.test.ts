import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  applySubtitleRecognition,
  requestSubtitleRecognition,
  type SubtitleRecognitionResult,
} from '@/lib/subtitle-recognition'
import type { TrackData } from '@/types/multitrack'

function recognitionResult(): SubtitleRecognitionResult {
  return {
    segments: [
      { start: 0.5, end: 1.25, text: 'First line' },
      { start: 1.25, end: 2, text: 'Second line' },
    ],
  }
}

function trackData(): TrackData {
  return {
    frame_rate: 24,
    total_length: 240,
    tracks: [
      {
        id: 'video-track',
        name: 'Video',
        type: 'video',
        color: 'var(--primary)',
        muted: false,
        locked: false,
        segments: [{
          id: 'video',
          start_frame: 24,
          end_frame: 96,
          color: 'var(--primary)',
          content: { media_type: 'video', source_type: 'input', file_path: 'clip.mp4' },
        }],
      },
    ],
  }
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe('applySubtitleRecognition', () => {
  it('creates a subtitle track from recognized timestamped text', () => {
    const updated = applySubtitleRecognition(trackData(), 'video', recognitionResult())
    const subtitleTrack = updated.tracks.find((track) => track.type === 'subtitle')

    expect(subtitleTrack).toMatchObject({
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
    })
    expect(subtitleTrack?.segments.map((segment) => ({
      start: segment.start_frame,
      end: segment.end_frame,
      text: segment.content.text,
      color: segment.color,
      style: segment.content.subtitle_style,
    }))).toEqual([
      {
        start: 36,
        end: 54,
        text: 'First line',
        color: '#9D4937',
        style: {
          font_size: 12,
          color: '#ffffff',
          outline_color: '#000000',
          background_color: 'rgba(0, 0, 0, 0)',
          background_opacity: 0.7,
          x: 0.125,
          y: 0.8,
          width: 0.75,
        },
      },
      {
        start: 54,
        end: 72,
        text: 'Second line',
        color: '#9D4937',
        style: {
          font_size: 12,
          color: '#ffffff',
          outline_color: '#000000',
          background_color: 'rgba(0, 0, 0, 0)',
          background_opacity: 0.7,
          x: 0.125,
          y: 0.8,
          width: 0.75,
        },
      },
    ])
  })

  it('appends recognition segments into an existing subtitle track', () => {
    const data = trackData()
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [{
        id: 'existing-subtitle',
        start_frame: 0,
        end_frame: 12,
        color: '#9D4937',
        content: { media_type: 'subtitle', text: 'Existing' },
      }],
    })

    const updated = applySubtitleRecognition(data, 'video', recognitionResult())

    expect(updated.tracks.find((track) => track.type === 'subtitle')?.segments).toHaveLength(3)
  })
})

describe('requestSubtitleRecognition', () => {
  it('sends the segment frame window so the backend can recognize only the trimmed source span', async () => {
    const data = trackData()
    const segment = data.tracks[0].segments[0]
    segment.start_frame = 48
    segment.end_frame = 96
    segment.origin_start_frame = 24
    const fetchMock = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({ segments: [] }),
    } as Response)

    await requestSubtitleRecognition(segment, data.frame_rate)

    const body = JSON.parse(String(fetchMock.mock.calls[0][1]?.body))
    expect(body).toMatchObject({
      media_type: 'video',
      file_path: 'clip.mp4',
      fps: 24,
      start_frame: 48,
      end_frame: 96,
      origin_start_frame: 24,
    })
  })
})
