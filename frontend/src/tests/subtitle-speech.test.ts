import { describe, expect, it, vi } from 'vitest'
import {
  applySubtitleSpeechAudio,
  requestSubtitleSpeechAudio,
} from '@/lib/subtitle-speech'
import type { MultiTrack, TrackData } from '@/types/multitrack'

function audioTrack(id: string, segments: MultiTrack['segments'] = []): MultiTrack {
  return {
    id,
    name: id,
    type: 'audio',
    color: 'var(--highlight)',
    muted: false,
    solo: false,
    volume_db: 0,
    locked: false,
    segments,
  }
}

function baseData(tracks: MultiTrack[]): TrackData {
  return {
    tracks,
    total_length: 120,
    frame_rate: 24,
    muted: false,
    volume_db: 0,
  }
}

describe('subtitle speech audio', () => {
  it('requests generated speech with subtitle text and speech settings', async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        file_path: 'easy_media/hello.wav',
        source_type: 'output',
        message: 'done',
      }),
    })
    vi.stubGlobal('fetch', fetchMock)

    await requestSubtitleSpeechAudio({
      text: '字幕123',
      model: 'VoxCPM2',
      prompt: '四川话',
      cfg: 2.3,
      steps: 12,
      referenceAudio: 'refs/voice.wav',
      referenceAudioSourceType: 'input',
    })

    expect(fetchMock).toHaveBeenCalledWith('/easy-media/subtitles/speech', expect.objectContaining({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    }))
    expect(JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body))).toMatchObject({
      text: '字幕123',
      model: 'VoxCPM2',
      prompt: '四川话',
      cfg: 2.3,
      steps: 12,
      reference_audio_path: 'refs/voice.wav',
      reference_audio_source_type: 'input',
    })
    vi.unstubAllGlobals()
  })

  it('adds generated speech to an existing free audio track starting at the subtitle start', () => {
    const data = baseData([
      audioTrack('audio-a', [{
        id: 'existing',
        start_frame: 80,
        end_frame: 100,
        color: 'var(--highlight)',
        content: { media_type: 'audio' },
      }]),
    ])

    const updated = applySubtitleSpeechAudio(data, {
      subtitleSegmentId: 'subtitle-a',
      startFrame: 24,
      endFrame: 72,
      filePath: 'easy_media/hello.wav',
      duration: 1,
    })

    const generated = updated.tracks[0].segments.find((segment) => segment.content.file_path === 'easy_media/hello.wav')
    expect(updated.tracks).toHaveLength(1)
    expect(generated).toMatchObject({
      start_frame: 24,
      end_frame: 48,
      content: {
        media_type: 'audio',
        source_type: 'output',
        file_path: 'easy_media/hello.wav',
        duration: 1,
      },
    })
  })

  it('uses generated speech duration when checking for audio track overlap', () => {
    const data = baseData([
      audioTrack('audio-a', [{
        id: 'after-generated-speech',
        start_frame: 60,
        end_frame: 70,
        color: 'var(--highlight)',
        content: { media_type: 'audio' },
      }]),
    ])

    const updated = applySubtitleSpeechAudio(data, {
      subtitleSegmentId: 'subtitle-a',
      startFrame: 24,
      endFrame: 72,
      filePath: 'easy_media/hello.wav',
      duration: 1,
    })

    expect(updated.tracks).toHaveLength(1)
    expect(updated.tracks[0].segments).toEqual(expect.arrayContaining([
      expect.objectContaining({
        start_frame: 24,
        end_frame: 48,
        content: expect.objectContaining({
          source_type: 'output',
          file_path: 'easy_media/hello.wav',
        }),
      }),
    ]))
  })

  it('creates a new audio track when existing audio overlaps generated speech duration', () => {
    const data = baseData([
      audioTrack('audio-a', [{
        id: 'overlap',
        start_frame: 40,
        end_frame: 50,
        color: 'var(--highlight)',
        content: { media_type: 'audio' },
      }]),
    ])

    const updated = applySubtitleSpeechAudio(data, {
      subtitleSegmentId: 'subtitle-a',
      startFrame: 24,
      endFrame: 72,
      filePath: 'easy_media/hello.wav',
      duration: 1.5,
    })

    expect(updated.tracks).toHaveLength(2)
    expect(updated.tracks[1]).toMatchObject({
      type: 'audio',
      segments: [expect.objectContaining({
        start_frame: 24,
        end_frame: 60,
        content: expect.objectContaining({
          source_type: 'output',
          file_path: 'easy_media/hello.wav',
        }),
      })],
    })
  })
})
