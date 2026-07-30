import { describe, expect, it } from 'vitest'
import { createSubtitleSegmentsFromSrt } from '@/lib/subtitle-srt'

describe('createSubtitleSegmentsFromSrt', () => {
  it('matches SRT cue timing to the dropped timeline range', () => {
    const segments = createSubtitleSegmentsFromSrt(
      [
        '\uFEFF1',
        '00:00:02,000 --> 00:00:03,500',
        'First line',
        'continues',
        '',
        '2',
        '00:00:04.000 --> 00:00:05,000',
        'Second line',
      ].join('\r\n'),
      {
        color: '#9D4937',
        createId: (index) => `subtitle-${index}`,
        frameRate: 24,
        startFrame: 120,
        subtitleStyle: {
          font_size: 32,
          color: '#ffffff',
          background_color: 'transparent',
          background_opacity: 0.7,
          x: 0.5,
          y: 0.85,
          width: 0.9,
        },
      },
    )

    expect(segments).toEqual([
      expect.objectContaining({
        id: 'subtitle-0',
        start_frame: 120,
        end_frame: 156,
        content: expect.objectContaining({
          media_type: 'subtitle',
          text: 'First line\ncontinues',
        }),
      }),
      expect.objectContaining({
        id: 'subtitle-1',
        start_frame: 168,
        end_frame: 192,
        content: expect.objectContaining({
          media_type: 'subtitle',
          text: 'Second line',
        }),
      }),
    ])
    expect(segments[0]?.content.subtitle_style).not.toBe(segments[1]?.content.subtitle_style)
  })

  it('rejects SRT files without valid subtitle cues', () => {
    expect(() => createSubtitleSegmentsFromSrt('not an SRT file', {
      color: '#9D4937',
      createId: () => 'subtitle',
      frameRate: 24,
      startFrame: 0,
      subtitleStyle: {
        font_size: 32,
        color: '#ffffff',
        background_color: 'transparent',
        background_opacity: 0.7,
        x: 0.5,
        y: 0.85,
        width: 0.9,
      },
    })).toThrow('No valid SRT subtitle cues found')
  })

  it('sorts out-of-order cues before matching them to the dropped range', () => {
    const segments = createSubtitleSegmentsFromSrt(
      [
        '2',
        '00:00:04,000 --> 00:00:05,000',
        'Later',
        '',
        '1',
        '00:00:02,000 --> 00:00:03,000',
        'Earlier',
      ].join('\n'),
      {
        color: '#9D4937',
        createId: (index) => `subtitle-${index}`,
        frameRate: 24,
        startFrame: 120,
        subtitleStyle: {
          font_size: 32,
          color: '#ffffff',
          background_color: 'transparent',
          background_opacity: 0.7,
          x: 0.5,
          y: 0.85,
          width: 0.9,
        },
      },
    )

    expect(segments.map((segment) => ({
      start: segment.start_frame,
      end: segment.end_frame,
      text: segment.content.text,
    }))).toEqual([
      { start: 120, end: 144, text: 'Earlier' },
      { start: 168, end: 192, text: 'Later' },
    ])
  })
})
