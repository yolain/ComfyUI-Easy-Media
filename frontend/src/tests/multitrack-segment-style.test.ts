import { describe, expect, it } from 'vitest'
import { getSegmentTrackPresentation } from '@/lib/multitrack-segment-style'

describe('multitrack segment style utilities', () => {
  it('uses video treatment only for video tracks', () => {
    expect(getSegmentTrackPresentation('video')).toMatchObject({
      showThumbnail: true,
      showWaveform: true,
      waveformSource: 'decoded-media',
    })

    expect(getSegmentTrackPresentation('audio')).toMatchObject({
      showThumbnail: false,
      showWaveform: true,
      waveformSource: 'decoded-media',
    })

    expect(getSegmentTrackPresentation('subtitle')).toMatchObject({
      showThumbnail: false,
      showWaveform: false,
      waveformSource: null,
    })

    expect(getSegmentTrackPresentation('task')).toMatchObject({
      backgroundColor: 'var(--multitrack-task-bg)',
      showThumbnail: false,
      showWaveform: false,
      waveformSource: null,
      textClassName: 'text-[8px]',
    })
  })
})
