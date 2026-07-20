import { describe, expect, it } from 'vitest'
import { getTrackSegmentGaps } from '@/components/widgets/multitrack/TrackGapAddButton'

describe('getTrackSegmentGaps', () => {
  it('finds uncovered ranges after merging overlapping segments', () => {
    expect(getTrackSegmentGaps([
      { start_frame: 0, end_frame: 100 },
      { start_frame: 20, end_frame: 30 },
      { start_frame: 120, end_frame: 140 },
    ])).toEqual([{ startFrame: 100, endFrame: 120 }])
  })

  it('includes the gap before the first segment but leaves trailing space to the append button', () => {
    expect(getTrackSegmentGaps([
      { start_frame: 24, end_frame: 48 },
      { start_frame: 48, end_frame: 72 },
    ])).toEqual([{ startFrame: 0, endFrame: 24 }])
  })
})
