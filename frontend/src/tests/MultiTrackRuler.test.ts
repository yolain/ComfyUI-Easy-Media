import { describe, expect, it } from 'vitest'
import { buildTicks } from '@/components/widgets/multitrack/MultiTrackRuler'

describe('MultiTrackRuler', () => {
  it('keeps time labels sparse enough for narrow timelines', () => {
    const ticks = buildTicks(40 * 24, 24, 350)
    const labeledTicks = ticks.filter((tick) => tick.label)

    expect(labeledTicks.map((tick) => tick.label)).toEqual([
      '00:00:00',
      '00:10:00',
      '00:20:00',
      '00:30:00',
      '00:40:00',
    ])
  })

  it('does not add an end label when it would overlap the previous label', () => {
    const ticks = buildTicks(22 * 24, 24, 260)
    const labels = ticks.filter((tick) => tick.label).map((tick) => tick.label)

    expect(labels).toEqual(['00:00:00', '00:05:00', '00:10:00', '00:15:00', '00:20:00'])
  })

  it('uses frame precision to avoid duplicate labels inside the same second', () => {
    const labels = buildTicks(3 * 24, 24, 720)
      .filter((tick) => tick.label)
      .map((tick) => tick.label)

    expect(new Set(labels).size).toBe(labels.length)
    expect(labels).toContain('00:01:00')
  })
})
