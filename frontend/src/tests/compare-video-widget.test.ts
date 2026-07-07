import { describe, expect, it } from 'vitest'
import { parseCompareVideoPayload } from '@/components/widgets/compareVideoWidget'

const source = { filename: 'source.mp4', type: 'temp' as const }
const output = { filename: 'output.mp4', type: 'temp' as const }

describe('compare video payload parsing', () => {
  it('reads a direct compare_videos payload', () => {
    expect(parseCompareVideoPayload({
      compare_videos: { source, output, frame_count: 12 },
    })).toMatchObject({ source, output, frame_count: 12 })
  })

  it('reads a payload nested in ComfyUI executed output', () => {
    expect(parseCompareVideoPayload({
      node: 8,
      output: { compare_videos: { source, output, fps: 24 } },
    })).toMatchObject({ source, output, fps: 24 })
  })

  it('reads a payload nested in a ui object', () => {
    expect(parseCompareVideoPayload({
      ui: { compare_videos: { source, duration: 1.5 } },
    })).toMatchObject({ source, duration: 1.5 })
  })

  it('accepts list-wrapped compare_videos values', () => {
    expect(parseCompareVideoPayload({
      output: { compare_videos: [{ source, output }] },
    })).toMatchObject({ source, output })
  })
})
