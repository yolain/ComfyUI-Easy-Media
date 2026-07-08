import { describe, expect, it } from 'vitest'
import cameraCases from '../../../tests/fixtures/panorama_camera_cases.json'
import {
  DEFAULT_PANORAMA_VIEW,
  horizontalToVerticalFov,
  isValidPanoramaView,
  normalizePanoramaView,
  panoramaDirectionToUv,
  panoramaLookAt,
} from '@/lib/panorama-camera'

describe('panorama camera contract', () => {
  it('normalizes persisted view values', () => {
    expect(normalizePanoramaView({
      version: 1,
      projection: 'equirectangular',
      yaw: 540,
      pitch: 100,
      hfov: 10,
      aspect_ratio: 2,
    })).toEqual({
      version: 1,
      projection: 'equirectangular',
      yaw: -180,
      pitch: 100,
      hfov: 30,
      aspect_ratio: 2,
    })
  })

  it('wraps vertical rotation instead of clamping it at the poles', () => {
    expect(normalizePanoramaView({ ...DEFAULT_PANORAMA_VIEW, pitch: 270 }).pitch).toBe(-90)
  })

  it('converts horizontal FOV using the actual canvas aspect', () => {
    expect(horizontalToVerticalFov(90, 2)).toBeCloseTo(53.130102, 5)
  })

  it.each(cameraCases)('maps $name direction to equirectangular UV', ({ yaw, pitch, u, v }) => {
    expect(panoramaDirectionToUv(yaw, pitch)).toEqual({ u, v })
  })

  it('uses a right-handed look direction with positive yaw toward positive Z', () => {
    expect(panoramaLookAt(0, 0)).toEqual({ x: 1, y: 0, z: 0 })
    expect(panoramaLookAt(90, 0)).toMatchObject({ x: expect.closeTo(0), y: 0, z: 1 })
  })

  it('falls back from invalid persisted metadata', () => {
    expect(normalizePanoramaView({ version: 2 })).toEqual(DEFAULT_PANORAMA_VIEW)
    expect(isValidPanoramaView({ version: 2 })).toBe(false)
  })

  it('rejects invalid aspect ratios', () => {
    expect(isValidPanoramaView({
      ...DEFAULT_PANORAMA_VIEW,
      aspect_ratio: 0,
    })).toBe(false)
  })
})
