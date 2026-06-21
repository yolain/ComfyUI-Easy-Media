import type { MultiTrackPanoramaView } from '@/types/multitrack'

export const DEFAULT_PANORAMA_VIEW: MultiTrackPanoramaView = {
  version: 1,
  projection: 'equirectangular',
  yaw: 0,
  pitch: 0,
  hfov: 90,
  aspect_ratio: 1,
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value))
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value)
}

export function normalizeYaw(yaw: number): number {
  return ((yaw + 180) % 360 + 360) % 360 - 180
}

export function isValidPanoramaView(value: unknown): value is MultiTrackPanoramaView {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const candidate = value as Record<string, unknown>
  return candidate.version === 1 &&
    candidate.projection === 'equirectangular' &&
    isFiniteNumber(candidate.yaw) &&
    isFiniteNumber(candidate.pitch) &&
    isFiniteNumber(candidate.hfov) &&
    isFiniteNumber(candidate.aspect_ratio) &&
    candidate.aspect_ratio > 0
}

export function normalizePanoramaView(value: unknown): MultiTrackPanoramaView {
  if (!isValidPanoramaView(value)) return { ...DEFAULT_PANORAMA_VIEW }
  return {
    version: 1,
    projection: 'equirectangular',
    yaw: normalizeYaw(value.yaw),
    pitch: normalizeYaw(value.pitch),
    hfov: clamp(value.hfov, 30, 120),
    aspect_ratio: value.aspect_ratio,
  }
}

export function horizontalToVerticalFov(hfov: number, aspectRatio: number): number {
  if (!Number.isFinite(aspectRatio) || aspectRatio <= 0) {
    throw new RangeError('Panorama aspect ratio must be a positive finite number')
  }
  const horizontalRadians = clamp(hfov, 30, 120) * Math.PI / 180
  return 2 * Math.atan(Math.tan(horizontalRadians / 2) / aspectRatio) * 180 / Math.PI
}

export function panoramaDirectionToUv(yaw: number, pitch: number): { u: number; v: number } {
  return {
    u: normalizeYaw(yaw) / 360 + 0.5,
    v: 0.5 - clamp(pitch, -90, 90) / 180,
  }
}

export function panoramaLookAt(yaw: number, pitch: number): { x: number; y: number; z: number } {
  const yawRadians = normalizeYaw(yaw) * Math.PI / 180
  const pitchRadians = normalizeYaw(pitch) * Math.PI / 180
  const horizontalLength = Math.cos(pitchRadians)
  return {
    x: horizontalLength * Math.cos(yawRadians),
    y: Math.sin(pitchRadians),
    z: horizontalLength * Math.sin(yawRadians),
  }
}
