import { describe, expect, it } from 'vitest'
import { canEnableSharedTaskImage, synchronizeSharedTaskImages, taskImageReferenceIndex, togglePreviousFrame } from '@/lib/task-image-utils'
import type { MultiTrack, MultiTrackTaskImage } from '@/types/multitrack'

describe('previous frame reference', () => {
  it('retains its identity and slot when toggled and when shared references synchronize', () => {
    const clean: MultiTrackTaskImage = { id: 'clean', source_type: 'input', file_path: 'clean.png', shared_reference: true }
    const added = togglePreviousFrame([clean])
    const previous = added[1]
    const images = [previous, clean]
    const disabled = togglePreviousFrame(images)
    expect(disabled[0]).toEqual({ ...previous, muted: true })
    expect(togglePreviousFrame(disabled)[0]).toEqual({ ...previous, muted: false })
    expect(taskImageReferenceIndex(images, previous.id)).toBe(0)
    expect(canEnableSharedTaskImage([], previous)).toBe(false)
    const tracks = [{ id: 'tasks', type: 'task', segments: [
      { id: 'first', content: { images: [clean] } }, { id: 'second', content: { images } },
    ] }] as MultiTrack[]
    const result = synchronizeSharedTaskImages(tracks)
    expect(result[0].segments[1].content.images?.map((image) => image.id)).toEqual([previous.id, clean.id])
    expect(result[0].segments[0].content.images).toHaveLength(1)
  })
})
