import { describe, expect, it } from 'vitest'
import {
  applySmartSplit,
  applySmartSplitToMatchingTasks,
  hasMatchingTaskSegment,
  splitTrackSegmentAtFrame,
  type SmartSplitResult,
} from '@/lib/smart-split'
import type { TrackData } from '@/types/multitrack'

function result(ranges: Array<[number, number]>): SmartSplitResult {
  return {
    ranges,
  }
}

function trackData(): TrackData {
  return {
    frame_rate: 24,
    total_length: 300,
    tracks: [
      {
        id: 'task-track', name: 'Task', type: 'task', color: 'var(--primary)', muted: false, locked: false,
        segments: [
          {
            id: 'linked-task', start_frame: 24, end_frame: 264, color: 'var(--primary)',
            content: {
              media_type: 'none', text: 'linked', system_prompt: 'system',
              images: [{ id: 'image', source_type: 'input', file_path: 'image.png' }],
            },
          },
          { id: 'other-task', start_frame: 0, end_frame: 24, color: 'var(--primary)', content: { media_type: 'none', text: 'other' } },
        ],
      },
      {
        id: 'video-track', name: 'Video', type: 'video', color: 'var(--secondary)', muted: false, locked: false,
        segments: [{
          id: 'video', start_frame: 24, end_frame: 264, color: 'var(--secondary)',
          content: { media_type: 'video', source_type: 'input', file_path: 'clip.mp4' },
        }],
      },
    ],
  }
}

describe('applySmartSplit', () => {
  it('maps detector frames to timeline frames and splits linked task ranges', () => {
    const updated = applySmartSplit(trackData(), 'video', result([[0, 60], [60, 120], [120, 240]]))
    const video = updated.tracks.find((track) => track.type === 'video')!
    const task = updated.tracks.find((track) => track.type === 'task')!

    expect(video.segments.map(({ start_frame, end_frame }) => [start_frame, end_frame])).toEqual([
      [24, 84], [84, 144], [144, 264],
    ])
    expect(video.segments.map((segment) => segment.origin_start_frame)).toEqual([24, 24, 24])
    expect(task.segments.filter((segment) => segment.content.text === 'linked')
      .map(({ start_frame, end_frame }) => [start_frame, end_frame])).toEqual([
      [24, 84], [84, 144], [144, 264],
    ])
    expect(task.segments.find((segment) => segment.id === 'other-task')).toMatchObject({ start_frame: 0, end_frame: 24 })
    expect(task.segments.every((segment) => segment.origin_start_frame === undefined)).toBe(true)
    const clonedTasks = task.segments.filter((segment) => segment.content.text === 'linked')
    expect(clonedTasks.every((segment) => (
      segment.content.system_prompt === 'system' && segment.content.images?.[0]?.file_path === 'image.png'
    ))).toBe(true)
    expect(clonedTasks[0].content.images).not.toBe(clonedTasks[1].content.images)
  })

  it('preserves an existing source trim origin when splitting an already-trimmed video', () => {
    const data = trackData()
    data.tracks[1].segments[0].origin_start_frame = 0

    const updated = applySmartSplit(data, 'video', result([[0, 60], [60, 240]]))

    expect(updated.tracks[1].segments.map((segment) => segment.origin_start_frame)).toEqual([0, 0, 0])
    expect(updated.tracks[1].segments.map(({ start_frame, end_frame }) => [start_frame, end_frame])).toEqual([
      [24, 60], [60, 240], [240, 264],
    ])
  })

  it('leaves data unchanged when no internal cut was detected', () => {
    const data = trackData()
    expect(applySmartSplit(data, 'video', result([[0, 240]]))).toBe(data)
  })

  it('splits only matching task segments and leaves the video unchanged', () => {
    const data = trackData()
    expect(hasMatchingTaskSegment(data, 'video')).toBe(true)

    const updated = applySmartSplitToMatchingTasks(data, 'video', result([[0, 120], [120, 240]]))

    expect(updated.tracks[1].segments).toEqual(data.tracks[1].segments)
    expect(updated.tracks[0].segments.filter((segment) => segment.content.text === 'linked')
      .map(({ start_frame, end_frame }) => [start_frame, end_frame])).toEqual([
      [24, 144], [144, 264],
    ])
  })

  it('manually cuts media with a shared source origin and clones task content', () => {
    const data = trackData()
    const videoResult = splitTrackSegmentAtFrame(data, 'video', 84)
    expect(videoResult.tracks[1].segments.map((segment) => ({
      range: [segment.start_frame, segment.end_frame],
      origin: segment.origin_start_frame,
    }))).toEqual([
      { range: [24, 84], origin: 24 },
      { range: [84, 264], origin: 24 },
    ])

    const taskResult = splitTrackSegmentAtFrame(data, 'linked-task', 84)
    const tasks = taskResult.tracks[0].segments.filter((segment) => segment.content.text === 'linked')
    expect(tasks).toHaveLength(2)
    expect(tasks.every((segment) => segment.content.images?.[0]?.file_path === 'image.png')).toBe(true)
    expect(tasks[0].content.images).not.toBe(tasks[1].content.images)
  })
})
