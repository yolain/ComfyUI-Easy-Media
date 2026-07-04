import { describe, expect, it } from 'vitest'
import {
  addDefaultTaskSegmentIfRangeEmpty,
  calculateTotalLength,
  cloneMultiTrackSegment,
  collectMultiTrackPreviewResolutionInput,
  createDefaultTrackData,
  createMultiTrackAudioContent,
  createMultiTrackVideoContent,
  deleteSegmentsWithLinkedTasks,
  deleteSegmentWithLinkedTasks,
  distributeMultiTrackSegmentsEvenly,
  formatMultiTrackDurationTimecode,
  formatMultiTrackTime,
  getActivePreviewVideoSegment,
  getActivePreviewAudioSources,
  getMultiTrackTaskModeLabel,
  getSelectedMultiTrackSegment,
  frameToSeconds,
  getSegmentDragPlaceholder,
  getSegmentDragPreviewSegments,
  MULTITRACK_TASK_MODES,
  moveSelectedSegments,
  moveSegmentBetweenCompatibleTracks,
  multiTrackDbToLinearGain,
  normalizeTrackData,
  parseMultiTrackDurationTimecode,
  parseMultiTrackPreviewResolution,
  remapTrackDataFrameRate,
  secondsToFrame,
  snapTimeToFrame,
  updateMultiTrackSegmentContent,
  updateMultiTrackSegmentDuration,
} from '@/lib/multitrack-utils'

describe('multitrack utilities', () => {
  it('evenly distributes multitrack segments across a half-open frame range', () => {
    const source = {
      id: 'source',
      start_frame: 0,
      end_frame: 1,
      color: 'var(--multitrack-task-bg)',
      content: { media_type: 'none' as const, task_mode: 'default' as const },
    }
    const segments = [
      { ...source, id: 'third', start_frame: 8, end_frame: 10 },
      { ...source, id: 'first', start_frame: 0, end_frame: 2 },
      { ...source, id: 'second', start_frame: 3, end_frame: 7 },
    ]

    expect(distributeMultiTrackSegmentsEvenly(segments, 10).map((segment) => ({
      id: segment.id,
      start_frame: segment.start_frame,
      end_frame: segment.end_frame,
    }))).toEqual([
      { id: 'first', start_frame: 0, end_frame: 4 },
      { id: 'second', start_frame: 4, end_frame: 7 },
      { id: 'third', start_frame: 7, end_frame: 10 },
    ])
  })

  it('deep-clones a segment after its source and shifts following segments', () => {
    const source = {
      id: 'source',
      start_frame: 0,
      end_frame: 3,
      color: 'var(--multitrack-task-bg)',
      content: {
        media_type: 'none' as const,
        task_mode: 'default' as const,
        images: [{ id: 'image', source_type: 'input' as const, file_path: 'image.png' }],
      },
    }
    const following = { ...source, id: 'following', start_frame: 3, end_frame: 5 }
    const result = cloneMultiTrackSegment([source, following], source.id)

    expect(result).not.toBeNull()
    expect(result?.addedDuration).toBe(3)
    expect(result?.segments.map((segment) => [segment.id, segment.start_frame, segment.end_frame])).toEqual([
      ['source', 0, 3],
      [result?.clonedSegmentId, 3, 6],
      ['following', 6, 8],
    ])
    expect(result?.segments[1].content).toEqual(source.content)
    expect(result?.segments[1].content).not.toBe(source.content)
  })

  it('formats and validates minute-second-frame duration timecodes', () => {
    expect(formatMultiTrackDurationTimecode(65.5, 24)).toBe('01:05:12')
    expect(parseMultiTrackDurationTimecode('01:05:12', 24)).toBe(65.5)
    expect(parseMultiTrackDurationTimecode('00:60:00', 24)).toBeNull()
    expect(parseMultiTrackDurationTimecode('00:01:24', 24)).toBeNull()
    expect(parseMultiTrackDurationTimecode('0:01:00', 24)).toBeNull()
    expect(parseMultiTrackDurationTimecode('00:00:00', 24)).toBeNull()
  })

  it('serializes direct video URLs as URL media', () => {
    expect(createMultiTrackVideoContent('https://example.com/clips/shot.mp4', 'input')).toEqual({
      media_type: 'video',
      source_type: 'url',
      file_path: undefined,
      local_path: undefined,
      url: 'https://example.com/clips/shot.mp4',
      file_name: 'shot.mp4',
    })
  })

  it('serializes connected audio slots as slot media', () => {
    expect(createMultiTrackAudioContent('__slot__:audio2', 'input')).toMatchObject({
      media_type: 'audio',
      source_type: 'slot',
      slot_name: 'audio2',
      file_name: 'audio2',
      volume_db: 0,
    })
  })

  it('creates default data with task and video tracks', () => {
    const data = createDefaultTrackData()

    expect(data.frame_rate).toBe(24)
    expect(data.total_length).toBe(120)
    expect(data.muted).toBe(false)
    expect(data.volume_db).toBe(0)
    expect(data.tracks.every((track) => track.solo === false && track.volume_db === 0)).toBe(true)
    expect(data.tracks.map((track) => track.type)).toEqual(['task', 'video'])
    expect(data.tracks.map((track) => track.name)).toEqual(['Task 0', 'Video 0'])
    expect(data.tracks[0].task_mode).toBe('default')
  })

  it('uses only volume_db and muted for normalized audio settings', () => {
    const normalized = normalizeTrackData({
      muted: false,
      volume: 0.5,
      volume_db: -12,
      tracks: [{
        id: 'audio',
        name: 'Audio 1',
        type: 'audio',
        color: 'var(--multitrack-audio-bg)',
        muted: false,
        locked: false,
        segments: [{
          id: 'audio-segment',
          start_frame: 0,
          end_frame: 24,
          color: 'var(--multitrack-audio-bg)',
          content: { media_type: 'audio', volume: 0, volume_db: -4, muted: false },
        }],
      }],
      total_length: 24,
      frame_rate: 24,
    } as unknown as Parameters<typeof normalizeTrackData>[0])

    expect(normalized.volume_db).toBe(-12)
    expect('volume' in normalized).toBe(false)
    expect(normalized.tracks[0]).toMatchObject({ name: 'Audio 0', solo: false, volume_db: 0 })
    expect(normalized.tracks[0].segments[0].content).toMatchObject({ muted: false, volume_db: -4 })
    expect('volume' in normalized.tracks[0].segments[0].content).toBe(false)
    expect(multiTrackDbToLinearGain(6)).toBeCloseTo(1.995, 2)
  })

  it('maintains task modes separately from media track types', () => {
    expect(MULTITRACK_TASK_MODES).toEqual(['default', 'ref', 'edit'])
    expect(getMultiTrackTaskModeLabel('default', (key) => key)).toBe('multitrackTaskModes.default')
    expect(getMultiTrackTaskModeLabel('ref', (key) => key)).toBe('multitrackTaskModes.ref')
    expect(getMultiTrackTaskModeLabel('edit', (key) => key)).toBe('multitrackTaskModes.edit')
  })

  it('migrates legacy segment tracks to task tracks', () => {
    const normalized = normalizeTrackData({
      tracks: [
        {
          id: 'track1',
          name: 'Segment 1',
          type: 'segment',
          color: 'var(--muted)',
          muted: false,
          locked: false,
          segments: [],
        },
      ],
      total_length: 120,
      frame_rate: 24,
    } as unknown as Parameters<typeof normalizeTrackData>[0])

    expect(normalized).toMatchObject({ muted: false, volume_db: 0 })
    expect(normalized.tracks[0]).toMatchObject({
      name: 'Task 0',
      type: 'task',
      task_mode: 'default',
      color: 'var(--multitrack-task-bg)',
    })
  })

  it('normalizes invalid task segment modes and image lists', () => {
    const normalized = normalizeTrackData({
      tracks: [
        {
          id: 'task',
          name: 'Task 1',
          type: 'task',
          task_mode: 'edit',
          color: 'var(--multitrack-task-bg)',
          muted: false,
          locked: false,
          segments: [
            {
              id: 'task-segment',
              start_frame: 0,
              end_frame: 2,
              color: 'var(--multitrack-task-bg)',
              content: {
                media_type: 'none',
                task_mode: 'unknown',
                text: 'Prompt',
                images: 'not-an-array',
              },
            },
          ],
        },
      ],
      total_length: 120,
      frame_rate: 24,
    } as unknown as Parameters<typeof normalizeTrackData>[0])

    expect(normalized.tracks[0].task_mode).toBe('edit')
    expect(normalized.tracks[0].segments[0].content).toMatchObject({
      task_mode: 'default',
      text: 'Prompt',
      images: [],
    })
  })

  it('does not migrate legacy second-based multitrack time fields', () => {
    const normalized = normalizeTrackData({
      tracks: [
        {
          id: 'video',
          name: 'Video 1',
          type: 'video',
          color: 'var(--primary)',
          muted: false,
          locked: false,
          segments: [
            {
              id: 'legacy-seconds',
              start_time: 1,
              end_time: 2,
              color: 'var(--primary)',
              content: { media_type: 'video' },
            },
          ],
        },
      ],
      total_duration: 5,
      frame_rate: 24,
    } as unknown as Parameters<typeof normalizeTrackData>[0])

    expect(normalized.total_length).toBe(120)
    expect(normalized.tracks[0].segments[0]).toMatchObject({
      start_frame: 0,
      end_frame: 1,
    })
  })

  it('derives normalized total length from segment ranges instead of stale serialized data', () => {
    const normalized = normalizeTrackData({
      tracks: [
        {
          id: 'task',
          name: 'Task 0',
          type: 'task',
          color: 'var(--multitrack-task-bg)',
          muted: false,
          locked: false,
          segments: [
            {
              id: 'task-1',
              start_frame: 0,
              end_frame: 29,
              color: 'var(--multitrack-task-bg)',
              content: { media_type: 'none' },
            },
            {
              id: 'task-2',
              start_frame: 29,
              end_frame: 86,
              color: 'var(--multitrack-task-bg)',
              content: { media_type: 'none' },
            },
          ],
        },
        {
          id: 'video',
          name: 'Video 0',
          type: 'video',
          color: 'var(--primary)',
          muted: false,
          locked: false,
          segments: [{
            id: 'video-1',
            start_frame: 0,
            end_frame: 86,
            color: 'var(--primary)',
            content: { media_type: 'video' },
          }],
        },
      ],
      total_length: 120,
      frame_rate: 16,
    })

    expect(normalized.total_length).toBe(86)
    expect(formatMultiTrackTime(normalized.total_length, {
      frameRate: normalized.frame_rate,
      showFrames: true,
    })).toBe('00:05:06')
  })

  it('finds the active preview video segment for default and selected-video modes', () => {
    const data = createDefaultTrackData()
    const videoTrack = {
      ...data.tracks[1],
      segments: [
        {
          id: 'first-video',
          start_frame: 0,
          end_frame: 48,
          color: data.tracks[1].color,
          content: { media_type: 'video' as const, duration: 2, file_name: 'a.mp4' },
        },
        {
          id: 'second-video',
          start_frame: 96,
          end_frame: 168,
          origin_start_frame: 96,
          color: data.tracks[1].color,
          content: { media_type: 'video' as const, duration: 3, file_name: 'b.mp4' },
        },
      ],
    }
    const trackData = { ...data, tracks: [data.tracks[0], videoTrack], total_length: 168 }

    expect(getActivePreviewVideoSegment(trackData, 24, null)?.segment.id).toBe('first-video')
    expect(getActivePreviewVideoSegment(trackData, 72, null)).toBeNull()
    expect(getActivePreviewVideoSegment(trackData, 120, 'second-video')).toMatchObject({
      trackId: videoTrack.id,
      localTime: 1,
    })
    expect(getActivePreviewVideoSegment(trackData, 72, 'second-video')).toBeNull()

    videoTrack.segments[1].origin_start_frame = 48
    expect(getActivePreviewVideoSegment(trackData, 120, 'second-video')).toMatchObject({
      localTime: 3,
    })
  })

  it('collects all active audio sources and applies track solo, mute, and dB settings', () => {
    const data = createDefaultTrackData()
    data.volume_db = 1
    data.tracks[1] = {
      ...data.tracks[1],
      id: 'video-track',
      volume_db: 2,
      segments: [{
        id: 'video-audio',
        start_frame: 0,
        end_frame: 48,
        color: data.tracks[1].color,
        content: { media_type: 'video', source_type: 'input', file_path: 'video.mp4', volume_db: 3 },
      }],
    }
    data.tracks.push({
      id: 'audio-track',
      name: 'Audio 1',
      type: 'audio',
      color: 'var(--multitrack-audio-bg)',
      muted: false,
      solo: false,
      volume_db: -1,
      locked: false,
      segments: [{
        id: 'audio-segment',
        start_frame: 0,
        end_frame: 48,
        color: 'var(--multitrack-audio-bg)',
        content: { media_type: 'audio', source_type: 'input', file_path: 'audio.wav', volume_db: -2 },
      }],
    })

    expect(getActivePreviewAudioSources(data, 24, null).map((source) => ({
      id: source.segment.id,
      localTime: source.localTime,
      volumeDb: source.volumeDb,
    }))).toEqual([
      { id: 'video-audio', localTime: 1, volumeDb: 6 },
      { id: 'audio-segment', localTime: 1, volumeDb: -2 },
    ])

    data.tracks[2].solo = true
    expect(getActivePreviewAudioSources(data, 24, null).map((source) => source.segment.id)).toEqual(['audio-segment'])
    data.tracks[2].muted = true
    expect(getActivePreviewAudioSources(data, 24, null)).toEqual([])
  })

  it('parses preview resolution aspect ratios from fixed, custom, and auto modes', () => {
    expect(parseMultiTrackPreviewResolution({
      resolution: '1920 x 1080 (16:9)',
      resize_method: 'crop',
    }, null)).toMatchObject({
      width: 1920,
      height: 1080,
      resizeMethod: 'crop',
      mode: 'fixed',
    })

    expect(parseMultiTrackPreviewResolution({
      resolution: 'width x height (custom)',
      width: 544,
      height: 960,
      resize_method: 'pad',
    }, null)).toMatchObject({
      width: 544,
      height: 960,
      resizeMethod: 'pad',
      mode: 'custom',
    })

    expect(parseMultiTrackPreviewResolution({
      resolution: 'width x height (auto)',
      resize_method: 'stretch',
    }, { width: 1280, height: 720 })).toMatchObject({
      width: 1280,
      height: 720,
      resizeMethod: 'stretch',
      mode: 'auto',
    })
  })

  it('collects dynamic combo resolution values from sibling node widgets', () => {
    const input = collectMultiTrackPreviewResolutionInput({
      widgets: [
        { name: 'resolution', value: ['width x height (custom)'] },
        { name: 'resolution.resize_method', value: ['crop'] },
        { name: 'resolution.width', value: [1024] },
        { name: 'resolution.height', value: [576] },
      ],
    })

    expect(input).toEqual({
      resolution: 'width x height (custom)',
      resize_method: 'crop',
      width: 1024,
      height: 576,
    })
    expect(parseMultiTrackPreviewResolution(input, null)).toMatchObject({
      width: 1024,
      height: 576,
      resizeMethod: 'crop',
      mode: 'custom',
    })
  })

  it('uses backend-style shortest and longest resolution inference from first video metadata', () => {
    expect(parseMultiTrackPreviewResolution({
      resolution: 'width x height (shortest)',
      resize_method: 'pad',
      resize_to_pixel: 640,
    }, { width: 1280, height: 720 })).toMatchObject({
      width: 1138,
      height: 640,
      mode: 'shortest',
    })

    expect(parseMultiTrackPreviewResolution({
      resolution: 'width x height (longest)',
      resize_method: 'pad',
      resize_to_pixel: 640,
    }, { width: 1280, height: 720 })).toMatchObject({
      width: 640,
      height: 360,
      mode: 'longest',
    })
  })

  it('returns selected segment context with its owning track', () => {
    const data = createDefaultTrackData()
    const taskSegment = {
      id: 'task-selected',
      start_frame: 0,
      end_frame: 2,
      color: data.tracks[0].color,
      content: { media_type: 'none' as const, task_mode: 'edit' as const },
    }
    const trackData = {
      ...data,
      tracks: [
        { ...data.tracks[0], segments: [taskSegment] },
        data.tracks[1],
      ],
    }

    expect(getSelectedMultiTrackSegment(trackData, 'task-selected')).toMatchObject({
      trackId: data.tracks[0].id,
      trackType: 'task',
      segment: taskSegment,
    })
    expect(getSelectedMultiTrackSegment(trackData, 'missing')).toBeNull()
    expect(getSelectedMultiTrackSegment(trackData, null)).toBeNull()
  })

  it('updates selected segment content without changing other segments', () => {
    const data = createDefaultTrackData()
    const taskSegment = {
      id: 'task-selected',
      start_frame: 0,
      end_frame: 2,
      color: data.tracks[0].color,
      content: { media_type: 'none' as const, task_mode: 'default' as const, text: '' },
    }
    const videoSegment = {
      id: 'video-keep',
      start_frame: 0,
      end_frame: 2,
      color: data.tracks[1].color,
      content: { media_type: 'video' as const, file_name: 'clip.mp4' },
    }
    const trackData = {
      ...data,
      tracks: [
        { ...data.tracks[0], segments: [taskSegment] },
        { ...data.tracks[1], segments: [videoSegment] },
      ],
    }

    const updated = updateMultiTrackSegmentContent(trackData, 'task-selected', {
      task_mode: 'ref',
      text: 'New prompt',
    })

    expect(updated.tracks[0].segments[0].content).toMatchObject({
      task_mode: 'ref',
      text: 'New prompt',
    })
    expect(updated.tracks[1].segments[0]).toBe(videoSegment)
  })

  it('updates segment duration on frame boundaries and recalculates total duration', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [
      {
        id: 'task-matching',
        start_frame: 24,
        end_frame: 48,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
      {
        id: 'task-overlapping',
        start_frame: 24,
        end_frame: 36,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
    ]
    const videoSegment = {
      id: 'video-duration',
      start_frame: 24,
      end_frame: 48,
      color: data.tracks[1].color,
      content: { media_type: 'video' as const, duration: 10 },
    }
    const trackData = {
      ...data,
      tracks: [
        data.tracks[0],
        { ...data.tracks[1], segments: [videoSegment] },
      ],
    }

    const updated = updateMultiTrackSegmentDuration(trackData, 'video-duration', 2.03, 24)

    expect(updated.tracks[1].segments[0]).toMatchObject({
      start_frame: 24,
      end_frame: 73,
    })
    expect(updated.tracks[0].segments.map((segment) => ({
      id: segment.id,
      start_frame: segment.start_frame,
      end_frame: segment.end_frame,
    }))).toEqual([
      { id: 'task-matching', start_frame: 24, end_frame: 73 },
      { id: 'task-overlapping', start_frame: 24, end_frame: 36 },
    ])
    expect(updated.total_length).toBe(120)
  })

  it('keeps duration edits from overlapping the next segment', () => {
    const data = createDefaultTrackData()
    const trackData = {
      ...data,
      tracks: [
        data.tracks[0],
        {
          ...data.tracks[1],
          segments: [
            {
              id: 'selected',
              start_frame: 24,
              end_frame: 48,
              color: data.tracks[1].color,
              content: { media_type: 'video' as const },
            },
            {
              id: 'next',
              start_frame: 60,
              end_frame: 84,
              color: data.tracks[1].color,
              content: { media_type: 'video' as const },
            },
          ],
        },
      ],
    }

    const updated = updateMultiTrackSegmentDuration(trackData, 'selected', 10, 24)

    expect(updated.tracks[1].segments[0].end_frame).toBe(60)
  })

  it('converts seconds and frames using the current frame rate', () => {
    expect(secondsToFrame(1.5, 24)).toBe(36)
    expect(frameToSeconds(36, 24)).toBe(1.5)
  })

  it('snaps arbitrary frame values to integer frame boundaries', () => {
    expect(snapTimeToFrame(1.02, 24)).toBe(1)
    expect(snapTimeToFrame(1.5, 24)).toBe(2)
  })

  it('remaps segment frames when changing frame rate without changing displayed time', () => {
    const data = createDefaultTrackData()
    const trackData = {
      ...data,
      tracks: [
        data.tracks[0],
        {
          ...data.tracks[1],
          segments: [
            {
              id: 'video-remap',
              start_frame: 24,
              end_frame: 48,
              color: data.tracks[1].color,
              content: { media_type: 'video' as const },
            },
          ],
        },
      ],
    }

    const updated = remapTrackDataFrameRate(trackData, 30)

    expect(updated.frame_rate).toBe(30)
    expect(updated.total_length).toBe(150)
    expect(updated.tracks[1].segments[0]).toMatchObject({
      start_frame: 30,
      end_frame: 60,
    })
    expect(formatMultiTrackTime(updated.tracks[1].segments[0].start_frame, { frameRate: updated.frame_rate, showFrames: true })).toBe('00:01:00')

    expect(remapTrackDataFrameRate(data, 16).total_length).toBe(80)
  })

  it('formats multitrack time as clock labels with optional frame precision', () => {
    expect(formatMultiTrackTime(5.9)).toBe('00:05')
    expect(formatMultiTrackTime(65)).toBe('01:05')
    expect(formatMultiTrackTime(3661)).toBe('01:01:01')
    expect(formatMultiTrackTime(6, { frameRate: 24, showFrames: true })).toBe('00:00:06')
    expect(formatMultiTrackTime(33, { frameRate: 24, showFrames: true })).toBe('00:01:09')
    expect(formatMultiTrackTime(1572, { frameRate: 24, showFrames: true })).toBe('01:05:12')
    expect(formatMultiTrackTime(0, { frameRate: 24, showFrames: true })).toBe('00:00:00')
  })

  it('derives total length from the latest segment end frame', () => {
    const data = createDefaultTrackData()
    const videoTrack = data.tracks[1]
    videoTrack.segments = [
      {
        id: 'seg1',
        start_frame: 120,
        end_frame: 205,
        color: 'var(--primary)',
        content: {
          media_type: 'video',
          file_name: 'clip.mp4',
        },
      },
    ]

    expect(calculateTotalLength(data.tracks)).toBe(205)
  })

  it('uses five seconds at the current frame rate as the timeline minimum', () => {
    const data = createDefaultTrackData()
    data.tracks[1].segments = [{
      id: 'short',
      start_frame: 0,
      end_frame: 86,
      color: 'var(--primary)',
      content: { media_type: 'video' },
    }]

    expect(calculateTotalLength(data.tracks, 16)).toBe(86)
    expect(calculateTotalLength(data.tracks, 20)).toBe(100)
  })

  it('adds a default task segment when a video range has no task coverage', () => {
    const data = createDefaultTrackData()
    const updated = addDefaultTaskSegmentIfRangeEmpty(data.tracks, 2, 5)
    const taskTrack = updated[0]

    expect(taskTrack.segments).toHaveLength(1)
    expect(taskTrack.segments[0]).toMatchObject({
      start_frame: 2,
      end_frame: 5,
      color: 'var(--multitrack-task-bg)',
      content: {
        media_type: 'none',
        task_mode: 'default',
      },
    })
  })

  it('does not add a default task segment when the range already has task coverage', () => {
    const data = createDefaultTrackData()
    const withTask = addDefaultTaskSegmentIfRangeEmpty(data.tracks, 2, 5)
    const updated = addDefaultTaskSegmentIfRangeEmpty(withTask, 3, 4)

    expect(updated[0].segments).toHaveLength(1)
  })

  it('deletes overlapping task segments when deleting a video segment', () => {
    const data = createDefaultTrackData()
    const taskTrack = {
      ...data.tracks[0],
      segments: [
        {
          id: 'task-before',
          start_frame: 0,
          end_frame: 2,
          color: data.tracks[0].color,
          content: { media_type: 'none' as const, task_mode: 'default' as const },
        },
        {
          id: 'task-linked',
          start_frame: 2,
          end_frame: 5,
          color: data.tracks[0].color,
          content: { media_type: 'none' as const, task_mode: 'default' as const },
        },
        {
          id: 'task-after',
          start_frame: 5,
          end_frame: 7,
          color: data.tracks[0].color,
          content: { media_type: 'none' as const, task_mode: 'default' as const },
        },
      ],
    }
    const videoTrack = {
      ...data.tracks[1],
      segments: [
        {
          id: 'video-delete',
          start_frame: 2,
          end_frame: 5,
          color: data.tracks[1].color,
          content: { media_type: 'video' as const, duration: 3 },
        },
      ],
    }

    const updated = deleteSegmentWithLinkedTasks([taskTrack, videoTrack], 'video-delete')

    expect(updated[0].segments.map((segment) => segment.id)).toEqual(['task-before', 'task-after'])
    expect(updated[1].segments).toHaveLength(0)
  })

  it('does not delete linked tasks when deleting a non-video segment', () => {
    const data = createDefaultTrackData()
    const taskTrack = {
      ...data.tracks[0],
      segments: [
        {
          id: 'task-delete',
          start_frame: 2,
          end_frame: 5,
          color: data.tracks[0].color,
          content: { media_type: 'none' as const, task_mode: 'default' as const },
        },
      ],
    }
    const videoTrack = {
      ...data.tracks[1],
      segments: [
        {
          id: 'video-keep',
          start_frame: 2,
          end_frame: 5,
          color: data.tracks[1].color,
          content: { media_type: 'video' as const, duration: 3 },
        },
      ],
    }

    const updated = deleteSegmentWithLinkedTasks([taskTrack, videoTrack], 'task-delete')

    expect(updated[0].segments).toHaveLength(0)
    expect(updated[1].segments.map((segment) => segment.id)).toEqual(['video-keep'])
  })

  it('deletes multiple selected segments through the same linked-task rules', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [
      {
        id: 'task-linked',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
      {
        id: 'task-selected',
        start_frame: 2,
        end_frame: 4,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
    ]
    data.tracks[1].segments = [
      {
        id: 'video-selected',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 2 },
      },
      {
        id: 'video-keep',
        start_frame: 2,
        end_frame: 4,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 2 },
      },
    ]

    const updated = deleteSegmentsWithLinkedTasks(data.tracks, ['video-selected', 'task-selected'])

    expect(updated[0].segments).toHaveLength(0)
    expect(updated[1].segments.map((segment) => segment.id)).toEqual(['video-keep'])
  })

  it('packs later primary video segments forward after deleting selected middle clips', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [
      {
        id: 'task-first',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
      {
        id: 'task-middle',
        start_frame: 2,
        end_frame: 5,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
      {
        id: 'task-last',
        start_frame: 5,
        end_frame: 7,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
    ]
    data.tracks[1].segments = [
      {
        id: 'video-first',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 2 },
      },
      {
        id: 'video-middle',
        start_frame: 2,
        end_frame: 5,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 3 },
      },
      {
        id: 'video-last',
        start_frame: 5,
        end_frame: 7,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 2 },
      },
    ]

    const updated = deleteSegmentsWithLinkedTasks(data.tracks, ['video-middle'])

    expect(updated[1].segments.map((segment) => [segment.id, segment.start_frame, segment.end_frame])).toEqual([
      ['video-first', 0, 2],
      ['video-last', 2, 4],
    ])
    expect(updated[0].segments.map((segment) => [segment.id, segment.start_frame, segment.end_frame])).toEqual([
      ['task-first', 0, 2],
      ['task-last', 2, 4],
    ])
  })

  it('moves segments only between compatible tracks and clamps to avoid overlaps', () => {
    const data = createDefaultTrackData()
    const firstVideoTrack = data.tracks[1]
    const secondVideoTrack = {
      ...firstVideoTrack,
      id: 'video2',
      segments: [
        {
          id: 'existing',
          start_frame: 4,
          end_frame: 6,
          color: firstVideoTrack.color,
          content: { media_type: 'video' as const, duration: 2 },
        },
      ],
    }
    const moving = {
      id: 'moving',
      start_frame: 0,
      end_frame: 2,
      color: firstVideoTrack.color,
      content: { media_type: 'video' as const, duration: 2 },
    }

    const moved = moveSegmentBetweenCompatibleTracks(
      [
        data.tracks[0],
        { ...firstVideoTrack, id: 'video1', segments: [moving] },
        secondVideoTrack,
      ],
      'moving',
      'video2',
      3.5,
      24,
    )

    expect(moved[1].segments).toHaveLength(0)
    expect(moved[2].segments.map((segment) => segment.id)).toEqual(['moving', 'existing'])
    expect(moved[2].segments[0]).toMatchObject({
      start_frame: 0,
      end_frame: 2,
    })
    expect(moved[2].segments[1]).toMatchObject({
      start_frame: 2,
      end_frame: 4,
    })

    const blocked = moveSegmentBetweenCompatibleTracks(moved, 'moving', data.tracks[0].id, 0, 24)
    expect(blocked).toBe(moved)
  })

  it('repacks same-track segments from zero when their order changes', () => {
    const data = createDefaultTrackData()
    const firstVideoTrack = data.tracks[1]
    const moved = moveSegmentBetweenCompatibleTracks(
      [
        data.tracks[0],
        {
          ...firstVideoTrack,
          id: 'video1',
          segments: [
            {
              id: 'first',
              start_frame: 0,
              end_frame: 2,
              color: firstVideoTrack.color,
              content: { media_type: 'video' as const, duration: 2 },
            },
            {
              id: 'second',
              start_frame: 2,
              end_frame: 5,
              color: firstVideoTrack.color,
              content: { media_type: 'video' as const, duration: 3 },
            },
          ],
        },
      ],
      'first',
      'video1',
      4,
      24,
    )

    expect(moved[1].segments.map((segment) => segment.id)).toEqual(['second', 'first'])
    expect(moved[1].segments.map((segment) => [segment.start_frame, segment.end_frame])).toEqual([
      [0, 3],
      [3, 5],
    ])
  })

  it('moves selected segments together while preserving each track type rules', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [
      {
        id: 'task-first',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
      {
        id: 'task-second',
        start_frame: 2,
        end_frame: 5,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
    ]
    data.tracks[1].segments = [
      {
        id: 'video-first',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 2 },
      },
      {
        id: 'video-second',
        start_frame: 2,
        end_frame: 5,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 3 },
      },
    ]
    data.tracks.push({
      id: 'audio-track',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      solo: false,
      volume_db: 0,
      locked: false,
      segments: [
        {
          id: 'audio-first',
          start_frame: 0,
          end_frame: 2,
          color: 'var(--highlight)',
          content: { media_type: 'audio', duration: 2 },
        },
        {
          id: 'audio-second',
          start_frame: 3,
          end_frame: 5,
          color: 'var(--highlight)',
          content: { media_type: 'audio', duration: 2 },
        },
      ],
    })

    const moved = moveSelectedSegments(
      data.tracks,
      ['task-first', 'video-first', 'audio-first'],
      'video-first',
      data.tracks[1].id,
      4,
      24,
    )

    expect(moved[0].segments.map((segment) => segment.id)).toEqual(['task-second', 'task-first'])
    expect(moved[1].segments.map((segment) => segment.id)).toEqual(['video-second', 'video-first'])
    expect(moved[2].segments.map((segment) => [segment.id, segment.start_frame, segment.end_frame])).toEqual([
      ['audio-second', 3, 5],
      ['audio-first', 5, 7],
    ])
  })

  it('moves subtitle segments like audio segments without packing gaps', () => {
    const data = createDefaultTrackData()
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [
        {
          id: 'subtitle-first',
          start_frame: 0,
          end_frame: 2,
          color: '#9D4937',
          content: { media_type: 'subtitle', text: 'First' },
        },
        {
          id: 'subtitle-second',
          start_frame: 6,
          end_frame: 8,
          color: '#9D4937',
          content: { media_type: 'subtitle', text: 'Second' },
        },
      ],
    })

    const moved = moveSegmentBetweenCompatibleTracks(
      data.tracks,
      'subtitle-first',
      'subtitle-track',
      4,
      24,
    )

    expect(moved[2].segments.map((segment) => [segment.id, segment.start_frame, segment.end_frame])).toEqual([
      ['subtitle-first', 4, 6],
      ['subtitle-second', 6, 8],
    ])
  })

  it('previews subtitle drag placement without packing gaps', () => {
    const data = createDefaultTrackData()
    data.tracks.push({
      id: 'subtitle-track',
      name: 'Subtitle 1',
      type: 'subtitle',
      color: '#9D4937',
      muted: false,
      locked: false,
      segments: [
        {
          id: 'subtitle-first',
          start_frame: 0,
          end_frame: 2,
          color: '#9D4937',
          content: { media_type: 'subtitle', text: 'First' },
        },
        {
          id: 'subtitle-second',
          start_frame: 6,
          end_frame: 8,
          color: '#9D4937',
          content: { media_type: 'subtitle', text: 'Second' },
        },
      ],
    })

    const placeholder = getSegmentDragPlaceholder(
      data.tracks,
      'subtitle-first',
      'subtitle-track',
      4,
      24,
    )

    expect(placeholder).toMatchObject({
      segmentId: 'subtitle-first',
      targetTrackId: 'subtitle-track',
      start_frame: 4,
      end_frame: 6,
    })
    expect(getSegmentDragPreviewSegments(data.tracks, placeholder!, 24)).toEqual([
      expect.objectContaining({ id: 'subtitle-second', start_frame: 6, end_frame: 8 }),
    ])
  })

  it('moves matching task ranges together with reordered video segments', () => {
    const data = createDefaultTrackData()
    data.tracks[0].segments = [
      {
        id: 'task-first',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
      {
        id: 'task-second',
        start_frame: 2,
        end_frame: 5,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
    ]
    data.tracks[1].segments = [
      {
        id: 'video-first',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 2 },
      },
      {
        id: 'video-second',
        start_frame: 2,
        end_frame: 5,
        color: data.tracks[1].color,
        content: { media_type: 'video', duration: 3 },
      },
    ]

    const moved = moveSegmentBetweenCompatibleTracks(
      data.tracks,
      'video-first',
      data.tracks[1].id,
      4,
      data.frame_rate,
    )

    expect(moved[0].segments.map((segment) => [segment.id, segment.start_frame, segment.end_frame])).toEqual([
      ['task-first', 3, 5],
      ['task-second', 0, 3],
    ])
  })

  it('keeps audio drop positions and source-track gaps while preventing overlaps', () => {
    const data = createDefaultTrackData()
    const audioTrack = {
      ...data.tracks[1],
      type: 'audio' as const,
      id: 'audio1',
      segments: [
        {
          id: 'keep',
          start_frame: 8,
          end_frame: 10,
          color: data.tracks[1].color,
          content: { media_type: 'audio' as const },
        },
        {
          id: 'moving',
          start_frame: 14,
          end_frame: 17,
          color: data.tracks[1].color,
          content: { media_type: 'audio' as const },
        },
      ],
    }
    const targetTrack = {
      ...audioTrack,
      id: 'audio2',
      segments: [{
        id: 'existing',
        start_frame: 12,
        end_frame: 15,
        color: audioTrack.color,
        content: { media_type: 'audio' as const },
      }],
    }

    const moved = moveSegmentBetweenCompatibleTracks(
      [data.tracks[0], audioTrack, targetTrack],
      'moving',
      'audio2',
      11,
      24,
    )

    expect(moved[1].segments.map((segment) => [segment.id, segment.start_frame, segment.end_frame])).toEqual([
      ['keep', 8, 10],
    ])
    expect(moved[2].segments.map((segment) => [segment.id, segment.start_frame, segment.end_frame])).toEqual([
      ['moving', 11, 14],
      ['existing', 14, 17],
    ])
    expect(getSegmentDragPlaceholder(
      [data.tracks[0], audioTrack, targetTrack],
      'moving',
      'audio2',
      20,
      24,
    )).toMatchObject({ start_frame: 20, end_frame: 23 })
  })

  it('computes drag placeholder only for compatible target tracks', () => {
    const data = createDefaultTrackData()
    const videoTrack = {
      ...data.tracks[1],
      id: 'video1',
      segments: [
        {
          id: 'first',
          start_frame: 0,
          end_frame: 2,
          color: data.tracks[1].color,
          content: { media_type: 'video' as const, duration: 2 },
        },
        {
          id: 'second',
          start_frame: 2,
          end_frame: 5,
          color: data.tracks[1].color,
          content: { media_type: 'video' as const, duration: 3 },
        },
      ],
    }

    expect(getSegmentDragPlaceholder([data.tracks[0], videoTrack], 'first', data.tracks[0].id, 4, 24)).toBeNull()
    expect(getSegmentDragPlaceholder([data.tracks[0], videoTrack], 'first', 'video1', 4, 24)).toEqual({
      segmentId: 'first',
      targetTrackId: 'video1',
      insertIndex: 1,
      start_frame: 3,
      end_frame: 5,
    })
    expect(getSegmentDragPlaceholder([data.tracks[0], videoTrack], 'second', 'video1', 0, 24)).toEqual({
      segmentId: 'second',
      targetTrackId: 'video1',
      insertIndex: 0,
      start_frame: 0,
      end_frame: 3,
    })
  })

  it('temporarily repacks visible target-track segments around the drag placeholder', () => {
    const data = createDefaultTrackData()
    const videoTrack = {
      ...data.tracks[1],
      id: 'video1',
      segments: [
        {
          id: 'first',
          start_frame: 0,
          end_frame: 2,
          color: data.tracks[1].color,
          content: { media_type: 'video' as const, duration: 2 },
        },
        {
          id: 'second',
          start_frame: 2,
          end_frame: 5,
          color: data.tracks[1].color,
          content: { media_type: 'video' as const, duration: 3 },
        },
      ],
    }
    const placeholder = getSegmentDragPlaceholder([data.tracks[0], videoTrack], 'second', 'video1', 0, 24)

    data.tracks[0].segments = [
      {
        id: 'task-first',
        start_frame: 0,
        end_frame: 2,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
      {
        id: 'task-second',
        start_frame: 2,
        end_frame: 5,
        color: data.tracks[0].color,
        content: { media_type: 'none', task_mode: 'default' },
      },
    ]

    expect(placeholder).not.toBeNull()
    expect(getSegmentDragPreviewSegments([data.tracks[0], videoTrack], placeholder!, 24)?.map((segment) => ({
      id: segment.id,
      start_frame: segment.start_frame,
      end_frame: segment.end_frame,
    }))).toEqual([
      { id: 'first', start_frame: 3, end_frame: 5 },
      { id: 'task-first', start_frame: 3, end_frame: 5 },
      { id: 'task-second', start_frame: 0, end_frame: 3 },
    ])
  })
})
