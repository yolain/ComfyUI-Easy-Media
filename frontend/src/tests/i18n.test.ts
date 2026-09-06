import { describe, expect, it } from 'vitest'
import { translate } from '@/lib/i18n'

describe('multitrack audio lock translations', () => {
  it('explains video timing and audio-track priority in the video lock tooltip', () => {
    expect(translate('zh', 'multitrack.videoAudioLockTooltip')).toContain('视频轨道画面的时间线')
    expect(translate('zh', 'multitrack.videoAudioLockTooltip')).toContain('音频轨道将优先')
    expect(translate('en', 'multitrack.videoAudioLockTooltip')).toContain('audio track takes priority')
  })
})
