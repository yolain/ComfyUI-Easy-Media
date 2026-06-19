import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { LocaleContext } from '@/lib/i18n'
import { PreviewFloatingToolbar } from '@/components/widgets/multitrack/PreviewFloatingToolbar'

describe('PreviewFloatingToolbar', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      unobserve() {}
      disconnect() {}
    })
  })

  it('updates global audio settings when no video segment is selected', () => {
    const onGlobalSettingsChange = vi.fn()
    render(
      <PreviewFloatingToolbar
        globalMuted={false}
        globalVolume={1}
        frameRate={24}
        selectedVideoVolume={null}
        selectedVideoDuration={null}
        onGlobalSettingsChange={onGlobalSettingsChange}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Audio settings' }))
    fireEvent.click(screen.getByRole('button', { name: 'Mute preview audio' }))

    expect(onGlobalSettingsChange).toHaveBeenCalledWith({ muted: true })
  })

  it('updates selected video volume when a video segment is selected', () => {
    const onSelectedSegmentContentChange = vi.fn()
    render(
      <PreviewFloatingToolbar
        globalMuted={false}
        globalVolume={1}
        frameRate={24}
        selectedVideoVolume={0.8}
        selectedVideoDuration={3}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={onSelectedSegmentContentChange}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Audio settings' }))
    fireEvent.change(screen.getByLabelText('Volume'), { target: { value: '0.25' } })

    expect(onSelectedSegmentContentChange).toHaveBeenCalledWith({ volume: 0.25 })
  })

  it('updates frame rate and selected video duration from speed settings', () => {
    const onGlobalSettingsChange = vi.fn()
    const onSelectedSegmentDurationChange = vi.fn()
    const { rerender } = render(
      <PreviewFloatingToolbar
        globalMuted={false}
        globalVolume={1}
        frameRate={24}
        selectedVideoVolume={0.8}
        selectedVideoDuration={3}
        onGlobalSettingsChange={onGlobalSettingsChange}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={onSelectedSegmentDurationChange}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Speed settings' }))
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Frame rate' }), { key: 'ArrowRight' })
    rerender(
      <PreviewFloatingToolbar
        globalMuted={false}
        globalVolume={1}
        frameRate={25}
        selectedVideoVolume={0.8}
        selectedVideoDuration={3}
        onGlobalSettingsChange={onGlobalSettingsChange}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={onSelectedSegmentDurationChange}
      />,
    )
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Frame rate' }), { key: 'ArrowRight' })
    fireEvent.change(screen.getByLabelText('Duration'), { target: { value: '4.5' } })

    expect(onGlobalSettingsChange).toHaveBeenCalledWith({ frame_rate: 30 })
    expect(onSelectedSegmentDurationChange).toHaveBeenCalledWith(4.5)
  })

  it('uses locale messages for preview toolbar labels', () => {
    render(
      <LocaleContext.Provider value="zh">
        <PreviewFloatingToolbar
          globalMuted={false}
          globalVolume={1}
          frameRate={24}
          selectedVideoVolume={null}
          selectedVideoDuration={null}
          onGlobalSettingsChange={vi.fn()}
          onSelectedSegmentContentChange={vi.fn()}
          onSelectedSegmentDurationChange={vi.fn()}
        />
      </LocaleContext.Provider>,
    )

    fireEvent.click(screen.getByRole('button', { name: '音频设置' }))
    expect(screen.getByRole('button', { name: '静音预览音频' })).not.toBeNull()
    expect(screen.getByLabelText('音量')).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: '速度设置' }))
    expect(screen.getByRole('slider', { name: '帧率' })).not.toBeNull()
  })
})
