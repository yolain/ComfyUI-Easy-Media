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
        globalVolumeDb={0}
        frameRate={24}
        selectedMediaVolumeDb={null}
        selectedMediaMuted={false}
        selectedMediaDuration={null}
        onGlobalSettingsChange={onGlobalSettingsChange}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={vi.fn()}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Audio settings' }))
    expect(screen.getAllByText('Audio (g)').length).toBeGreaterThan(0)
    fireEvent.click(screen.getByRole('button', { name: 'Mute preview audio' }))

    expect(onGlobalSettingsChange).toHaveBeenCalledWith({ muted: true })
  })

  it('keeps the selected video volume number input and slider in sync', () => {
    const onSelectedSegmentContentChange = vi.fn()
    const props = {
      globalMuted: false,
      globalVolumeDb: 0,
      frameRate: 24,
      selectedMediaMuted: false,
      selectedMediaDuration: 3,
      onGlobalSettingsChange: vi.fn(),
      onSelectedSegmentContentChange,
      onSelectedSegmentDurationChange: vi.fn(),
    }
    const { rerender } = render(
      <PreviewFloatingToolbar
        {...props}
        selectedMediaVolumeDb={0.8}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Audio settings' }))
    expect(screen.getAllByText('Audio').length).toBeGreaterThan(0)
    const volumeInput = screen.getByRole('spinbutton', { name: 'Volume' })
    const muteButton = screen.getByRole('button', { name: 'Mute preview audio' })
    expect(volumeInput.parentElement?.className).toContain('cursor-pointer')
    expect(muteButton.className).toContain('cursor-pointer')
    fireEvent.change(volumeInput, { target: { value: '2.5' } })
    expect(onSelectedSegmentContentChange).toHaveBeenCalledWith({ volume_db: 2.5 })

    rerender(<PreviewFloatingToolbar {...props} selectedMediaVolumeDb={2.5} />)
    const slider = screen.getByRole('slider', { name: 'Volume' })
    expect(slider.closest('.cursor-pointer')).not.toBeNull()
    expect(slider.getAttribute('aria-valuenow')).toBe('2.5')

    fireEvent.keyDown(slider, { key: 'ArrowRight' })
    expect(onSelectedSegmentContentChange).toHaveBeenLastCalledWith({ volume_db: 2.6 })
  })

  it('updates frame rate and commits a formatted selected video duration on Enter', () => {
    const onGlobalSettingsChange = vi.fn()
    const onSelectedSegmentDurationChange = vi.fn()
    const { rerender } = render(
      <PreviewFloatingToolbar
        globalMuted={false}
        globalVolumeDb={0}
        frameRate={24}
        selectedMediaVolumeDb={0.8}
        selectedMediaMuted={false}
        selectedMediaDuration={3}
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
        globalVolumeDb={0}
        frameRate={25}
        selectedMediaVolumeDb={0.8}
        selectedMediaMuted={false}
        selectedMediaDuration={3}
        onGlobalSettingsChange={onGlobalSettingsChange}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={onSelectedSegmentDurationChange}
      />,
    )
    fireEvent.keyDown(screen.getByRole('slider', { name: 'Frame rate' }), { key: 'ArrowRight' })
    const durationInput = screen.getByLabelText('Duration')
    expect(durationInput.getAttribute('type')).toBe('text')
    expect(durationInput.getAttribute('value')).toBe('00:03:00')
    fireEvent.change(durationInput, { target: { value: '00:04:15' } })
    expect(onSelectedSegmentDurationChange).not.toHaveBeenCalled()
    fireEvent.keyDown(durationInput, { key: 'Enter' })

    expect(onGlobalSettingsChange).toHaveBeenCalledWith({ frame_rate: 30 })
    expect(onSelectedSegmentDurationChange).toHaveBeenCalledWith(4.6)
  })

  it('commits duration on blur and restores an invalid timecode', () => {
    const onSelectedSegmentDurationChange = vi.fn()
    render(
      <PreviewFloatingToolbar
        globalMuted={false}
        globalVolumeDb={0}
        frameRate={24}
        selectedMediaVolumeDb={0.8}
        selectedMediaMuted={false}
        selectedMediaDuration={3}
        onGlobalSettingsChange={vi.fn()}
        onSelectedSegmentContentChange={vi.fn()}
        onSelectedSegmentDurationChange={onSelectedSegmentDurationChange}
      />,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Speed settings' }))
    const durationInput = screen.getByLabelText('Duration')
    fireEvent.change(durationInput, { target: { value: '00:05:12' } })
    fireEvent.blur(durationInput)
    expect(onSelectedSegmentDurationChange).toHaveBeenCalledWith(5.5)

    fireEvent.change(durationInput, { target: { value: '00:61:00' } })
    fireEvent.blur(durationInput)
    expect(durationInput.getAttribute('value')).toBe('00:03:00')
    expect(onSelectedSegmentDurationChange).toHaveBeenCalledTimes(1)
  })

  it('uses locale messages for preview toolbar labels', () => {
    render(
      <LocaleContext.Provider value="zh">
        <PreviewFloatingToolbar
          globalMuted={false}
          globalVolumeDb={0}
          frameRate={24}
          selectedMediaVolumeDb={null}
          selectedMediaMuted={false}
          selectedMediaDuration={null}
          onGlobalSettingsChange={vi.fn()}
          onSelectedSegmentContentChange={vi.fn()}
          onSelectedSegmentDurationChange={vi.fn()}
        />
      </LocaleContext.Provider>,
    )

    fireEvent.click(screen.getByRole('button', { name: '音频设置' }))
    expect(screen.getAllByText('总音频').length).toBeGreaterThan(0)
    expect(screen.getByRole('button', { name: '静音预览音频' })).not.toBeNull()
    expect(screen.getByRole('spinbutton', { name: '音量' })).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: '速度设置' }))
    expect(screen.getByRole('slider', { name: '帧率' })).not.toBeNull()
  })
})
