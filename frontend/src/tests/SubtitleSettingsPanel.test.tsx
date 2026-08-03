import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { SubtitleSettingsPanel } from '@/components/widgets/multitrack/SubtitleSettingsPanel'

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: ({ onChange }: { onChange: (value: string) => void }) => (
    <button type="button" onClick={() => onChange('voice.wav')}>
      Mock media selector
    </button>
  ),
}))

const baseStyle = {
  font_size: 24,
  color: '#ffffff',
  outline_color: '#000000',
  background_color: 'transparent',
  background_opacity: 0.7,
  x: 0.15,
  y: 0.8,
  width: 0.7,
}

describe('SubtitleSettingsPanel', () => {
  beforeEach(() => {
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      unobserve() {}
      disconnect() {}
    })
    vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
  })

  it('edits subtitle text and style from a side panel', () => {
    const onTextChange = vi.fn()
    const onStyleChange = vi.fn()

    render(
      <SubtitleSettingsPanel
        text="Hello"
        style={baseStyle}
        onTextChange={onTextChange}
        onStyleChange={onStyleChange}
      />,
    )

    expect(screen.getByTestId('subtitle-settings-panel')).not.toBeNull()
    expect(screen.getByRole('tab', { name: /Text/ })).not.toBeNull()
    expect(screen.getByRole('tab', { name: /Speech/ })).not.toBeNull()

    fireEvent.change(screen.getByRole('textbox', { name: 'Subtitle text' }), { target: { value: 'Updated line' } })
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Font size' }), { target: { value: '32' } })
    fireEvent.click(screen.getByRole('button', { name: 'Apply Yellow subtitle style' }))
    fireEvent.change(screen.getByRole('spinbutton', { name: 'X (%)' }), { target: { value: '20' } })
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Y (%)' }), { target: { value: '75' } })
    fireEvent.change(screen.getByRole('spinbutton', { name: 'Width (%)' }), { target: { value: '60' } })

    expect(onTextChange).toHaveBeenCalledWith('Updated line')
    expect(onStyleChange).toHaveBeenCalledWith({ font_size: 32 })
    expect(onStyleChange).toHaveBeenCalledWith({
      color: '#ffd60a',
      outline_color: '#000000',
      background_color: 'transparent',
    })
    expect(onStyleChange).toHaveBeenCalledWith({ x: 0.2 })
    expect(onStyleChange).toHaveBeenCalledWith({ y: 0.75 })
    expect(onStyleChange).toHaveBeenCalledWith({ width: 0.6 })
    expect(screen.queryByText('Outline color')).toBeNull()
    expect(screen.queryByText('Background color')).toBeNull()
  })

  it('keeps subtitle textarea wheel scrolling inside the settings panel', () => {
    const onCanvasWheel = vi.fn()
    render(
      <div
        onWheelCapture={(event) => {
          const target = event.target as HTMLElement
          const captureElement = target.closest('[data-capture-wheel="true"]')
          if (!captureElement?.contains(document.activeElement)) onCanvasWheel()
        }}
      >
        <SubtitleSettingsPanel
          text={'Line 1\nLine 2\nLine 3\nLine 4\nLine 5'}
          style={baseStyle}
          onTextChange={vi.fn()}
          onStyleChange={vi.fn()}
        />
      </div>,
    )

    const textarea = screen.getByRole('textbox', { name: 'Subtitle text' })
    textarea.focus()
    const keepsNativeScroll = fireEvent.wheel(textarea, { deltaY: 100 })

    expect(keepsNativeScroll).toBe(true)
    expect(onCanvasWheel).not.toHaveBeenCalled()
  })

  it('disables background opacity without a background and updates it for background presets', () => {
    const onStyleChange = vi.fn()
    const { rerender } = render(
      <SubtitleSettingsPanel
        text="Hello"
        style={baseStyle}
        onTextChange={vi.fn()}
        onStyleChange={onStyleChange}
      />,
    )

    expect((screen.getByRole('spinbutton', { name: 'Background opacity (%)' }) as HTMLInputElement).disabled).toBe(true)

    fireEvent.click(screen.getByRole('button', { name: 'Apply Black subtitle style' }))
    expect(onStyleChange).toHaveBeenCalledWith({
      color: '#ffffff',
      outline_color: 'transparent',
      background_color: '#000000',
    })

    rerender(
      <SubtitleSettingsPanel
        text="Hello"
        style={{ ...baseStyle, background_color: '#000000' }}
        onTextChange={vi.fn()}
        onStyleChange={onStyleChange}
      />,
    )

    const opacityInput = screen.getByRole('spinbutton', { name: 'Background opacity (%)' })
    expect((opacityInput as HTMLInputElement).disabled).toBe(false)
    expect(opacityInput.getAttribute('value')).toBe('70')

    fireEvent.change(opacityInput, { target: { value: '45' } })
    expect(onStyleChange).toHaveBeenCalledWith({ background_opacity: 0.45 })
  })

  it('marks the matching preset selected and supports no-outline preset', () => {
    const onStyleChange = vi.fn()
    render(
      <SubtitleSettingsPanel
        text="Hello"
        style={baseStyle}
        onTextChange={vi.fn()}
        onStyleChange={onStyleChange}
      />,
    )

    const presetGrid = screen.getByRole('button', { name: 'Apply White subtitle style' }).parentElement
    expect(presetGrid?.className).toContain('grid-cols-7')
    expect(screen.getByRole('button', { name: 'Apply White subtitle style' }).getAttribute('aria-pressed')).toBe('true')

    fireEvent.click(screen.getByRole('button', { name: 'Apply No outline subtitle style' }))

    expect(onStyleChange).toHaveBeenCalledWith({
      color: '#ffffff',
      outline_color: 'transparent',
      background_color: 'transparent',
    })
  })

  it('shows speech controls with default values', () => {
    render(
      <SubtitleSettingsPanel
        text="Hello"
        style={baseStyle}
        onTextChange={vi.fn()}
        onStyleChange={vi.fn()}
      />,
    )

    fireEvent.mouseDown(screen.getByRole('tab', { name: /Speech/ }))

    expect(screen.getByRole('combobox', { name: 'Model' })).not.toBeNull()
    expect(screen.getByText('VoxCPM2')).not.toBeNull()
    expect(screen.getByRole('textbox', { name: 'Control prompt' }).getAttribute('placeholder')).toBe('Control prompt: Describe language, dialect, emotion, gender, etc. here.')
    expect(screen.getByRole('spinbutton', { name: 'CFG' }).getAttribute('value')).toBe('2.0')
    expect(screen.getByRole('spinbutton', { name: 'Steps' }).getAttribute('value')).toBe('10')
    expect(screen.getByRole('button', { name: 'Add reference audio' })).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Speech' }).className).toContain('bg-highlight')
  })

  it('selects and clears reference audio from the speech tab', () => {
    render(
      <SubtitleSettingsPanel
        text="Hello"
        style={baseStyle}
        onTextChange={vi.fn()}
        onStyleChange={vi.fn()}
      />,
    )

    fireEvent.mouseDown(screen.getByRole('tab', { name: /Speech/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Add reference audio' }))
    fireEvent.click(screen.getByRole('button', { name: 'Mock media selector' }))

    expect(screen.getByText('voice.wav')).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Reselect reference audio' })).not.toBeNull()
    expect(screen.getByRole('button', { name: 'Reselect reference audio' }).querySelector('.lucide-rotate-ccw')).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Clear reference audio' }))

    expect(screen.getByRole('button', { name: 'Add reference audio' })).not.toBeNull()
  })

  it('plays and pauses an existing reference audio preview', async () => {
    const play = vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue(undefined)
    const pause = vi.spyOn(HTMLMediaElement.prototype, 'pause').mockImplementation(() => {})
    const view = render(
      <SubtitleSettingsPanel
        text="Hello"
        style={baseStyle}
        onTextChange={vi.fn()}
        onStyleChange={vi.fn()}
        speechSettings={{
          model: 'VoxCPM2',
          prompt: '',
          cfg: 2,
          steps: 10,
          referenceAudio: 'voices/sample.wav',
          referenceAudioSourceType: 'output',
        }}
      />,
    )

    fireEvent.mouseDown(screen.getByRole('tab', { name: /Speech/ }))
    const audio = screen.getByTestId('reference-audio-preview') as HTMLAudioElement
    expect(audio.getAttribute('src')).toBe('/view?filename=sample.wav&type=output&subfolder=voices')

    fireEvent.click(screen.getByRole('button', { name: 'Play reference audio' }))
    await waitFor(() => expect(play).toHaveBeenCalledOnce())
    expect(screen.getByRole('button', { name: 'Pause reference audio' })).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Pause reference audio' }))
    expect(pause).toHaveBeenCalled()
    expect(screen.getByRole('button', { name: 'Play reference audio' })).not.toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'Play reference audio' }))
    await waitFor(() => expect(play).toHaveBeenCalledTimes(2))
    fireEvent.ended(audio)
    expect(screen.getByRole('button', { name: 'Play reference audio' })).not.toBeNull()

    pause.mockClear()
    view.rerender(
      <SubtitleSettingsPanel
        text="Hello"
        style={baseStyle}
        onTextChange={vi.fn()}
        onStyleChange={vi.fn()}
        speechSettings={{
          model: 'VoxCPM2',
          prompt: '',
          cfg: 2,
          steps: 10,
          referenceAudio: '',
          referenceAudioSourceType: 'input',
        }}
      />,
    )
    await waitFor(() => expect(pause).toHaveBeenCalledOnce())
  })

  it('previews a legacy URL reference directly', () => {
    render(
      <SubtitleSettingsPanel
        text="Hello"
        style={baseStyle}
        onTextChange={vi.fn()}
        onStyleChange={vi.fn()}
        speechSettings={{
          model: 'VoxCPM2',
          prompt: '',
          cfg: 2,
          steps: 10,
          referenceAudio: 'https://example.com/reference.wav',
          referenceAudioSourceType: 'input',
        }}
      />,
    )

    expect(screen.getByTestId('reference-audio-preview').getAttribute('src'))
      .toBe('https://example.com/reference.wav')
  })

  it('keeps the speech button loading until generation settles', async () => {
    let resolveGeneration: (() => void) | undefined
    const onGenerateSpeech = vi.fn(() => new Promise<void>((resolve) => {
      resolveGeneration = resolve
    }))
    render(
      <SubtitleSettingsPanel
        text="Hello"
        style={baseStyle}
        onTextChange={vi.fn()}
        onStyleChange={vi.fn()}
        onGenerateSpeech={onGenerateSpeech}
      />,
    )

    fireEvent.mouseDown(screen.getByRole('tab', { name: /Speech/ }))
    fireEvent.click(screen.getByRole('button', { name: 'Speech' }))

    const button = screen.getByRole('button', { name: 'Speech' }) as HTMLButtonElement
    expect(button.disabled).toBe(true)
    expect(button.querySelector('.animate-spin')).not.toBeNull()

    resolveGeneration?.()

    await waitFor(() => {
      expect((screen.getByRole('button', { name: 'Speech' }) as HTMLButtonElement).disabled).toBe(false)
    })
    expect(onGenerateSpeech).toHaveBeenCalledWith(expect.objectContaining({
      model: 'VoxCPM2',
      cfg: 2,
      steps: 10,
    }))
  })
})
