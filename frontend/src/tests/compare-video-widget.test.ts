import { createElement } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { CompareVideoWidget, parseCompareVideoPayload } from '@/components/widgets/compareVideoWidget'

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

describe('CompareVideoWidget', () => {
  function widgetProps(node: object, onChange = vi.fn()) {
    return {
      app: {
        api: {},
        canvas: { ds: { scale: 1 } },
        ui: { settings: { settingsValues: {} } },
      } as never,
      node,
      value: { save_output: false, filename_prefix: 'ComfyUI' },
      onChange,
      inputName: 'compare_video',
      widget: {} as never,
    }
  }

  it('starts with audio enabled and uses output audio for comparison', () => {
    vi.stubGlobal('ResizeObserver', class {
      observe() {}
      unobserve() {}
      disconnect() {}
    })
    const node = {
      id: 1,
      __easyMediaCompareVideos: { source, output },
    }
    render(createElement(CompareVideoWidget, widgetProps(node)))

    const videos = document.querySelectorAll('video')
    expect(videos).toHaveLength(2)
    expect(videos[0].muted).toBe(true)
    expect(videos[1].muted).toBe(false)
    expect(screen.getByRole('button', { name: 'Mute audio' })).not.toBeNull()
  })

  it('edits save settings in the empty state', () => {
    const onChange = vi.fn()
    render(createElement(CompareVideoWidget, widgetProps({ id: 2 }, onChange)))

    const saveOutputCheckbox = screen.getByRole('checkbox', { name: 'Save output video' })
    expect(saveOutputCheckbox.getAttribute('aria-checked')).toBe('false')
    fireEvent.click(saveOutputCheckbox)
    expect(onChange).toHaveBeenCalledWith({ save_output: true, filename_prefix: 'ComfyUI' })

    fireEvent.change(screen.getByRole('textbox', { name: 'Save prefix' }), { target: { value: 'renders/final' } })
    expect(onChange).toHaveBeenCalledWith({ save_output: false, filename_prefix: 'renders/final' })
  })

  it('shows output settings and download controls after preview', () => {
    const onChange = vi.fn()
    let clickedAnchor: HTMLAnchorElement | null = null
    const anchorClick = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function captureAnchor(this: HTMLAnchorElement) {
      clickedAnchor = this
    })
    render(createElement(CompareVideoWidget, widgetProps({
      id: 3,
      __easyMediaCompareVideos: { source, output },
    }, onChange)))

    expect(screen.getByRole('button', { name: 'Download output video' })).not.toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Output settings' }))
    fireEvent.change(screen.getByRole('textbox', { name: 'Save prefix' }), { target: { value: 'compare' } })
    expect(onChange).toHaveBeenCalledWith({ save_output: false, filename_prefix: 'compare' })

    fireEvent.click(screen.getByRole('button', { name: 'Download output video' }))
    expect(anchorClick).toHaveBeenCalledOnce()
    expect(clickedAnchor).not.toBeNull()
    expect((clickedAnchor as HTMLAnchorElement | null)?.download).toBe('output.mp4')
    expect((clickedAnchor as HTMLAnchorElement | null)?.href).toContain('/view?filename=output.mp4')
    anchorClick.mockRestore()
  })

  it('pins the toolbar to the bottom and reveals it on hover or keyboard focus', () => {
    render(createElement(CompareVideoWidget, widgetProps({
      id: 5,
      __easyMediaCompareVideos: { source, output },
    })))

    const toolbar = document.querySelector('[data-compare-video-toolbar]')
    expect(toolbar).not.toBeNull()
    expect(toolbar?.className).toContain('absolute')
    expect(toolbar?.className).toContain('bottom-0')
    expect(toolbar?.className).toContain('opacity-0')
    expect(toolbar?.className).toContain('transition-[opacity,transform]')
    expect(toolbar?.className).toContain('group-hover:opacity-100')
    expect(toolbar?.className).toContain('group-focus-within:opacity-100')
  })

  it('hides the download control when no output video is connected', () => {
    render(createElement(CompareVideoWidget, widgetProps({
      id: 4,
      __easyMediaCompareVideos: { source },
    })))

    expect(screen.queryByRole('button', { name: 'Download output video' })).toBeNull()
  })
})
