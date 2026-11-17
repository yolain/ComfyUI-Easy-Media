import { describe, expect, it, vi } from 'vitest'
import { fireEvent, render, screen, cleanup } from '@testing-library/react'
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import { DEFAULT_BEAUTY_PRESET, normalizeBeautyPreset } from '@/lib/face-beauty'
import { preserveFaceBeautyNodeSize } from '@/lib/face-beauty-node-size'
import { FaceBeautyWidget } from '@/components/widgets/FaceBeautyWidget'
import { Button } from '@/components/ui/button'
import { CompareVideo } from '@/components/widgets/compareVideoWidget'

const { downloadModel } = vi.hoisted(() => ({ downloadModel: vi.fn() }))
vi.mock('@/hooks/use-face-beauty-preview', () => ({
  useFaceBeautyPreview: () => ({ preview: null, error: '', loading: false, canvasRef: { current: null }, connected: true, hasImageConnection: true, downloadModel }),
}))
vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: ({ onSourceChange }: { onSourceChange: (selection: { filePath: string; sourceType: string }) => void }) =>
    <Button onClick={() => onSourceChange({ filePath: 'portrait.png', sourceType: 'input' })}>Choose test portrait</Button>,
}))
vi.mock('@/lib/comfy-api' , () => ({ $error: vi.fn() }))
beforeAll(() => { vi.stubGlobal("ResizeObserver", class { observe() {} unobserve() {} disconnect() {} }) })
afterEach(cleanup)
const app = { ui: { settings: { settingsValues: {} } } } as unknown as ComfyApp

describe('face beauty preset', () => {
  it('round trips all settings and the selected source', () => {
    const value = { ...DEFAULT_BEAUTY_PRESET, parameters: { ...DEFAULT_BEAUTY_PRESET.parameters, eyes: .4 }, source: { source_type: 'input', file_path: 'portrait.png' } }
    expect(normalizeBeautyPreset(JSON.parse(JSON.stringify(value)))).toEqual(value)
    expect(normalizeBeautyPreset({ parameters: { eyes: 9, smooth: -2 } }).parameters).toMatchObject({ eyes: 1, smooth: 0 })
  })
  it.each([null, { version: 2 }, { source: 0 }, { colors: { lip: null } }, { parameters: { eyes: NaN } }, { parameters: [] }, { source: { source_type: 'local', file_path: '/etc/test' } }])('rejects malformed presets', value => {
    expect(() => normalizeBeautyPreset(value)).toThrow()
  })
  it('restores older presets with neutral makeup and validates saved colors', () => {
    const preset = normalizeBeautyPreset({ parameters: { eyes: .3 }, colors: { lip: '#AA3355', blush: '#ff9f80' } })
    expect(preset.parameters).toMatchObject({ eyes:.3, eye_bright:0, under_eye:0, lip_tint:0, blush:0 })
    expect(preset.colors.lip).toBe('#aa3355')
    expect(() => normalizeBeautyPreset({ colors: { lip:'red' } })).toThrow()
  })
  it('assigns a default size while preserving creation hooks', () => {
    const original = vi.fn()
    class Node {
      size: [number, number] = [300, 200]
      onNodeCreated = original
    }
    Node.prototype.onNodeCreated = original
    // Instance fields would shadow the prototype in the test, so use a plain prototype instance.
    const node = Object.create(Node.prototype) as Node
    preserveFaceBeautyNodeSize(Node, { name: 'easy faceBeautify' })
    node.onNodeCreated()
    expect(original).toHaveBeenCalledOnce()
    expect(node.size).toEqual([960, 640])
    node.size = [1100, 720]
    expect(node.size).toEqual([1100, 720])
  })
})

describe('image comparison reuses the existing comparison modes', () => {
  it('renders images and a live canvas without video playback controls', () => {
    const { container } = render(<CompareVideo app={app} sourceUrl="/before.png" outputUrl="/after.png" mediaType="image" outputContent={<canvas data-testid="beauty-canvas" />} />)
    expect(container.querySelectorAll('video')).toHaveLength(0)
    expect(container.querySelector('img')?.src).toContain('/before.png')
    expect(screen.getByTestId('beauty-canvas')).toBeDefined()
    expect(container.querySelector('[data-compare-video-toolbar]')).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'A/B' }))
    expect(screen.getByRole('button', { name: 'Source' })).toBeDefined()
  })
})


describe('face beauty toolbar', () => {
  it('keeps portrait selection available with an IMAGE connection and switches the processing source', () => {
    const onChange = vi.fn()
    render(<FaceBeautyWidget app={app} value={DEFAULT_BEAUTY_PRESET} onChange={onChange} node={{}} widget={{} as never} inputName="preset" />)
    expect(screen.getAllByRole('heading').map(heading => heading.textContent)).toEqual(['Face'])
    const toolbar = document.querySelector('[data-face-beauty-toolbar]')!
    const sliders = document.querySelector('[data-face-beauty-sliders]')!
    expect(sliders.contains(toolbar)).toBe(false)
    expect(sliders.classList.contains('overflow-y-auto')).toBe(true)
    expect(toolbar.classList.contains('shrink-0')).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Skin & makeup' }))
    expect(screen.getByRole('slider', { name: 'Eye & teeth whiten' }).getAttribute('aria-valuemin')).toBe('0')
    const select = screen.getByRole('button', { name: 'Select portrait' }) as HTMLButtonElement
    expect(select.disabled).toBe(false)
    const download = screen.getByRole('button', { name: 'Download face model' })
    expect(download.textContent).toBe('')
    expect(download.hasAttribute('title')).toBe(false)
    fireEvent.click(download)
    expect(downloadModel).toHaveBeenCalledOnce()
    fireEvent.click(screen.getByRole('button', { name: 'Lip color' }))
    const colorInput = screen.getByRole('textbox', { name: 'Hex value' })
    expect(colorInput).toBeDefined()
    fireEvent.keyDown(colorInput, { key: 'Escape' })
    fireEvent.click(select)
    fireEvent.click(screen.getByRole('button', { name: 'Choose test portrait' }))
    expect(onChange).toHaveBeenCalledWith(expect.objectContaining({ source_mode: 'picker', source: { source_type: 'input', file_path: 'portrait.png' } }))
  })
})


describe('face beauty categories', () => {
  it('shows face controls and keeps category buttons outside scrolling content', () => {
    const {container} = render(<FaceBeautyWidget app={app} value={DEFAULT_BEAUTY_PRESET} onChange={vi.fn()} node={{}} widget={{} as never} inputName="preset" />)
    expect(screen.getAllByRole('slider').map(slider=>slider.getAttribute('aria-label'))).toEqual([
      'Slim face','Small face','Narrow face','Temples','Cheekbones','Jaw','Chin length','Pointed chin',
      'Contour smoothing','Jawline','V face','Face vertical position','Face horizontal position','Face rotation',
    ])
    const categories = screen.getByRole('group',{name:'Adjustment categories'})
    expect(container.querySelector('[data-face-beauty-sliders]')?.contains(categories)).toBe(false)
    expect(screen.queryByRole('slider',{name:'Small head'})).toBeNull()
    expect(screen.queryByRole('slider',{name:'Hairline'})).toBeNull()
    fireEvent.click(screen.getByRole('button',{name:'Eyes'}))
    expect(screen.getAllByRole('slider')).toHaveLength(11)
    expect(screen.getByRole('slider',{name:'Eye size'})).toBeDefined()
  })
  it('switches categories without losing preset values', () => {
    const onChange = vi.fn()
    const props = {app,onChange,node:{},widget:{} as never,inputName:'preset'}
    const {rerender} = render(<FaceBeautyWidget {...props} value={DEFAULT_BEAUTY_PRESET} />)
    const slider = screen.getByRole('slider',{name:'Small face'})
    fireEvent.keyDown(slider,{key:'ArrowRight'})
    const changed = onChange.mock.calls.at(-1)![0]
    expect(changed.parameters.small_face).toBe(.01)
    rerender(<FaceBeautyWidget {...props} value={changed} />)
    fireEvent.click(screen.getByRole('button',{name:'Nose'}))
    expect(screen.getAllByRole('slider')).toHaveLength(7)
    fireEvent.click(screen.getByRole('button',{name:'Face'}))
    expect(screen.getByRole('slider',{name:'Small face'}).getAttribute('aria-valuenow')).toBe('0.01')
  })
})
