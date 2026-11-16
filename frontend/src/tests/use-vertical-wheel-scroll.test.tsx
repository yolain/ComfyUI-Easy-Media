import { useRef } from 'react'
import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useVerticalWheelScroll } from '@/hooks/use-vertical-wheel-scroll'

function Panel({ cancelInCapture = false, onCanvasWheel = () => {} }: {
  cancelInCapture?: boolean
  onCanvasWheel?: () => void
}) {
  const ref = useRef<HTMLDivElement>(null)
  useVerticalWheelScroll(ref)
  return <div onWheelCapture={event => { if (cancelInCapture) event.preventDefault() }} onWheel={onCanvasWheel}>
    <div data-testid="toolbar" />
    <div ref={ref} data-testid="sliders"><span role="slider" tabIndex={0} aria-label="Effect" data-testid="slider-content" /></div>
  </div>
}
function setDimensions(element: HTMLElement, scrollHeight = 1000, clientHeight = 200) {
  Object.defineProperties(element, { scrollHeight:{configurable:true,value:scrollHeight}, clientHeight:{configurable:true,value:clientHeight} })
}
afterEach(cleanup)

describe('vertical scrolling inside node widgets', () => {
  it('runs before a host capture listener that truncates propagation while a slider is focused', () => {
    const hostCapture = vi.fn((event: Event) => {
      event.preventDefault()
      event.stopImmediatePropagation()
    })
    document.addEventListener('wheel', hostCapture, { capture:true, passive:false })
    try {
      render(<Panel />)
      const sliders = screen.getByTestId('sliders')
      const thumb = screen.getByRole('slider', {name:'Effect'})
      setDimensions(sliders)
      thumb.focus()
      expect(document.activeElement).toBe(thumb)
      fireEvent.wheel(thumb,{deltaY:120})
      expect(sliders.scrollTop).toBe(120)
      expect(document.activeElement).toBe(thumb)
      expect(hostCapture).not.toHaveBeenCalled()
    } finally {
      document.removeEventListener('wheel', hostCapture, true)
    }
  })

  it('does not hijack an outside wheel event just because its slider keeps focus', () => {
    render(<Panel />)
    const sliders = screen.getByTestId('sliders')
    setDimensions(sliders)
    screen.getByRole('slider',{name:'Effect'}).focus()
    const outside = new WheelEvent('wheel',{deltaY:100,bubbles:true,cancelable:true})
    fireEvent(screen.getByTestId('toolbar'),outside)
    expect(outside.defaultPrevented).toBe(false)
    expect(sliders.scrollTop).toBe(0)
  })

  it('routes wheel events only to the panel under the pointer when several nodes exist', () => {
    render(<><Panel /><Panel /></>)
    const [first,second] = screen.getAllByTestId('sliders')
    setDimensions(first); setDimensions(second)
    fireEvent.wheel(second,{deltaY:100})
    expect(first.scrollTop).toBe(0)
    expect(second.scrollTop).toBe(100)
  })

  it('scrolls when nodeV2 has already cancelled the wheel default', () => {
    const canvasWheel = vi.fn()
    render(<Panel cancelInCapture onCanvasWheel={canvasWheel} />)
    const sliders = screen.getByTestId('sliders')
    setDimensions(sliders)
    const wheel = new WheelEvent('wheel',{deltaY:120,bubbles:true,cancelable:true})
    fireEvent(screen.getByTestId('slider-content'),wheel)
    expect(wheel.defaultPrevented).toBe(true)
    expect(sliders.scrollTop).toBe(120)
    expect(canvasWheel).not.toHaveBeenCalled()
    expect(screen.getByTestId('toolbar').scrollTop).toBe(0)
  })

  it('cancels the legacy default to avoid scrolling twice and clamps both boundaries', () => {
    render(<Panel />)
    const sliders = screen.getByTestId('sliders')
    setDimensions(sliders)
    const wheel = new WheelEvent('wheel',{deltaY:900,bubbles:true,cancelable:true})
    fireEvent(sliders,wheel)
    expect(wheel.defaultPrevented).toBe(true)
    expect(sliders.scrollTop).toBe(800)
    fireEvent.wheel(sliders,{deltaY:-1000})
    expect(sliders.scrollTop).toBe(0)
  })

  it('converts line and page wheel units', () => {
    render(<Panel />)
    const sliders = screen.getByTestId('sliders')
    setDimensions(sliders)
    fireEvent.wheel(sliders,{deltaY:2,deltaMode:1})
    expect(sliders.scrollTop).toBe(32)
    fireEvent.wheel(sliders,{deltaY:1,deltaMode:2})
    expect(sliders.scrollTop).toBe(232)
  })

  it.each([{ctrlKey:true},{metaKey:true},{deltaY:0,deltaX:100}])('leaves zoom or horizontal gestures alone: %o', modifiers => {
    render(<Panel />)
    const sliders = screen.getByTestId('sliders')
    setDimensions(sliders)
    const wheel = new WheelEvent('wheel',{deltaY:100,bubbles:true,cancelable:true,...modifiers})
    fireEvent(sliders,wheel)
    expect(wheel.defaultPrevented).toBe(false)
    expect(sliders.scrollTop).toBe(0)
  })

  it('removes the wheel listener on unmount', () => {
    const {unmount} = render(<Panel />)
    const sliders = screen.getByTestId('sliders')
    setDimensions(sliders)
    unmount()
    // Reattach the detached DOM so a leaked window listener would receive it.
    document.body.append(sliders)
    fireEvent.wheel(sliders,{deltaY:100})
    expect(sliders.scrollTop).toBe(0)
    sliders.remove()
  })
})
