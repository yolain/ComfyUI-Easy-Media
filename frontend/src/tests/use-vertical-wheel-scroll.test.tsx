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
    <div ref={ref} data-testid="sliders"><span data-testid="slider-content" /></div>
  </div>
}
function setDimensions(element: HTMLElement, scrollHeight = 1000, clientHeight = 200) {
  Object.defineProperties(element, { scrollHeight:{configurable:true,value:scrollHeight}, clientHeight:{configurable:true,value:clientHeight} })
}
afterEach(cleanup)

describe('vertical scrolling inside node widgets', () => {
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
    fireEvent.wheel(sliders,{deltaY:100})
    expect(sliders.scrollTop).toBe(0)
  })
})
