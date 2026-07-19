import { render } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TimelineWidget } from '@/components/widgets/TimelineWidget'
import { createDefaultTimelineData } from '@/lib/timeline-utils'
import type { ReactWidgetProps } from '@/lib/create-react-widget'
import type { TimelineData } from '@/types/timeline'

vi.mock('@/hooks/use-canvas-scale', () => ({
  useCanvasScale: () => 1,
}))

vi.mock('@/hooks/use-element-width', () => ({
  useElementWidth: () => 480,
}))

vi.mock('@/components/widgets/timeline/MaintainTrack', () => ({
  MaintainTrack: () => <div />,
}))

vi.mock('@/components/widgets/timeline/AudioTrack', () => ({
  AudioTrack: () => <div />,
}))

vi.mock('@/components/widgets/timeline/EditPanel', () => ({
  EditPanel: () => <div />,
}))

vi.mock('@/components/widgets/timeline/Toolbar', () => ({
  Toolbar: () => <div />,
}))

vi.mock('@/components/widgets/timeline/TimelineRuler', () => ({
  TimelineRuler: () => <div />,
}))

vi.mock('@/components/ui/tooltip', () => ({
  TooltipProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}))

describe('TimelineWidget', () => {
  it('stays enabled and leaves connection events unchanged when prompt_override is connected', () => {
    const onConnectionsChange = vi.fn()
    const node = {
      inputs: [{ name: 'prompt_override', link: 42 }],
      onConnectionsChange,
    }
    const widget = { disabled: false }

    render(
      <TimelineWidget
        value={createDefaultTimelineData()}
        onChange={vi.fn()}
        inputName="timeline"
        node={node as ReactWidgetProps<TimelineData>['node']}
        widget={widget as unknown as ReactWidgetProps<TimelineData>['widget']}
        app={{
          ui: { settings: { settingsValues: {} } },
        } as ReactWidgetProps<TimelineData>['app']}
      />,
    )

    expect(widget.disabled).toBe(false)
    expect(node.onConnectionsChange).toBe(onConnectionsChange)
  })
})
