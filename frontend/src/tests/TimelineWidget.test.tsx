import { fireEvent, render, screen, waitFor } from '@testing-library/react'
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
  MaintainTrack: ({ track, onSelectedIdChange }: {
    track: TimelineData['tracks'][number]
    onSelectedIdChange: (id: string) => void
  }) => (
    <button
      type="button"
      onClick={(event) => {
        event.stopPropagation()
        onSelectedIdChange(track.segments[0]?.id)
      }}
    >
      select segment
    </button>
  ),
}))

vi.mock('@/components/widgets/timeline/AudioTrack', () => ({
  AudioTrack: () => <div />,
}))

vi.mock('@/components/widgets/timeline/EditPanel', () => ({
  EditPanel: () => <div data-testid="edit-panel" />,
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

  it('lets the segment editor render outside the widget bounds while a segment is selected', async () => {
    const value = createDefaultTimelineData()
    value.tracks[0].segments = [{
      id: 'segment-1',
      start_frame: 0,
      end_frame: 120,
      content: { text: '', images: [], type: 'flf' },
      color: value.tracks[0].color,
    }]

    const { container } = render(
      <div className="comfyui-react-widget comfyui-easy-media">
        <TimelineWidget
          value={value}
          onChange={vi.fn()}
          inputName="timeline"
          node={{} as ReactWidgetProps<TimelineData>['node']}
          widget={{} as ReactWidgetProps<TimelineData>['widget']}
          app={{
            ui: { settings: { settingsValues: {} } },
          } as ReactWidgetProps<TimelineData>['app']}
        />
      </div>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'select segment' }))

    expect(screen.getByTestId('edit-panel')).not.toBeNull()
    await waitFor(() => {
      expect(container.firstElementChild?.className).toContain('-timeline-editor-open')
    })
  })
})
