import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { TrackArea } from '@/components/widgets/multitrack/TrackArea'
import { createDefaultTrackData } from '@/lib/multitrack-utils'

vi.mock('@/components/widgets/multitrack/VideoTrack', () => ({
  VideoTrack: () => <div data-testid="video-track" />,
}))

vi.mock('@/components/widgets/multitrack/AudioTrack', () => ({
  AudioTrack: ({ node, app }: { node: unknown; app: unknown }) => (
    <div data-testid="audio-track">{node && app ? 'connected' : 'missing'}</div>
  ),
}))

vi.mock('@/components/widgets/multitrack/MultiTrackSegmentBlock', () => ({
  MultiTrackSegmentBlock: () => null,
}))

describe('TrackArea track controls', () => {
  it('renders the add-track bar with only audio enabled', () => {
    const onAddTrack = vi.fn()
    render(
      <TooltipProvider>
        <TrackArea
          data={createDefaultTrackData()}
          width={480}
          currentTime={0}
          canvasScale={1}
          selectedSegmentId={null}
          node={{}}
          app={{}}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onAddTrack={onAddTrack}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    const addTrackLabel = screen.getByText('Add track:')
    expect(addTrackLabel.parentElement?.className).toContain('border-b')
    expect((screen.getByRole('button', { name: 'Add video track' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Add subtitle track' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Add audio track' }))
    expect(onAddTrack).toHaveBeenCalledWith('audio')
  })

  it('passes ComfyUI graph context to each audio track', () => {
    const data = createDefaultTrackData()
    data.tracks.push({
      id: 'audio-track',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      locked: false,
      segments: [],
    })

    render(
      <TooltipProvider>
        <TrackArea
          data={data}
          width={480}
          currentTime={0}
          canvasScale={1}
          selectedSegmentId={null}
          node={{ inputs: [] }}
          app={{ graph: {} }}
          onAddVideo={vi.fn()}
          onAddAudio={vi.fn()}
          onAddTrack={vi.fn()}
          onReplaceVideo={vi.fn()}
          onAddTaskSegment={vi.fn()}
          onSelectSegment={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onDistributeTaskSegments={vi.fn()}
          onCloneTaskSegment={vi.fn()}
          onResizeSegment={vi.fn()}
          onMoveSegment={vi.fn()}
          onSmartSplit={vi.fn()}
          onSmartSplitTasks={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    expect(screen.getByTestId('audio-track').textContent).toBe('connected')
  })
})
