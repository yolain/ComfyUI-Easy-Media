import type { ReactNode } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { AudioTrack } from '@/components/widgets/multitrack/AudioTrack'
import type { MultiTrack } from '@/types/multitrack'

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: ({ slotItems, onChange }: {
    slotItems: Array<{ value: string }>
    onChange: (value: string) => void
  }) => (
    <button type="button" onClick={() => onChange(slotItems[0]?.value ?? '')}>
      {slotItems.map((item) => item.value).join(',')}
    </button>
  ),
}))

vi.mock('@/components/widgets/multitrack/MultiTrackSegmentBlock', () => ({
  MultiTrackSegmentBlock: () => null,
}))

vi.mock('@/components/widgets/multitrack/TrackAudioControls', () => ({
  TrackAudioControls: ({ icon }: { icon: ReactNode }) => <div>{icon}</div>,
}))

describe('AudioTrack', () => {
  it('shows connected audio inputs in the slot selector', () => {
    const track: MultiTrack = {
      id: 'audio-track',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      locked: false,
      segments: [],
    }
    const sourceNode = {
      type: 'LoadAudio',
      outputs: [{ shape: 0 }],
      widgets_values: ['voice.wav'],
    }
    const node = { inputs: [{ name: 'audio', type: 'AUDIO', link: 7 }] }
    const app = {
      graph: {
        links: { 7: { origin_id: 3, origin_slot: 0 } },
        getNodeById: () => sourceNode,
      },
    }

    const onAddAudio = vi.fn()
    render(
      <TooltipProvider>
        <AudioTrack
          track={track}
          totalLength={120}
          frameRate={24}
          width={480}
          canvasScale={1}
          selectedSegmentId={null}
          node={node}
          app={app}
          onAddAudio={onAddAudio}
          onSelectSegment={vi.fn()}
          onDeleteSegment={vi.fn()}
          onDeleteTrack={vi.fn()}
          onTrackAudioSettingsChange={vi.fn()}
          onResizeSegment={vi.fn()}
          onMoveSegment={vi.fn()}
          onDragPreviewChange={vi.fn()}
          onDragPreviewEnd={vi.fn()}
          cutMode={false}
          onCutSegment={vi.fn()}
        />
      </TooltipProvider>,
    )

    fireEvent.click(screen.getByRole('button', { name: 'Add audio' }))
    fireEvent.click(screen.getByRole('button', { name: '__slot__:audio' }))

    expect(onAddAudio).toHaveBeenCalledWith(
      'audio-track',
      '__slot__:audio',
      'input',
      '/view?filename=voice.wav&type=input&subfolder=',
    )
  })
})
