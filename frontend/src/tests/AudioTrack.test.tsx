import type { ComponentProps, ReactNode } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { TooltipProvider } from '@/components/ui/tooltip'
import { AudioTrack } from '@/components/widgets/multitrack/AudioTrack'
import type { MultiTrack } from '@/types/multitrack'

vi.mock('@/components/widgets/mediaSelector/MediaSelector', () => ({
  MediaSelector: ({ value, defaultTab, slotItems, onChange }: {
    value: string
    defaultTab: string
    slotItems: Array<{ value: string }>
    onChange: (value: string, source?: 'input' | 'output') => void
  }) => (
    <button
      type="button"
      data-default-tab={defaultTab}
      data-slot-items={slotItems.map((item) => item.value).join(',')}
      onClick={() => onChange(value ? 'replacement.wav' : slotItems[0]?.value ?? 'new.wav', 'input')}
    >
      {value ? `replace ${value}` : slotItems.map((item) => item.value).join(',') || 'new.wav'}
    </button>
  ),
}))

vi.mock('@/components/widgets/multitrack/MultiTrackSegmentBlock', () => ({
  MultiTrackSegmentBlock: ({ segment, audioLocked, onAudioLockToggle, sharedReference, onSharedReferenceToggle, onDoubleClick }: {
    segment: { id: string; content?: { shared_reference?: boolean } }
    audioLocked?: boolean
    onAudioLockToggle?: (locked: boolean) => void
    sharedReference?: boolean
    onSharedReferenceToggle?: (enabled: boolean) => void
    onDoubleClick?: (segmentId: string, event: React.MouseEvent) => void
  }) => (
    <div>
      <button type="button" data-audio-locked={String(audioLocked === true)} onClick={() => onAudioLockToggle?.(!audioLocked)} onDoubleClick={(event) => onDoubleClick?.(segment.id, event)}>
        {segment.id}
      </button>
      <button type="button" aria-label={`shared-${segment.id}`} data-shared-reference={String(sharedReference === true)} onClick={() => onSharedReferenceToggle?.(!sharedReference)} />
    </div>
  ),
}))

vi.mock('@/components/widgets/multitrack/TrackAudioControls', () => ({
  TrackAudioControls: ({ icon }: { icon: ReactNode }) => <div>{icon}</div>,
}))

function renderAudioTrack(track: MultiTrack, props?: Partial<ComponentProps<typeof AudioTrack>>) {
  return render(
    <TooltipProvider>
      <AudioTrack
        track={track}
        totalLength={120}
        frameRate={24}
        width={480}
        canvasScale={1}
        selectedSegmentIds={new Set()}
        node={null}
        app={null}
        onAddAudio={vi.fn()}
        onReplaceAudio={vi.fn()}
        onSelectSegment={vi.fn()}
        onDeleteSegment={vi.fn()}
        onDeleteTrack={vi.fn()}
        onTrackAudioSettingsChange={vi.fn()}
        onResizeSegment={vi.fn()}
        onResizeSegmentPreview={vi.fn()}
        onMoveSegment={vi.fn()}
        onDragPreviewChange={vi.fn()}
        onDragPreviewEnd={vi.fn()}
        cutMode={false}
        onCutSegment={vi.fn()}
        {...props}
        onCloneSegment={props?.onCloneSegment ?? vi.fn()}
      />
    </TooltipProvider>,
  )
}

describe('AudioTrack', () => {
  it('adds audio using the exact internal gap range', () => {
    const track: MultiTrack = {
      id: 'audio-track',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      locked: false,
      segments: [
        {
          id: 'first', start_frame: 0, end_frame: 24, color: 'var(--highlight)',
          content: { media_type: 'audio', source_type: 'input', file_path: 'first.wav' },
        },
        {
          id: 'second', start_frame: 72, end_frame: 96, color: 'var(--highlight)',
          content: { media_type: 'audio', source_type: 'input', file_path: 'second.wav' },
        },
      ],
    }
    const onAddAudio = vi.fn()

    renderAudioTrack(track, { onAddAudio })
    const gapButton = screen.getByTestId('track-gap-add-24-72')
    expect(gapButton.getAttribute('aria-haspopup')).toBe('dialog')
    expect(gapButton.getAttribute('data-state')).toBe('closed')
    fireEvent.click(gapButton)
    expect(gapButton.getAttribute('data-state')).toBe('open')
    fireEvent.click(screen.getByRole('button', { name: 'new.wav' }))

    expect(onAddAudio).toHaveBeenCalledWith('audio-track', 'new.wav', 'input', undefined, 24, 72)
  })

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
    renderAudioTrack(track, { node, app, onAddAudio })

    fireEvent.click(screen.getByRole('button', { name: 'Add audio' }))
    fireEvent.click(screen.getByRole('button', { name: '__slot__:audio' }))

    expect(onAddAudio).toHaveBeenCalledWith(
      'audio-track',
      '__slot__:audio',
      'input',
      '/view?filename=voice.wav&type=input&subfolder=',
    )
  })

  it('hides the delete track control when an audio track has segments', () => {
    const track: MultiTrack = {
      id: 'audio-track',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      locked: false,
      segments: [{
        id: 'audio-segment',
        start_frame: 0,
        end_frame: 120,
        color: 'var(--highlight)',
        content: {
          media_type: 'audio',
          source_type: 'input',
          file_path: '__slot__:audio',
          file_name: 'voice.wav',
        },
      }],
    }

    renderAudioTrack(track)

    const addButton = screen.getByRole('button', { name: 'Add audio' })
    const actionGroup = addButton.parentElement

    expect(screen.queryByRole('button', { name: 'Delete Audio 0' })).toBeNull()
    expect(actionGroup?.classList.contains('flex')).toBe(true)
    expect(actionGroup?.classList.contains('gap-1')).toBe(true)
    expect(actionGroup?.style.left).toBe('486px')
  })

  it('marks every segment when its MiniMax audio track is locked', () => {
    const onTrackAudioSettingsChange = vi.fn()
    const track: MultiTrack = {
      id: 'audio-track', name: 'Audio 0', type: 'audio', color: 'var(--highlight)',
      muted: false, locked: false, audio_locked: true,
      segments: [
        { id: 'first', start_frame: 0, end_frame: 24, color: 'var(--highlight)', content: { media_type: 'audio' } },
        { id: 'second', start_frame: 24, end_frame: 48, color: 'var(--highlight)', content: { media_type: 'audio' } },
      ],
    }

    renderAudioTrack(track, { audioLockEnabled: true, onTrackAudioSettingsChange })

    expect(screen.getByRole('button', { name: 'first' }).dataset.audioLocked).toBe('true')
    expect(screen.getByRole('button', { name: 'second' }).dataset.audioLocked).toBe('true')
    fireEvent.click(screen.getByRole('button', { name: 'first' }))
    expect(onTrackAudioSettingsChange).toHaveBeenCalledWith('audio-track', { audio_locked: false })
  })

  it('toggles the shared reference for the selected audio segment', () => {
    const onSharedReferenceChange = vi.fn()
    const track: MultiTrack = {
      id: 'audio-track', name: 'Audio 0', type: 'audio', color: 'var(--highlight)',
      muted: false, locked: false,
      segments: [{
        id: 'voice', start_frame: 0, end_frame: 24, color: 'var(--highlight)',
        content: { media_type: 'audio', shared_reference: true },
      }],
    }

    renderAudioTrack(track, { audioLockEnabled: true, onSharedReferenceChange })

    const sharedButton = screen.getByRole('button', { name: 'shared-voice' })
    expect(sharedButton.dataset.sharedReference).toBe('true')
    fireEvent.click(sharedButton)
    expect(onSharedReferenceChange).toHaveBeenCalledWith('audio-track', 'voice', false)
  })

  it('opens the current audio in the media selector on double click and replaces it', () => {
    const track: MultiTrack = {
      id: 'audio-track',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      locked: false,
      segments: [{
        id: 'audio-segment',
        start_frame: 24,
        end_frame: 120,
        color: 'var(--highlight)',
        content: {
          media_type: 'audio',
          source_type: 'output',
          file_path: 'renders/original.wav',
          file_name: 'original.wav',
        },
      }],
    }
    const onReplaceAudio = vi.fn()

    renderAudioTrack(track, { onReplaceAudio })
    fireEvent.doubleClick(screen.getByRole('button', { name: 'audio-segment' }), {
      clientX: 20,
      clientY: 10,
    })

    const selector = screen.getByRole('button', { name: 'replace renders/original.wav' })
    expect(selector.getAttribute('data-default-tab')).toBe('outputs')
    fireEvent.click(selector)

    expect(onReplaceAudio).toHaveBeenCalledWith(
      'audio-track',
      'audio-segment',
      'replacement.wav',
      'input',
      undefined,
    )
  })

  it('refreshes connected audio slots when the reselect popover opens', () => {
    const track: MultiTrack = {
      id: 'audio-track',
      name: 'Audio 0',
      type: 'audio',
      color: 'var(--highlight)',
      muted: false,
      locked: false,
      segments: [{
        id: 'audio-segment',
        start_frame: 0,
        end_frame: 48,
        color: 'var(--highlight)',
        content: { media_type: 'audio', source_type: 'input', file_path: 'original.wav' },
      }],
    }
    const sourceNode = {
      type: 'LoadAudio',
      outputs: [{ shape: 0 }],
      widgets_values: ['connected.wav'],
    }
    const node = { inputs: [{ name: 'audio', type: 'AUDIO', link: null as number | null }] }
    const app = {
      graph: {
        links: {} as Record<number, { origin_id: number; origin_slot: number }>,
        getNodeById: () => sourceNode,
      },
    }

    renderAudioTrack(track, { node, app })
    node.inputs[0].link = 7
    app.graph.links[7] = { origin_id: 3, origin_slot: 0 }
    fireEvent.doubleClick(screen.getByRole('button', { name: 'audio-segment' }))

    expect(screen.getByRole('button', { name: 'replace original.wav' }).getAttribute('data-slot-items'))
      .toContain('__slot__:audio')
  })
})
