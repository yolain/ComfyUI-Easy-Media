import { describe, expect, it } from 'vitest'
import { computeSlotItems } from '@/lib/timeline-utils'

function makeGraph(
  mediaType: 'IMAGE' | 'AUDIO' | 'VIDEO',
  legacy = false,
  connectedInputIndexes = [1],
  skipEmpty = mediaType !== 'AUDIO',
) {
  const rootNode = {
    outputs: [{ shape: 0, link: null }],
    imgs: mediaType === 'IMAGE' ? [{ currentSrc: '/view?filename=ref.png' }] : undefined,
    widgets_values: mediaType === 'AUDIO' ? ['voice.wav'] : undefined,
    type: mediaType === 'AUDIO' ? 'LoadAudio' : `Load${mediaType}`,
  }
  const listNode = {
    outputs: [{ shape: 6 }],
    inputs: [
      { name: 'skip_empty', type: 'BOOLEAN', link: null },
      { name: `${mediaType.toLowerCase()}1`, type: mediaType, link: legacy && connectedInputIndexes.includes(1) ? 0 : undefined },
      { name: `${mediaType.toLowerCase()}2`, type: mediaType, link: null },
      { name: `${mediaType.toLowerCase()}3`, type: mediaType, link: null },
    ],
    widgets_values: [skipEmpty],
    getInputLink: legacy ? undefined : (index: number) => connectedInputIndexes.includes(index)
      ? { origin_id: 1, origin_slot: 0 }
      : null,
  }
  const nodes: Record<number, object> = { 1: rootNode, 2: listNode }
  const graph = {
    links: {
      0: { origin_id: 1, origin_slot: 0 },
      7: { origin_id: 2, origin_slot: 0 },
    },
    getNodeById: (id: number) => nodes[id],
  }
  return {
    app: { graph },
    node: {
      inputs: [{ name: mediaType.toLowerCase(), type: mediaType, link: legacy ? 7 : undefined }],
      getInputLink: legacy ? undefined : () => ({ origin_id: 2, origin_slot: 0 }),
    },
  }
}

describe('computeSlotItems', () => {
  it.each([
    ['image', 'IMAGE', '__slot__:image1'],
    ['audio', 'AUDIO', '__slot__:audio1'],
    ['video', 'VIDEO', '__slot__:video1'],
  ] as const)('discovers connected %s list inputs without relying on isConnected', (mediaType, graphType, expected) => {
    const { node, app } = makeGraph(graphType)

    expect(computeSlotItems(node, app, mediaType).map((item) => item.value)).toEqual([expected])
  })

  it('traces link id zero when resolving a slot preview', () => {
    const { node, app } = makeGraph('IMAGE')

    expect(computeSlotItems(node, app, 'image')[0]?.img).toBe('/view?filename=ref.png')
  })

  it('keeps compatibility with legacy input.link graphs', () => {
    const { node, app } = makeGraph('VIDEO', true)

    expect(computeSlotItems(node, app, 'video')[0]?.value).toBe('__slot__:video1')
  })

  it.each([
    ['image', 'IMAGE', '__slot__:image1'],
    ['video', 'VIDEO', '__slot__:video1'],
  ] as const)('uses the compacted output index for sparse %s lists that skip empty inputs', (mediaType, graphType, expected) => {
    const { node, app } = makeGraph(graphType, false, [2], true)

    expect(computeSlotItems(node, app, mediaType)[0]?.value).toBe(expected)
  })

  it('preserves the declared input index when an audio list fills empty inputs', () => {
    const { node, app } = makeGraph('AUDIO', false, [2], false)

    expect(computeSlotItems(node, app, 'audio')[0]?.value).toBe('__slot__:audio2')
  })
})
