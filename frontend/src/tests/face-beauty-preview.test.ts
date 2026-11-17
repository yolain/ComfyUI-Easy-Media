import { act, cleanup, renderHook, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { useFaceBeautyPreview, type BeautyNode } from '@/hooks/use-face-beauty-preview'
import { DEFAULT_BEAUTY_PRESET, type BeautyPreset, type BeautyPreview } from '@/lib/face-beauty'

vi.mock('@/lib/face-beauty-renderer', () => ({ createBeautyRenderer: vi.fn() }))

function preview(filename: string, inputConnected: boolean): BeautyPreview {
  return { filename, type:'temp', geometry:{handles:[],regions:[]}, face_found:false, input_connected:inputConnected }
}
afterEach(() => { cleanup(); vi.unstubAllGlobals() })

describe('face beauty preview source changes', () => {
  it('clears the previous preview when a different IMAGE is connected', () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch',fetchMock)
    const node: BeautyNode = { inputs:[{ name:'image',link:1 }] }
    const { result } = renderHook(() => useFaceBeautyPreview(node,DEFAULT_BEAUTY_PRESET))
    act(() => node.onExecuted?.({face_beauty:[preview('first.png',true)]}))
    expect(result.current.preview?.filename).toBe('first.png')
    act(() => {
      node.inputs![0].link = 2
      node.onConnectionsChange?.()
    })
    expect(result.current.preview).toBeNull()
    expect(fetchMock).not.toHaveBeenCalled()
    act(() => node.onExecuted?.({face_beauty:[preview('second.png',true)]}))
    expect(result.current.preview?.filename).toBe('second.png')
  })

  it('discards a slow preparation response after selecting another portrait', async () => {
    let resolveFirst!: (response: Response) => void
    const first = new Promise<Response>(resolve => { resolveFirst = resolve })
    const fetchMock = vi.fn().mockReturnValueOnce(first).mockResolvedValueOnce({ok:true,json:async()=>preview('second.png',false)})
    vi.stubGlobal('fetch',fetchMock)
    const node: BeautyNode = {}
    const preset: BeautyPreset = {...DEFAULT_BEAUTY_PRESET,source:{source_type:'input',file_path:'first.png'}}
    const { result,rerender } = renderHook(({value}) => useFaceBeautyPreview(node,value),{initialProps:{value:preset}})
    const firstSignal = fetchMock.mock.calls[0][1].signal as AbortSignal
    rerender({value:{...preset,source:{source_type:'input',file_path:'second.png'}}})
    await waitFor(() => expect(result.current.preview?.filename).toBe('second.png'))
    expect(firstSignal.aborted).toBe(true)
    await act(async () => { resolveFirst({ok:true,json:async()=>preview('first.png',false)} as Response); await first })
    expect(result.current.preview?.filename).toBe('second.png')
  })

  it('restores node callbacks when the widget unmounts', () => {
    const originalExecuted = vi.fn()
    const originalConnections = vi.fn()
    const node: BeautyNode = {onExecuted:originalExecuted,onConnectionsChange:originalConnections}
    const {unmount} = renderHook(() => useFaceBeautyPreview(node,DEFAULT_BEAUTY_PRESET))
    unmount()
    expect(node.onExecuted).toBe(originalExecuted)
    expect(node.onConnectionsChange).toBe(originalConnections)
  })
})
