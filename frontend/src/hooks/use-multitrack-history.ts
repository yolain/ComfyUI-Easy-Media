import { useCallback, useLayoutEffect, useRef, useState } from 'react'
import type { TrackData } from '@/types/multitrack'

const MULTITRACK_HISTORY_LIMIT = 50

interface MultiTrackHistoryState {
  past: TrackData[]
  future: TrackData[]
}

function serializeTrackData(data: TrackData): string {
  return JSON.stringify(data)
}

export function useMultiTrackHistory(data: TrackData, onChange: (value: TrackData) => void) {
  const [history, setHistory] = useState<MultiTrackHistoryState>({ past: [], future: [] })
  const dataRef = useRef(data)
  const expectedSerializedRef = useRef(serializeTrackData(data))
  const lastPropSerializedRef = useRef(serializeTrackData(data))
  const pendingLocalSerializedRef = useRef(new Set<string>())
  const staleLocalSerializedRef = useRef(new Set<string>())

  const serialized = serializeTrackData(data)

  useLayoutEffect(() => {
    const pendingLocalSerialized = pendingLocalSerializedRef.current
    if (pendingLocalSerialized.size > 0) {
      if (pendingLocalSerialized.has(serialized)) {
        pendingLocalSerialized.delete(serialized)
        if (pendingLocalSerialized.size === 0) staleLocalSerializedRef.current.clear()
        lastPropSerializedRef.current = serialized
        expectedSerializedRef.current = serialized
        dataRef.current = data
        return
      }
      if (serialized === lastPropSerializedRef.current || staleLocalSerializedRef.current.has(serialized)) return
      pendingLocalSerialized.clear()
      staleLocalSerializedRef.current.clear()
      expectedSerializedRef.current = serialized
      lastPropSerializedRef.current = serialized
      dataRef.current = data
      setHistory({ past: [], future: [] })
      return
    }

    dataRef.current = data
    if (serialized !== expectedSerializedRef.current) {
      expectedSerializedRef.current = serialized
      lastPropSerializedRef.current = serialized
      setHistory({ past: [], future: [] })
      return
    }
    lastPropSerializedRef.current = serialized
  }, [data, serialized])

  const commitChange = useCallback((nextData: TrackData) => {
    const currentData = dataRef.current
    const currentSerialized = serializeTrackData(currentData)
    const nextSerialized = serializeTrackData(nextData)
    if (nextSerialized === currentSerialized) return

    expectedSerializedRef.current = nextSerialized
    pendingLocalSerializedRef.current.add(nextSerialized)
    staleLocalSerializedRef.current.add(currentSerialized)
    dataRef.current = nextData
    setHistory((current) => ({
      past: [...current.past, currentData].slice(-MULTITRACK_HISTORY_LIMIT),
      future: [],
    }))
    onChange(nextData)
  }, [onChange])

  const undo = useCallback(() => {
    setHistory((current) => {
      const previous = current.past.at(-1)
      if (!previous) return current

      const currentData = dataRef.current
      const currentSerialized = serializeTrackData(currentData)
      const previousSerialized = serializeTrackData(previous)
      expectedSerializedRef.current = previousSerialized
      pendingLocalSerializedRef.current.add(previousSerialized)
      staleLocalSerializedRef.current.add(currentSerialized)
      dataRef.current = previous
      onChange(previous)

      return {
        past: current.past.slice(0, -1),
        future: [currentData, ...current.future].slice(0, MULTITRACK_HISTORY_LIMIT),
      }
    })
  }, [onChange])

  const redo = useCallback(() => {
    setHistory((current) => {
      const next = current.future[0]
      if (!next) return current

      const currentData = dataRef.current
      const currentSerialized = serializeTrackData(currentData)
      const nextSerialized = serializeTrackData(next)
      expectedSerializedRef.current = nextSerialized
      pendingLocalSerializedRef.current.add(nextSerialized)
      staleLocalSerializedRef.current.add(currentSerialized)
      dataRef.current = next
      onChange(next)

      return {
        past: [...current.past, currentData].slice(-MULTITRACK_HISTORY_LIMIT),
        future: current.future.slice(1),
      }
    })
  }, [onChange])

  return {
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,
    commitChange,
    undo,
    redo,
  }
}
