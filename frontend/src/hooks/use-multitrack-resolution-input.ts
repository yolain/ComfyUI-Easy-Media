import { useEffect, useState } from 'react'
import {
  collectMultiTrackPreviewResolutionInput,
  type MultiTrackPreviewResolutionInput,
} from '@/lib/multitrack-utils'

const RESOLUTION_POLL_INTERVAL_MS = 250

function resolutionInputSignature(input: MultiTrackPreviewResolutionInput): string {
  try {
    return JSON.stringify(input)
  } catch (error: unknown) {
    console.error('[useMultiTrackResolutionInput] failed to serialize resolution input:', error)
    return String(Date.now())
  }
}

export function useMultiTrackResolutionInput(node: unknown): MultiTrackPreviewResolutionInput {
  const [resolutionInput, setResolutionInput] = useState(() => collectMultiTrackPreviewResolutionInput(node))

  useEffect(() => {
    let nextInput = collectMultiTrackPreviewResolutionInput(node)
    let currentSignature = resolutionInputSignature(nextInput)
    setResolutionInput((current) => (
      resolutionInputSignature(current) === currentSignature ? current : nextInput
    ))

    const timer = window.setInterval(() => {
      nextInput = collectMultiTrackPreviewResolutionInput(node)
      const nextSignature = resolutionInputSignature(nextInput)
      if (nextSignature === currentSignature) return
      currentSignature = nextSignature
      setResolutionInput(nextInput)
    }, RESOLUTION_POLL_INTERVAL_MS)

    return () => window.clearInterval(timer)
  }, [node])

  return resolutionInput
}
