import { useEffect } from 'react'

interface InputSlot {
  name: string
  link: number | null
}

interface NodeWithConnections {
  inputs?: InputSlot[]
  onConnectionsChange?: (...args: unknown[]) => void
}

interface DisableableWidget {
  disabled?: boolean
}

function isInputConnected(node: NodeWithConnections | null | undefined, inputName: string): boolean {
  if (!node?.inputs) return false
  const slot = node.inputs.find((input) => input.name === inputName)
  return slot?.link !== null && slot?.link !== undefined
}

export function useNodeInputConnectionDisabled(
  node: NodeWithConnections | null | undefined,
  widget: DisableableWidget | null | undefined,
  inputName: string,
): void {
  useEffect(() => {
    if (!widget) return

    const updateConnectionStatus = () => {
      widget.disabled = isInputConnected(node, inputName)
    }
    updateConnectionStatus()

    if (!node) return

    const prevOnConnectionsChange = node.onConnectionsChange?.bind(node)
    node.onConnectionsChange = (...args: unknown[]) => {
      prevOnConnectionsChange?.(...args)
      updateConnectionStatus()
    }

    return () => {
      node.onConnectionsChange = prevOnConnectionsChange ?? undefined
    }
  }, [node, widget, inputName])
}
