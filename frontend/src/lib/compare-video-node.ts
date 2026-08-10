interface CompareVideoWidget {
  name?: string
}

interface CompareVideoNodeInstance {
  hideOutputImages?: boolean
  onDrawBackground?: (context: CanvasRenderingContext2D) => void
  widgets?: CompareVideoWidget[]
  removeWidget?: (widget: CompareVideoWidget) => void
  videoContainer?: HTMLElement
  imgs?: unknown[]
}

interface CompareVideoNodeType {
  prototype: object
}

interface CompareVideoNodeDefinition {
  name?: string
}

export function suppressCompareVideoDefaultPreview(
  nodeType: CompareVideoNodeType,
  nodeData: CompareVideoNodeDefinition,
): void {
  if (nodeData.name !== 'easy compareVideos') return
  const prototype = nodeType.prototype as CompareVideoNodeInstance
  prototype.hideOutputImages = true
  prototype.onDrawBackground = function hideCompareVideoOutputPreview() {
    const defaultPreview = this.widgets?.find((widget) => widget.name === 'video-preview')
    if (defaultPreview && this.removeWidget) {
      try {
        this.removeWidget(defaultPreview)
      } catch (error) {
        console.error('[CompareVideoWidget] failed to remove the default video preview:', error)
      }
    }

    this.videoContainer?.replaceChildren()
    this.videoContainer = undefined
    this.imgs = undefined
  }
}
