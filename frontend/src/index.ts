import { addInlineStyles } from "@/lib/add-stylesheet";
import { preserveTimelineEditorNodeHeight } from "@/lib/timeline-node-size";
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import type { TimelineData } from '@/types/timeline'

declare const __COMFY_EASY_MEDIA_GLOBAL_CSS__: string;

declare global {
  // eslint-disable-next-line no-var
  var comfyAPI: { app: { app: ComfyApp } } | undefined
}

const [{ createReactWidget }, { TimelineWidget }, { createDefaultTimelineData }] = await Promise.all([
  import('@/lib/create-react-widget'),
  import('@/components/widgets'),
  import('@/lib/timeline-utils'),
]);

const DEFAULT_TIMELINE_VALUE = JSON.stringify(createDefaultTimelineData())

globalThis.comfyAPI!.app.app.registerExtension({
  name: 'Comfy.EasyMedia.widgets',

  async setup() {
    addInlineStyles(
      __COMFY_EASY_MEDIA_GLOBAL_CSS__,
      `${import.meta.env.PROJECT_NAME}-globals`,
    );
  },

  beforeRegisterNodeDef(nodeType, nodeData) {
    preserveTimelineEditorNodeHeight(nodeType, nodeData)
  },

  getCustomWidgets() {
    return {
      TIMELINE: createReactWidget<TimelineData>(TimelineWidget, {
        defaultValue: DEFAULT_TIMELINE_VALUE,
        domWidgetOptions: {
          getMinHeight: () => 180,
        },
      }),
    }
  },
})
