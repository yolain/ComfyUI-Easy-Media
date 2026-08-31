import { addInlineStyles } from "@/lib/add-stylesheet";
import { getEasyMediaSyncPlayMenuItems, installEasyMediaSyncPlay } from "@/lib/sync-play";
import { preserveTimelineEditorNodeSize } from "@/lib/timeline-node-size";
import { preserveCompareVideoNodeSize } from "@/lib/compare-video-node-size";
import { preserveVideoCombineNodeSize } from '@/lib/project-video-combine-node-size';
import type { ComfyApp } from '@comfyorg/comfyui-frontend-types'
import type { TimelineData } from '@/types/timeline'
import type { TrackData } from '@/types/multitrack'
import type { CompareVideoSettings } from '@/components/widgets/compareVideoWidget'
import type { ProjectData } from '@/types/project'
import { DEFAULT_PROJECT_DATA } from '@/types/project'
import { suppressCompareVideoDefaultPreview } from '@/lib/compare-video-node'

declare const __COMFY_EASY_MEDIA_GLOBAL_CSS__: string;

declare global {
  // eslint-disable-next-line no-var
  var comfyAPI: { app: { app: ComfyApp } } | undefined
}

const [
  { createReactWidget },
  { TimelineWidget, MultiTrackWidget, CompareVideoWidget, PromptEnhancerAccountWidget, ProjectVideoCombineWidget },
  { createDefaultTimelineData },
  { createDefaultTrackData },
] = await Promise.all([
  import('@/lib/create-react-widget'),
  import('@/components/widgets'),
  import('@/lib/timeline-utils'),
  import('@/lib/multitrack-utils'),
]);

const DEFAULT_TIMELINE_VALUE = JSON.stringify(createDefaultTimelineData())
const DEFAULT_TRACK_DATA_VALUE = JSON.stringify(createDefaultTrackData())
const DEFAULT_COMPARE_VIDEO_VALUE = JSON.stringify({
  save_output: true,
  filename_prefix: 'ComfyUI',
  watch_output_history: false,
})

const comfyApp = globalThis.comfyAPI!.app.app

comfyApp.registerExtension({
  name: 'Comfy.EasyMedia.widgets',

  async setup() {
    addInlineStyles(
      __COMFY_EASY_MEDIA_GLOBAL_CSS__,
      `${import.meta.env.PROJECT_NAME}-globals`,
    );
  },

  beforeRegisterNodeDef(nodeType, nodeData) {
    preserveTimelineEditorNodeSize(nodeType, nodeData)
    preserveCompareVideoNodeSize(nodeType, nodeData)
    preserveVideoCombineNodeSize(nodeType, nodeData)
    installEasyMediaSyncPlay(nodeType, nodeData)
    suppressCompareVideoDefaultPreview(nodeType, nodeData)
  },

  getNodeMenuItems(node) {
    const locale = comfyApp.ui?.settings?.settingsValues?.['Comfy.Locale']
    return getEasyMediaSyncPlayMenuItems(node, comfyApp.canvas, locale)
  },

  getCustomWidgets() {
    return {
      TIMELINE: createReactWidget<TimelineData>(TimelineWidget, {
        defaultValue: DEFAULT_TIMELINE_VALUE,
        domWidgetOptions: {
          getMinHeight: () => 180,
        },
      }),
      TRACK_DATA: createReactWidget<TrackData>(MultiTrackWidget, {
        defaultValue: DEFAULT_TRACK_DATA_VALUE,
        domWidgetOptions: {
          getMinHeight: () => 320,
        },
      }),
      PROJECT_DATA: createReactWidget<ProjectData>(ProjectVideoCombineWidget, {
        defaultValue: JSON.stringify(DEFAULT_PROJECT_DATA),
        domWidgetOptions: {
          getMinHeight: () => 520,
          hideOnZoom: false,
          serialize: true,
        },
      }),
      EASY_COMPARE_VIDEO: createReactWidget<CompareVideoSettings>(CompareVideoWidget, {
        defaultValue: DEFAULT_COMPARE_VIDEO_VALUE,
        domWidgetOptions: {
          getMinHeight: () => 360,
          hideOnZoom: false,
          serialize: true,
        },
      }),
      EASY_API_ACCOUNT: createReactWidget<string>(PromptEnhancerAccountWidget, {
        defaultValue: '',
        height: 44,
        domWidgetOptions: {
          getMinHeight: () => 44,
          getMaxHeight: () => 44,
          hideOnZoom: false,
          serialize: true,
        },
      }),
    }
  },
})
