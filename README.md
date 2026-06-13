# ComfyUI Easy Media

<div align="center">
<a href="./README.md"><img src="https://img.shields.io/badge/🇬🇧English-0b8cf5"></a>
<a href="./README_CN.md"><img src="https://img.shields.io/badge/🇨🇳中文简体-e9e9e9"></a>
<br>
</div>

![Poster](https://github.com/user-attachments/assets/6c76433e-1893-4709-8738-acbed4438757)

A ComfyUI custom node package for streamlined media loading and video pipeline assembly. Provides intuitive nodes that simplify media resource editing and loading with user-friendly parameters, making it easier to build and configure video processing workflows.


## Installation

```bash
cd Your_ComfyUI_Path/custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-Media.git
```

## Example Workflows

After installing, open ComfyUI and find the bundled example workflows in the **Templates** panel on the left sidebar — look for entries under **ComfyUI-Easy-Media**.

## Changelog

### v1.0.4

- **[SaveVideo]** Added `hide&save` option to hide the output video node while saving the video
- **[Timeline Editor: App Mode]** Added time range parsing support like `[0-5s]` for `prompt_override`
- **[Timeline Editor: UI Mode]** Added drag-and-drop image import for sub-tracks
- **[Timeline Editor: UI Mode]** Fixed issue where modifying clip duration in main track should proportionally adjust sub-track images
- **[Timeline Editor: UI Mode]** Fixed audio preview display error after importing audio subdirectory

### v1.0.3

- **[Bernini Temporary Solution]** Added `Bernini conditioning` and `Bernini Model Patch` nodes, providing a temporary solution before ComfyUI officially supports Bernini
- **[Timeline Editor: UI Mode]** Fixed an issue where `node height` would reset to the default value when `canvas refresh` or `resolution option` was switched
- **[Timeline Editor: UI Mode]** Fixed an issue where clip content could not be edited in some cases when using `Overall Edit` prompt mode
- **[Timeline Editor: UI Mode]** Fixed the issue where nodes and track heights didn't adapt automatically; added `Clone Clip` to the right-click menu for better compatibility with wan2's `berinini` and `LTX2.3 R2V`
- **[LTXV Reference Video]** New node for multi-reference LoRA [model_url](https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference)

### v1.0.2

- **[Timeline Editor: App Mode]** Fixed an issue where, if `prompt_override` was not written strictly according to the prompt format, the default duration was not evenly distributed across clips
- **[Timeline Editor: App Mode]** Fixed an issue where filling the entire timeline with a single audio clip required filtering out empty audio before making a determination
- **[Timeline Editor: UI Mode]** Fixed an issue where the output resources and sorting were incorrect when a single clip contained different formats
</details>

<details>
<summary><b>v1.0.1</b></summary>

- **[Workflow]** Added wan2.2 loop segment example workflow
- **[Frontend Optimization]** Added + button to insert segments before or after the selected segment, and fixed some known bugs
- **[Bug Fix]** Fixed incorrect image import paths from output and subdirectories, which caused images and outputs to be filtered out in the editor
</details>

<details>
<summary><b>v1.0.0</b></summary>

- **Important Changes** `Duration and frame rate` input only takes effect after `blur` (i.e., press Enter or click outside to confirm changes, reducing errors)
- **Important Changes** Duration input step changes: `4` when format is frames, `1` when format is seconds
- **Important Changes** Segment duration editing no longer affects other segments; if the total exceeds the timeline length, the timeline will auto-expand to fit all segments
- The automatic height adjustment of tracks in the Timeline Editor has been adjusted; users must now double-click image and audio clips to open the media selection interface, thereby preventing frequent pop-ups caused by accidental operations.
- Added dynamic parameter injection support for prompt template format + media input using the timeline editor
</details>

## Features

### Timeline Editor

> I believe the media timeline editor component is better suited as a standalone module node for greater versatility. This node focuses on media import/editing and timeline-related functionality, providing better support for video pipeline creation across different models.<br>
The editor can be used for single video segment generation (e.g., combined with PromptRelay), as well as segmented generation. Each segment can be combined with different model video pipelines for text-only generation, single image generation, first/last frame generation, multi-frame generation, reference-based generation, etc.

![timelineEditor](https://github.com/user-attachments/assets/d7c9e894-6e7e-488c-90fb-d3aa8310419d)

#### Dynamic Parameter Injection (05-23)

> If you want to dynamically invoke the timeline editor via `agents` or `app`, a method is available: input media assets into the corresponding input ports of the timeline editor (`prompt_override`, `image`, `audio`). When `prompt_override` is injected, it will override the segment data in the timeline editor. However, compared to directly editing segment content via the visual interface, the dynamic parameter injection method has limitations—for example, it is not convenient to control audio duration and ranges. `prompt_override` provides a prompt formatting template specification, similar to `promptRelay + seedance2.0` dynamic prompts. See the example below for details.

![dynamicInput](https://github.com/user-attachments/assets/eef6798e-a68d-4724-8e72-69b1a13825dd)

**Optional Parameters**:

- `prompt_override`: Due to ComfyUI's force_input compatibility issues, when force_input exists, custom widgets cannot be accessed. Therefore, the parameter type is currently set to `AnyType`. It is recommended to connect using a regular string type node.
- `image`: Input image resource list; it is recommended to use the newly added `easy makeImageList` node to create image lists.
- `audio`: Input audio resource list; if a segment only needs one audio, connect the audio directly to the audio input port. If multiple audio segments are needed, it is recommended to use the newly added `easy makeAudioList` node to create audio lists.

**Prompt Example**:

```text
@image1 @audio1 镜头晃动，老者正望着光亮处神色慌张地喊话：别学那玩意，别连线啊。[0-120]|@image2 @audio2 镜头缓慢推进，男人正在操作电脑，说道：有意思，这ComfyUI能火，我指定得学它 [121-296]
```

- `[0-120]` and `[121-296]` represent the start and end frame ranges of segments on the timeline, in frames. If no time range is specified, the total duration set on the original timeline editor will be equally distributed.
- Segments are separated by `|`, representing different time periods. Each segment can contain `media placeholder`, `text prompt`, and `start-end frame range`.
- Image injection: Supports `@image{n}`, `@img{n}`, `@图{n}`, `@图片{n}`, `@图像{n}` as placeholders to inject image resources, where `{n}` represents the n-th image in the image list (starting from 1). For example, `@image1` will inject the first image from the image list.
- Audio injection: Supports `@audio{n}`, `@音频{n}` as placeholders to inject audio resources, where `{n}` represents the n-th audio in the audio list (starting from 1). For example, `@audio1` will inject the first audio from the audio list.

**Adding Media via Timeline Editor Input Ports**:
> If you only want to pass parameters via the `image` or `audio` input ports and do not want to use `prompt_override`, you can use the `slot` method to associate media in the image or audio adding section with the media from the corresponding input port. This way, when executing workflow tasks, the input media resources will automatically be associated with the corresponding segments in the timeline editor.
(Note: The preview displayed on the timeline editor traces back to the resources of the corresponding nodes that initially loaded the images or audio. If you use cropping or truncation nodes between the loading and timeline editor workflow to process the original media, the backend will also execute this processing; however, the frontend preview display shows the initial state.)

![dynamicInput2](https://github.com/user-attachments/assets/6dd84d52-1fd3-4b27-a890-2a0e22cecda4)

### SaveVideo

![SaveVideo](https://github.com/user-attachments/assets/30e2dcc3-9ed3-4d5f-bb15-69e50c3e8fca)
> Integrated and enhanced the video saving node from the SaveVideoRGBA node package. Supports video export with customizable output path, filename prefix, frame rate, and other parameters.

### Merge Videos From Paths

> Load video files from a list of file paths (or URLs) and concatenate them into a single video output.

**Installing FFmpeg** is recommended for best performance and transition quality:

```bash
# macOS (Homebrew)
brew install ffmpeg

# Windows — download a full build (includes xfade filter):
# https://ffmpeg.org/download.html
# Recommended: BtbN or gyan.dev full builds

# Linux (Ubuntu/Debian)
sudo apt install ffmpeg
```

### MultiTrack Editor

> Planned...


## Development & Testing

1. Create a `config.yaml` file in the ComfyUI-Easy-Media directory and add the following content to enable frontend development mode:

```yaml
WEB_VERSION: dev
```

2. Navigate to the frontend directory and compile the development code for debugging:

```shell
cd frontend && bun install && bun run dev
```

3. After modifying the code, compile for production:

```shell
bun run build:release
```

## Node List

| Node ID | Description |
| ------- | ----------- |
| easy timelineEditor | Load media timeline (prompt, image, audio tracks) and output structured data |
| easy timelineInfoOutput | Output timeline info including formatted prompt, dimensions, and image indexes |
| easy timelineSegmentOutput | Output specific segment data from the timeline |
| easy timelineSegmentCount | Output the total number of segments in the timeline |
| easy makeImageList | Combine multiple image inputs into an image list |
| easy makeAudioList | Combine multiple audio inputs into an audio list |
| easy imageIndexesToIntList | Convert comma-separated image index string to integer list |
| easy saveVideo | Save images and optional audio as video file |
| easy mergeVideos | Concatenate multiple compatible VIDEO segments |
| easy mergeVideosFromPaths | Load and concatenate videos from file path list |
| LTXVAddGuidesFromBatchIndexes | Add guide images from batch images to specified frame indexes of latent variables |

## Source of Inspiration

- [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)
- [ComfyUI-PromptRelay](https://github.com/kijai/ComfyUI-PromptRelay)
- [ComfyUI-Licon-MSR](https://github.com/liconstudio/ComfyUI-Licon-MSR)