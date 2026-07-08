<div align="center">

<img src="https://github.com/user-attachments/assets/fb602a3c-4a2a-48da-8c44-d36417f4633b" height="120">
<h1>ComfyUI-Easy-Media</h1>

[中文文档](./README_CN.md) | [Changelog](./CHANGELOG.md)

A ComfyUI custom node package for streamlined media loading and video pipeline assembly. Provides intuitive nodes that simplify media resource editing and loading with user-friendly parameters, making it easier to build and configure video processing workflows.

[![][github-release-shield]][github-release-link]
[![][github-stars-shield]][github-stars-link]
[![][github-forks-shield]][github-forks-link]
[![][github-license-shield]][github-license-link]

<img src="https://github.com/user-attachments/assets/493947f1-3fff-4503-b2d3-408591b7597f" style="width:100%">
</div>

## 📦 Installation

> [!IMPORTANT]
> It is strongly recommended that before installing this node package, you first ensure that `FFmpeg` has already been installed in your system environment

```bash
cd Your_ComfyUI_Path/custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-Media.git
```

## ✏️ Example Workflows

After installing, open ComfyUI and find the bundled example workflows in the **Templates** panel on the left sidebar — look for entries under **ComfyUI-Easy-Media**.



## ✨ Features

### 🎞️ MultiTrack Editor

#### Comparison

![Compare](https://github.com/user-attachments/assets/e7a30db8-48b3-480a-a211-a2633b4b1243)

> **Tips:** The advantage of the multi-track editor is its decoupling design — it is used solely for media editing and loading, and is not bound to any model. Users can freely choose any model node to process the media data output by the multi-track editor.

#### Overview

![multiTrackEditor](https://github.com/user-attachments/assets/8d4dd7a0-361a-4e19-814a-f19d9b2f31cb)

#### Tracks

| Track Type | Description |
| - | - |
| Task Track | Supports multiple task type definitions such as t2v, i2v, r2v, v2v |
| Video Track | Import and manage video clips, supporting multi-segment video stitching and intelligent segmentation |
| Audio Track | Import and manage audio clips, supporting multi-segment audio stitching |
| Subtitle Track | Add subtitles recognized from audio/video |

- Task segments are the core of this node; workflows can be designed for automatic looping based on the number of task track segments
- When adding video clips to the video track, corresponding task segments will be automatically added with matching duration
- Selecting a task segment allows you to set image, task type, and user prompt / system prompt (defaults exist based on task type, or you can write your own)
- The MultiTrack Info Output node outputs video dimensions, total frame count, frame rate, and task count
- The MultiTrack Task Output node outputs user prompt & system prompt for corresponding segments; users can decide whether to connect LLM nodes for prompt expansion or use images in segments for reverse inference

#### Use Cases

| Scenario | Description | Requirements |
|----------|-------------|-------------|
| Video Generation | wan/bernini/ltx t2v, i2v, r2v | Task track segments only |
| Video Editing | bernini v2v, bernini vi2v, wan animate, ltx video replace, ltx iclora edit/inpaint/outpaint | Video track segments + task track segments |
| Video Reference | wan scail2, wan animate, ltx iclora guide | Video track segments + task track segments |
| Video Dubbing | wan infinititalk, longcat avatar, ltx ai2v | Task track segments + audio track segments |
| Video Subtitles | - | Task track segments + subtitle track segments |

- Only the most common open-source model generation types are listed; theoretically any video model pipeline can use the multi-track editor as a preprocessing tool

#### Optional Models

| Scenario | Description | Download | Local Path | Prerequisites 
| - | - | - | - | - | 
| **Video Subtitles (Whisper)** | Audio/video recognition to generate subtitles | [Whisper Large V3](https://huggingface.co/Comfy-Org/HuMo_ComfyUI/tree/main/split_files/audio_encoders) | models/audio_encoders/ | `pip install openai-whisper` |
| **Video Subtitles (Qwen3)** | Audio/video recognition to generate subtitles | [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) <br> [Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | models/Qwen3-ASR/ | `pip install qwen-asr torchaudio` |
| **Subtitle Narration** | Convert subtitles to speech voiceover | [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | models/voxcpm/ | `pip install voxcpm` |
| **Shot Detection** | Intelligently segment video shots | [OmniShotCut](https://huggingface.co/uva-cv-lab/OmniShotCut/resolve/main/OmniShotCut_ckpt.pth) | models/checkpoints | - |

> **Note:** Some models support automatic download via the built-in Easy-Media model download interface. Model files will be placed in the `ComfyUI/models/` directory.

### 🎞️ Subtitle To Video

![SubtitleToVideo](https://github.com/user-attachments/assets/58f90eb7-d671-437d-8adf-d8a04a3e261e)

### 🎞️ Compare Videos

![CompareVideos](https://github.com/user-attachments/assets/3bad558c-c5f4-411d-ba4c-b2edee9b9f11)

### 🎞️ Timeline Editor

![timelineEditor](https://github.com/user-attachments/assets/d7c9e894-6e7e-488c-90fb-d3aa8310419d)

<details>
<summary>Dynamic Parameter Injection</summary>

> If you want to dynamically invoke the timeline editor via `agents` or `app`, a method is available: input media assets into the corresponding input ports of the timeline editor (`prompt_override`, `image`, `audio`, `video`). When `prompt_override` is injected, it will override the segment data in the timeline editor. However, compared to directly editing segment content via the visual interface, the dynamic parameter injection method has limitations—for example, it is not convenient to control audio duration and ranges. `prompt_override` provides a prompt formatting template specification, similar to `promptRelay + seedance2.0` dynamic prompts. See the example below for details.

![dynamicInput](https://github.com/user-attachments/assets/eef6798e-a68d-4724-8e72-69b1a13825dd)

**Optional Parameters**:

- `prompt_override`: Due to ComfyUI's force_input compatibility issues, when force_input exists, custom widgets cannot be accessed. Therefore, the parameter type is currently set to `AnyType`. It is recommended to connect using a regular string type node.
- `image`: Input image resource list; it is recommended to use the newly added `easy makeImageList` node to create image lists.
- `video`: Input video resource list; if a segment only needs one video, connect the video directly to the video input port. If multiple video segments are needed, it is recommended to use the newly added `easy makeVideoList` node to create video lists.
- `audio`: Input audio resource list; if a segment only needs one audio, connect the audio directly to the audio input port. If multiple audio segments are needed, it is recommended to use the newly added `easy makeAudioList` node to create audio lists.


**Prompt Example**:

```
@图片1 @音频1 镜头晃动，老者正望着光亮处神色慌张地喊话： 别学那玩意，别连线啊。 [0-120] | @图片2 @音频2 镜头缓慢推进，男人正在操作电脑，说道：有意思，这ComfyUI能火，我指定得学它 [121-241]
```

- [0-120] and [121-241] represent the start and end frame ranges of segments on the timeline, in frames (frame), also supports [0-5s] [5-10s] writing, in seconds. If no time range is specified, the total duration set on the original timeline editor will be equally distributed.
- Segments are separated by `|`, representing different time periods. Each segment can contain `media placeholder`, `text prompt`, and `start-end frame range`.
- Image injection: Supports `@image{n}`, `@img{n}`, `@图{n}`, `@图片{n}`, `@图像{n}` as placeholders to inject image resources, where `{n}` represents the n-th image in the image list (starting from 1). For example, `@image1` will inject the first image from the image list.
- Video injection: Supports `@video{n}`, `@视频{n}` as placeholders to inject video resources, where `{n}` represents the n-th video in the video list (starting from 1). For example, `@video1` will inject the first video from the video list.
- Audio injection: Supports `@audio{n}`, `@音频{n}` as placeholders to inject audio resources, where `{n}` represents the n-th audio in the audio list (starting from 1). For example, `@audio1` will inject the first audio from the audio list.


**Adding Media via Timeline Editor Input Ports**:
> If you only want to pass parameters via the `image` or `audio`, `video` input ports and do not want to use `prompt_override`, you can use the `slot` method to associate media in the image or audio adding section with the media from the corresponding input port. This way, when executing workflow tasks, the input media resources will automatically be associated with the corresponding segments in the timeline editor.
(Note: The preview displayed on the timeline editor traces back to the resources of the corresponding nodes that initially loaded the images or audio. If you use cropping or truncation nodes between the loading and timeline editor workflow to process the original media, the backend will also execute this processing; however, the frontend preview display shows the initial state.)

![dynamicInput2](https://github.com/user-attachments/assets/6dd84d52-1fd3-4b27-a890-2a0e22cecda4)
</details>

### 🎞️ SaveVideo

![SaveVideo](https://github.com/user-attachments/assets/30e2dcc3-9ed3-4d5f-bb15-69e50c3e8fca)
> Integrated and enhanced the video saving node from the SaveVideoRGBA node package. Supports video export with customizable output path, filename prefix, frame rate, and other parameters.

### 🎞️ Merge Videos From Paths

> Load video files from a list of file paths (or URLs) and concatenate them into a single video output.

The `trim_frame_count` parameter defaults to `-1`, which keeps all frames of the merged video. When set to a value greater than `0`, the node calculates the duration based on the merged video's frame rate and uses FFmpeg to trim the final video.



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
| easy audioMerge | Merge or concatenate up to six audio inputs |
| easy makeVideoList | Combine multiple video inputs into a video list |
| easy compareVideos | Preview source and output VIDEO inputs side by side with an interactive comparison slider |
| easy imageIndexesToIntList | Convert comma-separated image index string to integer list |
| easy saveVideo | Save images and optional audio as video file |
| easy getAudioFromVideo | Extract audio from a VIDEO input |
| easy mergeVideos | Concatenate multiple compatible VIDEO segments |
| easy mergeVideosFromPaths | Load and concatenate videos from file path list, optionally trimming the merged output by frame count |
| easy multiTrackEditor | Multi-track editor for editing and transferring multi-track media data |
| easy multiTrackInfoOutput | Output multi-track dimensions, duration, frame rate, and task count |
| easy multiTrackTaskOutput | Output multi-track task segment prompts and task-ranged media |
| easy multiTrackAddSubtitleToVideo | Add subtitle track to video track |
| easy makeRefsCompositeBySam3 | Detect subject in prompt using SAM3 and composite reference images onto canvas |
| easy splitImages | Split an image list or batch into multiple single-image outputs |
| easy matchLine | Return zero-based index of the first line containing matching text |
| LTXVAddGuidesFromBatchIndexes | Add guide images from batch images to specified frame indexes of latent variables |
| LTXVMakeRefVideo | Expand a reference image batch into an IC-LoRA reference video |
| BerniniModelPatch | Add Bernini context latent support for Wan model |
| BerniniConditioning | Bernini context conditioning for video/image condition injection |

## Credits

- [OmniShotCut](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut)
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Whisper](https://github.com/openai/whisper)
- [VoxCPM2](https://github.com/OpenBMB/VoxCPM)

## Source of Inspiration

- [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)
- [ComfyUI-PromptRelay](https://github.com/kijai/ComfyUI-PromptRelay)
- [ComfyUI-Licon-MSR](https://github.com/liconstudio/ComfyUI-Licon-MSR)
- [ComfyUI-RH-Bernini](https://github.com/RH-RunningHub/ComfyUI-RH-Bernini)


<!-- LINK GROUP -->
[github-forks-link]: https://github.com/yolain/ComfyUI-Easy-Media/network/members
[github-forks-shield]: https://img.shields.io/github/forks/yolain/ComfyUI-Easy-Media?color=8ae8ff&labelColor=black&style=flat-square
[github-license-link]: https://github.com/yolain/ComfyUI-Easy-Media/blob/master/LICENSE
[github-license-shield]: https://img.shields.io/github/license/yolain/ComfyUI-Easy-Media?color=white&labelColor=black&style=flat-square
[github-release-link]: https://github.com/yolain/ComfyUI-Easy-Media/releases
[github-release-shield]: https://img.shields.io/github/v/release/yolain/ComfyUI-Easy-Media?color=f2ff59&labelColor=black&style=flat-square
[github-stars-link]: https://github.com/yolain/ComfyUI-Easy-Media/network/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/yolain/ComfyUI-Easy-Media?color=ffcb47&labelColor=black&style=flat-square
