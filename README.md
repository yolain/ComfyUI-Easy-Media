<div align="center">

<img src="https://github.com/user-attachments/assets/fb602a3c-4a2a-48da-8c44-d36417f4633b" height="120">
<h1>ComfyUI-Easy-Media</h1>

[中文文档](./README_CN.md) | [Changelog](./CHANGELOG.md)

A ComfyUI custom node package for streamlined media loading and video pipeline assembly. Provides intuitive nodes that simplify media resource editing and loading with user-friendly parameters, making it easier to build and configure video processing workflows.

[![][github-release-shield]][github-release-link]
[![][github-stars-shield]][github-stars-link]
[![][github-forks-shield]][github-forks-link]
[![][github-license-shield]][github-license-link]
[![][workflow-shield]][workflow-link]

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

<a id="multitrack-workflow-skill"></a>

### 🤖 Generate MultiTrack Workflows with a Skill

The package includes the `easy-media-multitrack-workflow` Skill. Give Codex your assets and requirements in natural language to create a workflow from the bundled template or modify an existing one, automatically configuring the timeline, prompts, continuity modes, and sampling settings.

```text
Use $easy-media-multitrack-workflow to create three 5-second tasks from these reference images and prompts.
Use Shot for the first segment and Context for the next two, then generate a new workflow JSON.
```

By default, the Skill only generates JSON. When explicitly asked to run the workflow, it can upload assets and submit it to ComfyUI. See the [Skill documentation](./skills/easy-media-multitrack-workflow/README.md) for installation and detailed usage.

## ✨ Features

### 🎞️ (New) MultiTrack Project Pipeline

![pipeline](https://github.com/user-attachments/assets/0170ae6f-149e-41ba-9bdf-03ec31b0625e)

Introduced in v1.3.0, the MultiTrack Project Pipeline connects timeline arrangement, segment generation, context continuity, and final video assembly. **MultiTrack Project currently supports MiniMax H3**; MultiTrack Editor remains independent of any model and can still feed other model workflows.

Use the [workflow generation Skill](#multitrack-workflow-skill) to configure this pipeline through natural language, either from a blank template or by modifying an existing workflow.

```text
MultiTrack Editor ── TRACKS_INFO ──→ MultiTrack Project ── PROJECT_NAME ──→ MultiTrack Project Video Combine ── VIDEO ──→ SaveVideo
                                          ↑
                                    model_loader
                              (optional model_loader_2nd)
```

> [!IMPORTANT]
> **v1.3.0 loads media on demand.** When a timeline has no Slot references, the editor no longer loads all images, audio, and video in advance. Its `IMAGES`, `AUDIO`, and `VIDEO` outputs are `None` by design. The project pipeline only needs `TRACKS_INFO`; older workflows that consume media directly should use [MultiTrack Task Output](#multitrack-lazy-loading).

#### 1. MultiTrack Editor: Tasks and Continuity

Select the `MiniMax` format, set the target dimensions and frame rate, enter prompts for each task segment, and add reference images, video, or audio as needed. See the [MultiTrack Editor overview](#multitrack-editor) below for tracks, task modes, and media editing.

The new **continuity mode** determines how a task follows the previous segment. It is configured separately from task modes such as text-to-video, image-to-video, and reference-to-video:

| Continuity Mode | Generation Behavior | Use Cases |
|-----------------|---------------------|-----------|
| **Shot (`shot`)** | Generates independently, without inheriting motion or audio context from the previous segment | New shots, scene changes, and deliberate cuts |
| **Context (`context`)** | Uses the tail of the previous result's audio/video latent to continue motion and sound | Continuous action, long takes, and ongoing audio |
| **Character Swap Context (`context_swap`)** | Uses disposable tapered noise on the previous video context in both sampling passes while preserving its audio | Character or appearance replacement that should retain the previous motion |

The first segment starts in Shot mode. Set subsequent segments individually or select multiple tasks to change them together. For example, “Shot → Context → Context → Shot” creates three connected segments followed by a new shot. This setting affects **generation**, rather than adding a crossfade during assembly. Context does not guarantee seamless continuity across arbitrary scene or prompt changes.

#### 2. MultiTrack Project: Encoding, Sampling, and Segment Loops

Connect the editor's `TRACKS_INFO` to `tracks_info`, then connect a `model_loader` containing the H3 model, CLIP, video VAE, and audio VAE. The node automatically expands the tasks in sequence, so no manual segment-index loop is needed:

```text
Read task and load media on demand → Encode prompts and reference conditioning → First pass
    → [Optional: upscale video latent → Second pass] → Decode audio/video → Trim context overlap
    → Save segment and context → Process next task → Output project name
```

Each segment uses its own prompts, references, and continuity mode. With Context enabled, the first pass uses the previous segment's context at the corresponding resolution; the second pass uses its final high-resolution result to preserve the join. Each segment is saved before the next one runs. You can choose a starting segment and count, or save a first-pass preview and resume the second pass later. See [MultiTrack Project details](#multitrack-project) for settings, upscaler dependencies, and the context implementation's source and adaptations.

#### 3. MultiTrack Project Video Combine: Preview, Select, and Export

Connect `PROJECT_NAME` to the combine node to preview saved segments, choose a version for each, and assemble the final video. **Auto combine** produces a complete video in one workflow run. Turn it off to review results first, then click **Combine** to run only assembly and downstream saving without re-running the model. If a segment has multiple generated versions, select two for synchronized comparison before choosing the final version. See [MultiTrack Project Video Combine](#multitrack-project-video-combine) for details.

<a id="multitrack-editor"></a>

### 🎞️ MultiTrack Editor


> **Tips:** The advantage of the multi-track editor is its decoupling design — it is used solely for media editing and loading, and is not bound to any model. Users can freely choose any model node to process the media data output by the multi-track editor.

#### Overview

![multiTrackEditor](https://github.com/user-attachments/assets/45dc72fd-d2bc-4df9-9e46-5c3e7fc6aa62)

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

<a id="multitrack-lazy-loading"></a>

#### v1.3.0 Lazy Media Loading and Workflow Migration

The editor passes time ranges, prompts, media locations, and track settings through `TRACKS_INFO`. Downstream output nodes load the required media when processing each task, reducing the cost of decoding and assembling an entire long timeline in advance. Metadata probing, frontend thumbnails, and waveform previews are separate from loading complete media for workflow execution.

| Timeline Sources | Editor Outputs | How to Access Media |
|------------------|----------------|---------------------|
| Files or URLs, **without Slot references** | `TRACKS_INFO` is available; media outputs are `None` | Connect `TRACKS_INFO` to MultiTrack Project or MultiTrack Task Output for loading on demand |
| **Slot references** to upstream image, audio, or video inputs | Retains immediate loading and media outputs; requests only the input types actually referenced | Continue using the editor's media outputs and companion output nodes |

If an older workflow connects directly to the editor's media outputs, insert `MultiTrack Task Output (easy multiTrackTaskOutput)`:

- **Process one segment at a time:** Connect `TRACKS_INFO` and set `task_index = 0, 1, 2…` to retrieve media for the corresponding task range.
- **Retrieve all media at once:** Set `task_index = -1` to output the entire timeline's media, replacing the editor's former full-media output behavior. This restores the resource cost of loading the full timeline.
- **Read only dimensions, frame rate, total frames, and task count:** Use `MultiTrack Info Output`; complete media loading is unnecessary for this metadata.
- **Use the project pipeline:** MultiTrack Project calls Task Output internally, so no extra Task Output node is needed between the editor and the project.

#### v1.3.x Task and Audio Settings

- **Task track:** Adds Shot / Context / Character Swap Context continuity modes. Select multiple task segments to change their task mode, continuity mode, and reference image size together.
- **Lock audio:** In MultiTrack Project, the input audio constrains visual generation, and the delivered video uses the original task audio instead of regenerating it. This differs from using audio only as a reference.
- **Reuse audio:** Share the same reference audio across tasks without copying it under every segment. Up to 15 seconds are used, without truncating it to a shorter task duration.

#### Use Cases

| Scenario | Description | Requirements |
|----------|-------------|-------------|
| Video Generation | MiniMax H3/wan/bernini/ltx t2v, i2v, r2v | Task track segments only |
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

<a id="multitrack-project"></a>

### 🎞️ MultiTrack Project

![multiTrackProject](https://github.com/user-attachments/assets/b4cc13a9-5e64-4361-8b3a-ddf900d04a94)

`easy multitrackProject` manages segment-by-segment generation for MiniMax H3 projects. The editor's dimensions are the **first-pass dimensions**; dual sampling scales up from this size for the second pass. Project files are stored in `ComfyUI/output/easy_media/projects/<project_name>/`, including segment media, project records, and context latents for continuation.

#### Encoding and Sampling

1. **Read and encode the task:** Load its prompts and media, build conditioning for text-to-video, first/last frames, last-frame-only, or multimedia references according to the task mode, and create audio/video latents. When audio is locked, encode it and apply sampling constraints.
2. **First pass:** Both `sampling_mode = single` and `dual` generate at the editor's configured dimensions. The first-pass size is not reduced based on `upscale_by`.
3. **Upscale and second pass (`dual` only):** Multiply each configured dimension by `upscale_by`, then align to the nearest multiple of 32 using `round(dimension * upscale_by / 32) * 32`. Upscale the first-pass video latent to this size, recombine it with the audio latent, and sample again. Rebuild reference conditioning when dimensions change. Setting `upscale_by = 1.000` skips resizing but still allows a second pass.
4. **Decode and save:** Decode video and audio. For Context segments, trim the repeated guide prefix and extra trailing frames required by the temporal grid before saving the segment.
5. **Continue to the next segment:** Save context and project records, then process subsequent tasks automatically. Shot starts a new shot; Context inherits the previous segment. Finally, output `PROJECT_NAME` for the combine node.

| Setting | Description |
|---------|-------------|
| `model_loader` | First-pass H3 model and shared CLIP, video VAE, and audio VAE; video projects also require the audio VAE |
| `model_loader_2nd` | Optional second-pass H3 model; defaults to the first-pass model. Even when connected, encoding and VAEs still come from the first-pass loader |
| `sampling_plan` | Built-in presets such as `ultra_light`, `light`, `medium`, and `high` select samplers and sigmas for Turbo / non-Turbo models; use `custom` for manual settings |
| `sampler` / `sigmas` | Connect both to override first-pass sampling. Second-pass overrides use `sampler_2nd` / `sigmas_2nd`, also as a pair. `custom` requires both inputs for every sampling pass that runs |
| `upscale_by` | Second-pass scale relative to the editor dimensions; default `1.250`, three decimal places, step `0.001` |
| `disable_2nd_noise` | Disables added second-pass noise; it does not skip the second pass |
| `1st_pass_only` | In `dual` mode, runs and checkpoints **only the first pass of the first selected segment**. Disable it on the next run to resume that segment at the second pass |

Alignment uses Python `round`, matching [the H3 latent upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_3d.py); it is not always upward, and exact ties round to the even integer. For example, `1344 × 768` at `1.250` becomes `1680 × 960` before alignment and **`1664 × 960`** for the second pass. Both upscaling paths use this size, and project records and combined exports retain the resulting dimensions. Single-pass generation and first-pass-only previews keep the editor dimensions; audio-only projects (`32 × 32`) do not upscale. Existing workflows retain their saved multiplier rather than automatically adopting the new default.

To customize presets, copy [h3_sample.json.example](./presets/h3_sample.json.example) to `presets/h3_sample.json` and edit it. **The current built-in dual-sampling presets use separate first- and second-pass sigma schedules.** Do not assume that `light` always splits sigmas and leaves the first pass incomplete. A custom preset with `split_step` explicitly splits one sigma schedule across both passes. Existing custom files take precedence, so review your configuration after upgrading.

#### Second-Pass Upscalers and Dependencies

| `upscale_model` | Upscaling Path | Dependencies |
|-----------------|----------------|--------------|
| An H3 latent upscaler model | Calls `MinimaxH3LatentUpscaler3D` to upscale the video latent directly to the scaled, aligned second-pass dimensions | Install [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) and place the [H3 upscaler weights](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) in `ComfyUI/models/latent_upscale_models/` |
| `None` | Decode the first-pass video with the VAE → Resize images → Re-encode with the VAE → Second pass | Requires `ImageResizeKJv2` from [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) |

`upscale_model` selects **H3 latent upscaler weights**, which serve a different purpose from the H3 generation model in `model_loader_2nd`. Direct latent upscaling avoids the intermediate video VAE decode/encode round trip, but the second pass still runs at the target resolution; its VRAM requirements do not decrease proportionally. Selecting an upscaler model without installing the required node raises an error rather than silently switching paths.

For example, download `minimax_h3_latent_upscaler_3d_fp16.safetensors` from the model repository above, place it in the specified directory, restart ComfyUI, and select it under `upscale_model`. Its documentation specifies a 1–4× scaling range. Follow the selected model's supported range even if the project parameter accepts larger values.

#### Context Implementation: Source and Adaptations

Context conditioning is based on [NikoDemon80 / ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context), adapted inside Easy Media for project loops and dual sampling. The current hard-continuity implementation requires native H3 audio/video keyframe support in ComfyUI (0.34.0+). Check ComfyUI compatibility when upgrading.

In addition to attaching the previous segment's context conditioning, the project pipeline applies these adaptations:

- **First-pass hard continuity:** Preserves native audio/video keyframes and existing multimedia references, while copying the previous segment's audio/video tail into the current starting latent. Separate video and audio masks lock and gradually release the copied region, allowing new content to emerge around the join.
- **Separate low- and high-resolution context:** In dual sampling, the first pass inherits the previous segment's first-pass context. After upscaling, the second pass copies the previous final high-resolution video tail into the current high-resolution latent, rather than merely enlarging the low-resolution join. Context second-pass sampling freezes the current audio to retain continuity established in the first pass.
- **Context-specific second-pass sigmas:** Built-in presets provide `sigmas_2nd_context = 0.50, 0.30, 0.14, 0.06, 0.0` for second passes with previous-segment context. Explicit custom second-pass sampler or sigma inputs prevent this substitution.
- **Synchronized trimming and clean continuation sources:** By default, the project takes the previous segment's last 22 frames as context and reserves 34 extra generation frames to satisfy H3's temporal grid. After decoding, it removes the repeated prefix and excess tail, retaining the task's required frame count. It then re-encodes context from the actual delivered audio/video range so subsequent segments do not inherit discarded tail frames.
- **Memory-bounded high-resolution context:** Before the next segment starts, the high-resolution continuation latent is reduced to the 22-frame video/audio tail, detached from the full sampling result, and moved to CPU. Project artifacts store the same context-only high-resolution payload after second-pass completion. The complete low-resolution first-pass latent remains saved so a `1st pass only` run can resume its second pass later; runtime context handoffs use a separate trimmed copy.
- **Automatic handoff and resuming across runs:** Context passes automatically between segments in one run. When starting from a later segment, the project loads context from the previous segment's active saved version, without manually wiring Save / Load Latent nodes.

> **Note:** Context continuation requires the previous segment's saved latents; an MP4 alone is insufficient. After changing editor dimensions, scaling factor, or the previous segment's version, check the downstream context chain. Regenerate first-pass checkpoints and context latents created with the old reduced-first-pass sizing before resuming with the new sizing behavior. Existing later segments are not automatically regenerated when an earlier segment changes.

#### Generation Ranges, Regeneration, and Version Retention

| Setting | Behavior |
|---------|----------|
| `project_name` | Identifies the project directory and records; use the same name when resuming |
| `segment_start_number` | Starting segment, **counting from 1**, unlike the zero-based `task_index` in MultiTrack Task Output |
| `segment_count` | Maximum segments to generate in this run; `-1` processes all remaining segments from the start |
| `project_save = new` | Preserves existing results in the same project and adds versions for regenerated segments, allowing comparison |
| `project_save = override` | Replaces the corresponding segment version. With `segment_count = -1`, clears saved segments from the start onward before regeneration; a first-pass checkpoint being resumed is preserved |

For example, to regenerate only segment 3, set `segment_start_number = 3` and `segment_count = 1`. Choose `project_save = new` to retain the old result for comparison. If segment 3 uses Context, the project must contain compatible context from segment 2. After accepting the new segment 3, regenerate any later Context segments that depend on it.

<a id="multitrack-project-video-combine"></a>

### 🎞️ MultiTrack Project Video Combine

![MultiTrackProjectVideoCombine](https://github.com/user-attachments/assets/976dd4fd-8b69-4adb-8aee-d2269123502a)

`easy multitrackProjectVideoCombine` reads saved video segments from a project. Preview them sequentially on the timeline **without first creating a merged file**. Use the project selector to switch projects or refresh saved results.

#### Automatic and Manual Combining

- **Auto combine (enabled by default):** After project generation finishes, reads the current segments, concatenates them in timeline order, and outputs `VIDEO` and `FILENAME_PREFIX`. Connect an output node such as `SaveVideo` to save the final video.
- **Manual combine:** With Auto combine disabled, the project still generates and saves individual segments, while the combine node updates the preview without sending a merged result downstream. Review and select your clips, wait for the ComfyUI queue to empty, then click **Combine**. Only this node and its downstream nodes are queued; upstream encoding and sampling are not repeated.
- **Saving requirements:** Connect a downstream video save output before combining manually. The combine node provides a temporary merged video; the save node controls the final path and filename. Use `FILENAME_PREFIX` as a naming prefix if desired.

#### Segment Versions and Comparison

After regenerating with `project_save = new`, click a timeline segment to open its video file list:

1. **Select one video:** Use that version for project preview and assembly.
2. **Select two videos:** Enter synchronized comparison to review results from different seeds, prompts, or sampling settings. Each segment allows at most two selected versions at a time.
3. **Keep one after comparing:** Deselect the unwanted version, then combine manually. Selecting two is only for comparison: assembly still uses the primary version (the first in the current selection list), not the comparison layout or both versions together.

Segments display their Shot / Context mode to help review joins. Selecting an older version changes the assembled clip but does not repair already-generated downstream context. If versions end with different motion or audio, regenerate the affected later segments. Deleting a segment version or an entire project also deletes associated files and context; make sure they are no longer needed for resuming.

> **Scope:** This node is for video projects. Audio-only projects with editor dimensions set to `32 × 32` cannot use it to combine video.

### 🎞️ Subtitle To Video

![SubtitleToVideo](https://github.com/user-attachments/assets/58f90eb7-d671-437d-8adf-d8a04a3e261e)

### 🎞️ Compare Videos

![CompareVideos](https://github.com/user-attachments/assets/3bad558c-c5f4-411d-ba4c-b2edee9b9f11)

---

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

<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Node ID</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">🎞️ Multi-Track Editor</td>
      <td>easy multiTrackEditor</td>
      <td>Edit multi-track timelines and pass track information; defer media loading to downstream nodes when there are no Slot references</td>
    </tr>
    <tr>
      <td>easy multiTrackInfoOutput</td>
      <td>Output multi-track dimensions, duration, frame rate, and task count</td>
    </tr>
    <tr>
      <td>easy multiTrackTaskOutput</td>
      <td>Load and output task prompts and media on demand; task_index = -1 outputs media for the entire timeline</td>
    </tr>
    <tr>
      <td>easy multiTrackAddSubtitleToVideo</td>
      <td>Add subtitle track to video track</td>
    </tr>
    <tr>
      <td rowspan="7">🎬 MiniMax H3</td>
      <td>easy minimaxH3ToVideo</td>
      <td>Build MiniMax H3 text-to-video, reference-to-video, or first/last-frame conditioning and latent inputs</td>
    </tr>
    <tr>
      <td>easy MiniMaxH3ReferenceToVideoBridge</td>
      <td>Bridge node for H3 reference conditioning without Autogrow expansion</td>
    </tr>
    <tr>
      <td>easy MiniMaxH3MotionContextHard</td>
      <td>Apply H3 context conditioning with hard video/audio latent continuity</td>
    </tr>
    <tr>
      <td>easy MiniMaxH3HiResContinuity</td>
      <td>Copy previous high-resolution video tail into current upscaled latent</td>
    </tr>
    <tr>
      <td>easy removeH3MotionContextLatent</td>
      <td>Remove H3 Motion Context latent files after a loop finishes</td>
    </tr>
    <tr>
      <td>easy multitrackProject</td>
      <td>Build and execute multi-track MiniMax H3 project with optional first/second-pass sampling</td>
    </tr>
    <tr>
      <td>easy multitrackProjectVideoCombine</td>
      <td>Preview project segments, select versions, compare two videos, and combine automatically or manually</td>
    </tr>
    <tr>
      <td rowspan="5">🎞️ LTX Video</td>
      <td>LTXVAddGuidesFromBatchIndexes</td>
      <td>Add guide images from batch images to specified frame indexes of latent variables</td>
    </tr>
    <tr>
      <td>LTXVMakeRefVideo</td>
      <td>Expand a reference image batch into an IC-LoRA reference video</td>
    </tr>
    <tr>
      <td>easy ltxMultiTrackEncode</td>
      <td>Build Prompt Relay conditioning and LTX video/audio latents</td>
    </tr>
    <tr>
      <td>easy ltxI2VInplaceAndUpsample</td>
      <td>Optionally upscale an LTX video latent and apply an image guide in place</td>
    </tr>
    <tr>
      <td>easy ltxSamplerSimple</td>
      <td>Sample combined LTX audio/video latents and crop video guides</td>
    </tr>
    <tr>
      <td rowspan="4">🎞️ Timeline Editor</td>
      <td>easy timelineEditor</td>
      <td>Load media timeline (prompt, image, audio tracks) and output structured data</td>
    </tr>
    <tr>
      <td>easy timelineInfoOutput</td>
      <td>Output timeline info including formatted prompt, dimensions, and image indexes</td>
    </tr>
    <tr>
      <td>easy timelineSegmentOutput</td>
      <td>Output specific segment data from the timeline</td>
    </tr>
    <tr>
      <td>easy timelineSegmentCount</td>
      <td>Output the total number of segments in the timeline</td>
    </tr>
    <tr>
      <td rowspan="8">📋 Media List Operations</td>
      <td>easy makeImageList</td>
      <td>Combine multiple image inputs into an image list</td>
    </tr>
    <tr>
      <td>easy makeAudioList</td>
      <td>Combine multiple audio inputs into an audio list</td>
    </tr>
    <tr>
      <td>easy splitAudios</td>
      <td>Split an audio list into multiple single-audio outputs</td>
    </tr>
    <tr>
      <td>easy audioMerge</td>
      <td>Merge or concatenate up to six audio inputs</td>
    </tr>
    <tr>
      <td>easy makeVideoList</td>
      <td>Combine multiple video inputs into a video list</td>
    </tr>
    <tr>
      <td>easy splitVideos</td>
      <td>Split a video list into multiple single-video outputs</td>
    </tr>
    <tr>
      <td>easy imageIndexesToIntList</td>
      <td>Convert comma-separated image index string to integer list</td>
    </tr>
    <tr>
      <td>easy splitImages</td>
      <td>Split an image list or batch into multiple single-image outputs</td>
    </tr>
    <tr>
      <td rowspan="5">🎬 Video Operations</td>
      <td>easy saveVideo</td>
      <td>Save images and optional audio as video file</td>
    </tr>
    <tr>
      <td>easy getAudioFromVideo</td>
      <td>Extract audio from a VIDEO input</td>
    </tr>
    <tr>
      <td>easy mergeVideos</td>
      <td>Concatenate multiple compatible VIDEO segments</td>
    </tr>
    <tr>
      <td>easy mergeVideosFromPaths</td>
      <td>Load and concatenate videos from file path list, optionally trimming the merged output by frame count</td>
    </tr>
    <tr>
      <td>easy compareVideos</td>
      <td>Preview source and output VIDEO inputs side by side with an interactive comparison slider</td>
    </tr>
    <tr>
      <td rowspan="2">📝 Subtitle</td>
      <td>easy recognizeSubtitle</td>
      <td>Recognize subtitles with Qwen3-ASR or Whisper Large V3; configure SRT/timestamp output, sentence length, and model unloading</td>
    </tr>
    <tr>
      <td>easy addSubtitleToVideo</td>
      <td>Normalize multiline SRT, timestamp, or bracket-formatted text and burn it into a video</td>
    </tr>
    <tr>
      <td rowspan="1">🖼️ Reference & Image</td>
      <td>easy makeRefsCompositeBySam3</td>
      <td>Detect subject in prompt using SAM3 and composite reference images onto canvas</td>
    </tr>
    <tr>
      <td rowspan="2">🔧 Utility</td>
      <td>easy matchLine</td>
      <td>Return zero-based index of the first line containing matching text</td>
    </tr>
    <tr>
      <td>easy apiWorkflowGate</td>
      <td>Determine if the workflow is an API call and pass through preceding input items</td>
    </tr>
    <tr>
      <td rowspan="1">🗣️ Speech to Video (S2V)</td>
      <td>easy berniniS2VConditioning</td>
      <td>Unified Bernini + Wan S2V conditioning: original optional single-speaker audio, spatially masked single-speaker audio, or optional sequential two-speaker audio</td>
    </tr>
  </tbody>
</table>

## Credits

- [OmniShotCut](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut)
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Whisper](https://github.com/openai/whisper)
- [VoxCPM2](https://github.com/OpenBMB/VoxCPM)
- [Bernini S2V](https://huggingface.co/rzgar/Bernini-R-S2V)
- [H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
- [MiniMax H3 Chained Character Swap](https://github.com/MacroSony/minimax-h3-chained-character-swap)
- [MiniMax H3 Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

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
[workflow-shield]: https://img.shields.io/badge/💻-Workflows-efff30?color=e92759&labelColor=black&style=flat-square
[workflow-link]:https://www.runninghub.ai/user-center/1847825328541474818/userPost?inviteCode=rh-v1623
