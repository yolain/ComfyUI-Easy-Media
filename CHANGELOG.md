# Changelog

---

## [1.2.0] - 2026-08-21

### ✨ New Features

- **Compare Video**: Added `Side-by-side Compare` mode, support for directly selecting media for comparison, and `Watch Output History` mode
- **Media Selector**: Added support for multi-selecting image items and selecting all images in current path
- **Multi-Track Prompt Enhancer**: Added this node, supports prompt enhancement for models like h3-context-ir, supports local model usage
- **Multi-Track Editor**: Added user prompt A/B output, supports selecting different user prompt outputs
- **Multi-Track Editor**: Added user prompt reference functionality, supports referencing resources via `<Picture 1>`, `<Audio 1>`, `<Video 1>`, `@图片1` and similar methods, supports multi-element combinations

### 🐛 Bug Fixes

- **Multi-Track Editor**: When two adjacent task segments are connected, modifying the front segment's duration should increase the total duration
- **Media Selector**: Fixed breadcrumbs should support recursive subdirectories
- **Media Selector**: Fixed image re-selection issue


## [1.1.4] - 2026-08-05

### ✨ Features

- **MultiTrack Editor**: Add `MiniMax` video format
- **MiniMax H3**: Add `easy minimaxH3ToVideo` for text-to-video, reference-to-video, and first/last-frame conditioning
- **Media List Utilities**: Add `easy splitAudios` and `easy splitVideos` to split media lists into independent outputs
- **MultiTrack Editor**: Video, audio, and subtitle tracks support dragging the left-side icon to adjust sorting; each track type is limited to a maximum of 3 tracks

### 🐛 Bug Fixes

- **Multitrack Editor**: Fixed an issue where, in `MiniMax` format, the `Multitrack Editor` and `Multitrack Task Output` would output empty clips when neither the video track nor the audio track contained any clips.
- **MultiTrack Editor**: Fix `TRACK_INFO` output when duration is less than 5 seconds — total duration should be the actual total duration of task segments, not the default 5 seconds
- **MultiTrack Editor**: Task track segments support free placement while preserving gaps; new task segments can be added directly between segments to fill gaps
- **MultiTrack Editor**: Fix issue where dragging to trim from segment edge hot zone caused the trim frame to shift based on mouse position or canvas zoom level
- **MultiTrack Editor**: Fix incorrect time ruler click position calculation after canvas zoom, causing playhead positioning and toolbar left/right trim time offset issues
- **MultiTrack Task Output**: `MiniMax` format trims audio/video from the next task segment's start point to the media track segment's actual end; the last task trims from its own start to the media's effective end, without outputting trailing black frames or silent audio
- **MultiTrack Info Output**: Total duration is calculated as the sum of task segment durations, automatically skipping gaps between task segments
- **MultiTrack Editor**: Fix node height recalculation when adding or removing tracks — should recalculate node height instead of adapting to the preview area height

## [1.1.3] - 2026-07-30

### ✨ Features

- **MultiTrack Editor**: Add task markers, allowing multiple task segments to be used as one loop task
- **MultiTrack Editor**: Add track overview, allowing segments to be expanded to view prompts and reference images
- **MultiTrack Editor**: Support importing SRT files to subtitle track
- **MultiTrack Audio Output**: Add cropped audio output mode, outputting cropped audio clips and starting frame number for easier S2V usage
- **LTX Workflow Simplification**: Add `easy ltxMultiTrackEncode` and multiple simplified LTX workflow nodes

### 🐛 Bug Fixes

- **MultiTrack Task Output**: Should output empty system prompt when prompt format is `default` or `promptRelay`
- **MultiTrack Editor**: Fix issue where only the first audio was output when inputting audio list in App mode
- **MultiTrack Editor**: Fix issue where segments cannot be added at the beginning or between segments on audio and subtitle tracks

## [1.1.2] - 2026-07-12

### ✨ Features

- **MultiTrack to S2V Output**: Outputs cropped audio clips and starting frame number for easier S2V usage

### 🐛 Bug Fixes

- **Media Selector**: Fix issue where media selector did not stay in subdirectory after reopening when subdirectory was selected last time
- **LTXV Reference Video**: Optimizing the Use of `LTXVMakeRefVideo`
- **MultiTrack Editor**: Fixed inconsistent left-side crop behavior compared to standard editing tools
- **MultiTrack Editor**: Fixed incorrect waveform display after cropping audio clips

---

## [1.1.1] - 2026-07-11

### 🐛 Bug Fixes

- **MultiTrack Editor**: Added empty state prompt message, removed 720 panorama feature from image items, added single image preview
- **Merge Videos From Paths**: Optimize video processing and add audio option
- **Compare Video**: Fix mute issue by default, and add option to save video to reduce the need for extra video save nodes

---

## [1.1.0] — 2026-07-09

### ✨ Features

- **MultiTrack Editor**: Add initial version of multitrack editor with supporting nodes, supporting multitrack video, audio editing, segment editing and preview
- **Media Selector**: Add directory store cache for media selector to solve the problem of frequent fetching of list data from backend
- **Split Image**: Support image list or image batch type image splitting, applicable to `Bernini multi-reference` scenario
- **Merge Videos From Paths**: Add `frame_count` to support clipping

### 🐛 Bug Fixes

- **MultiTrack Editor、TimelineEditor**: Fixed the default width and height when creating nodes
- **Media Selector**: Fix resource sorting should be by `name`, `creation time`, `folder first`
- **Media Selector**: Fix issues where keyword is not cleared when entering subdirectory after searching

---

## [1.0.4] — 2026-06-16

### ✨ Features

- **Save Video**: Add `hide&save` option to hide output video node output while saving video
- **Timeline Editor (App Mode)**: Add `[0-5s]` time range parsing support for `prompt_override`
- **Timeline Editor (UI Mode)**: Sub-track supports `drag and drop to import images`

### 🐛 Bug Fixes

- **Timeline Editor (UI Mode)**: Fix the issue where sub-track image should proportionally adjust duration when segment duration is modified in main track
- **Timeline Editor (UI Mode)**: Fix incorrect audio preview display after importing audio subdirectory

---

## [1.0.3] — 2026-06-06

### ✨ Features

- **Bernini Temporary Solution**: Add `Bernini conditioning` and `Bernini Model Patch` nodes as a temporary solution before ComfyUI official Bernini support
- **LTXV Reference Video**: Add new node for multi-reference Lora [model](https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference)

### 🐛 Bug Fixes

- **Timeline Editor (UI Mode)**: Fix `node height` being reset to default when `canvas refresh` and `resolution option` are switched
- **Timeline Editor (UI Mode)**: Fix issue where segment content cannot be edited in some cases under `overall editing` prompt mode
- **Timeline Editor (UI Mode)**: Fix adaptive node and track height issues, add `clone segment` in right-click menu for `wan2.1 bernini` and `LTX2.3 R2V` usage

---

## [1.0.2] — 2026-05-31

### 🐛 Bug Fixes

- **Timeline Editor (App Mode)**: Fix issue where segments should evenly distribute default duration when `prompt_override` is not strictly in prompt format
- **Timeline Editor (App Mode)**: Fix issue where only one audio segment is used to fill the entire timeline - need to filter out empty audio first
- **Timeline Editor (UI Mode)**: Fix resource output and sorting errors when a single segment contains different formats

---

## [1.0.1] — 2026-05-27

### ✨ Features

- **Workflow**: Add wan2.2 loop segment example workflow
- **Frontend**: Add `+` button when segment is selected to add segments before or after, and fix some known bugs

### 🐛 Bug Fixes

- Fix incorrect image links imported from output and subdirectory, causing images and outputs to be filtered out in editor

---

## [1.0.0] — 2026-05-25

### 💥 BREAKING CHANGES

- `Duration & Frame Rate` input only takes effect on `blur` (must press enter or click outside input box to confirm changes)
- `Duration Input` step change: step is `4` when format is frame count, `1` when format is seconds
- Segment duration editing no longer affects other segments - if total duration exceeds main track after modification, main track will automatically adapt to the sum of all segments

### ✨ Features

- **Timeline Editor**: Track auto-adapt height, image and audio segments require double-click to enter media selection interface to avoid accidental triggering
- **Timeline Editor**: Add dynamic parameter injection settings, support prompt template format + multimedia input for timeline editor usage
