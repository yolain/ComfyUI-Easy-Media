# Changelog

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