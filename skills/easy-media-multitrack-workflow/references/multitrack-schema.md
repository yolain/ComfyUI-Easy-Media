# MultiTrack 数据与编排规则

本参考用于创建或调整 `easy multiTrackEditor` 的 TRACK_DATA。当前项目的前端类型定义位于 `frontend/src/types/multitrack.ts`，实际执行逻辑位于 `nodes/basic.py`。

## 动态分辨率

`resolution` 是 `easy multiTrackEditor` 的 DynamicCombo widget。它不属于 TRACK_DATA；修改时必须按所选模式重建对应的子 widget，并同步工作流已有的 `widgets_values_named`。使用 `patch_workflow.py` 的 `editor.resolution` 写入，不要手工假设数组下标。

| 需求 | `resolution` 标签 | 必需子字段 |
| --- | --- | --- |
| 输出固定预设尺寸 | 如 `1344 x 768 (16:9)` | 可选 `resize_method`，默认 `stretch` |
| 明确自定义宽高 | `width x height (custom)` | `width`、`height`，可选 `resize_method` |
| 跟随首个视频比例 | `width x height (auto)` | 可选 `resize_method` |
| 固定短边或长边 | `width x height (shortest)` / `width x height (longest)` | `resize_to_pixel`，可选 `resize_method` |
| 按比例和总像素预算 | `width x height (megapixels)` | `aspect_ratio`、`megapixels` |

选择顺序：用户给出精确宽高时用已有 fixed preset，若没有对应预设则用 custom；只给横竖屏或画幅比例与质量/像素预算时用 megapixels；要求保持源视频比例时用 auto；要求“短边/长边为 N”时用 shortest/longest。用户没有表达分辨率意图时保留当前设置。模型 format 的默认尺寸只是参考，不能覆盖用户明确要求。

### 自然语言中的 MP 与比例

把不区分大小写的 `mp`、`Mpx`、`megapixel(s)` 或“百万像素”视为 megapixels 单位。`0.9mp`、`1 MP`、`0.8 megapixels` 等表达直接选择 megapixels 模式，并原样提取数值。同句中的比例映射为 DynamicCombo 的完整 `aspect_ratio` 值：

| 用户比例 | `aspect_ratio` |
| --- | --- |
| `1:1` | `1:1 (Square)` |
| `2:3` | `2:3 (Portrait Photo)` |
| `3:2` | `3:2 (Photo)` |
| `3:4` | `3:4 (Portrait Standard)` |
| `4:3` | `4:3 (Standard)` |
| `9:16` | `9:16 (Portrait Widescreen)` |
| `16:9` | `16:9 (Widescreen)` |
| `21:9` | `21:9 (Ultrawide)` |

例如“生成 16:9 0.9mp 大小的视频”无需追问，直接规范化为：

```json
{
  "resolution": "width x height (megapixels)",
  "aspect_ratio": "16:9 (Widescreen)",
  "megapixels": 0.9
}
```

MP 数值不得换算为宽高或四舍五入。若只有 MP：当前已经是 megapixels 模式则保留其 `aspect_ratio`；否则可根据明确的横屏/竖屏语义选择 `16:9`/`9:16`，仍无法确定时询问。精确宽高、比例与 MP 互相矛盾时也应询问。

custom 示例：

```json
{
  "resolution": "width x height (custom)",
  "width": 1024,
  "height": 576,
  "resize_method": "crop"
}
```

可用 `resize_method` 为 `stretch`、`resize`、`pad`、`pad (white)`、`pad_edge`、`pad_edge_pixel`、`crop`、`pillarbox_blur`。需要保留完整画面时优先 pad/resize，需要铺满画布且允许裁边时选 crop；不要在用户未提出时擅自改变已有 resize method。

## 时间语义

- `start_frame` 包含，`end_frame` 不包含；片段帧数为 `end_frame - start_frame`。
- 秒转帧使用 `round(seconds * frame_rate)`，每个片段至少 1 帧。
- 同一轨道中的片段按 `start_frame` 排序且不得重叠；允许间隙。不同同类轨道可独立编排。
- 有片段时，`total_length` 应为所有轨道最大 `end_frame`。无片段时可保留编辑器当前默认长度。
- 修改 FPS 时，应按时间比例重映射所有片段边界、`origin_start_frame` 和 task marker，而不是只改 `frame_rate`。

## 顶层对象

```json
{
  "muted": false,
  "volume_db": 0,
  "task_markers": [],
  "task_overview": false,
  "tracks": [],
  "total_length": 240,
  "frame_rate": 24
}
```

轨道 `type` 为 `task`、`video`、`audio` 或 `subtitle`。保留现有未知字段，以兼容新版本。

## 通用轨道与片段

```json
{
  "id": "uuid",
  "name": "Video 0",
  "type": "video",
  "color": "var(--primary)",
  "muted": false,
  "solo": false,
  "volume_db": 0,
  "locked": false,
  "segments": [
    {
      "id": "uuid",
      "start_frame": 0,
      "end_frame": 120,
      "color": "var(--primary)",
      "content": {}
    }
  ]
}
```

新建轨道时优先使用项目已有颜色 token：task 为 `var(--multitrack-task-bg)`，video 为 `var(--primary)`。保留现有轨道自己的颜色。

## 任务片段

```json
{
  "media_type": "none",
  "task_mode": "ref",
  "continuity_mode": "shot",
  "ref_image_size": "match",
  "images": [],
  "user_prompt": "用户提示词",
  "system_prompt": "",
  "user_prompt_variant": "a",
  "user_prompt_b": "",
  "muted": false,
  "volume_db": 0
}
```

- `task_mode`: `default`、`l2v`、`ref`、`edit`。
- `continuity_mode`: `shot` 或 `context`。
- `ref_image_size`: `match` 或 `max`。
- `images` 最多 9 张，顺序决定 Picture 引用顺序。
- 编辑已有任务时保留未要求改变的 `system_prompt`、prompt variant、连续性和参考图策略。
- 若当前 variant 为 `b`，应修改 `user_prompt_b`；否则修改 `user_prompt`。不要无意切换 variant。

任务参考图：

```json
{
  "id": "uuid",
  "source_type": "input",
  "file_path": "codex/session-123/reference.png",
  "file_name": "reference.png"
}
```

`source_type` 可为 `input`、`output`、`local`、`url` 或 `slot`。字段必须与来源匹配：input/output 用 `file_path`，local 用 `local_path`，url 用 `url`，slot 用 `slot_name`。不要把本机绝对路径标成 `input`。

## 视频与音频片段

```json
{
  "media_type": "video",
  "source_type": "input",
  "file_path": "codex/session-123/clip.mp4",
  "file_name": "clip.mp4",
  "duration": 5.0,
  "muted": false,
  "volume_db": 0,
  "speed": 1
}
```

音频把 `media_type` 改为 `audio`。素材原始时长可用 `ffprobe` 获取；时间线时长按用户目标、裁切和速度决定。若用户要求素材与任务片段对齐，应复用相同的 `[start_frame, end_frame)`，不要仅依靠相近秒数。

## 音频锁定决策

`audio_locked` 是音轨级字段，用来指定 MultiTrack Project 的主音频。它会让生成任务沿用该音频，并按它与任务片段的重叠范围调度视频帧。它与轨道的 `locked` 不同：`locked` 只禁止时间线编辑，不能代替 `audio_locked`。

```json
{
  "id": "music-track",
  "name": "Master Song",
  "type": "audio",
  "audio_locked": true,
  "muted": false,
  "segments": [
    {
      "id": "song-segment",
      "start_frame": 0,
      "end_frame": 240,
      "content": {
        "media_type": "audio",
        "source_type": "input",
        "file_path": "codex/mv/song.wav",
        "file_name": "song.wav"
      }
    }
  ]
}
```

同一个 TrackData 最多只能有一条 `audio_locked: true` 的音轨。切换锁定音轨时，应把所有其他 audio track 的 `audio_locked` 显式设为 `false`。不要把 `audio_locked` 写进音频片段的 `content`；旧版片段级字段仅用于兼容迁移。

根据上下文按以下顺序判断：

1. 用户明确指定“锁定”“主音轨”“沿用原声”“不要重生成音频”时，使用其指定音轨。
2. MV、音乐视频、歌词视频、卡点视频、舞蹈视频、按歌曲节拍或完整歌曲时长生成画面时，将歌曲/配乐主轨锁定。
3. 旁白、播客或对白是视频时长与镜头节奏的唯一基准时，也可锁定该旁白主轨。
4. 普通背景音乐、环境声、音效、角色参考音频或临时占位音频不自动视为主锁定音轨，除非用户语义表明它负责驱动生成。
5. 已有一条锁定音轨且用户未要求更换时，优先保留；但仍要检查它是否与本次要生成的任务片段范围相交。

有多条音轨时先利用名称、文件名、素材时长、轨道内容和用户描述消歧。例如 `Master Song`、`完整歌曲` 通常优先于 `SFX`、`环境声`。如果仍有两条或更多合理候选，停止生成补丁并询问用户，至少列出每个候选的轨道 ID、名称、文件名和时间范围。不得按轨道顺序、第一条音轨或最长时长擅自决定。

锁定前还要检查：

- 锁定音轨至少包含一个 `media_type: "audio"` 的片段。
- 锁定音频片段应与本次目标 task segment 的 `[start_frame, end_frame)` 相交；不相交时，先调整时间线或询问用户。
- 作为主音频使用时通常保持 `muted: false`。不要为了“锁定”而修改无关的 solo、音量或轨道编辑锁状态。
- 如果上下文明确不需要保留任何原音频，应确保所有 audio track 的 `audio_locked` 为 `false`。

## 说话人参考音频

`speaker_reference` 是音频片段 `content` 中的布尔字段。MiniMax 模式会提取被标记片段对应的源音频（从源文件开头起，最多 15 秒）作为声音/说话人参考，并让所有被处理的 task segment 沿用该参考；它不要求在时间线上与任务片段重叠。

```json
{
  "id": "voice-reference-segment",
  "start_frame": 0,
  "end_frame": 120,
  "content": {
    "media_type": "audio",
    "source_type": "input",
    "file_path": "codex/voice/reference.wav",
    "file_name": "reference.wav",
    "speaker_reference": true
  }
}
```

决策规则：

- 用户说“设为说话人”“所有片段沿用这个声音”“参考这个人的音色/语速/表达”时，标记其指定的音频片段。
- 标记必须位于 audio track 的 audio segment 上。同一音轨最多一个；切换时把该音轨其他片段显式设为 `false`。常规单说话人工作流默认只保留一个全局候选；多说话人工作流只有在节点图确实支持且用户明确要求时才在不同音轨上保留多个。
- 这是参考而不是时间线原声复用：不要仅因为设置了说话人就同时设置 `audio_locked`。只有用户还要求原样保留该音频并让它驱动时间线时，才分别评估 `audio_locked`。
- MiniMax 的说话人参考会作为每个任务的参考音频处理，该音轨不再按普通时间线音轨语义混入对应任务。若用户既要音色参考又要可听的时间线音频，应使用独立音轨/片段表达两个角色，并在交付摘要中说明。
- 非 MiniMax format 不应自动添加该标记；已有标记在用户未要求改变时保留，但要说明当前 format 不会消费它。
- 多个候选无法消歧时，列出片段 ID、音轨名称、文件名、时长或帧范围，请用户指定；不要按第一段或最长片段选择。

## 自动编排决策

- 用户给出明确镜头/片段长度时优先采用；否则按素材时长或提示词中的时间信息推导，并说明推导。
- “每段 N 秒”意味着连续片段 `[0,N*fps)`、`[N*fps,2N*fps)`……；不要在边界额外加 1 帧。
- 将提示词放入对应 task segment；参考图放入该 task segment 的 `images`；主视频和音频放入各自 media track。
- 对 MV 或其他主音频驱动项目，在完成时间线编排后锁定唯一主音轨，并复核它覆盖目标任务范围。
- 默认保留轨道间空隙。只有用户要求紧凑排列时才消除间隙并顺移后续片段。
- 调整已有片段时尽量保留 segment ID；新建片段、轨道、图片项才生成新 UUID。

## MultiTrack Project 参数

- `segment_start_number` 为从 1 开始的任务片段编号；执行逻辑使用 `segment_start_number - 1` 转换为内部索引。
- `segment_count = -1` 表示从起始索引处理到末尾；非负数限制本次任务片段数量。
- `project_save`: `new` 或 `override`。在 `override` 且 count 为 `-1` 时，执行逻辑会删除起始索引之后的已保存片段再生成；修改前向用户清楚说明。
- `sampling_mode`: `single` 或 `dual`；`sampling_plan` 必须是当前节点可用预设。不要仅凭另一份流程中的值假设当前环境支持。
- 编辑器分辨率是一采尺寸；`dual` 不会反算缩小一采。二采宽高分别为 `round(编辑器尺寸 * upscale_by / 32) * 32`，采用 Python `round` 就近对齐，恰好居中时取偶数，不是固定向上对齐。
- `upscale_by` 是 Project 的独立参数，默认 `1.250`，三位小数，步长 `0.001`；旧工作流保留已保存的值。两条放大路径及合并导出均使用放大并对齐后的尺寸。例如 `1344 × 768`、倍率 `1.250` 的二采为 `1664 × 960`。若用户指定最终输出尺寸，应同时核对编辑器尺寸与倍率，不能将编辑器尺寸直接当作双采样的最终尺寸。
- 单采样、仅一采预览保持编辑器尺寸；`upscale_by = 1.000` 不放大但可二采；纯音频项目（`32 × 32`）不放大。旧版反算缩小尺寸的一采检查点和上下文潜空间应重新生成后再按新规则续跑。
- `1st_pass_only` 仅适用于显式的一阶段检查点流程；不要自动开启。

## 补丁 Plan

Plan 只包含实际变更。`node_id` 可省略并由拓扑定位；`editor.resolution` 是完整 DynamicCombo 设置，`editor.track_data` 是完整 TRACK_DATA。不要把整份节点或工作流放入 plan。

```json
{
  "editor": {
    "resolution": {
      "resolution": "width x height (megapixels)",
      "aspect_ratio": "16:9 (Widescreen)",
      "megapixels": 0.9
    },
    "track_data": { "tracks": [], "total_length": 120, "frame_rate": 24 },
    "format": "MiniMax"
  },
  "project": {
    "project_save": "override",
    "segment_start_number": 1,
    "segment_count": -1,
    "sampling_plan": "medium",
    "sampling_mode": "single"
  }
}
```
