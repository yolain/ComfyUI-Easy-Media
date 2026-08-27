# MultiTrack 数据与编排规则

本参考用于创建或调整 `easy multiTrackEditor` 的 TRACK_DATA。当前项目的前端类型定义位于 `frontend/src/types/multitrack.ts`，实际执行逻辑位于 `nodes/basic.py`。

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

## 自动编排决策

- 用户给出明确镜头/片段长度时优先采用；否则按素材时长或提示词中的时间信息推导，并说明推导。
- “每段 N 秒”意味着连续片段 `[0,N*fps)`、`[N*fps,2N*fps)`……；不要在边界额外加 1 帧。
- 将提示词放入对应 task segment；参考图放入该 task segment 的 `images`；主视频和音频放入各自 media track。
- 默认保留轨道间空隙。只有用户要求紧凑排列时才消除间隙并顺移后续片段。
- 调整已有片段时尽量保留 segment ID；新建片段、轨道、图片项才生成新 UUID。

## MultiTrack Project 参数

- `segment_start_index` 为从 0 开始的任务片段索引。
- `segment_count = -1` 表示从起始索引处理到末尾；非负数限制本次任务片段数量。
- `project_save`: `new` 或 `override`。在 `override` 且 count 为 `-1` 时，执行逻辑会删除起始索引之后的已保存片段再生成；修改前向用户清楚说明。
- `sampling_mode`: `single` 或 `dual`；`sampling_plan` 必须是当前节点可用预设。不要仅凭另一份流程中的值假设当前环境支持。
- `1st_pass_only` 仅适用于显式的一阶段检查点流程；不要自动开启。
