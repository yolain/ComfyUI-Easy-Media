---
name: easy-media-multitrack-workflow
description: "Create, edit, and optionally run ComfyUI Easy Media MultiTrack workflows from a blank bundled template or a user-provided workflow, including media upload, timeline arrangement, resolution, audio roles, and MultiTrack Project settings."
---

# Easy Media MultiTrack 工作流

根据用户需求生成一份可复核的新工作流 JSON；用户明确要求实际生成视频时，也可以上传素材并提交到 ComfyUI 执行。

## 选择基线

只有两条路径：

1. **空模板生成**：用户明确要求新建、从空模板开始，或没有提供可编辑工作流时，使用 `assets/templates/v1.3.0-blank-workflow.json`。需要模板节点替换时阅读 [references/template-workflow.md](references/template-workflow.md)。
2. **已有工作流生成**：用户提供了工作流/模板时，以该文件为唯一基线，只打补丁；保留未涉及的节点、连线、节点 ID、分组、布局和配置。即使其节点图与内置模板不同，也不得用内置模板覆盖。

用户同时提供工作流又明确要求“从空模板重建”时，遵循空模板路径；否则优先已有工作流路径。

## 编辑流程

1. 保留源文件，运行 `scripts/patch_workflow.py inspect WORKFLOW.json`，确认连接的 `easy multiTrackEditor` / `easy multitrackProject`、分辨率、format、FPS、轨道和工程参数。
2. 需要时间线、分辨率或音频角色时阅读 [references/multitrack-schema.md](references/multitrack-schema.md)，生成完整的新 `track_data`。空模板必须用目标内容替换空 TRACK_DATA；已有对象保留 ID 和未知字段，只为新增对象生成 UUID。
3. 仅当用户明确要求上传或执行时，阅读 [references/upload-api.md](references/upload-api.md) 并把本地/URL 素材交给 ComfyUI。使用用户地址，未给则使用 `http://127.0.0.1:8188`；TRACK_DATA 只写接口返回的媒体路径。
4. 用户明确要求受支持的 loader/attention 替换时，按 [references/template-workflow.md](references/template-workflow.md) 原位处理；其他情况不改节点图。
5. 按 [references/multitrack-schema.md](references/multitrack-schema.md#补丁-plan) 生成只包含实际变更的 plan，先 dry-run，再写入新文件并重新 inspect：

   ```bash
   python3 scripts/patch_workflow.py apply SOURCE.json --plan PLAN.json
   python3 scripts/patch_workflow.py apply SOURCE.json --plan PLAN.json --output RESULT.json --write
   python3 scripts/patch_workflow.py inspect RESULT.json
   ```

6. 复核节点数、link 数和非目标节点不变量。脚本报告目标歧义、schema 不兼容或媒体无效时停止，不绕过校验。

## 关键决策

- **分辨率**：用户未表达意图时保留现值。`16:9 0.9mp` 等表达选择 megapixels 模式，并映射完整 `aspect_ratio`；精确尺寸、auto、短/长边和 custom 规则见 schema 参考。
- **主音频**：需要原样沿用并驱动任务时长/节奏时，设置唯一轨道级 `audio_locked: true`。
- **说话人参考**：需要所有 MiniMax 任务沿用某段声音特征时，在该音频片段设置 `content.speaker_reference: true`。它不等于 `audio_locked`；同一音轨最多一个。
- **工程范围**：排队前复核 `segment_start_number`、`segment_count`、`project_save` 和 sampling 设置，尤其是 `override` 的覆盖范围。

## 可选：提交生成

“创建/生成一个工作流”只授权生成 JSON；“运行、执行、开始生成、提交队列、直接出片”等明确意图才授权上传素材并执行。

用户要求执行时，阅读 [references/execution-api.md](references/execution-api.md)：使用用户地址，未提供则使用 `http://127.0.0.1:8188`；所有素材必须上传到同一实例。UI workflow 不能原样 POST 到 `/prompt`，必须先由 ComfyUI 前端序列化为 API-format prompt，再提交并记录 `prompt_id`。等待到成功、失败或用户要求的停止条件，不以“已排队”冒充“已生成”。

## 交付

说明基线路径、输出文件、目标节点 ID、分辨率、时间线摘要、媒体路径、锁定音轨/说话人参考、工程参数和图结构是否保持。若已执行，再报告 ComfyUI 地址、`prompt_id`、最终状态和输出摘要。
