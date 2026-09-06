---
name: easy-media-multitrack-workflow
description: "Create, edit, and optionally run ComfyUI Easy Media MultiTrack workflows from a blank bundled template or a user-provided workflow, including media upload, timeline arrangement, subject or partial video replacement continuity, audio roles, resolution, and MultiTrack Project settings."
---

# Easy Media MultiTrack 工作流

根据用户需求生成一份可复核的新工作流 JSON；用户明确要求实际生成视频时，也可以上传素材并提交到 ComfyUI 执行。

## 选择基线

只有两条路径：

1. **空模板生成**：用户明确要求新建、从空模板开始，或没有提供可编辑工作流时，使用 `assets/templates/v1.3.0-blank-workflow.json`。需要模板节点替换时阅读 [references/template-workflow.md](references/template-workflow.md)。
2. **已有工作流生成**：用户提供了工作流/模板时，以该文件为唯一基线，只打补丁；保留未涉及的节点、连线、节点 ID、分组、布局和配置。即使其节点图与内置模板不同，也不得用内置模板覆盖。

用户同时提供工作流又明确要求“从空模板重建”时，遵循空模板路径；否则优先已有工作流路径。

## 编辑流程

1. 保留源文件，运行 `scripts/patch_workflow.py inspect WORKFLOW.json`，确认连接的 `easy multiTrackEditor` / `easy multitrackProject`、分辨率、format、FPS、轨道和项目参数。
2. 需要时间线、分辨率或音频角色时阅读 [references/multitrack-schema.md](references/multitrack-schema.md)，生成完整的新 `track_data`。空模板必须用目标内容替换空 TRACK_DATA；已有对象保留 ID 和未知字段，只为新增对象生成 UUID。
3. 仅当用户明确要求上传或执行时，阅读 [references/upload-api.md](references/upload-api.md) 并把本地/URL 素材交给 ComfyUI。使用用户地址，未提供地址时先读取 Easy Media 节点包 `config.yaml` 的 `COMFYUI_URL`，文件或字段不存在（含空值）时才回退至 `http://127.0.0.1:8188`；TRACK_DATA 只写接口返回的媒体路径。
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
- **上下文主体替换**：用户提供视频作为主体或局部替换参考时，将全部目标 task segment 的 `content.continuity_mode` 设为 `context_swap`，并在该视频轨设置 `audio_locked: true` 以保留画面时间线。未另设音频锁时沿用视频原声；另有锁定 audio track 时优先用其音频替换。用户未指定片段长度时，默认把任务范围按连续 10 秒拆分，末段使用剩余时长；完整规则见 schema 参考。
- **主音频/视频原声**：需要原样沿用并驱动任务时长/节奏时，在目标 audio 或 video 轨设置 `audio_locked: true`。同类型最多锁一条；video 与 audio 可各锁一条，并存时 audio 优先执行音频锁定，video 继续约束画面时间线。
- **公用媒体**：上下文分段需要使用同一张图片、同一段音频或同一段视频作为参考时，将对应媒体设置为 `shared_reference: true`。任务图片把字段写在 `content.images[]` 项上；音频/视频把字段写在媒体片段的 `content` 上。公用图片会排在各任务私有图片之前且每个任务合计仍最多 9 张；同一音频或视频轨最多一个公用片段，公用音频最多使用源文件开头 15 秒。它是跨任务参考，不等于轨道级 `audio_locked`。`speaker_reference` 仅作为旧工作流兼容字段，新工作流不再写入。
- **项目范围**：排队前复核 `segment_start_number`、`segment_count`、`project_save` 和 sampling 设置，尤其是 `override` 的覆盖范围。

## 可选：提交生成

“创建/生成一个工作流”只授权生成 JSON；“运行、执行、开始生成、提交队列、直接出片”等明确意图才授权上传素材并执行。

用户要求在 ComfyUI 中打开或执行时，阅读 [references/execution-api.md](references/execution-api.md)，使用 `scripts/submit_workflow.py` 提交 **UI workflow JSON**。默认新开工作流 tab，再通过前端原生运行入口入队；只要求打开时传 `--no-queue`，明确要求覆盖当前 tab 时传 `--mode replace`。不改用 API-format prompt 或直接 POST `/prompt`。未指定目标且只有一个在线页面时自动选择；多个页面时列出客户端让用户指定，不猜测。使用用户地址，未提供地址时先读取 Easy Media 节点包 `config.yaml` 的 `COMFYUI_URL`，文件或字段不存在（含空值）时才回退至 `http://127.0.0.1:8188`；所有素材必须上传到同一实例。执行后记录 `request_id` 和 `prompt_id`，等待到成功、失败或用户要求的停止条件，不以“已排队”冒充“已生成”。

## 交付

说明基线路径、输出文件、目标节点 ID、分辨率、时间线摘要、媒体路径、主体替换衔接模式、锁定媒体轨/公用媒体、项目参数和图结构是否保持。若已执行，再报告 ComfyUI 地址、`prompt_id`、最终状态和输出摘要。
