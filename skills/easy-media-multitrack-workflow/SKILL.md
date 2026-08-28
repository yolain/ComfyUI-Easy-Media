---
name: easy-media-multitrack-workflow
description: "Edit or create a ComfyUI Easy Media MultiTrack workflow from Codex context: use the bundled v1.3.0 template when requested, upload or reference media, arrange timeline/task segments and prompts, choose a master locked-audio track, safely substitute supported loader/attention nodes, and patch MultiTrack Project settings."
---

# Easy Media 多轨工作流编辑

编辑模式以用户当前提供的工作流 JSON 为唯一基线，输出一份保留现有节点、连线、分组和布局的新工作流。用户明确要求新建工作流或没有现有流程时，使用内置 v1.3.0 模板资产作为基线，再根据上下文完整替换多轨内容。

## 核心边界

- 只修改目标 `easy multiTrackEditor` 的 `widgets_values`，以及与它连接的 `easy multitrackProject` 的多轨工程参数。
- 除非用户明确要求替换不可用或不想要的模型/attention 节点，不新增、删除、替换或重排任何工作流节点、连线、分组、额外信息或节点 ID。受支持的原位替换必须遵循 [references/template-workflow.md](references/template-workflow.md)。
- 不通过“第一个同类型节点”选择目标。优先沿 `easy multitrackProject.tracks_info` 的 link 反查编辑器；多个候选无法消歧时，列出节点 ID 并请用户指定。
- 默认写入新文件；禁止先复制样例流程再写入多轨数据。即使用户工作流已与样例节点不同，也必须从用户当前文件打补丁。
- 上传和运行属于外部状态变更。仅在用户要求或完成请求确有必要时执行；编辑参数不等于自动排队运行。

## 工作方式

1. 选择模式。用户有当前工作流且没有要求新建时，使用当前文件；用户明确要求新建或没有流程时，阅读 [references/template-workflow.md](references/template-workflow.md)，使用 `assets/templates/v1.3.0-blank-workflow.json`。模板的时间线为空，必须写入完整的目标 TRACK_DATA 后才能交付。
2. 运行 `scripts/patch_workflow.py inspect WORKFLOW.json`，确认目标编辑器、连接的工程节点、当前 FPS、轨道和工程参数。先保留原文件。
3. 需要上传本地素材时，阅读 [references/upload-api.md](references/upload-api.md)。记录接口返回的相对 input 路径，不要凭本地文件名猜测上传结果。
4. 编辑时间线或插入媒体时，阅读 [references/multitrack-schema.md](references/multitrack-schema.md)。根据用户要求生成新的 `track_data`：保留未涉及的轨道、片段、ID 和字段；只为真正新增的对象生成 UUID。新建模式必须完整替换模板 TRACK_DATA。
5. 根据用户上下文判断音频是否是驱动视频时长与节奏的主音频。MV、歌词视频、音乐驱动剪辑、按歌曲节拍生成，或用户明确要求保留原音乐/旁白时，应锁定对应音轨。将该音轨设置为 `audio_locked: true`，并把其他音轨设置为 `false`。若有多条合理候选且上下文不能唯一确定，必须先列出候选音轨的 ID、名称和素材摘要并询问用户，不得自行选择。详细规则见 [references/multitrack-schema.md](references/multitrack-schema.md#音频锁定决策)。
6. 若用户要求更换 `MiniMaxH3HybridLoader` 或 `ModelAttentionBackend`，先用 `scripts/customize_template.py` 做受限原位替换；不要手工猜测节点 schema。未明确要求时保留当前节点图。
7. 创建补丁 plan，并先 dry-run：

   ```bash
   python3 scripts/patch_workflow.py apply CURRENT.json --plan PLAN.json
   ```

8. 检查报告中的目标节点、参数差异和校验结果，再写入新文件：

   ```bash
   python3 scripts/patch_workflow.py apply CURRENT.json --plan PLAN.json --output EDITED.json --write
   ```

9. 重新 inspect 输出文件。对节点数、link 数和所有非目标节点做不变量检查；脚本若报告歧义或 schema 不兼容，不要绕过。

## Plan 格式

只包含实际要改的部分。`node_id` 可省略，由拓扑自动定位。

```json
{
  "editor": {
    "node_id": 29,
    "track_data": { "tracks": [], "total_length": 120, "frame_rate": 24 },
    "format": "MiniMax"
  },
  "project": {
    "node_id": 15,
    "project_save": "override",
    "segment_start_index": 0,
    "segment_count": -1,
    "sampling_plan": "medium",
    "sampling_mode": "single"
  }
}
```

`editor.track_data` 是完整的新多轨数据，但它只替换所选编辑器中的 TRACK_DATA widget。不要把整份节点或整份样例工作流放入 plan。`project` 只接受脚本白名单中的 MultiTrack Project widget 字段。

## 交付

说明输出文件、目标编辑器/工程节点 ID、上传后的媒体路径、时间线片段摘要、锁定音轨及判断依据，以及修改过的工程参数。明确说明节点图是否保持不变。若用户随后要求运行，再使用当前 ComfyUI 的队列接口或 UI，并在排队前复核锁定音轨、`segment_start_index`、`segment_count` 与保存模式。
