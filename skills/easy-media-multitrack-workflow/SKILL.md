---
name: easy-media-multitrack-workflow
description: "Edit an existing ComfyUI Easy Media MultiTrack workflow from Codex context: upload or reference media, arrange timeline/task segments and prompts, and patch MultiTrack Project sampling ranges without replacing the user's graph."
---

# Easy Media 多轨工作流编辑

以用户当前提供的工作流 JSON 为唯一基线，输出一份保留现有节点、连线、分组和布局的新工作流。样例、空白流程或历史流程只用于理解结构，绝不能作为输出模板覆盖当前工作流。

## 核心边界

- 只修改目标 `easy multiTrackEditor` 的 `widgets_values`，以及与它连接的 `easy multitrackProject` 的多轨工程参数。
- 除非用户明确要求，不新增、删除、替换或重排任何工作流节点、连线、分组、额外信息或节点 ID。
- 不通过“第一个同类型节点”选择目标。优先沿 `easy multitrackProject.tracks_info` 的 link 反查编辑器；多个候选无法消歧时，列出节点 ID 并请用户指定。
- 默认写入新文件；禁止先复制样例流程再写入多轨数据。即使用户工作流已与样例节点不同，也必须从用户当前文件打补丁。
- 上传和运行属于外部状态变更。仅在用户要求或完成请求确有必要时执行；编辑参数不等于自动排队运行。

## 工作方式

1. 运行 `scripts/patch_workflow.py inspect WORKFLOW.json`，确认目标编辑器、连接的工程节点、当前 FPS、轨道和工程参数。先保留原文件。
2. 需要上传本地素材时，阅读 [references/upload-api.md](references/upload-api.md)。记录接口返回的相对 input 路径，不要凭本地文件名猜测上传结果。
3. 编辑时间线或插入媒体时，阅读 [references/multitrack-schema.md](references/multitrack-schema.md)。根据用户要求生成新的 `track_data`：保留未涉及的轨道、片段、ID 和字段；只为真正新增的对象生成 UUID。
4. 创建补丁 plan，并先 dry-run：

   ```bash
   python3 scripts/patch_workflow.py apply CURRENT.json --plan PLAN.json
   ```

5. 检查报告中的目标节点、参数差异和校验结果，再写入新文件：

   ```bash
   python3 scripts/patch_workflow.py apply CURRENT.json --plan PLAN.json --output EDITED.json --write
   ```

6. 重新 inspect 输出文件。对节点数、link 数和所有非目标节点做不变量检查；脚本若报告歧义或 schema 不兼容，不要绕过。

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

说明输出文件、目标编辑器/工程节点 ID、上传后的媒体路径、时间线片段摘要，以及修改过的工程参数。明确说明节点图是否保持不变。若用户随后要求运行，再使用当前 ComfyUI 的队列接口或 UI，并在排队前复核 `segment_start_index`、`segment_count` 与保存模式。
