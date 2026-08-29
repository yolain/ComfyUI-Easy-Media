# ComfyUI 提交与执行

仅在用户明确要求运行、执行、开始生成、提交队列或直接产出媒体时使用本参考。仅要求创建/修改工作流 JSON 不触发执行。

## 地址与预检

- 用户提供 ComfyUI 地址时使用该地址；只给 `host:port` 时补 `http://`。
- 未提供地址时使用 `http://127.0.0.1:8188`。
- 不因连接失败猜测其他地址，也不从本地地址自动切换到远程实例。
- 先请求 `GET /object_info`。连接失败、鉴权失败或目标节点不存在时停止并报告。
- 图片、视频和音频必须上传到同一实例；上传规则见 [upload-api.md](upload-api.md)。远程实例需要凭据时，使用用户提供或当前已授权会话中的凭据，不编造 token。

## UI workflow 与 API prompt

`patch_workflow.py` 输出的是 ComfyUI UI workflow JSON（包含 `nodes`、`links`、布局等），而 `POST /prompt` 的 `prompt` 字段要求 API-format prompt：节点 ID 映射到 `{"class_type": "...", "inputs": {...}}`。

不要把 UI workflow 原样放进 `prompt`。有两种正确方式：

1. 在目标 ComfyUI 前端加载编辑后的 UI workflow，让前端执行 `graphToPrompt` 后提交队列。这是只有 UI workflow 时的默认方式，也能正确执行 DynamicCombo 和自定义 widget 的序列化。
2. 如果已经获得由 ComfyUI 前端导出的 API-format prompt，可以直接调用 `/prompt`。提交体应同时保留 UI workflow 元数据：

```json
{
  "prompt": {
    "14": {
      "class_type": "easy multiTrackEditor",
      "inputs": {}
    }
  },
  "extra_data": {
    "extra_pnginfo": {
      "workflow": { "nodes": [], "links": [] }
    }
  }
}
```

上例只展示外层格式，不能作为可运行 prompt。不要靠 `widgets_values` 下标手写 API prompt；DynamicCombo、虚拟节点、bypass/mute 和前端扩展都应交给当前 ComfyUI 前端序列化。

## 提交与完成判定

1. 提交 `POST /prompt`，检查 HTTP 状态、响应中的 `prompt_id`、`error` 和 `node_errors`。400 或存在校验错误时不重试相同 payload，先修复工作流。
2. 记录 `prompt_id`。可用 `GET /queue` 判断它是 pending 还是 running。
3. 用 `GET /history/{prompt_id}` 等待终态。只有对应记录的 `status.completed` 为 `true` 且 `status.status_str` 为 `success` 才算生成完成；`error` 必须作为失败报告。
4. 从 history 的 `outputs` 汇总生成文件。不要仅凭 POST 成功、进入 queue 或 WebSocket 开始执行就宣称完成。
5. 轮询应使用有界间隔并持续给用户状态更新；用户未要求长时间监控且任务超时仍未结束时，返回 `prompt_id` 和当前队列状态，不取消任务。取消或清空队列需要用户明确要求。

## 执行前复核

- 输出工作流已通过 dry-run、write 后 inspect 和不变量检查。
- `resolution` 与 DynamicCombo 子字段一致。
- 目标 task 范围、`project_save`、sampling 参数符合本次执行意图。
- `audio_locked` 唯一且覆盖目标任务；`speaker_reference` 指向预期片段。
- 所有 input 媒体路径确实存在于当前 ComfyUI 实例。
