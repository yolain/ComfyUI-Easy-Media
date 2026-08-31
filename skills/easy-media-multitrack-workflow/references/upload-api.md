# 媒体上传与路径记录

本参考用于把 Codex 上下文中的素材放入目标 ComfyUI input 目录。优先使用用户给出的 ComfyUI 地址；未提供地址时先读取 Easy Media 节点包 `config.yaml` 的 `COMFYUI_URL`，文件或字段不存在（含空值）时才回退至 `http://127.0.0.1:8188`。同一次工作流执行的上传、排队和历史查询必须使用同一实例。

地址解析与配置文件定位见 [提交参考](execution-api.md#地址与预检)。以下 curl 示例中的本机地址需替换为解析后的地址。

## 上传到 input 子目录

项目的前端统一通过 ComfyUI `/upload/image` 接口上传图片、音频和视频，并传递 input 子目录：

```bash
curl --fail-with-body --silent --show-error \
  -X POST http://127.0.0.1:8188/upload/image \
  -F 'image=@/absolute/path/to/clip.mp4' \
  -F 'type=input' \
  -F 'overwrite=false' \
  -F 'subfolder=codex/session-name'
```

响应通常为：

```json
{"name":"clip.mp4","subfolder":"codex/session-name","type":"input"}
```

写入 TRACK_DATA 的 `file_path` 必须使用响应拼出的相对路径 `subfolder/name`，`source_type` 为 `input`。不要写上传前的本机绝对路径，也不要假设同名文件未被服务端改名。

## 通用根目录上传

`POST /easy-media/upload` 接受任意媒体文件并写入 input 根目录，但当前实现不支持子目录：

```bash
curl --fail-with-body --silent --show-error \
  -X POST http://127.0.0.1:8188/easy-media/upload \
  -F 'file=@/absolute/path/to/audio.wav'
```

响应为 `{"file_name":"..."}`。只有 `/upload/image` 不适用且用户不需要子目录时才使用此接口。

## URL 素材

`POST /easy-media/download-url` 会把 URL 下载到 input 根目录：

```bash
curl --fail-with-body --silent --show-error \
  -X POST http://127.0.0.1:8188/easy-media/download-url \
  -H 'Content-Type: application/json' \
  --data '{"url":"https://example.com/media.mp4"}'
```

如果用户希望保持远程引用而非下载，可在 TRACK_DATA 中使用 `source_type: "url"` 与 `url` 字段。不要在没有用户授权时把私有素材发送到远程服务。

## 上传后检查

- 为本次编辑选择稳定且不含 `..` 的相对子目录，例如 `codex/<project-or-date>`。
- 逐个检查 HTTP 状态和 JSON 响应；失败时停止，不把失败素材写入时间线。
- 同一素材只上传一次，复用返回路径。
- 可通过 `GET /easy-media/media/recent?source=inputs&type=all&subfolder=...` 检查近期文件。
- 上传属于外部写入；dry-run 工作流补丁不会回滚已上传文件。
