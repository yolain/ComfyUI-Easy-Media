# ComfyUI UI 工作流提交与执行

仅创建/修改 JSON 不触发提交；要求在界面打开时只加载，明确要求运行、执行或生成媒体时才自动入队。

## 地址与预检

- 地址优先级：用户明确提供的地址（`--url`）→ Easy Media 节点包 `config.yaml` 的 `COMFYUI_URL` → `http://127.0.0.1:8188`。文件、字段不存在或值为空时才回退；只给 `host:port` 时补 `http://`。配置无效或连接失败时报告错误，不切换实例。
- 脚本在仓库内或以符号链接安装时自动定位节点包配置；复制安装时，从节点包根目录或其子目录运行，或传 `--config /path/to/ComfyUI-Easy-Media/config.yaml`。其他目录默认检查当前目录的 `config.yaml`；不扫描机器上其他实例。
- 存在配置文件时需要 PyYAML，可使用 ComfyUI 的 Python 环境。仅读取 `COMFYUI_URL`，不要输出整个配置文件中的 API key。上传、打开/入队和状态查询共用解析后的地址；用户未指定地址时不要主动传默认 `--url` 覆盖配置。
- 安装包含此接口的 Easy Media 后重启 ComfyUI，并刷新浏览器页面。必须保持至少一个已加载工作流的 ComfyUI 页面在线。
- 请求 `GET /object_info` 检查目标节点；素材必须上传到同一实例，见 [upload-api.md](upload-api.md)。
- 接口沿用 ComfyUI 的访问权限，不提供独立认证；勿把未鉴权的 ComfyUI 暴露到公网。代理鉴权需使用已有授权凭据；脚本不接管浏览器登录。

## 从 UI 打开并运行

在 skill 目录执行：

```bash
python3 scripts/submit_workflow.py RESULT.json
```

脚本只提交带 `nodes`、`links`、布局和控件值的 **UI workflow JSON**。前端先加载到可见画布，再调用与“运行”按钮相同的原生入队入口，保留自定义控件和种子的提交钩子。不要手写 API prompt，也不要直接请求 `/prompt`。

- 默认 `--mode new_tab`，新建工作流 tab；`--mode replace` 替换当前 tab 的画布，不自动保存或覆盖磁盘文件。
- `--no-queue` 只打开，不运行。默认自动运行，因此只有打开意图时必须带此参数。
- `--name 名称` 指定 tab 名称前缀；新 tab 附加请求 ID，避免与已打开流程重名。
- `--clients` 列出在线页面；一个页面时可省略 `--client-id`，多个页面必须指定目标，不能选择第一个或广播。
- `client_id` 是本接口为每个页面生成的 ID；`session_id` 是对应 ComfyUI WebSocket ID。同一个页面里的多个工作流 tab 不计为多个客户端。
- `--request-id ID` 提供重试标识；脚本在 POST 前把它输出到 stderr。相同 ID 和相同参数只返回已有请求，不重复执行；相同 ID 更换参数会报错。
- `--status ID` 查询已有请求，不再次提交。`--timeout` 仅控制等待浏览器回执的秒数，不代表生成时限，不取消已经提交的任务。

提交开始后到返回 `queued` / `loaded` 前，不要在目标页面手动切换工作流或连续点击运行；冲突会停止自动提交并报告错误。

## HTTP 接口

`GET /easy-media/workflow/clients` 返回 `{"clients": [{"client_id": "...", "session_id": "...", "title": "...", "last_seen": 0}]}`。

`POST /easy-media/workflow/submit`，要求 `Content-Type: application/json`：

```json
{
  "workflow": { "nodes": [], "links": [], "version": 0.4 },
  "mode": "new_tab",
  "auto_queue": true,
  "name": "My workflow",
  "request_id": "unique-request-id",
  "client_id": "client-id-from-clients-endpoint"
}
```

这里的空图只展示格式，运行时必须替换成完整工作流。`workflow` 必填，其余字段可选；`mode` 默认 `new_tab`、`auto_queue` 默认 `true`。只有一个在线页面时可省略 `client_id`。

返回 `request_id`、目标 `client_id` 和初始 `status`。用 `GET /easy-media/workflow/submissions/{request_id}` 查询：

| status | 含义 |
|---|---|
| `pending` | 等待前端领取，尚未打开 |
| `loading` | 已领取，正在加载或入队 |
| `loaded` | 已打开，未要求入队 |
| `queued` | 原生入队成功，附带 `prompt_id`；不代表生成完成 |
| `failed` | 加载失败或尚未入队就停止，附带 `error` |
| `unknown` | 执行结果未确认；可能已入队，先检查 queue/history，不能自动重跑 |

HTTP 400 表示参数错误，409 表示多个页面未指定目标、目标忙或请求 ID 冲突，503 表示没有可用前端或服务容量已满。前端轮询领取并回传结果，不向全部 WebSocket 广播。无回执时不会重新派发已领取的工作流。

请求状态只保存在当前 ComfyUI 进程内：完成记录保留一小时；未领取请求 60 秒后失败，领取后超过 120 秒无回执变成 `unknown`。`unknown` 会阻止向同一页面再次自动提交；先检查队列和历史，确认后可刷新页面重新注册。重启或记录过期后不能依赖原 ID 去重。

## 生成完成判定

1. 保存 `request_id`，等待到 `queued` 并记录 `prompt_id`。`loaded`、`pending`、`loading` 都不表示已运行。
2. 用 `GET /queue` 查看 pending/running；用 `GET /history/{prompt_id}` 等待终态。
3. 只有对应记录的 `status.completed` 为 `true` 且 `status.status_str` 为 `success` 才算成功；错误/中断按失败报告。
4. 从 history 的 `outputs` 汇总文件。用户未要求长期监控且仍在执行时，返回 ID 和当前状态，不取消任务；取消或清空队列需要明确要求。

## 执行前复核

- 工作流通过 dry-run、write 后 inspect 和不变量检查。
- 分辨率与 DynamicCombo 子字段一致；目标 task 范围、`project_save`、sampling 符合用户要求。
- `audio_locked` 唯一且覆盖目标任务；`shared_reference` 指向预期的公用图片、音频或视频，旧 `speaker_reference` 仅用于兼容迁移。
- 媒体路径存在于当前 ComfyUI 实例；覆盖当前 tab 时用户已要求替换。
