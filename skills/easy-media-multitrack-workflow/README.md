# Easy Media MultiTrack Workflow Skill

这是 ComfyUI Easy Media 节点包附带的 Codex Skill，用于根据当前对话上下文安全编辑已有的多轨工作流 JSON。

## 它能做什么

- 将本地图片、视频和音频上传到 ComfyUI `input` 目录或指定子目录。
- 自动创建和编排任务、视频、音频时间线片段。
- 把用户提示词、系统提示词和参考图片放入对应任务片段。
- 调整帧率、片段起止位置、时长、连续性模式和参考图策略。
- 调整 `easy multitrackProject` 的采样方案、单/双阶段模式、开始片段索引和本次处理片段数。
- 在包含多个多轨编辑器时，通过节点连线定位实际连接到工程节点的编辑器。
- 保留用户当前工作流中的节点、连线、节点 ID、布局和其他配置，只替换目标多轨编辑器与多轨工程参数。

Skill 默认先执行 dry-run，并输出一份新的工作流文件，不会直接覆盖用户原始 JSON。

## 安装

### 方法一：复制到 Codex Skills 目录

在 Easy Media 节点包根目录运行：

```bash
mkdir -p ~/.codex/skills
cp -R skills/easy-media-multitrack-workflow ~/.codex/skills/
```

如果目标目录中已经安装了旧版本，请先备份或移走旧目录，再复制新版本。不要把新目录嵌套成：

```text
~/.codex/skills/easy-media-multitrack-workflow/easy-media-multitrack-workflow
```

正确入口应为：

```text
~/.codex/skills/easy-media-multitrack-workflow/SKILL.md
```

### 方法二：开发环境使用符号链接

如果希望节点包更新后立即使用最新版 Skill，可在节点包根目录运行：

```bash
mkdir -p ~/.codex/skills
ln -s "$(pwd)/skills/easy-media-multitrack-workflow" \
  ~/.codex/skills/easy-media-multitrack-workflow
```

目标位置已有同名文件或目录时，`ln` 会停止并报错，不会覆盖它。安装完成后，新建一个 Codex 任务以刷新可用 Skill 列表。

## 使用方法

可以显式调用：

```text
使用 $easy-media-multitrack-workflow，把这些图片和视频上传到 input/project-a，
按每段 5 秒编排到当前工作流，并把每条提示词放入对应任务片段。
```

也可以直接描述任务；当请求涉及 Easy Media 多轨工作流时，Codex 可以自动选择该 Skill。例如：

```text
基于我当前的工作流，只修改多轨编辑器：从第 2 个任务片段开始运行 3 段，
使用 medium 采样方案，不要替换或删除现有节点。
```

建议同时提供：

- 当前需要编辑的工作流 JSON。
- 素材文件或可访问的绝对路径。
- 每个片段的提示词及期望时长。
- 目标帧率、片段排列方式和参考图对应关系。
- 是否需要实际上传素材，以及 ComfyUI 服务地址；未说明时通常使用 `http://127.0.0.1:8188` 作为本地示例地址。

## 安全行为

Skill 始终把用户当前工作流作为源文件。空白流程、示例流程和历史流程仅用于理解结构，不会被整份复制到用户工作流中。

补丁脚本会检查：

- 目标编辑器是否确实连接到 `easy multitrackProject`。
- 时间线片段是否排序、重叠或包含无效帧范围。
- 轨道与片段 ID 是否重复。
- 媒体类型及任务参考图数量是否有效。
- 修改前后的节点数、连线和非目标节点内容是否保持不变。

上传媒体和提交 ComfyUI 队列属于外部状态变更。Skill 不会因为用户只要求修改 JSON 就自动运行工作流。

## 目录结构

```text
easy-media-multitrack-workflow/
├── SKILL.md
├── README.md
├── agents/openai.yaml
├── references/
│   ├── multitrack-schema.md
│   └── upload-api.md
└── scripts/patch_workflow.py
```

`SKILL.md` 是 Codex 入口；`README.md` 面向安装和使用者；`references` 保存数据结构与接口说明；`scripts/patch_workflow.py` 负责安全定位节点、校验补丁并生成新的工作流文件。
