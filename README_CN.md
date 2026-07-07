<div align="center">

<img src="https://github.com/user-attachments/assets/fb602a3c-4a2a-48da-8c44-d36417f4633b" height="120">
<h1>ComfyUI-Easy-Media</h1>

[English Docs](./README.md) | [变更日志](./CHANGELOG_CN.md)

这是一个用于简化媒体加载和视频处理管道构建的 ComfyUI 自定义节点包。它提供了直观的节点，通过用户友好的参数简化媒体资源的编辑与加载，从而更轻松地构建和配置视频处理工作流。

[![][github-release-shield]][github-release-link]
[![][github-stars-shield]][github-stars-link]
[![][github-forks-shield]][github-forks-link]
[![][github-license-shield]][github-license-link]

<img src="https://github.com/user-attachments/assets/e12f219c-b4c7-47ce-96fb-23103c621720" style="width:100%">
</div>


## 📦 安装

> [!IMPORTANT]
> 强烈建议您在安装此节点包前，先确保您的系统环境中已经安装了`FFmpeg`

```bash
cd 你的ComfyUI路径/custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-Media.git
```

## ✏️ 示例工作流

安装完成后，打开 ComfyUI，在左侧侧边栏的 `Templates（模板）` 面板中即可找到内置的示例工作流，查找 `ComfyUI-Easy-Media` 相关条目。

## ✨ 核心功能

### 🎞️ 多轨编辑器 MultiTrack Editor

![multiTrackEditor](https://github.com/user-attachments/assets/fc9ebcc6-d5e6-4f43-9825-6432c17d340d)

#### 轨道

| 轨道类型    | 功能描述                                           |
|-------------|----------------------------------------------------|
| 任务轨道    | 支持t2v、i2v、r2v、v2v等多种任务类型定义                |
| 视频轨道    | 导入并管理视频片段，支持多段视频拼接、智能分割镜头           |
| 音频轨道    | 导入并管理音频片段，支持多段音频拼接                   |
| 字幕轨道    | 添加或从音视频中识别            |

- 任务片段是该节点的核心，工作流可根据任务轨道中片段的数量，设计自动循环执行
- 视频轨道添加视频片段时也将自动添加对应时长的任务片段
- 选中任务片段可设置图片、任务类型、用户提示词/系统提示词（根据任务类型会有默认值也可以自行编写）
- 在 `多轨信息输出` 节点将输出视频的宽高尺寸、视频总帧数、帧率、任务数量
- 在 `多轨任务输出` 节点将输出对应片段任务的 用户提示词&系统提示词，用户可自行抉择是否外接LLM节点以进行提示词扩写或结合片段中图像进行反推


#### 适用场景

| 场景 | 描述 | 条件 
|------|------|------|
| 视频生成 | wan/bernini/ltx t2v、i2v、r2v | 任务轨道有片段即可 
| 视频编辑 | bernini v2v、bernini vi2v、wan animate、ltx video replace、ltx iclora edit/inpaint/outpaint | 视频轨道片段及任务轨道片段必要
| 视频参考 | wan scail2、wan animate、ltx iclora guide | 视频轨道片段及任务轨道片段必要
| 视频配音 | wan infinititalk、longcat avatar、ltx ai2v | 任务轨道片段和音频轨道片段必要
| 视频字幕 | - | 任务轨道有片段即可
| 字幕朗读 | - | 任务轨道与字幕轨道有片段即可

- 仅统计了热门开源模型常见的生成类型，理论上任何视频模型流程都可以通过多轨编辑器作为前置处理工具

#### 额外模型（可选）

| 场景 | 功能说明 | 下载地址 | 本地路径 | 前置依赖
|------|----------|----------|----------|-------------|
| **视频字幕** | 音视频识别生成字幕 | [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) <br>[Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | models/Qwen3-ASR/ | `pip install qwen-asr torchaudio` |
| **字幕朗读** | 字幕转语音配音 | [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | models/voxcpm/ |  `pip install voxcpm` |
| **镜头检测** | 智能分割视频镜头 | [OmniShotCut](https://huggingface.co/uva-cv-lab/OmniShotCut/resolve/main/OmniShotCut_ckpt.pth) | models/checkpoints | - |

> **提示：** 部分模型支持通过 Easy-Media 内置的模型下载接口自动下载，模型文件将放置在 `ComfyUI/models/` 目录下。

#### 字幕烧录到视频

![SubtitleToVideo](https://github.com/user-attachments/assets/58f90eb7-d671-437d-8adf-d8a04a3e261e)



### 🎞️ 媒体时间线编辑器 Timeline Editor

![timelineEditor](https://github.com/user-attachments/assets/d7c9e894-6e7e-488c-90fb-d3aa8310419d)

#### 动态参数注入 (05-23)

> 如果你想通过`agents`或`app`方式动态地调用时间线编辑器，目前提供了一种方案，你可以将媒体素材输入到时间线编辑器的对应输入端口（`prompt_override`、`image`、`audio`）中。当`prompt_override`注入时，它会覆盖时间线编辑器中的片段数据。但相比于可视化界面直接编辑片段内容，动态参数注入的方式有局限性，例如无法很方便的控制音频时长和出现的范围，`prompt_override`提供了一种提示词格式化模板的规范写法，类似于 `promptRelay + seedance2.0` 动态提示词结合，具体可参考下方的示例。

![dynamicInput](https://github.com/user-attachments/assets/eef6798e-a68d-4724-8e72-69b1a13825dd)

**可选参数**：
- `prompt_override`：由于ComfyUI存在force_input兼容性问题，当force_input存在时自定义部件将无法被获取，所以目前该参数的类型被设置为了`AnyType`，建议使用常规的字符串类型节点进行连入即可。
- `image`：输入的图片资源列表，建议使用新增加的`easy makeImageList`节点来创建图片列表。
- `audio`：输入的音频资源列表，若片段只需要一段音频直接将音频连接到audio输入口即可，如需要多段音频则建议使用新增加的`easy makeAudioList`节点来创建音频列表。


**提示词示例**：

```
@图片1 @音频1 镜头晃动，老者正望着光亮处神色慌张地喊话： 别学那玩意，别连线啊。 [0-120] | @图片2 @音频2 镜头缓慢推进，男人正在操作电脑，说道：有意思，这ComfyUI能火，我指定得学它 [121-241]
```

- [0-120] 和 [121-241] 表示时间轴上片段的起止帧范围，单位为帧（frame），也支持 [0-5s] [5-10s] 这种写法，单位为秒。如果不指定时间范围，默认会均分原先时间轴编辑器上设置的总时长。
- 片段之间使用 `|` 分隔，表示不同的时间段。每个片段可以包含`媒体占位符`、`文本提示词`和`起止帧范围`。
- 图片注入：支持 `@image{n}`、`@img{n}`、`@图{n}`、`@图片{n}`、`@图像{n}` 作为占位符来注入图片资源, 其中`{n}`表示图片列表中的第n张图（从1开始计数）。例如，`@image1`将注入图片列表中的第一张图。
- 音频注入：支持 `@audio{n}`、`@音频{n}` 作为占位符来注入音频资源, 其中`{n}`表示音频列表中的第n段音频（从1开始计数）。例如，`@audio1`将注入音频列表中的第一段音频。


**使用时间线编辑器添加输入口的媒体**：
> 如果你只想通过`image`或`audio`输入端口传递参数，不想使用`prompt_override`，你也可以在添加图片或添加音频的地方使用`slot`方式关联到对应输入端口的媒体，这样在执行工作流任务时，便会自动将输入的媒体资源关联到时间线编辑器中对应的片段上。
（注意：时间编辑器上显示的预览是溯源到最初加载图片或加载音频的节点中对应资源的，如果你在加载与时间编辑器流程之间使用了裁剪或者截断等节点对原始媒体进行处理，后端同样会执行这一块的处理，只是前端的预览显示是初始加载的状态。）

![dynamicInput2](https://github.com/user-attachments/assets/6dd84d52-1fd3-4b27-a890-2a0e22cecda4)


### 🎞️ 从路径合并视频 MergeVideoFromPath

> 该节点可以从指定路径加载视频文件，并将它们合并成一个视频输出。

`截取帧数` 默认值为 `-1`，表示保留合并后的全部帧；当设置为大于 `0` 的数值时，节点会按合并后视频帧率换算为时长，并使用 FFmpeg 截取得到最终视频。

**推荐安装 FFmpeg** 以获得最佳性能和转场质量：

```bash
# macOS (Homebrew)
brew install ffmpeg

# Windows — 下载完整构建版（包含 xfade 滤镜）：
# https://ffmpeg.org/download.html
# 推荐使用 BtbN 或 gyan.dev 提供的完整版（full build）

# Linux (Ubuntu/Debian)
sudo apt install ffmpeg
```


### 保存视频 SaveVideo

![保存视频](https://github.com/user-attachments/assets/30e2dcc3-9ed3-4d5f-bb15-69e50c3e8fca)
> 已整合 SaveVideoRGBA 节点包的视频保存节点，并进行了功能完善。支持视频导出，可自定义输出路径、文件名前缀、帧率等参数。



## 开发测试

1. 在 ComfyUI-Easy-Media 目录下创建一个 `config.yaml` 文件，添加以下内容，表示使用前端开发环境：

```yaml
WEB_VERSION: dev
```

2. 进入前端目录编译开发环境代码进行调试：

```shell
cd frontend && bun install && bun run dev
```

3. 修改代码后，编译正式环境：

```shell
bun run build:release
```

## 节点列表

| 节点 ID | 描述 |
|---------|------|
| easy timelineEditor | 加载媒体时间线（prompt、图片、音频轨道）并输出结构化数据 |
| easy timelineInfoOutput | 输出时间线信息，包括格式化的 prompt、尺寸和图片索引 |
| easy timelineSegmentOutput | 输出时间线的特定片段数据 |
| easy timelineSegmentCount | 输出时间线中的片段总数 |
| easy makeImageList | 将多个图片输入组合成图片列表 |
| easy makeAudioList | 将多个音频输入组合成音频列表 |
| easy makeVideoList | 将多个视频输入组合成视频列表 |
| easy imageIndexesToIntList | 将逗号分隔的图片索引字符串转换为整数列表 |
| easy saveVideo | 将图片和可选音频保存为视频文件 |
| easy getAudioFromVideo | 从 VIDEO 输入中提取音频 |
| easy mergeVideos | 串联多个兼容的 VIDEO 片段 |
| easy mergeVideosFromPaths | 从文件路径列表加载并串联视频 |
| easy multiTrackEditor | 多轨编辑器，编辑和传递多轨媒体数据 |
| easy multiTrackInfoOutput | 输出多轨维度、时长、帧率和任务数量 |
| easy multiTrackTaskOutput | 输出多轨任务段的提示词和任务范围媒体 |
| easy multiTrackAddSubtitleToVideo | 将字幕轨道添加到视频轨道中 |
| easy makeRefsCompositeBySam3 | 使用 SAM3 检测提示的主体并组合参考图到画布 |
| easy splitImages | 将图像列表或批次拆分为多个单图像输出 |
| easy matchLine | 返回包含匹配文本的第一行的零基索引 |
| LTXVAddGuidesFromBatchIndexes | 从批量图像添加引导图到潜在变量的指定帧索引 |
| LTXVMakeRefVideo | 将参考图像批次扩展为 IC-LoRA 参考视频 |
| BerniniModelPatch | 为 Wan 模型添加 Bernini 上下文潜在支持 |
| BerniniConditioning | Bernini 上下文条件处理，用于视频/图像条件注入 |

## Credits

- [OmniShotCut](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut)
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [VoxCPM2](https://github.com/OpenBMB/VoxCPM)

## Source of Inspiration

- [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)
- [ComfyUI-PromptRelay](https://github.com/kijai/ComfyUI-PromptRelay)
- [ComfyUI-Licon-MSR](https://github.com/liconstudio/ComfyUI-Licon-MSR)
- [ComfyUI-RH-Bernini](https://github.com/RH-RunningHub/ComfyUI-RH-Bernini)


<!-- LINK GROUP -->
[github-forks-link]: https://github.com/yolain/ComfyUI-Easy-Media/network/members
[github-forks-shield]: https://img.shields.io/github/forks/yolain/ComfyUI-Easy-Media?color=8ae8ff&labelColor=black&style=flat-square
[github-license-link]: https://github.com/yolain/ComfyUI-Easy-Media/blob/master/LICENSE
[github-license-shield]: https://img.shields.io/github/license/yolain/ComfyUI-Easy-Media?color=white&labelColor=black&style=flat-square
[github-release-link]: https://github.com/yolain/ComfyUI-Easy-Media/releases
[github-release-shield]: https://img.shields.io/github/v/release/yolain/ComfyUI-Easy-Media?color=f2ff59&labelColor=black&style=flat-square
[github-stars-link]: https://github.com/yolain/ComfyUI-Easy-Media/network/stargazers
[github-stars-shield]: https://img.shields.io/github/stars/yolain/ComfyUI-Easy-Media?color=ffcb47&labelColor=black&style=flat-square
