# ComfyUI Easy Media

<div align="center">
<a href="./README.md"><img src="https://img.shields.io/badge/🇬🇧English-e9e9e9"></a>
<a href="./README_CN.md"><img src="https://img.shields.io/badge/🇨🇳中文简体-0b8cf5"></a>
<br>
</div>

![Poster](https://github.com/user-attachments/assets/6c76433e-1893-4709-8738-acbed4438757)

这是一个用于简化媒体加载和视频处理管道构建的 ComfyUI 自定义节点包。它提供了直观的节点，通过用户友好的参数简化媒体资源的编辑与加载，从而更轻松地构建和配置视频处理工作流。

## 安装

```bash
cd 你的ComfyUI路径/custom_nodes
git clone https://github.com/yolain/ComfyUI-Easy-Media.git
```

## 示例工作流

安装完成后，打开 ComfyUI，在左侧侧边栏的 **Templates（模板）** 面板中即可找到内置的示例工作流，查找 **ComfyUI-Easy-Media** 相关条目。

## 更新日志

### v1.0.4

- **【保存视频】** 增加 `hide&save` 选项，支持在保存视频的同时隐藏输出视频节点的输出
- **【时间线编辑器：App模式】** 为 `prompt_override` 添加 [0-5s] 这种时间范围的解析支持
- **【时间线编辑器：UI模式】** 子轨道添加`拖拽导入图片`的功能
- **【时间线编辑器：UI模式】** 修复主轨中片段时长修改后子轨道如有图片应等比调整对应时长
- **【时间线编辑器：UI模式】** 修复音频子目录导入后音频预览显示错误的问题

### v1.0.3

- **【Bernini临时方案】** 添加了`Bernini conditioning`和 `Bernini Model Patch`节点, 在未更新到ComfyUI支持Bernini时，提供了一个暂时方案
- **【时间线编辑器：UI模式】** 修复`节点高度`在`画布刷新`和`分辨率选项`切换时会被恢复成默认值的bug
- **【时间线编辑器：UI模式】** 修复`整体编辑`提示词模式下，有些情况下不能编辑片段内容的bug
- **【时间线编辑器：UI模式】** 修复节点和轨道高度自适应的问题，右键菜单新增`克隆片段`，以方便 wan2的`berinini`和 `LTX2.3 R2V` 使用
- **【LTXV制作参考视频】** 针对于多参考Lora[模型地址](https://huggingface.co/LiconStudio/LTX-2.3-Multiple-Subject-Reference)新添加的节点

<details>
<summary><b>v1.0.2</b></summary>

- **【时间线编辑器：App模式】** 修复当 prompt_override 未严格按照提示词格式书写时，理应按片段均分默认时长
- **【时间线编辑器：App模式】** 修复只用一段音频填补完整时间线，需过滤掉空音频再进行判断
- **【时间线编辑器：UI模式】** 修复当同个片段中包含不同格式时，输出资源与排序错误的问题
</details>

<details>
<summary><b>v1.0.1</b></summary>

- 【工作流】添加wan2.2 循环片段示例工作流
- 【前端优化】添加选中片段时出现+号可以往前或往后增加片段，并修复了一些已知bug
- 【Bug修复】从output和子目录导入的图片链接有错误，导致编辑器中图片和输出都被过滤。
</details>

<details>
<summary><b>v1.0.0</b></summary>

- 【重要调整】 `时长与帧率`修改的输入框只有在`失焦`后才会生效（即修改数字后需要enter键确认或点击输入框以外区域才能变更成功，减少出错概率 ）
- 【重要调整】`时长输入框`步进变更，当格式为帧数时步进为`4`、秒数时步进为`1`
- 【重要调整】 片段时长编辑不再影响其他片段，如修改后所有片段相加的时长超出总时长，则总时长将自动适配到所有片段的时长总和。
- 调整了时间编辑器中轨道自动适配高度，图像和音频片段需要双击才能进入媒体选择界面，避免误操作导致的频繁弹窗
- 增加了动态参数注入设定，支持提示词模板格式+多媒体传入的方式使用时间轴编辑器
</details>

## 核心功能

### 媒体时间线编辑器 Timeline Editor

> 我认为媒体时间线编辑器组件更适合作为单独的模块节点来使用，会更具有通用性。此节点更聚焦于媒体的导入/编辑、时间轴相关的功能与交互，可以更好地为不同模型的视频流水线创作提供有利的帮助。<br>
编辑器可用于视频单段的生成（如结合PromptRealy），也可用作分段生成，每一段可结合不同模型的视频流水线进行纯文本生成、单图生成、首尾帧生成、多帧生成、参考生成等）

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


### 从路径合并视频 MergeVideoFromPath

> 该节点可以从指定路径加载视频文件，并将它们合并成一个视频输出。

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


### 多轨道音视频编辑器 MultiTrack Editor

> 规划中...



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
| easy imageIndexesToIntList | 将逗号分隔的图片索引字符串转换为整数列表 |
| easy saveVideo | 将图片和可选音频保存为视频文件 |
| easy mergeVideos | 串联多个兼容的 VIDEO 片段 |
| easy mergeVideosFromPaths | 从文件路径列表加载并串联视频 |
| LTXVAddGuidesFromBatchIndexes | 从批量图像添加引导图到潜在变量的指定帧索引 |

## Source of Inspiration

- [WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI)
- [ComfyUI-PromptRelay](https://github.com/kijai/ComfyUI-PromptRelay)
- [ComfyUI-Licon-MSR](https://github.com/liconstudio/ComfyUI-Licon-MSR)
