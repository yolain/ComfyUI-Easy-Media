<div align="center">

<img src="https://github.com/user-attachments/assets/fb602a3c-4a2a-48da-8c44-d36417f4633b" height="120">
<h1>ComfyUI-Easy-Media</h1>

[English Docs](./README.md) | [变更日志](./CHANGELOG_CN.md)

这是一个用于简化媒体加载和视频处理管道构建的 ComfyUI 自定义节点包。它提供了直观的节点，通过用户友好的参数简化媒体资源的编辑与加载，从而更轻松地构建和配置视频处理工作流。

[![][github-release-shield]][github-release-link]
[![][github-stars-shield]][github-stars-link]
[![][github-forks-shield]][github-forks-link]
[![][github-license-shield]][github-license-link]
[![][workflow-shield]][workflow-link]

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

<a id="multitrack-workflow-skill"></a>

### 🤖 使用 Skill 自动生成多轨工作流

节点包附带 `easy-media-multitrack-workflow` Skill，可让 Codex 根据素材和自然语言要求，从内置模板新建或修改已有工作流，自动编排时间线、提示词、衔接模式与采样参数。

```text
使用 $easy-media-multitrack-workflow，根据这些参考图和提示词创建三个各 5 秒的任务，
第一段用分镜，后两段用上下文，生成新的工作流 JSON。
```

默认只生成 JSON；明确要求运行时可上传素材并提交 ComfyUI 执行。安装方法与详细用法见 [Skill 说明](./skills/easy-media-multitrack-workflow/README.md)。

## ✨ 核心功能

### 🎞️ (新)多轨项目流水线 MultiTrack Project Pipeline

![pipeline](https://github.com/user-attachments/assets/0170ae6f-149e-41ba-9bdf-03ec31b0625e)

v1.3.0 新增多轨项目流水线，将时间线编排、逐段生成、上下文衔接和成片合并串起来。目前 **多轨项目节点适配 MiniMax H3**；多轨编辑器本身仍与模型解耦，可以继续接入其他模型工作流。

如果希望通过自然语言自动配置这套流水线，可使用 [工作流生成 Skill](#multitrack-workflow-skill)，从空模板生成或在已有工作流上修改。

```text
多轨编辑器 ── TRACKS_INFO ──→ 多轨项目 ── PROJECT_NAME ──→ 多轨项目视频合并 ── VIDEO ──→ 保存视频
                               ↑
                         model_loader
                    （可选 model_loader_2nd）
```

> [!IMPORTANT]
> **v1.3.0 改为按需加载媒体。** 时间线未引用 Slot 资源时，编辑器不再提前加载整条时间线的图像、音频和视频，`IMAGES`、`AUDIO`、`VIDEO` 输出为 `None`，这是正常行为。使用项目流水线只需连接 `TRACKS_INFO`；旧工作流如需直接获取媒体，请改接 [多轨任务输出](#multitrack-lazy-loading)。

#### 1. 多轨编辑器：编排任务与衔接方式

选择 `MiniMax` 格式，设置目标尺寸、帧率，为各个任务片段填写提示词，并按需要添加参考图、视频或音频。轨道、任务模式与媒体编辑操作见下方 [多轨编辑器介绍](#multitrack-editor)。

新增的 **衔接模式** 决定当前任务如何接续上一段，与文生视频、图生视频、参考生视频等任务模式分别设置：

| 衔接模式 | 生成方式 | 适用场景 |
|----------|----------|----------|
| **分镜（`shot`）** | 当前片段独立生成，不继承上一段的运动和声音上下文 | 新镜头、换场景、需要明显切镜的段落 |
| **上下文（`context`）** | 取上一段生成结果的尾部音视频潜空间，引导当前片段继续运动与声音 | 连续动作、长镜头、需要延续声音的段落 |
| **角色替换上下文（`context_swap`）** | 一采与二采均使用经过一次性锥形噪声处理的视频上下文，音频上下文保持不变 | 需要继承上一段动作，但替换角色或外观的片段 |

第一段作为起始镜头使用分镜模式；后续片段可逐段切换，也可多选任务片段统一调整。例如“分镜 → 上下文 → 上下文 → 分镜”表示先连续生成三段，再开启一个新镜头。此设置影响**生成阶段**，并非合并时添加的淡入淡出转场；上下文也不能保证任意场景或提示词变化都能无缝接续。

#### 2. 多轨项目：编码、采样与逐段循环

将编辑器的 `TRACKS_INFO` 接入 `tracks_info`，再接入包含 H3 模型、CLIP、视频 VAE 和音频 VAE 的 `model_loader`。节点会根据任务顺序自动展开执行，不需要再手动搭建片段索引循环：

```text
读取当前任务并按需加载媒体 → 编码提示词与参考条件 → 一采
    →［可选：放大视频潜空间 → 二采］→ 解码音视频 → 裁剪上下文重叠部分
    → 保存片段与上下文 → 处理下一任务 → 输出项目名称
```

每段都按自己的提示词、参考素材和衔接模式生成。开启上下文时，一采使用上一段对应分辨率的上下文，二采再使用上一段最终高分辨率结果维持接缝；保存完成后才继续下一段。可指定起始片段和生成数量，也可只生成一采预览后再续跑二采。参数、放大模型依赖及上下文实现的来源与调整见 [多轨项目详细说明](#multitrack-project)。

#### 3. 多轨项目视频合并：预览、选片与输出

将项目的 `PROJECT_NAME` 接入合并节点，便可预览已生成的片段、选择每段采用的版本，并将各段拼接为完整视频。**自动合并**适合一次运行直接出片；关闭后可先检查结果，再点 **合并** 单独执行合并与下游保存，无需重跑模型。同一片段有多个生成版本时，可选择两个视频进行同步对比，选定后再输出。具体操作见 [多轨项目视频合并](#multitrack-project-video-combine)。

<a id="multitrack-editor"></a>

### 🎞️ 多轨编辑器 MultiTrack Editor


> **提示：** 多轨编辑器的优势在于解耦，它只用来做媒体的编辑与加载，不与任何模型绑定，用户可以自由选择任何模型节点来处理多轨编辑器输出的媒体数据。

#### 概览

![multiTrackEditor](https://github.com/user-attachments/assets/45dc72fd-d2bc-4df9-9e46-5c3e7fc6aa62)

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

<a id="multitrack-lazy-loading"></a>

#### v1.3.0 媒体懒加载与旧工作流迁移

编辑器优先传递 `TRACKS_INFO` 中的时间范围、提示词、媒体地址及轨道配置，由下游输出节点在处理对应任务时再加载所需媒体，减少长时间线提前解码、拼接全部资源的开销。媒体元信息探测与前端缩略图、波形预览不等同于工作流中的完整媒体加载。

| 时间线的资源来源 | 编辑器输出行为 | 如何获取媒体 |
|------------------|----------------|--------------|
| 文件或 URL，**没有 Slot 引用** | 输出 `TRACKS_INFO`，媒体输出为 `None` | 将 `TRACKS_INFO` 接到多轨项目或多轨任务输出，由下游按需加载 |
| **有 Slot 引用**，引用上游输入的图像、音频或视频 | 保留即时加载及媒体输出行为；仅请求实际引用的媒体输入类型 | 可继续使用编辑器的媒体输出及配套输出节点 |

旧工作流如果直接连接编辑器的媒体输出，可在中间增加 `多轨任务输出（easy multiTrackTaskOutput）`：

- **逐段处理**：连接 `TRACKS_INFO`，设置 `task_index = 0、1、2…`，获取相应任务范围的媒体。
- **一次获取全部媒体**：设置 `task_index = -1`，获取整条时间线的媒体，替代旧版编辑器直接输出完整媒体的用法。这会恢复完整加载的资源开销。
- **只读尺寸、帧率、总帧数和任务数量**：使用 `多轨信息输出`，无需为了这些信息加载完整媒体。
- **使用多轨项目流水线**：项目内部会调用任务输出，不需要在编辑器和项目之间额外串接任务输出节点。

#### v1.3.x 任务与音频设置

- **任务轨道**：支持分镜 / 上下文 / 角色替换上下文三种衔接模式；多选任务片段时，可批量调整任务模式、衔接模式和参考图尺寸。
- **锁定音频**：用于多轨项目，以输入音频约束画面生成，交付视频使用原始任务音频，避免将其重新生成。与仅作为声音参考的音频用途不同。
- **复用音频**：同一参考音频可供多个任务使用，无需复制到每段下面；参考音频最多使用 15 秒，不因当前任务片段更短而被截断。


#### 适用场景

| 场景 | 描述 | 条件 
|------|------|------|
| 视频生成 | MiniMax H3/wan/bernini/ltx t2v、i2v、r2v | 任务轨道有片段即可
| 视频编辑 | bernini v2v、bernini vi2v、wan animate、ltx video replace、ltx iclora edit/inpaint/outpaint | 视频轨道片段及任务轨道片段必要
| 视频参考 | wan scail2、wan animate、ltx iclora guide | 视频轨道片段及任务轨道片段必要
| 视频配音 | wan infinititalk、longcat avatar、ltx ai2v | 任务轨道片段和音频轨道片段必要
| 视频字幕 | - | 任务轨道有片段即可
| 字幕朗读 | - | 任务轨道与字幕轨道有片段即可

- 仅统计了热门开源模型常见的生成类型，理论上任何视频模型流程都可以通过多轨编辑器作为前置处理工具

#### 额外模型（可选）

| 场景 | 功能说明 | 下载地址 | 本地路径 | 前置依赖
|------|----------|----------|----------|-------------|
| **视频字幕（Whisper）** | 音视频识别生成字幕 | [Whisper Large V3](https://huggingface.co/Comfy-Org/HuMo_ComfyUI/tree/main/split_files/audio_encoders) | models/audio_encoders/ | `pip install openai-whisper` |
| **视频字幕（Qwen3）** | 音视频识别生成字幕 | [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) <br>[Qwen3-ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | models/Qwen3-ASR/ | `pip install qwen-asr torchaudio` |
| **字幕朗读** | 字幕转语音配音 | [VoxCPM2](https://huggingface.co/openbmb/VoxCPM2) | models/voxcpm/ |  `pip install voxcpm` |
| **镜头检测** | 智能分割视频镜头 | [OmniShotCut](https://huggingface.co/uva-cv-lab/OmniShotCut/resolve/main/OmniShotCut_ckpt.pth) | models/checkpoints | - |

> **提示：** 部分模型支持通过 Easy-Media 内置的模型下载接口自动下载，模型文件将放置在 `ComfyUI/models/` 目录下。


<a id="multitrack-project"></a>

### 🎞️ 多轨项目 MultiTrack Project

![multiTrackProject](https://github.com/user-attachments/assets/b4cc13a9-5e64-4361-8b3a-ddf900d04a94)

`easy multitrackProject` 负责管理 MiniMax H3 项目的逐段生成。编辑器中的尺寸是**一采尺寸**，双采样在此基础上放大后进行二采；项目文件保存在 `ComfyUI/output/easy_media/projects/<project_name>/`，包含片段媒体、项目记录以及用于续接的上下文潜空间。

#### 编码与采样

1. **读取任务并编码**：按当前任务读取提示词和媒体，根据任务模式构建文生、首尾帧、尾帧或多媒体参考条件，并创建音视频潜空间。存在锁定音频时，将音频编码并施加采样约束。
2. **一采**：`sampling_mode = single` 和 `dual` 都按编辑器设定尺寸生成，不再根据 `upscale_by` 反推缩小一采尺寸。
3. **放大与二采（仅 `dual`）**：将设定的宽、高分别乘以 `upscale_by`，再按 `round(尺寸 * upscale_by / 32) * 32` 就近对齐到 32 的倍数。把一采的视频潜空间放大到此尺寸，与音频潜空间重新组合后进行二采；尺寸变化时重新构建对应的参考条件。`upscale_by = 1.000` 时不做尺寸放大，但仍可进行二采。
4. **解码与保存**：解码视频和音频；上下文片段会同步裁掉开头重复的引导部分及末尾用于时间网格对齐的多余帧，再保存当前片段。
5. **继续下一段**：保存上下文与项目记录后，自动处理后续任务。分镜模式重新开始一个镜头，上下文模式继承前一段；最终输出 `PROJECT_NAME` 供合并节点读取。

| 设置 | 说明 |
|------|------|
| `model_loader` | 一采 H3 模型与共用的 CLIP、视频 VAE、音频 VAE；视频项目也需要音频 VAE |
| `model_loader_2nd` | 可选的二采 H3 模型；不接时复用一采模型，接入后也仍使用一采加载器的编码器和 VAE |
| `sampling_plan` | 内置 `ultra_light`、`light`、`medium`、`high` 等预设，根据 Turbo / 非 Turbo 模型选择采样器和 sigmas；也可用 `custom` 自定义 |
| `sampler` / `sigmas` | 成对接入以覆盖一采采样设置；二采对应 `sampler_2nd` / `sigmas_2nd`，也需成对接入。`custom` 需要为实际运行的每个采样阶段提供这两个输入 |
| `upscale_by` | 相对于编辑器尺寸的二采放大倍率；默认 `1.250`，三位小数，步长 `0.001` |
| `disable_2nd_noise` | 控制是否禁用二采新增噪声，不代表跳过二采 |
| `1st_pass_only` | 在 `dual` 下只执行所选范围中**第一段的一采**并保存检查点；下次关闭此项，可从该段已有检查点继续二采 |

对齐使用 Python `round`，与 [H3 潜空间放大节点](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler/blob/main/nodes/minimax_h3_latent_upscaler_3d.py) 一致，并非固定向上对齐；恰好居中时取偶数。例如 `1344 × 768`、倍率 `1.250`，相乘后为 `1680 × 960`，二采对齐尺寸为 **`1664 × 960`**。两条放大路径采用相同尺寸，项目记录与合并导出也保留实际输出尺寸。单采样和仅一采预览保持编辑器尺寸；纯音频项目（`32 × 32`）不放大。旧工作流保留已保存的倍率，不会自动改成新默认值。

采样预设可参考仓库中的 [h3_sample.json.example](./presets/h3_sample.json.example)，复制为 `presets/h3_sample.json` 后修改。**当前内置双采样预设采用独立的一采、二采sigmas**；不应再将 `light` 一概理解成“拆分 sigmas、未完成一采”。如自定义预设使用 `split_step`，才会将同一sigmas拆到两个阶段。已有自定义文件会优先生效，升级后应检查自己的配置。

#### 二采放大模型与依赖

| `upscale_model` | 放大路径 | 依赖 |
|-----------------|----------|------|
| 选择 H3 潜空间放大模型 | 调用 `MinimaxH3LatentUpscaler3D`，直接把视频潜空间放大到乘倍率并对齐后的二采尺寸 | 安装 [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)，将 [H3 放大模型权重](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) 放入 `ComfyUI/models/latent_upscale_models/` |
| `None` | 一采视频 VAE 解码 → 图像缩放 → VAE 重新编码 → 二采 | 需要 [ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes) 的 `ImageResizeKJv2` 节点 |

这里的 `upscale_model` 是 **H3 潜空间放大权重**，与 `model_loader_2nd` 中负责二采的 H3 生成模型用途不同。直接放大潜空间可以省去中间的视频 VAE 解码 / 编码步骤，但二采仍在目标分辨率运行，不意味着高分辨率采样的显存开销也会同比下降。选择了模型但未安装对应节点时会报错，不会静默切换放大方式。

例如可从上述模型仓库下载 `minimax_h3_latent_upscaler_3d_fp16.safetensors`，放入指定目录后重启 ComfyUI，再在 `upscale_model` 中选择。该模型文档给出的放大范围为 1–4 倍；即使项目参数允许更大数值，也应遵循所选放大模型的支持范围。

#### 上下文方式的来源与调整

上下文条件逻辑基于 [NikoDemon80 / ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)，在 Easy Media 内部适配了项目循环和双采样。当前硬衔接实现要求 ComfyUI 支持 H3 原生音视频关键帧（ComfyUI 0.34.0+）；升级时应同时检查 ComfyUI 的兼容性。

相较于只给当前任务附加上一段的上下文条件，项目流水线还做了以下处理：

- **一采硬衔接**：保留原生音视频关键帧与原有多媒体参考，将上一段尾部音视频潜空间复制到当前采样起始潜空间；在复制区域内分别设置视频、音频的锁定与渐进释放掩码，让接缝附近逐渐进入新内容。
- **区分低分辨率与高分辨率上下文**：双采样时，一采继承上一段的一采上下文；二采放大后，再把上一段最终高分辨率视频尾部复制到当前高分辨率潜空间，避免仅放大低分辨率接缝。上下文二采会冻结当前音频，保留一采形成的声音连续性。
- **上下文专用二采Sigmas**：当前内置预设提供 `sigmas_2nd_context = 0.50, 0.30, 0.14, 0.06, 0.0`，用于已有前段上下文的二采；显式接入自定义二采采样器或 sigmas 时，不会再替换为该Sigmas。
- **同步裁剪与干净的续接源**：项目默认取上一段尾部 22 帧作为上下文，为满足 H3 时间网格额外预留 34 帧生成空间。解码后去掉重复开头和多余尾帧，保留当前任务所需帧数；再从实际交付的音视频范围重新编码上下文，避免连续续接时误用被裁掉的尾部。
- **限制高分辨率上下文占用**：下一片段开始前，高分辨率续接潜空间只保留 Motion Context 所需的尾部 22 帧音视频数据，与完整采样结果解除存储引用后转移到 CPU；二采完成后项目文件同样只保存这份高分辨率上下文包。低分辨率一采 latent 仍完整保存，保证“只执行一采”后可在下次运行中继续二采；运行时的上下文传递则使用单独的裁剪副本。
- **项目内自动传递与跨次续跑**：同一轮生成自动传递前段上下文；从中间片段开始生成时，从项目中读取前一段的活动版本上下文，无需手工连 Save / Load Latent 节点。

> **注意：** 上下文依赖前一段已保存的潜空间，只有 MP4 文件不足以恢复完整上下文。改变编辑器尺寸、放大倍率或更换前一段版本后，应重新检查后续上下文链。旧版按反算缩小尺寸生成的一采检查点和上下文潜空间，应重新生成后再按新尺寸规则续跑；已有后续片段不会因为前段改变而自动重新生成。

#### 指定范围、重生成与版本保留

| 设置 | 行为 |
|------|------|
| `project_name` | 指定项目目录与记录，续跑时使用相同项目名 |
| `segment_start_number` | 从第几段开始，**从 1 计数**；与多轨任务输出从 0 开始的 `task_index` 不同 |
| `segment_count` | 本轮最多生成多少段；`-1` 表示从起始段处理到末尾 |
| `project_save = new` | 在同一项目下保留既有结果，为重生成片段新增版本，便于后续对比 |
| `project_save = override` | 覆盖对应片段版本；与 `segment_count = -1` 搭配时，会先清理起始段及之后的已保存片段，再重生成（续跑的一采检查点会保留） |

例如只重做第 3 段，可设 `segment_start_number = 3`、`segment_count = 1`；想保留旧结果对比，再选 `project_save = new`。若第 3 段为上下文模式，需要项目里有第 2 段兼容的上下文。确认第 3 段的新版本后，依赖它的后续上下文片段也应重新生成。

<a id="multitrack-project-video-combine"></a>

### 🎞️ 多轨项目视频合并 MultiTrack Project Video Combine

![MultiTrackProjectVideoCombine](https://github.com/user-attachments/assets/976dd4fd-8b69-4adb-8aee-d2269123502a)


`easy multitrackProjectVideoCombine` 从项目读取已经保存的视频片段。预览可以按时间线连续播放各段，**无需先生成完整合并文件**；可通过项目选择器切换项目或刷新已保存的结果。

#### 自动合并与手动合并

- **自动合并（默认开启）**：执行工作流后，等待项目生成完成，读取当前项目片段并按时间线顺序拼接，输出 `VIDEO` 与 `FILENAME_PREFIX`。接入 `保存视频` 等输出节点即可保存成片。
- **手动合并**：关闭自动合并后，项目仍生成并保存各段，合并节点只更新项目预览，不向下游输出合并结果。检查并选好各段后，等待 ComfyUI 队列清空，再点击节点内的 **合并**；它只提交合并节点及其下游，不重新执行上游编码和采样。
- **保存要求**：手动合并前，需要连接下游视频保存输出节点。合并节点本身提供临时合并视频，最终输出路径和文件名由保存节点控制，可使用 `FILENAME_PREFIX` 作为命名前缀。

#### 同一片段的多个版本与对比

使用 `project_save = new` 重生成后，点击时间线中的片段，展开该片段的视频文件列表：

1. **选择一个视频**：将该版本用于项目预览和合并。
2. **选择两个视频**：进入同步对比预览，便于比较不同种子、提示词或采样设置的结果；每个片段最多同时选择两个版本。
3. **对比后保留一个**：取消不采用的版本，再手动合并。双选仅用于对比，合并仍使用主选版本（当前选择列表中的第一个），不会把对比画面或两个版本一起拼进成片。

片段上会标注分镜 / 上下文模式，便于检查接缝。选择旧版本用于合并只会改变选片，不会修复已生成的后续上下文；若版本间动作或声音结尾不同，需要重新生成相应后续段落。删除片段版本或整个项目会同时删除关联文件与上下文，操作前请确认不再需要续跑。

> **适用范围：** 此节点用于视频项目。编辑器尺寸设为 `32 × 32` 的纯音频项目不支持通过该节点合并视频。

### 🎞️ 字幕烧录到视频 Subtitle To Video

![SubtitleToVideo](https://github.com/user-attachments/assets/58f90eb7-d671-437d-8adf-d8a04a3e261e)

#### 🎞️ 对比视频 Compare Videos

![CompareVideos](https://github.com/user-attachments/assets/3bad558c-c5f4-411d-ba4c-b2edee9b9f11)

> 预览源视频和输出视频的输入，支持交互式对比滑块进行左右对比。


### 🎞️ 保存视频 SaveVideo

![保存视频](https://github.com/user-attachments/assets/30e2dcc3-9ed3-4d5f-bb15-69e50c3e8fca)
> 已整合 SaveVideoRGBA 节点包的视频保存节点，并进行了功能完善。支持视频导出，可自定义输出路径、文件名前缀、帧率等参数。

### 🎞️ 从路径合并视频 MergeVideoFromPath

> 从文件路径列表（或 URL）加载视频文件并将其拼接成单个视频输出。

`截取帧数` 参数默认值为 `-1`，表示保留合并后视频的全部帧；当设置为大于 `0` 的数值时，节点会按合并后视频的帧率换算时长，并使用 FFmpeg 截取最终视频。

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

<table>
  <thead>
    <tr>
      <th>分类</th>
      <th>节点 ID</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">🎞️ 多轨编辑器 Multi-Track Editor</td>
      <td>easy multiTrackEditor</td>
      <td>编辑多轨时间线并传递轨道信息；无 Slot 引用时延迟到下游加载媒体</td>
    </tr>
    <tr>
      <td>easy multiTrackInfoOutput</td>
      <td>输出多轨维度、时长、帧率和任务数量</td>
    </tr>
    <tr>
      <td>easy multiTrackTaskOutput</td>
      <td>按需加载并输出任务段提示词和媒体；task_index = -1 时输出整条时间线媒体</td>
    </tr>
    <tr>
      <td>easy multiTrackAddSubtitleToVideo</td>
      <td>将字幕轨道添加到视频轨道中</td>
    </tr>
    <tr>
      <td rowspan="7">🎬 MiniMax H3</td>
      <td>easy minimaxH3ToVideo</td>
      <td>构建 MiniMax H3 文生视频、参考生视频或首尾帧生视频的条件与潜空间输入</td>
    </tr>
    <tr>
      <td>easy MiniMaxH3ReferenceToVideoBridge</td>
      <td>用于 H3 参考条件构建的桥接节点，无需 Autogrow 展开</td>
    </tr>
    <tr>
      <td>easy MiniMaxH3MotionContextHard</td>
      <td>应用 H3 上下文条件并硬链接视频/音频潜空间连续性</td>
    </tr>
    <tr>
      <td>easy MiniMaxH3HiResContinuity</td>
      <td>将前一个高分辨率视频尾部复制到当前 upscale 潜空间中</td>
    </tr>
    <tr>
      <td>easy removeH3MotionContextLatent</td>
      <td>在循环结束后删除 H3 Motion Context 潜空间文件</td>
    </tr>
    <tr>
      <td>easy multitrackProject</td>
      <td>构建并执行多轨 MiniMax H3 项目，支持可选的第一/第二遍采样</td>
    </tr>
    <tr>
      <td>easy multitrackProjectVideoCombine</td>
      <td>预览项目片段、选择版本并双视频对比，支持自动或手动合并</td>
    </tr>
    <tr>
      <td rowspan="5">🎞️ LTX Video</td>
      <td>LTXVAddGuidesFromBatchIndexes</td>
      <td>从批量图像添加引导图到潜在变量的指定帧索引</td>
    </tr>
    <tr>
      <td>LTXVMakeRefVideo</td>
      <td>将参考图像批次扩展为 IC-LoRA 参考视频</td>
    </tr>
    <tr>
      <td>easy ltxMultiTrackEncode</td>
      <td>构建 Prompt Relay 条件并生成 LTX 视频/音频潜变量</td>
    </tr>
    <tr>
      <td>easy ltxI2VInplaceAndUpsample</td>
      <td>可选地对 LTX 视频潜变量进行 upscale 并应用图像引导</td>
    </tr>
    <tr>
      <td>easy ltxSamplerSimple</td>
      <td>对组合的 LTX 音视频潜变量进行采样并裁剪视频引导</td>
    </tr>
    <tr>
      <td rowspan="4">🎞️ 时间线编辑 Timeline Editor</td>
      <td>easy timelineEditor</td>
      <td>加载媒体时间线（prompt、图片、音频轨道）并输出结构化数据</td>
    </tr>
    <tr>
      <td>easy timelineInfoOutput</td>
      <td>输出时间线信息，包括格式化的 prompt、尺寸和图片索引</td>
    </tr>
    <tr>
      <td>easy timelineSegmentOutput</td>
      <td>输出时间线的特定片段数据</td>
    </tr>
    <tr>
      <td>easy timelineSegmentCount</td>
      <td>输出时间线中的片段总数</td>
    </tr>
    <tr>
      <td rowspan="8">📋 媒体列表操作 Media List Operations</td>
      <td>easy makeImageList</td>
      <td>将多个图片输入组合成图片列表</td>
    </tr>
    <tr>
      <td>easy makeAudioList</td>
      <td>将多个音频输入组合成音频列表</td>
    </tr>
    <tr>
      <td>easy splitAudios</td>
      <td>将音频列表拆分为多个独立音频输出</td>
    </tr>
    <tr>
      <td>easy audioMerge</td>
      <td>合并或拼接最多 6 个音频输入</td>
    </tr>
    <tr>
      <td>easy makeVideoList</td>
      <td>将多个视频输入组合成视频列表</td>
    </tr>
    <tr>
      <td>easy splitVideos</td>
      <td>将视频列表拆分为多个独立视频输出</td>
    </tr>
    <tr>
      <td>easy imageIndexesToIntList</td>
      <td>将逗号分隔的图片索引字符串转换为整数列表</td>
    </tr>
    <tr>
      <td>easy splitImages</td>
      <td>将图像列表或批次拆分为多个单图像输出</td>
    </tr>
    <tr>
      <td rowspan="4">🎬 视频操作 Video Operations</td>
      <td>easy saveVideo</td>
      <td>将图片和可选音频保存为视频文件</td>
    </tr>
    <tr>
      <td>easy getAudioFromVideo</td>
      <td>从 VIDEO 输入中提取音频</td>
    </tr>
    <tr>
      <td>easy mergeVideos</td>
      <td>串联多个兼容的 VIDEO 片段</td>
    </tr>
    <tr>
      <td>easy mergeVideosFromPaths</td>
      <td>从文件路径列表加载并串联视频</td>
    </tr>
    <tr>
      <td rowspan="2">📝 字幕 Subtitle</td>
      <td>easy recognizeSubtitle</td>
      <td>使用 Qwen3-ASR 或 Whisper Large V3 识别字幕，可设置 SRT/时间戳输出、每句长度和模型卸载</td>
    </tr>
    <tr>
      <td>easy addSubtitleToVideo</td>
      <td>将 SRT、时间戳或括号格式的多行字幕文本规范化并烧录到视频中</td>
    </tr>
    <tr>
      <td rowspan="1">🖼️ 参考图与图像 Reference & Image</td>
      <td>easy makeRefsCompositeBySam3</td>
      <td>使用 SAM3 检测提示的主体并组合参考图到画布</td>
    </tr>
    <tr>
      <td rowspan="2">🔧 工具 Utility</td>
      <td>easy matchLine</td>
      <td>返回包含匹配文本的第一行的零基索引</td>
    </tr>
    <tr>
      <td>easy apiWorkflowGate</td>
      <td>判断是否为 API 调用的工作流，透传前面输入项</td>
    </tr>
    <tr>
      <td rowspan="1">🗣️ 语音转视频 Speech to Video (S2V)</td>
      <td>easy berniniS2VConditioning</td>
      <td>统一 Bernini + Wan S2V 条件处理，保留可选单人全画面音频，并支持单人遮罩或可选双人顺序音频</td>
    </tr>
  </tbody>
</table>

## Credits

- [OmniShotCut](https://github.com/UVA-Computer-Vision-Lab/OmniShotCut)
- [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Whisper](https://github.com/openai/whisper)
- [VoxCPM2](https://github.com/OpenBMB/VoxCPM)
- [Bernini S2V](https://huggingface.co/rzgar/Bernini-R-S2V)
- [H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context)
- [MiniMax H3 Chained Character Swap](https://github.com/MacroSony/minimax-h3-chained-character-swap)
- [MiniMax H3 Latent Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

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
[workflow-shield]: https://img.shields.io/badge/💻-Workflows-efff30?color=e92759&labelColor=black&style=flat-square
[workflow-link]:https://www.runninghub.cn/user-center/1852215241684750337/userPost?inviteCode=14757185
