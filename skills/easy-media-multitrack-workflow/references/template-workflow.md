# v1.3.0 工作流模板与节点替换

模板资产位于 `assets/templates/v1.3.0-blank-workflow.json`。它保留 v1.3.0 的完整节点、连线、分组和布局，用于用户明确要求“新建工作流”或没有可编辑工作流时。

模板校验信息：ComfyUI workflow version `0.4`，21 个节点，26 条 links，SHA-256 `6ac8498b5a60689f1919387cec960550bdadb4d9bc9459f4be264cfc3c52a470`。

## 两种工作模式

### 编辑用户现有工作流

以用户当前文件为基线。默认只修改多轨编辑器和 MultiTrack Project 内容，不用模板覆盖现有节点。只有用户明确要求替换不可用/不想要的 loader 或 attention 节点时，才使用下述受限替换。

### 从模板新建工作流

模板的任务轨道和视频轨道均为空。新建流程时仍应通过 `patch_workflow.py` 写入完整的目标 TRACK_DATA，不要直接修改模板资产。

如果不需要替换模型节点，直接以模板为输入应用多轨 plan：

```bash
python3 scripts/patch_workflow.py apply \
  assets/templates/v1.3.0-blank-workflow.json \
  --plan PLAN.json \
  --output NEW-WORKFLOW.json \
  --write
```

根据用户环境替换 loader 或 attention 时，先生成定制模板，再应用多轨 plan。

## 模型加载器替换

模板主模型链为：

```text
MiniMaxH3HybridLoader (11)
→ ModelAttentionBackend (13)
→ LoraLoaderModelOnly (6)
→ ModelPreviewOverrideKJ (21)
→ easy modelLoaderPack (16)
→ easy multitrackProject.model_loader (15)
```

当用户没有 `MiniMaxH3HybridLoader`、明确不想使用它，或要求使用 ComfyUI 默认加载器时，可以把节点 11 原位替换为 `UNETLoader`。节点 ID、位置和现有 MODEL 输出 links 保持不变。模型文件名必须来自用户明确指定或当前 ComfyUI 可用模型列表；不要盲用模板二阶段节点 19 的模型文件。

```bash
python3 scripts/customize_template.py SOURCE.json \
  --output CUSTOMIZED.json \
  --replace-loader \
  --loader-node-id 11 \
  --unet-name minimax_h3_fl2va_pruned_int8_convrot.safetensors \
  --weight-dtype default
```

`UNETLoader` 仍需通过后续 VAE、CLIP 和 `easy modelLoaderPack` 组成 `FAST_MODEL_LOADER`；不要直接把 MODEL 接到 MultiTrack Project 的 `model_loader`。

## Attention 后端替换

`PathchSageAttentionKJ` 与 `MiniMaxH3MemoryEfficientSageAttentionPatch` 都接收并输出 MODEL，因此可在节点 13 原位替换，同时保持输入 link 51 和输出 link 56。

使用通用 Sage Attention：

```bash
python3 scripts/customize_template.py SOURCE.json \
  --output CUSTOMIZED.json \
  --attention-backend pathch-sage \
  --attention-node-id 13 \
  --sage-attention auto
```

使用 MiniMax H3 省显存 patch：

```bash
python3 scripts/customize_template.py SOURCE.json \
  --output CUSTOMIZED.json \
  --attention-backend minimax-memory-efficient \
  --attention-node-id 13
```

两个替换可以在同一次命令中组合。`customize_template.py` 只修改指定节点，拒绝覆盖源文件，并检查顶层 links、节点数量及其他节点内容保持不变。

## 选择原则

- 用户明确指定节点类型时遵循用户选择。
- 目标自定义节点缺失时，选择用户环境中已安装且兼容的替代项；无法确认可用节点时先询问。
- 不因模板提供某个模型、精度或 attention 后端就默认用户环境也有它。
- 替换后检查 MODEL 链完整，且 `easy modelLoaderPack` 仍收到 model、clip、video VAE 和 audio VAE。
- 只删除节点会留下断链。若用户要求移除而不是替换，必须重接生产者与消费者，并同步节点 input/output links；优先使用同语义的原位替换。
