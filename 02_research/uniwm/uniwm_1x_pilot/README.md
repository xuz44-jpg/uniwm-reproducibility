# UniWM-1X：20 个起点的开发预实验

更新：2026-10-21。关联 [研究计划 v1.3](../NAVIGATION_ACTION_FIDELITY_PLAN.md)。

## 规模与协议

- 已有 `1x200` 基线，直接推理，记忆为空；不训练新模型。
- **20 起点 × 4 动作 × 3 配对 seed = 240 张预测图**。
- 10 个左转、10 个右转；记录的转角绝对值在 0.05–0.20 rad。此轮是转向诊断，不代表全部导航运动。
- 原动作、零位移、只翻转 yaw 符号、只把 yaw 幅度乘 1.2。最后两者均保留 dx/dy；yaw 翻转不是完整 SE(2) 求逆。
- 每个起点的四种动作编码互异，且所有 action token 都存在于基线 tokenizer；反事实是否物理可执行仍未知。
- 沿用原生 5Hz、`delta[k]` 对 `frame[k+1]`。原始帧索引相差 6；传感器与动作的物理时间对齐仍须校准。
- 固定起点图、目标图与当前图。起点/目标先缩至 256；当前图经过原探针的 VQ 重建路径。目标图属于模型原生条件，可能透露未来场景；所有动作共享它。真实下一帧只用于参考，不作为下一步预测的输入。
- 沿用 FP16、末层 MLP FP32、VQ FP32、SDPA、batch 1。A30 虽支持 BF16，本轮不同时改变推理精度。

## 样本与边界

`manifest.json` 已冻结，SHA-256：
`379e048d7b66d149c5db0e3f37a720e0f0ad40e60a7b1a9e5722be838b3ce05b`。

从本地 510 条轨迹的排序、seed 42 的 shuffle 和前 200 条训练配置重建基线训练来源：154 个 `(shard, segment)` 分组。剩余 195 条轨迹来自 128 个未重叠分组；pilot 取其中 20 组，每组一个起点。选择仅依赖元数据、固定随机数和编码有效性，不依赖光流或生成表现。

这是依据代码与训练记录重建的来源划分，不是训练时保存的 manifest；不能保证房间、人物或全局预训练数据没有重叠。正式确认集须避开这 20 个开发来源。其他 checkpoint 的训练集合可能不同，不能直接宣称该清单对所有模型都干净。

**2026-10-21 加载审计更正：** 旧基线没有独立 `action_embed.safetensors`，但已在 `adapter_model.safetensors` 内保存完整输入 embedding，PEFT 会恢复。原 pilot 记录的 1,554 行动作输入哈希与保存的训练 checkpoint 完全一致。将初始化 seed 改为 17 后，全输入表及实际输出头仍逐元素匹配 checkpoint，动作映射也通过。因此原先“加载时重新随机初始化”的说法不成立；原始运行记录保留，错误说明由 审计记录 取代。

## 文件

| 文件 | 用途 |
|---|---|
| `build_manifest.py` | 从原仓库生成可复现清单；已存在的清单拒绝覆盖 |
| `manifest.json` | 20 个起点、源分组、动作、seed、原图哈希、保留来源 |
| `run_pilot.py` | 调用原仓库 loader/prompt，生成 240 张；支持显式、配置一致的续跑 |
| `analyze_pilot.py` | LPIPS 采样对照、RAFT 已知位移检查及真实/生成运动诊断 |
| `checkpoint_manifest.json` | 本地基线全部文件的尺寸与 SHA-256，传输完成后逐项比对 |
| `execute_when_ready.py` | 一次性队列：限时等待准备完成、校验、4 张技术冒烟、续跑与分析 |

每个生成单元保存独立 JSON 与 PNG。首次生成失败保留在分母，不自动换 seed 重试；OOM 等运行错误直接退出。输出目录存在时默认停止；`--resume` 会核对代码、清单和权重哈希。进程文件锁防止重复写入。

## 分析口径

主对比为 Original 与 yaw flip。分别报告：同动作不同 seed、跨动作独立 seed、跨动作同 seed 的 LPIPS。逐起点效应为前两者之差；20 个来源的 bootstrap 区间仅用于开发诊断，不是确认性检验。

RAFT Large 在原始/VQ 当前帧上检查 0、1、2、4、8、16、32、−8 px 的已知水平位移，256×256 图像，使用中央区域避免合成边界。真实帧同时检查 raw/raw、VQ/raw、VQ/VQ。估计光流不称为真实物理光流；未经时序与几何校准不报告 yaw 方向准确率，不计算缺少真实反事实未来的 effect error，也不因 LPIPS 区间含零就宣称模型忽略动作。

## 服务器与执行

在有权限的 GPU 环境运行。下面命令以独立工作目录 `external/uniwm-action-pilot` 为例。

首次检查：A30 24GB，约 21GB 空闲；已有语音服务占用约 2.7GB。`/data` 可用约 51GB（准备前），内存充足。安装与迁移都在独立目录，现有服务继续运行。完整推理能否与服务并存，须由冒烟时实际峰值确认。

基座固定为 `leloy/Anole-7b-v0.1-hf` 的 revision `96df52301e844d8a624a13953051ead4c008343b`；Transformers 使用原项目要求的 `leloykun/transformers` commit `dae439cb93e7c2ef61943554b7a2f4590c2118cd`。基线来自本地 `output/ckpts/1x200`。

A30 隔离环境已通过 UniWM、LPIPS、RAFT 的导入检查：PyTorch `2.9.1+cu128`、Transformers fork `4.45.0.dev0`、PEFT `0.12.0`。PyTorch 与原 V100 环境不同，后续须记录生成是否正常；导入成功不等于 GPU 推理或测量已经验证。

从 Drive 取回的精简权重可直接用于 `run_pilot.py`。`execute_when_ready.py` 是原始传输验证器，另需训练日志与状态文件。

运行命令（服务器）：

```bash
cd external/uniwm-action-pilot
venv/bin/python -u pilot/run_pilot.py \
  --repo repo --data data/1x_nav --checkpoint checkpoints/1x200 \
  --base-model models/anole --manifest pilot/manifest.json \
  --output runs/uniwm_1x_dev20_v1 --max-images 4
# 检查首个状态的四张图、seed、token、数值与显存后，续跑剩余 236 张：
venv/bin/python -u pilot/run_pilot.py \
  --repo repo --data data/1x_nav --checkpoint checkpoints/1x200 \
  --base-model models/anole --manifest pilot/manifest.json \
  --output runs/uniwm_1x_dev20_v1 --resume
venv/bin/python -u pilot/analyze_pilot.py --run runs/uniwm_1x_dev20_v1
```

2026-10-21 更新：权重已传齐并逐项校验。首个起点的四种动作均已生成 448×448 图像，共享同一 seed、使用不同动作文本；每张耗时 40.6–41.1 秒，峰值分配约 17.6 GiB、峰值保留约 19.0 GiB。三模型短训练验收期间暂停了后续调度，验收结束后已恢复队列（PID `984852`），续跑余下 236 张。训练验收使用独立输出，pilot 仍使用原始 `1x200`。实际状态以服务器日志为准。

队列先验证首个状态的四张图已完整解析为 448×448、动作不同且 seed 一致，再续跑 236 张。这个门槛只确认技术可运行；图像质量、测量分辨力和科学结论仍须人工复核。实际阶段以服务器 `pipeline_status.json` 为准，日志为 `logs/pipeline.log`、`logs/inference.log`、`logs/analysis.log`。权重传输与安装时间不包含在原先的 3–7 小时生成预算中。


## 2026-10-21 完成状态

服务器于 14:34（北京时间）完成生成和第一轮分析，240/240 张成功。`analysis_summary.json` 已取回；动作对比区间仍是开发集探索性结果，不能据此断言模型完全忽略动作。后续任务转由 `navigation-inference.service` 管理并优先让路语音服务。
