# Skill Annealing

> **A controlled study of how LLM post-training robustness changes when an Agent skill prompt is incomplete or absent.**

研究业务 Agent 在 Skill Prompt 不完整或缺失时的鲁棒性，并比较不同 Skill Exposure 训练顺序和混合方式。

这是一个可运行的研究镜像，不是一个已被证明有效的新算法。核心结论是：在当前合成任务和训练预算下，**Random-Mix 是更强、更稳定的 baseline；Naive Staged 没有显示出跨任务普适优势。**

![Stage C retention curve](figures/skill-retention-stage-c.png)

## 为什么做这个问题

业务 Agent 常依赖长 Skill Prompt 来说明规则、输出 schema 和边界条件。但实际运行中 Prompt 可能被压缩、部分缺失，甚至因为上下文预算不可用。这里不研究路由或完整生产 Agent，而是隔离一个 skill executor 问题：同一 case 在 Full、Partial、Minimal、No-Skill 输入下是否仍能产出正确、可解析的结构化决策？

## 数据与任务

- `refund_decision` 是合成的电商售后决策任务；gold 完全由确定性规则引擎生成。
- 每个 case 成对渲染为 Full / Partial / Minimal / No-Skill 四档 prompt。
- 模型输出包含 `decision`、`risk_level`、`need_human`、`reason_codes` 和解释字段。
- Structured Evaluator 按 exposure 校验 JSON、schema、decision、risk、人工介入与 reason-code F1。

公开样例是合成数据，不包含真实订单、企业规则、模型权重或原始预测。

## 实验矩阵

| 范围 | 设置 | 目的 |
| --- | --- | --- |
| 主 SFT | Qwen3.5-4B + LoRA SFT | 统一底座与参数比较 exposure 策略 |
| Full-only | 仅 Full Skill | 测量完整 prompt 依赖 |
| Random-Mix | 四档等量、seed 打散 | 强 multi-exposure baseline |
| Naive Staged | Full→Partial→Minimal/No-Skill | 测试顺序化 exposure |
| 2×2 控制 | 边际分布 × record order | 区分样本比例和连续训练/recency |
| 三 Skill 六臂 | refund / expense / warranty | 检查跨 Skill 方向一致性 |
| Gradient Audit | assistant-only loss + LoRA gradients | 描述 exposure 的局部梯度关系 |
| GRPO | rule/evaluator reward、100-step controlled scaffold | 链路验证，不是 RL 结论 |

跨 Skill 主要训练臂使用 1,800 条记录。关键 SFT 配置为 seed=42、LoRA r=8/α=16、all-linear、LR=1e-4、batch=1、grad accumulation=8、bf16、3 epochs；详见 [`configs/`](configs)。

## 核心结果与边界

Stage C 强约束评测共 384 条记录（每 exposure 96 条），指标为 decision accuracy：

| 方法 | No-Skill | 四档平均 |
| --- | ---: | ---: |
| Full-only | **0.00%** | — |
| Random-Mix | **90.62%** | **95.31%** |
| Naive Staged | 87.50% | 93.23% |

2×2 控制中，Stage C 的 shuffled−staged 分别为旧边际 **+2.08pp**、uniform 边际 **+7.03pp**；hard-boundary lite 中为 **+9.72pp** 和 **+8.33pp**。这些结果使“只因样本比例不同”不足以解释差异；更谨慎的解释是 naive staged order / recency 在此设置中不利。

三 Skill pilot 的 `Staged − Random` 是 Refund **+8.33pp**、Warranty **+1.82pp**、Expense **−37.76pp**。方向不一致，因此不主张 Naive Staged 普适优于 Random-Mix。

Refund Gradient Audit 有 **2,048** 配对组，跨 Skill 验证有 **1,536** 组。梯度指标对 prompt 形式敏感，未通过跨 Skill 预测门；不主张已经证明梯度机制。

![Mechanism ablation](figures/mechanism-ablation.png)

机制消融显示 Random-Mix 的 fact-mutation accuracy 为 81.67%，Naive Staged 为 73.89%；这支持更低的 prompt-token dependence 与 adapter-carried behavior，而不是“模型已把业务知识完全内化到参数”。全部数字、样本范围和证据文件见 [`docs/results-and-limitations.md`](docs/results-and-limitations.md)。

README 的逐项数字定位见 [`docs/evidence-index.md`](docs/evidence-index.md)。

## 负结果的价值

本项目把“输入暴露顺序”和“连续训练本身”拆开检查：在 2×2 控制里固定初始化、优化器、LR、LR schedule、训练预算、样本 multiset，并在 staged 组继承前一阶段 adapter；仅改变 record order 或 exposure 边际分布。结果没有支持 naive staged 的普适优势，反而把后续假设收窄为需保留 replay 的 schedule，而不是把负结果隐藏掉。

## 快速复现（无需模型）

需要 Python 3.10+。仓库无运行时第三方依赖；完整训练另需本地安装并合法使用 `ms-swift` 与模型。

```bash
python scripts/generate_mvp_data.py --sample-count 12 --output-dir data/examples/refund_decision_smoke
python scripts/export_ms_swift_data.py --source-dir data/examples/refund_decision_smoke --output-dir data/examples/ms_swift/refund_decision_smoke
python -m skill_annealing.refund_decision.evaluator --predictions data/examples/refund_decision_smoke/eval_prompts.jsonl --by-exposure --oracle
python scripts/smoke_test.py
pytest -q
```

构造公开的三种训练序列：

```bash
python scripts/build_exposure_mix.py --method full_only --records 48 --output data/examples/full_only.jsonl
python scripts/build_exposure_mix.py --method random_mix --records 48 --output data/examples/random_mix.jsonl
python scripts/build_exposure_mix.py --method naive_staged --records 48 --output data/examples/naive_staged.jsonl
```

`smoke_test.py` 只验证数据构造、导出和 oracle evaluator，不下载模型，也不声称完成了微调。完整 LoRA SFT 可在获得模型后执行：

```bash
MODEL_PATH=/path/to/your/Qwen3.5-4B DATASET=data/examples/ms_swift/refund_decision_smoke/annealed_sft.jsonl bash scripts/run_sft.sh
```

tiny / smoke 配置是 8–48 条合成记录；它可在 CPU 上完成数据和评测验证。完整 1,800-record LoRA SFT 需要 CUDA GPU 与 bf16 支持；耗时和显存取决于模型版本、序列长度和 `ms-swift` 版本，仓库不伪造通用成本数字。

## GRPO

[`scripts/refund_grpo_reward_plugin.py`](scripts/refund_grpo_reward_plugin.py) 和 [`configs/grpo-controlled-smoke.json`](configs/grpo-controlled-smoke.json) 保留 evaluator/rule reward 与 100-step controlled-run 参数。它们只表明曾验证 reward 接线和短程控制链路；缺少多 seed、长训练和 reward ablation，因此不能被解释成“GRPO 已经提升方法”。

## 仓库结构

```text
skill_annealing/    deterministic rule engine, prompt builder, evaluator, cross-skill utilities
configs/            LoRA SFT / GRPO public configs
scripts/            data builders, SFT entry point, gradient and result utilities
data/examples/      synthetic Full / Partial / Minimal / No-Skill samples
results/            redacted aggregate tables with provenance
figures/            experiment-design evidence figures
docs/               protocol, results, and limitations
tests/              focused regression tests
```

## 发布与许可证说明

运行 `python scripts/scan_public_release.py` 可执行保守的 key、私有路径和大文件扫描。模型权重、真实业务数据、服务器日志、checkpoint、optimizer state、tokens 和未脱敏预测均不在仓库。

暂未附加 LICENSE：虽然本镜像不分发任何第三方权重或数据，仍需要作者逐文件确认著作权与外部依赖许可。详见 [`NOTICE.md`](NOTICE.md)。
