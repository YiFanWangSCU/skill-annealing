# README 数字证据索引

本索引指向原研究工作区的只读证据文件；公开仓库只保留脱敏聚合值于 `results/core-results.json`。

| README 数字/主张 | 原始证据文件 | 定位键或段落 |
| --- | --- | --- |
| Full-only No-Skill 0.00%；Random-Mix No-Skill 90.62%、四档均值 95.31%；Naive Staged 四档均值 93.23% | `experiments/stagec_causal_ablation_20260629/summary.json` | `metrics.stage_c.old_*`、`matrix.stage_c` |
| Stage C 384 条、每 exposure 96 条 | `experiments/stagec_causal_ablation_20260629/summary.json` | `metrics.stage_c.*.count` / `by_exposure.*.count` |
| 2×2：+2.08、+7.03、+9.72、+8.33pp | `experiments/stagec_causal_ablation_20260629/summary.json` | `effects.stage_c`、`effects.hard_boundary` |
| fact mutation 81.67% vs 73.89%；No-Skill overall 95.83% vs 91.67% | `experiments/mechanism_ablation_s24_summary_20260622.json` | random_mix / annealed aggregate fields |
| 24 source cases → 960 prompts | `docs/final_report_refund_decision_mechanism_closure_20260623.md` | “S24 Setup” |
| 三 Skill Staged−Random：+8.33、+1.82、−37.76pp | `experiments/multiskill_prompt_robustness_pilot_20260724/frozen_discovery_analysis_20260725.md` | per-skill result table |
| 每个跨 Skill 训练臂 1,800 records；每 Skill 1,024 eval requests | `data/multiskill_prompt_robustness_pilot_20260724/build_summary.json` | `skills.*.train_record_count_per_arm` / `eval_request_count` |
| Refund gradient 2,048 groups | `experiments/paired_exposure_gradient_v0/phase2_core_gradient_20260724_formal_analysis.json` | `record_count` |
| Cross-skill gradient 1,536 groups、未通过预测门 | `experiments/multiskill_prompt_robustness_pilot_20260724/gradient_boundary_v0/g1_formal_gpu0_20260725T130403Z_1d67e050/analysis/g1_formal_analysis.json` | formal aggregate / gate conclusion |
| GRPO ng8 / bs8 / 100 steps、仅 controlled run | `experiments/grpo_control_random_mix_ng8_bs8_100s_pro6000_20260523_summary.json`、`experiments/grpo_control_annealed_ng8_bs8_100s_pro6000_20260523_summary.json` | run configuration and summaries |
| Qwen3.5-4B、LoRA r=8/alpha=16、LR=1e-4、batch=1、accum=8、bf16、3 epochs | `interview_pack_skill_annealing_20260630/09_experiment_config_appendix.md` and original SFT runbooks | SFT config appendix |

说明：该索引不把简历或面试文案当成结果来源；它们只被用于识别应核查的实验，再由配置、结果 JSON、报告和 runbook 交叉验证。
