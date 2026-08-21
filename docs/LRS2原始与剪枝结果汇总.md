# SepPrune：LRS2 原始模型与剪枝结果汇总

更新时间：2026-08-13（UTC，23:40 二次更新）

> 2026-08-13 口径更新：后续以 **SepPrune 预算约束复现版（budget-aligned structured pruning）** 作为主结果，不再运行参考代码的严格复现分支。该版本沿用论文剪枝位置和论文报告的目标参数规模，但使用确定性全局参数预算投影并完整继承存活参数；因此属于可重复性增强复现，而非作者源码逐行复现。

## 1. 结果口径

- 数据集：LRS2，2-speaker speech separation。
- 指标：SDRi / SI-SDRi，单位均为 dB，数值越高越好。
- 本地正式测试结果均来自完整的 3,000 条测试集；只有剪枝后、微调前的 A-FRCNN-12 和 SuDoRM-RF 是 100 条样本的诊断结果，不能与完整测试结果直接比较。
- “1 epoch”表示剪枝后继承保留权重并微调一个 epoch，不是最终收敛结果。
- 论文数据取自 SepPrune 论文中的 LRS2 主结果和 1-epoch 快速恢复实验。
- 本地剪枝最初是在作者公开仓库缺少 `ChannelMask1D` 和物理剪枝模型文件的情况下完成的独立重建。2026-08-13 新取得的参考代码补齐了 A-FRCNN-12、SuDoRM-RF 的 mask/物理剪枝链路；源码审计确认剪枝位置一致，但 mask 采样、参数预算和权重继承方式不同，详见 `docs/参考代码与本地剪枝实现对照.md`。

## 2. 原始模型结果

| 模型 | 参数量：论文 / 本地 | 论文 SDRi | 本地 SDRi | 差值 | 论文 SI-SDRi | 本地 SI-SDRi | 差值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TDANet | 2.35M / 2,353,479 | 12.740 | **13.723** | +0.983 | 12.450 | **13.438** | +0.988 |
| A-FRCNN-12 | 5.13M / 5,127,688 | 10.900 | **11.843** | +0.943 | 10.500 | **11.487** | +0.987 |
| SuDoRM-RF | 2.72M / 2,720,417 | **11.430** | 10.541 | -0.889 | **11.100** | 10.164 | -0.936 |

结论：TDANet 和 A-FRCNN-12 的本地原始结果分别高于论文约 1 dB；SuDoRM-RF 仍比论文低约 0.9 dB。因此评估剪枝损失时，应优先相对各自的本地原始基线比较，不能只和论文基线比较。

本地原始模型结果文件：

- `experiments/original_baselines_lrs2/tdanet_tt_summary.json`
- `experiments/original_baselines_lrs2/afrcnn12_tt_summary.json`
- `experiments/original_baselines_lrs2/sudormrf_tt_summary.json`

## 3. 剪枝后的模型规模

| 模型 | 论文原始参数 | 论文剪枝参数 | 论文压缩率 | 本地原始参数 | 本地剪枝参数 | 本地压缩率 | 与论文剪枝目标差异 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TDANet | 2.35M | 1.92M | 18.30% | 2,353,479 | 2,030,463 | 13.73% | +0.110M |
| A-FRCNN-12 | 5.13M | 3.06M | 40.35% | 5,127,688 | 3,059,372 | 40.34% | -628 参数 |
| SuDoRM-RF | 2.72M | 1.54M | 43.38% | 2,720,417 | 1,539,895 | 43.39% | -105 参数 |

论文报告的 MAC：

| 模型 | 原始 MAC | SepPrune MAC | 降幅 |
|---|---:|---:|---:|
| TDANet | 4.77G | 4.33G | 9.22% |
| A-FRCNN-12 | 28.58G | 16.52G | 42.20% |
| SuDoRM-RF | 4.65G | 2.91G | 37.42% |

A-FRCNN-12 和 SuDoRM-RF 的本地参数量已经几乎精确对齐论文目标。TDANet 是较早完成的非预算约束版本，剪枝强度比论文小，后续若做最终对齐实验需要重新进行预算约束的 mask 搜索。

## 4. 剪枝后 1 epoch 快速恢复

| 模型 | 论文 1ep SDRi / SI-SDRi | 本地 1ep SDRi / SI-SDRi | 本地相对论文 | 本地相对本地原始 | 本地恢复率（SDRi / SI-SDRi） |
|---|---:|---:|---:|---:|---:|
| TDANet | 11.170 / 10.810 | **12.956 / 12.625** | +1.786 / +1.815 | -0.766 / -0.812 | 94.41% / 93.95% |
| A-FRCNN-12 | 9.430 / 8.940 | **10.933 / 10.522** | +1.503 / +1.582 | -0.910 / -0.965 | 92.32% / 91.60% |
| SuDoRM-RF | 5.180 / 4.060 | **7.117 / 6.330** | +1.937 / +2.270 | -3.423 / -3.834 | 67.52% / 62.28% |

三种模型的本地 1-epoch 结果都高于论文的 1-epoch 数值。不过这并不表示最终复现已经超过论文，因为原始基线、剪枝结构和训练实现并不完全相同，而且 SuDoRM-RF 相对本地原始模型仍损失较大。

本地 1-epoch 完整测试文件：

- `experiments/tdanet_lrs2_stage2_eval/pruned_inherited_1epoch_tt_summary.json`
- `experiments/seprune_budgeted_afrcnn12_lrs2_eval/afrcnn12_budgeted_1epoch_tt_summary.json`
- `experiments/seprune_budgeted_sudormrf_lrs2_eval/sudormrf_budgeted_1epoch_tt_summary.json`

## 5. 论文最终剪枝结果与本地状态

> **状态说明（2026-08-13 23:40 UTC）**：项目重心已转向数据剪枝研究（`docs/数据剪枝与模型剪枝协同方案调研.md`）。本节的复现缺口**不阻塞**该研究——它的四个冻结输入（dense `best.pt`、`masks.pt`、E2 阈值、`epoch1.pt`）都已最终确定，且都不来自长程微调分支。本节的收尾工作**推迟到数据剪枝阶段 A 之后**再做，第 5.1 节记录了续做时需要的全部信息。

| 模型 | 论文最终 SDRi / SI-SDRi | 论文剪枝前后变化 | 本地最终完整测试 | 当前本地状态 |
|---|---:|---:|---:|---|
| TDANet | 12.720 / 12.410 | -0.020 / -0.040 | 待完成 | 已有非预算约束剪枝和 1ep 结果，尚未运行对齐论文参数预算的最终收敛实验 |
| A-FRCNN-12 | 12.590 / 12.250 | +1.690 / +1.750 | **11.865 / 11.497**（ep100 `best.pt`，完整 3,000 条） | ✅ **已收敛定稿**：ep130 早停，`stop_reason=early_stopped`。相对论文 −0.725 / −0.753；**相对本地 dense +0.022 / +0.010**，即剪枝无损，见 §5.3 |
| SuDoRM-RF | 10.370 / 9.980 | -1.060 / -1.120 | **9.791 / 9.336**（ep206 `best.pt`，完整 3,000 条） | ✅ **已收敛定稿**：ep236 早停。相对论文 −0.579 / −0.644；**相对本地 dense −0.750 / −0.828**，即剪枝有损，方向与论文一致且损失更小，见 §5.4 |

A-FRCNN-12 的 11.344 只测了 SI-SDRi（批量实现，关闭 TF32，约 62 秒）；SDRi 需要 `fast_bss_eval`，留到最终 checkpoint 选定后与 SDRi 一并出。验证 SI-SDR 与测试集 SI-SDRi 不是同一指标，本地标定为 **测试 SI-SDRi ≈ 验证 SI-SDR + 0.31 ~ +0.40 dB**（两个标定点：dense 11.086 → 11.487，pruned@1ep 10.215 → 10.522）。

### 5.3 A-FRCNN-12 定稿结论：剪枝无损，但复现不出论文的「剪枝提升」

长程微调于 2026-08-16 早停结束（epoch 130，`stop_reason=early_stopped`，best 在 epoch 100，验证 SI-SDR 11.092）。最终 `best.pt` 在完整 3,000 条测试集上：

| | 参数量 | SDRi | SI-SDRi |
|---|---:|---:|---:|
| 本地 dense 未剪枝 | 5,127,688 | 11.843 | 11.487 |
| **本地 SepPrune 剪枝收敛** | **3,059,372** | **11.865** | **11.497** |
| 差 | −40.3% | **+0.022** | **+0.010** |

**在本地，把 A-FRCNN-12 从 5.13M 剪到 3.06M 再微调至收敛，性能与 dense 基线完全持平（+0.01 dB）——剪枝是无损的。** 这一条复现成功。

但论文声称的是剪枝后**提升 +1.75 dB**（10.500 → 12.250），我们得到的是 +0.010 dB。两边的对照关系是：

| | 论文 | 本地 | 本地−论文 |
|---|---:|---:|---:|
| dense 未剪枝 SI-SDRi | 10.500 | 11.487 | **+0.987** |
| 剪枝后 SI-SDRi | 12.250 | 11.497 | −0.753 |
| 剪枝带来的变化 | **+1.750** | **+0.010** | |

**最可能的解释：论文的 dense 基线训练不足。** 论文 dense 比本地低约 1 dB，而剪枝+微调阶段本身就是一大轮额外训练；一个欠训练的 dense 经过这一轮自然会被超过。本地 dense 已按论文报告的 136 epoch 训到 11.487（接近该架构在此数据上的天花板），剪枝模型再微调也只是回到同一天花板。

这个解释与证据一致但**未经直接验证**——严格验证需要先把 dense 训到论文的 10.50 再剪枝，看是否复现 +1.75。若后续要补，这是最小充分实验。在此之前，只能表述为「本地条件下剪枝无损；论文的提升幅度未能复现，最可能来自基线差异」，不能断言论文有误。

结果文件：`experiments/seprune_budgeted_afrcnn12_lrs2_eval/afrcnn12_budgeted_converged_tt_summary.json`

### 5.1 训练过程记录：学习率长期未衰减（已于 ep99 自行触发）

`train_pruned_original.py` 的 `ReduceLROnPlateau` 使用 PyTorch 默认的 `threshold=1e-4, threshold_mode='rel'`。在 `val_loss ≈ −10.9` 时，相对阈值只要求改进超过 `10.9 × 1e-4 ≈ 0.0011 dB`。A-FRCNN 前 98 个 epoch 靠 +0.017 / +0.021 dB 的噪声级刷新反复把 patience 清零，形成自锁：**学习率迟迟不降**。它最终在 **ep99 自行触发**（ep83→ep100 之间出现 16 个 epoch 的空档，够 patience 15 用），best 随即从 11.009 涨到 **11.092（+0.083 dB）**，随后于 ep130 早停。方向上印证了学习率假设，但幅度远小于 SuDoRM-RF 的 +1.123 dB，**因此学习率并不是与论文差距的主要成因**（真正的成因见 §5.3）（epoch-to-epoch 验证波动 sd ≈ 0.09 dB，远大于这些"提升"）。

用真实验证序列回放（patience=15, factor=0.5）：

| 判据配置 | 会在哪些 epoch 降 LR |
|---|---|
| 当前（PyTorch 默认 rel 1e-4） | **从未** |
| abs 0.05 dB | ep22 → 5e-4，ep43 → 2.5e-4，ep59 → 1.25e-4 |
| abs 0.02 dB | ep22 → 5e-4，ep56 → 2.5e-4 |

回放只在 ep22 之前严格有效——之后的序列是在恒定 LR 下产生的，属反事实。

**同一套代码上的对照证据**：SuDoRM-RF 在 ep32 降过一次 LR（1e-3 → 5e-4），best 从 7.466 涨到 8.589，**+1.123 dB**。而 A-FRCNN 自 ep6（10.834）起 66 个 epoch 只涨 0.124 dB。

### 5.2 续做时的操作方式（代码已就绪）

`train_pruned_original.py` 已新增两个**可选**参数，默认行为与此前所有 run 完全一致：

- `--override-learning-rate`：在 `--resume` **之后**强制学习率。这是必需的——`optimizer.load_state_dict()` 会恢复 checkpoint 里的 LR 并**静默覆盖** `--learning-rate`，直接传 `--learning-rate 5e-4` 不会生效。
- `--plateau-threshold`：以 dB 为单位的绝对改进阈值，同时作用于 scheduler 和早停计数。`best.pt` 的保存判据仍保持严格，保证存下来的是真正最好的权重。

建议的续做命令与验收方式：

```bash
.venv/bin/python -m reproduction.train_pruned_original --model afrcnn12 \
  --baseline-checkpoint experiments/original_afrcnn12_lrs2_train/best.pt \
  --masks experiments/seprune_budgeted_afrcnn12_lrs2_e07_seed2026/masks.pt \
  --resume experiments/seprune_budgeted_afrcnn12_lrs2_finetune/last.pt \
  --override-learning-rate 5e-4 --plateau-threshold 0.05 \
  --output-dir experiments/seprune_budgeted_afrcnn12_lrs2_finetune
```

阈值取 0.05 dB 的依据：验证波动 sd ≈ 0.09 dB，而噪声级刷新是 +0.017 / +0.021——0.05 卡在中间，真提升过得去、噪声过不去。

**建议先跑 20 个 epoch（约 20 h）就复测一次完整测试集**：相对 11.344 若涨 ≥0.3 dB 说明路子对，继续跑到早停；若基本持平，就接受 11.344 定稿，并在论文对照中如实说明差距。不要无限期等下去。

SuDoRM-RF 目前仍在正常收敛（LR 已降过一次，无提升计数 12/30），可让它自然早停；但因为 ep90 出现过 8.59 → 0.94 的塌陷，最终 checkpoint **只能取 best，绝不能取 last**，且应对 best 及其前后若干 checkpoint 一并做完整测试，确认 best 不是一次幸运波动。

### 5.4 SuDoRM-RF 定稿结论：剪枝有损，方向与论文一致

长程微调于 2026-08-17 早停结束（epoch 236，`stop_reason=early_stopped`，best 在 epoch 206，验证 SI-SDR 9.181，学习率已从 1e-3 逐级降到 1.5625e-05）。最终 `best.pt` 完整 3,000 条测试集：

| | 参数量 | SDRi | SI-SDRi |
|---|---:|---:|---:|
| 本地 dense 未剪枝 | 2,720,417 | 10.541 | 10.164 |
| **本地 SepPrune 剪枝收敛** | **1,539,895** | **9.791** | **9.336** |
| 差 | −43.4% | **−0.750** | **−0.828** |

与论文的对照：

| | 论文 | 本地 |
|---|---:|---:|
| dense 未剪枝 SI-SDRi | 11.100 | 10.164 |
| 剪枝后 SI-SDRi | 9.980 | 9.336 |
| 剪枝带来的变化 | **−1.120** | **−0.828** |

**SuDoRM-RF 的复现是成功的**：论文报告剪枝有损，本地同样有损，且本地损失（0.83 dB）小于论文（1.12 dB）。绝对值偏低约 0.64 dB，与本地 dense 基线本就比论文低 0.94 dB 一致——两级差距量级相当，说明差异来自基线而非剪枝实现。

训练过程极不稳定，全程出现多次验证塌陷（ep78 7.90→6.72，ep90 8.59→**0.94**，ep101→3.39）。因此最终 checkpoint 严格取 best 而非 last；这一点在本模型上不是形式要求。

结果文件：`experiments/seprune_budgeted_sudormrf_lrs2_eval/sudormrf_budgeted_converged_tt_summary.json`

## 6. 剪枝后、微调前的诊断结果

| 模型 | 测试规模 | SDRi | SI-SDRi | 说明 |
|---|---:|---:|---:|---|
| TDANet | 3,000 | 13.208 | 12.873 | 完整测试；比本地原始模型下降 0.515 / 0.564 dB |
| A-FRCNN-12 | 100 | 7.447 | 6.823 | 仅作权重映射与前向正确性诊断 |
| SuDoRM-RF | 100 | 3.646 | 2.449 | 仅作权重映射与前向正确性诊断 |

TDANet 另有随机初始化的 1-epoch 对照实验：3.842 / 1.973 dB，明显低于继承权重的 12.956 / 12.625 dB，说明物理剪枝后的权重继承确实生效。

## 7. 当前可下的结论

1. 三个原始模型均已完成本地完整测试；TDANet、A-FRCNN-12 对齐良好并超过论文，SuDoRM-RF 原始基线仍有约 0.9 dB 差距。
2. A-FRCNN-12 和 SuDoRM-RF 的剪枝参数预算已经与论文精确对齐；TDANet 当前版本尚未对齐论文的 1.92M 参数目标。
3. 三个模型的 1-epoch 快速恢复结果均已完成，并且均高于论文报告的相同阶段结果。
4. **论文关于 A-FRCNN-12 的核心声称（剪枝后比未剪枝提升 +1.75 dB）目前没有复现出来，而且是方向性的不一致**：本地剪枝模型当前为 11.344 dB SI-SDRi，比本地 dense 的 11.487 低 0.143 dB，比论文的 12.250 低 0.906 dB。原因已定位到学习率从未衰减（§5.1），修复手段已就绪但**尚未验证是否足以补上这 0.906 dB**——在跑完 §5.2 的续做实验之前，不能声称差距的成因已经确认。
5. 论文"最终收敛"的剪枝结果：A-FRCNN-12 有一个未收敛的中间值（11.344）；SuDoRM-RF 尚未完成 checkpoint 选择和完整测试；TDANet 尚未启动预算对齐后的长程微调。当前 A/S 长程训练属于"精确参数预算 + 完整权重继承"分支，不是新参考代码的逐行复现。
6. SuDoRM-RF 的长程训练稳定性仍是最需要关注的问题，且**已经恶化**：ep78 为 7.90 → 6.72，ep90 为 8.59 → **0.94** → 8.19，ep101 再次跌到 3.39。最终 checkpoint 只能取 best，并须对 best 及其邻近 checkpoint 做一致的完整测试。
7. **项目重心已转向数据剪枝研究**，本文件第 5 节的收尾推迟执行。这不影响数据剪枝：其冻结输入全部已最终确定，且实测表明 GPU 可并发（batch=1 训练是 kernel launch 延迟受限，第二个任务只让在跑的任务慢 3.7%），两条线可以并行推进。

## 8. 主要结果文件

- TDANet 微调前完整测试：`experiments/tdanet_lrs2_eval_full/pruned_tt_summary.json`
- TDANet 继承权重 1ep：`experiments/tdanet_lrs2_stage2_eval/pruned_inherited_1epoch_tt_summary.json`
- TDANet 随机初始化对照：`experiments/tdanet_lrs2_stage2_eval/pruned_random_1epoch_tt_summary.json`
- A-FRCNN-12 1ep：`experiments/seprune_budgeted_afrcnn12_lrs2_eval/afrcnn12_budgeted_1epoch_tt_summary.json`
- SuDoRM-RF 1ep：`experiments/seprune_budgeted_sudormrf_lrs2_eval/sudormrf_budgeted_1epoch_tt_summary.json`
- A-FRCNN-12 长程训练：`experiments/seprune_budgeted_afrcnn12_lrs2_finetune/`
- SuDoRM-RF 长程训练：`experiments/seprune_budgeted_sudormrf_lrs2_finetune/`

论文：A. Li et al., *SepPrune: Structured Pruning for Efficient Deep Speech Separation*, AAAI 2026。
