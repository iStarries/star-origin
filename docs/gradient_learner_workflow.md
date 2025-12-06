# 梯度学习器（可信前景）完整工作流

本文档梳理如何在现有 CISS 训练脚本中启用并运行“只在可信前景像素上训练的梯度学习器”方案，涵盖配置、训练循环以及跨增量步骤的使用方式，便于命令行直接复现。

## 1. 前置配置
- 基础超参位于 `configs/config_voc.json` / `configs/config_ade.json`，梯度学习器开关及超参在 `hyperparameter.grad_learner` 下（默认关闭）。
- 常用 CLI 参数（`train_voc.py` / `train_ade.py` 共享）：
  - `--grad`：启用梯度学习器。
  - `--grad_hidden` / `--grad_layers`：MLP 隐藏维度与层数。
  - `--grad_samples`：每个 batch 随机采样的可信前景像素数。
  - `--grad_alpha` / `--grad_eta`：预测梯度缩放与模拟步长。
  - `--grad_lambda` / `--grad_eps`：fitness loss 权重与归一化平滑项。
  - `--grad_warmup`：仅训练主模型的 warmup epoch 数。
  - `--grad_lr`：梯度学习器的学习率。

## 2. 训练启动示例
以 VOC 增量训练为例，基于默认配置启用梯度学习器：
```bash
python train_voc.py -c configs/config_voc.json --task_name voc --task_step 1 --grad \
  --grad_samples 512 --grad_hidden 64 --grad_layers 2 --grad_alpha 0.5 --grad_eta 1.0 \
  --grad_lambda 1.0 --grad_eps 1e-6 --grad_lr 1e-3 --grad_warmup 0
```
如需 ADE，命令与参数相同，替换为 `train_ade.py -c configs/config_ade.json`。

## 3. 模型与优化器初始化
- 主模型 `f_θ` 按原有流程构建与分布式封装。
- 当 `--grad` 打开时，会构建一个输入/输出维度等于当前类别数的轻量级 MLP 梯度学习器 `h_ω`，支持 DataParallel/DDP；其优化器（Adam）独立于主模型，学习率可单独设定。

## 4. 每个 mini-batch 的训练流程
1. **主模型更新（常规损失）**
   - 前向得到 logits，计算既有的 CE/KD/正则等 `L_main`，反向更新 `θ`（与原基线一致）。
   - 若设置 warmup，当 epoch < `grad_warmup` 时跳过后续梯度学习器步骤。
2. **可信前景采样**
   - 构建 mask：`label != background` 且 `label != ignore`（255），覆盖当前步新类前景 + replay 旧类前景。
   - 随机采样至多 `grad_samples` 个像素，得到 `z_fg`（对应 logits）与 `y_fg`（类别索引去背景偏移）。
3. **真实梯度计算**
   - 在采样像素上计算前景交叉熵 `L_fg`，仅对 `z_fg` 求梯度 `g_true = ∂L_fg/∂z_fg`，不更新主模型。
   - 记录模长 `τ = ||g_true||_2` 供缩放使用。
4. **梯度预测与 fitness loss**
   - 用梯度学习器预测 `g_hat = h_ω(detach(z_fg))`，归一化并按 `α·τ` 缩放得到 `g_tilde`。
   - 模拟一步 logits 更新：`z_prime = detach(z_fg) - η·g_tilde`，在 `z_prime` 上计算交叉熵得到 `L_fit = λ_fit · CE(z_prime, y_fg)`。
   - 仅对 `ω` 反向：`optimizer_gl.step()`，主模型不受影响。

## 5. 指标与日志
- `loss_grad` 记录 `L_fit` 以便在日志中监控梯度学习器训练情况。
- 其他主模型指标与原有实现一致。

## 6. 跨增量步骤策略
- **Base step**：标签完整，可充分训练梯度学习器获得稳定初始 `ω`。
- **后续 step**：继承上一阶段的梯度学习器权重与优化器状态（由 checkpoint 自动恢复），在每个 batch 继续按“可信前景 + L_fit”微调；可通过 `--grad_warmup` 设定前若干 epoch 只训练主模型。

## 7. Checkpoint 与恢复
- 当启用梯度学习器时，保存/恢复包括 `ω` 参数与 `optimizer_gl` 状态，确保增量步骤间梯度学习器连贯。
- 继续训练或测试时，使用 `--resume <ckpt>` 自动加载对应组件。

## 8. 注意事项
- 梯度学习器只在可信前景像素上训练，不使用“不可信背景”；主模型其他损失对所有像素仍按原基线执行。
- 样本中若无前景（mask 为空）则本 batch 跳过梯度学习器更新，保持主模型训练正常进行。
- 仍可叠加其他可选模块（如 phase replay、pseudo gradient 对齐），开关与原配置兼容。
