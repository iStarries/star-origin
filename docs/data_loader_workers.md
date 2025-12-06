# 数据加载并行度使用指南

针对 `train_voc.py` 与 `train_ade.py` 支持的 `--workers` / `--num_workers` 覆盖项，8 大核 + 12 小核的 CPU 环境下推荐以下设置：

- **推荐值**：`--workers 12`（或 `--num_workers 12`），相当于将 DataLoader 的 `num_workers` 设为 12，通常能在不压满 CPU 的情况下加速数据读取。
- **更激进**：若磁盘 IO 充足且内存允许，可尝试 `--workers 16`，但应观察 CPU 占用与数据加载稳定性。
- **保守模式**：如同时跑多卡或需留资源给其他进程，可降到 `--workers 8`。

说明：
- 配置文件现在支持 `data_loader.args.num_workers_override`，可直接在 `configs/config_voc.json` / `configs/config_ade.json` 中写死默认并行度（默认填入 12），这样就无需在命令行反复加 `--workers`。若仍提供 CLI 覆盖项，则以 CLI 为准。
- 覆盖后的并行度在内部会自动裁剪到不超过实际 CPU 核心数。
- 建议在不同任务设置（如 VOC / ADE、基步 / 增量步）保持同一并行度，便于对齐训练耗时与吞吐。

示例命令：
```bash
python train_ade.py -c configs/config_ade.json \
  -d 0 --save_dir saved_ade --name exp_workers12 \
  --task_name 50-50 --task_setting overlap --task_step 0 \
  --lr 0.0025 --bs 12 --workers 12
```
