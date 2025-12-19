"""
评估侧阈值/间隔(margin)扫参脚本（不改训练，仅改变 pred->background 的门控规则）

原始规则：
  prob = sigmoid(logit)
  pred = argmax(prob) + 1
  if max_prob <= 0.5: pred = 0

本脚本支持：
  pred = top1 + 1
  if (max_prob <= tau) OR (max_prob - second_max_prob <= delta): pred = 0

用法示例：

CUDA_VISIBLE_DEVICES=0 python sweep_threshold_voc.py \
  -c /media/wyh/star/saved_voc/models/overlap_15-1_phase-append/step_5_20251218-154510/config.json \
  -r /media/wyh/star/saved_voc/models/overlap_15-1_phase-append/step_5_20251218-154510/checkpoint-epoch60.pth \
  --device 0 --test \
  --taus 0.3,0.35,0.4,0.45,0.5 \
  --deltas 0,0.02,0.05 \
  --focus 2,9,16,18,20 \
  --sort_by harmonic

只扫 tau（保持 delta=0）：
  --taus 0.3,0.35,0.4,0.45,0.5

只测单点：
  --taus 0.45 --deltas 0.02

A) 先只扫 tau（delta 固定 0）
CUDA_VISIBLE_DEVICES=0 python sweep_threshold_voc.py \
  -c /media/wyh/star/saved_voc/models/overlap_15-1_phase-append/step_5_20251218-154510/config.json \
  -r /media/wyh/star/saved_voc/models/overlap_15-1_phase-append/step_5_20251218-154510/checkpoint-epoch60.pth \
  --device 0 --test \
  --taus 0.30,0.35,0.40,0.45,0.50 \
  --deltas 0 \
  --focus 2,9,16,18,20 \
  --sort_by new

B) 再加 delta（抑制“背景误触发”）
CUDA_VISIBLE_DEVICES=0 python sweep_threshold_voc.py \
  -c /media/wyh/star/saved_voc/models/overlap_15-1_phase-append/step_5_20251218-154510/config.json \
  -r /media/wyh/star/saved_voc/models/overlap_15-1_phase-append/step_5_20251218-154510/checkpoint-epoch60.pth \
  --device 0 --test \
  --taus 0.35,0.40,0.45 \
  --deltas 0,0.02,0.05,0.08 \
  --focus 2,9,16,18,20 \
  --sort_by harmonic



"""

import argparse
import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
import torch.multiprocessing as mp

import models.model as module_arch
import utils.metric as module_metric
import data_loader.data_loaders as module_data
from trainer.trainer_voc import Trainer_base
from utils.parse_config import ConfigParser
from logger.logger import Logger


VOC_CLASS_NAMES = [
    "background",
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def parse_float_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip() != ""]


def parse_int_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main(config):
    ngpus_per_node = torch.cuda.device_count()
    if config["multiprocessing_distributed"]:
        config.config["world_size"] = ngpus_per_node * config["world_size"]
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        main_worker(None, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    if config["multiprocessing_distributed"]:
        config.config["rank"] = config["rank"] * ngpus_per_node + gpu
        dist.init_process_group(
            backend=config["dist_backend"],
            init_method=config["dist_url"],
            world_size=config["world_size"],
            rank=config["rank"],
        )
        rank = dist.get_rank()
    else:
        rank = 0
        config.config["rank"] = 0

    # logging
    logger = Logger(config.log_dir, rank=rank)
    logger.set_logger(f"sweep(rank{rank})", verbosity=2)

    # fix seeds
    SEED = config["seed"]
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # dataloader
    config.config['data_loader']['args'].pop('num_workers_override', None)
    dataset = config.init_obj("data_loader", module_data)
    test_loader = dataset.get_test_loader()

    # task labels (与你 eval/diagnose 一致：old 包含背景 0)
    task_step = config["data_loader"]["args"]["task"]["step"]
    old_classes, _ = dataset.get_task_labels(step=0)
    new_classes = []
    for i in range(1, task_step + 1):
        c, _ = dataset.get_task_labels(step=i)
        new_classes += c

    old_idx = list(set(old_classes + [0]))
    new_idx = new_classes
    num_class = dataset.n_classes + 1

    if rank == 0:
        logger.info(f"Old Classes: {old_classes}")
        logger.info(f"New Classes: {new_classes}")
        logger.info(f"old_idx (include bg=0): {sorted(old_idx)}")
        logger.info(f"new_idx: {sorted(new_idx)}")

    # model
    model = config.init_obj("arch", module_arch, **{"classes": dataset.get_per_task_classes()})

    # SyncBN if needed
    if config["multiprocessing_distributed"] and (config["arch"]["args"].get("norm_act", "") == "bn_sync"):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 用 Trainer_base 复用“resume checkpoint / device / DDP”等逻辑
    trainer = Trainer_base(
        model=model,
        optimizer=None,
        evaluator=(None, None),
        config=config,
        task_info=dataset.task_info(),
        data_loader=(None, None, test_loader),
        lr_scheduler=None,
        logger=logger,
        gpu=gpu,
    )
    model = trainer.model
    device = trainer.device

    # sweep configs from injected fields
    taus = config.config.get("taus", [0.5])
    deltas = config.config.get("deltas", [0.0])
    focus = config.config.get("focus", [])
    sort_by = config.config.get("sort_by", "harmonic")

    if len(taus) == 0:
        taus = [0.5]
    if len(deltas) == 0:
        deltas = [0.0]

    # 每个 (tau, delta) 一套 confusion + 计数
    settings = []
    for tau in taus:
        for delta in deltas:
            key = (float(tau), float(delta))
            settings.append(key)

    # torch confusion matrices on device (for easy dist.reduce)
    conf = {k: torch.zeros((num_class, num_class), dtype=torch.float64, device=device) for k in settings}
    total_valid = {k: torch.zeros((), dtype=torch.long, device=device) for k in settings}
    total_pred_bg = {k: torch.zeros((), dtype=torch.long, device=device) for k in settings}
    total_gate_bg = {k: torch.zeros((), dtype=torch.long, device=device) for k in settings}  # 触发门控(<=tau 或 diff<=delta)

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data["image"].to(device, non_blocking=True)
            lab = data["label"].to(device, non_blocking=True)  # [N,H,W]

            valid = (lab >= 0) & (lab < num_class)
            if valid.sum() == 0:
                continue

            logit, _, _ = model(img)
            prob = torch.sigmoid(logit)  # [N,C,H,W]  (C=20 for VOC, no bg channel)

            # top1/top2
            top2_vals, top2_idx = torch.topk(prob, k=2, dim=1)  # vals:[N,2,H,W], idx:[N,2,H,W]
            max_prob = top2_vals[:, 0, :, :]
            second_prob = top2_vals[:, 1, :, :]
            diff = max_prob - second_prob
            pred_top1 = top2_idx[:, 0, :, :] + 1  # [N,H,W], in 1..20

            gt_flat = lab[valid].long()

            for (tau, delta) in settings:
                # 门控：<=tau 或 diff<=delta -> bg
                gate = (max_prob <= tau) | (diff <= delta)
                pred = pred_top1.clone()
                pred[gate] = 0

                pred_flat = pred[valid].long()

                # confusion update: bincount(num_class*gt + pred)
                label = num_class * gt_flat + pred_flat
                binc = torch.bincount(label, minlength=num_class * num_class).to(torch.float64)
                conf[(tau, delta)] += binc.view(num_class, num_class)

                # stats
                v = valid.sum()
                total_valid[(tau, delta)] += v
                total_pred_bg[(tau, delta)] += (pred_flat == 0).sum()
                total_gate_bg[(tau, delta)] += gate[valid].sum()

            if rank == 0 and len(test_loader) > 0:
                period = max(1, len(test_loader) // 5)
                if batch_idx % period == 0:
                    logger.info(f"[{batch_idx}/{len(test_loader)}]")

    # distributed reduce
    if dist.is_available() and dist.is_initialized():
        for k in settings:
            dist.reduce(conf[k], dst=0)
            dist.reduce(total_valid[k], dst=0)
            dist.reduce(total_pred_bg[k], dst=0)
            dist.reduce(total_gate_bg[k], dst=0)

    if rank != 0:
        return

    # compute metrics (exactly reuse module_metric.Evaluator)
    results = []
    for (tau, delta) in settings:
        e = module_metric.Evaluator(num_class, old_idx, new_idx)
        e.confusion_matrix = conf[(tau, delta)].detach().cpu().numpy()
        miou = e.Mean_Intersection_over_Union()  # {'harmonic','old','new','overall','by_class'}

        v = int(total_valid[(tau, delta)].item())
        bg = int(total_pred_bg[(tau, delta)].item())
        gate = int(total_gate_bg[(tau, delta)].item())

        pred_bg_ratio = (bg / v * 100.0) if v > 0 else float("nan")
        gate_ratio = (gate / v * 100.0) if v > 0 else float("nan")

        results.append({
            "tau": tau,
            "delta": delta,
            "pred_bg%": pred_bg_ratio,
            "gate_bg%": gate_ratio,
            "miou_old": float(miou["old"]),
            "miou_new": float(miou["new"]),
            "miou_h": float(miou["harmonic"]),
            "miou_overall": float(miou["overall"]),
            "by_class": miou["by_class"],
        })

    # sort
    key_map = {
        "new": lambda r: r["miou_new"],
        "old": lambda r: r["miou_old"],
        "overall": lambda r: r["miou_overall"],
        "harmonic": lambda r: r["miou_h"],
    }
    sort_fn = key_map.get(sort_by, key_map["harmonic"])
    results.sort(key=sort_fn, reverse=True)

    logger.info("========== SWEEP RESULTS ==========")
    logger.info(f"sort_by = {sort_by}")
    logger.info("tau    delta  pred_bg%  gate_bg%  mIoU_old  mIoU_new  mIoU_harm  mIoU_overall")
    for r in results:
        logger.info(
            f"{r['tau']:<5.2f}  {r['delta']:<5.2f}  "
            f"{r['pred_bg%']:<8.3f} {r['gate_bg%']:<8.3f}  "
            f"{r['miou_old']:<8.2f} {r['miou_new']:<8.2f} {r['miou_h']:<9.2f} {r['miou_overall']:<11.2f}"
        )

    best = results[0]
    logger.info("========== BEST SETTING ==========")
    logger.info(
        f"BEST tau={best['tau']:.2f}, delta={best['delta']:.2f} | "
        f"pred_bg%={best['pred_bg%']:.3f} gate_bg%={best['gate_bg%']:.3f} | "
        f"mIoU_old={best['miou_old']:.2f} mIoU_new={best['miou_new']:.2f} "
        f"mIoU_h={best['miou_h']:.2f} mIoU_overall={best['miou_overall']:.2f}"
    )

    # optional focus class IoU under best setting
    if focus:
        by = best["by_class"]
        names = VOC_CLASS_NAMES if len(VOC_CLASS_NAMES) == num_class else [f"class_{i}" for i in range(num_class)]
        logger.info("========== FOCUS CLASS IoU (BEST) ==========")
        for c in focus:
            if 0 <= c < num_class:
                logger.info(f"[{c:2d}] {names[c]:<12} IoU = {by[c]:.2f}")

    logger.info("========== DONE ==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VOC sweep thresholds (eval-side)")
    parser.add_argument("-c", "--config", default=None, type=str, help="config file path")
    parser.add_argument("-r", "--resume", default=None, type=str, help="checkpoint path")
    parser.add_argument("-d", "--device", default=None, type=str, help="gpu id(s)")

    # sweep options（这些要由我们自己先解析出来）
    parser.add_argument("--taus", type=str, default="0.5", help="comma-separated taus, e.g. 0.3,0.4,0.5")
    parser.add_argument("--deltas", type=str, default="0", help="comma-separated deltas, e.g. 0,0.02,0.05")
    parser.add_argument("--focus", type=str, default="", help="comma-separated class ids to print IoU (best setting)")
    parser.add_argument("--sort_by", type=str, default="harmonic",
                        choices=["harmonic", "new", "old", "overall"])

    # 关键：先解析我们关心的 sweep 参数；其它（比如 --test）先当 unknown 丢掉即可
    sweep_args, _ = parser.parse_known_args()

    CustomArgs = collections.namedtuple("CustomArgs", "flags type action target", defaults=(None, float, None, None))
    options = [
        CustomArgs(["--multiprocessing_distributed"], action="store_true", target="multiprocessing_distributed"),
        CustomArgs(["--dist_url"], type=str, target="dist_url"),
        CustomArgs(["--name"], type=str, target="name"),
        CustomArgs(["--save_dir"], type=str, target="trainer;save_dir"),
        CustomArgs(["--test"], action="store_true", target="test"),
    ]

    # 交给 ConfigParser 再完整解析（包括 --test 等）
    config = ConfigParser.from_args(parser, options)

    # 把 sweep 参数注入 config，确保 mp.spawn 时子进程也能拿到
    config.config["taus"] = parse_float_list(sweep_args.taus) or [0.5]
    config.config["deltas"] = parse_float_list(sweep_args.deltas) or [0.0]
    config.config["focus"] = parse_int_list(sweep_args.focus)
    config.config["sort_by"] = sweep_args.sort_by

    main(config)

