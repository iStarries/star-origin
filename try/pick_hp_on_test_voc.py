"""
在 test 集上直接挑选评估侧超参（tau/delta/gate_mode/temperature），并输出最优设置。

默认预测规则（与你现有 eval/diagnose 一致的 sigmoid+argmax 路线）：
  prob = sigmoid(logit / T)
  top1 = argmax(prob) + 1

背景门控（按 gate_mode）：
  low  = (max_prob <= tau)
  marg = (max_prob - second_prob <= delta)

  gate_mode="or"  : bg = low OR  marg
  gate_mode="and" : bg = low AND marg
  gate_mode="tau" : bg = low
  gate_mode="none": bg = False (永不置背景；通常不建议，只用于对照)

用法示例（直接用 test 挑 harmonic 最优）：
CUDA_VISIBLE_DEVICES=0 python pick_hp_on_test_voc.py \
  -c /path/to/config.json \
  -r /path/to/checkpoint.pth \
  --device 0 --test \
  --taus 0.35,0.40,0.45,0.50 \
  --deltas 0,0.02,0.05,0.08 \
  --gate_modes or,and,tau \
  --temps 1.0 \
  --select harmonic \
  --focus 2,9,16,18,20 \
  --save_best /tmp/best_gate.json

只扫 tau（delta 固定 0）：
  --taus 0.30,0.35,0.40,0.45,0.50 --deltas 0

只测单点：
  --taus 0.45 --deltas 0.08 --gate_modes or --temps 1.0
"""

import argparse
import random
import collections
import json
import numpy as np
import torch
import torch.nn as nn
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


def parse_str_list(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip() != ""]


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


def _gate_mask(max_prob, second_prob, tau: float, delta: float, gate_mode: str):
    """返回 gate mask: True 表示置背景"""
    if gate_mode == "none":
        return torch.zeros_like(max_prob, dtype=torch.bool)
    low = (max_prob <= tau)
    if gate_mode == "tau":
        return low
    marg = ((max_prob - second_prob) <= delta)
    if gate_mode == "or":
        return low | marg
    if gate_mode == "and":
        return low & marg
    raise ValueError(f"Unknown gate_mode: {gate_mode}")


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
    logger.set_logger(f"pick_hp(rank{rank})", verbosity=2)

    # seeds
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

    # task labels (与你 eval_voc.py 一致：old 统计里包含 background=0)
    task_step = config["data_loader"]["args"]["task"]["step"]
    old_classes, _ = dataset.get_task_labels(step=0)
    new_classes = []
    for i in range(1, task_step + 1):
        c, _ = dataset.get_task_labels(step=i)
        new_classes += c

    old_idx = list(set(old_classes + [0]))
    new_idx = new_classes
    num_class = dataset.n_classes + 1  # include bg=0

    if rank == 0:
        logger.info(f"Old Classes: {old_classes}")
        logger.info(f"New Classes: {new_classes}")
        logger.info(f"old_idx (include bg=0): {sorted(old_idx)}")
        logger.info(f"new_idx: {sorted(new_idx)}")

    # model
    model = config.init_obj("arch", module_arch, **{"classes": dataset.get_per_task_classes()})
    if config["multiprocessing_distributed"] and (config["arch"]["args"].get("norm_act", "") == "bn_sync"):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # reuse Trainer_base for resume/ddp/device
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

    # sweep settings
    taus = config.config.get("taus", [0.45])
    deltas = config.config.get("deltas", [0.0])
    gate_modes = config.config.get("gate_modes", ["or"])
    temps = config.config.get("temps", [1.0])
    select = config.config.get("select", "harmonic")  # harmonic/new/old/overall
    focus = config.config.get("focus", [])
    save_best = config.config.get("save_best", "")

    if not taus:
        taus = [0.45]
    if not deltas:
        deltas = [0.0]
    if not gate_modes:
        gate_modes = ["or"]
    if not temps:
        temps = [1.0]

    settings = []
    for T in temps:
        for gm in gate_modes:
            for tau in taus:
                for delta in deltas:
                    settings.append((float(T), str(gm), float(tau), float(delta)))

    # accumulators
    conf = {k: torch.zeros((num_class, num_class), dtype=torch.float64, device=device) for k in settings}
    total_valid = {k: torch.zeros((), dtype=torch.long, device=device) for k in settings}
    total_pred_bg = {k: torch.zeros((), dtype=torch.long, device=device) for k in settings}
    total_gate_bg = {k: torch.zeros((), dtype=torch.long, device=device) for k in settings}

    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            img = data["image"].to(device, non_blocking=True)
            lab = data["label"].to(device, non_blocking=True)  # [N,H,W]
            valid = (lab >= 0) & (lab < num_class)
            if valid.sum() == 0:
                continue

            logit, _, _ = model(img)  # [N,C,H,W] (no bg channel)
            # 先把 GT/pred flatten 所需准备好
            gt_flat = lab[valid].long()
            v = valid.sum()

            # 为了避免重复 topk，按不同 T 分组计算
            # (T, ...) 同一批复用一次 prob/topk
            for T in temps:
                T = float(T)
                prob = torch.sigmoid(logit / T)

                top2_vals, top2_idx = torch.topk(prob, k=2, dim=1)
                max_prob = top2_vals[:, 0, :, :]
                second_prob = top2_vals[:, 1, :, :]
                pred_top1 = top2_idx[:, 0, :, :] + 1  # 1..C

                # 只取 valid 位置
                max_v = max_prob[valid]
                second_v = second_prob[valid]
                pred1_v = pred_top1[valid].long()

                for gm in gate_modes:
                    for tau in taus:
                        for delta in deltas:
                            key = (T, str(gm), float(tau), float(delta))
                            gate_v = _gate_mask(max_v, second_v, float(tau), float(delta), str(gm))
                            pred_v = pred1_v.clone()
                            pred_v[gate_v] = 0

                            # confusion update
                            label = num_class * gt_flat + pred_v
                            binc = torch.bincount(label, minlength=num_class * num_class).to(torch.float64)
                            conf[key] += binc.view(num_class, num_class)

                            # stats
                            total_valid[key] += v
                            total_pred_bg[key] += (pred_v == 0).sum()
                            total_gate_bg[key] += gate_v.sum()

            if rank == 0 and len(test_loader) > 0:
                period = max(1, len(test_loader) // 5)
                if batch_idx % period == 0:
                    logger.info(f"[{batch_idx}/{len(test_loader)}]")

    # reduce for DDP
    if dist.is_available() and dist.is_initialized():
        for k in settings:
            dist.reduce(conf[k], dst=0)
            dist.reduce(total_valid[k], dst=0)
            dist.reduce(total_pred_bg[k], dst=0)
            dist.reduce(total_gate_bg[k], dst=0)

    if rank != 0:
        return

    # compute metrics using your Evaluator
    results = []
    for k in settings:
        T, gm, tau, delta = k
        e = module_metric.Evaluator(num_class, old_idx, new_idx)
        e.confusion_matrix = conf[k].detach().cpu().numpy()
        miou = e.Mean_Intersection_over_Union()

        v = int(total_valid[k].item())
        bg = int(total_pred_bg[k].item())
        gate = int(total_gate_bg[k].item())

        pred_bg_ratio = (bg / v * 100.0) if v > 0 else float("nan")
        gate_ratio = (gate / v * 100.0) if v > 0 else float("nan")

        results.append({
            "T": T,
            "gate_mode": gm,
            "tau": tau,
            "delta": delta,
            "pred_bg%": pred_bg_ratio,
            "gate_bg%": gate_ratio,
            "miou_old": float(miou["old"]),
            "miou_new": float(miou["new"]),
            "miou_harmonic": float(miou["harmonic"]),
            "miou_overall": float(miou["overall"]),
            "by_class": miou["by_class"],
        })

    # selection
    select_key = {
        "harmonic": "miou_harmonic",
        "new": "miou_new",
        "old": "miou_old",
        "overall": "miou_overall",
    }[select]

    # tie-break: overall > new > old
    def sort_key(r):
        return (
            r[select_key],
            r["miou_overall"],
            r["miou_new"],
            r["miou_old"],
        )

    results.sort(key=sort_key, reverse=True)
    best = results[0]

    # print table
    logger.info("========== TEST-DRIVEN HP PICK RESULTS ==========")
    logger.info(f"select = {select}  (primary={select_key})")
    logger.info("T     gate  tau   delta  pred_bg%  gate_bg%  mIoU_old  mIoU_new  mIoU_harm  mIoU_overall")
    for r in results:
        logger.info(
            f"{r['T']:<5.2f} {r['gate_mode']:<5s} "
            f"{r['tau']:<5.2f} {r['delta']:<5.2f} "
            f"{r['pred_bg%']:<8.3f} {r['gate_bg%']:<8.3f} "
            f"{r['miou_old']:<8.2f} {r['miou_new']:<8.2f} "
            f"{r['miou_harmonic']:<9.2f} {r['miou_overall']:<11.2f}"
        )

    logger.info("========== BEST ON TEST ==========")
    logger.info(
        f"BEST: T={best['T']:.2f}, gate_mode={best['gate_mode']}, tau={best['tau']:.2f}, delta={best['delta']:.2f} | "
        f"pred_bg%={best['pred_bg%']:.3f}, gate_bg%={best['gate_bg%']:.3f} | "
        f"old={best['miou_old']:.2f}, new={best['miou_new']:.2f}, harm={best['miou_harmonic']:.2f}, overall={best['miou_overall']:.2f}"
    )

    # focus classes IoU under best
    if focus:
        names = VOC_CLASS_NAMES if len(VOC_CLASS_NAMES) == num_class else [f"class_{i}" for i in range(num_class)]
        logger.info("========== FOCUS CLASS IoU (BEST) ==========")
        by = best["by_class"]
        for c in focus:
            if 0 <= c < num_class:
                logger.info(f"[{c:2d}] {names[c]:<12} IoU = {by[c]:.2f}")

    # optional save
    if save_best:
        payload = {
            "best": {
                "T": best["T"],
                "gate_mode": best["gate_mode"],
                "tau": best["tau"],
                "delta": best["delta"],
                "pred_bg%": best["pred_bg%"],
                "gate_bg%": best["gate_bg%"],
                "miou_old": best["miou_old"],
                "miou_new": best["miou_new"],
                "miou_harmonic": best["miou_harmonic"],
                "miou_overall": best["miou_overall"],
            }
        }
        with open(save_best, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved best setting to: {save_best}")

    logger.info("========== DONE ==========")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pick eval-side HPs on TEST set (VOC)")

    parser.add_argument("-c", "--config", default=None, type=str, help="config file path")
    parser.add_argument("-r", "--resume", default=None, type=str, help="checkpoint path")
    parser.add_argument("-d", "--device", default=None, type=str, help="gpu id(s)")

    # hp search space
    parser.add_argument("--taus", type=str, default="0.45", help="comma-separated taus")
    parser.add_argument("--deltas", type=str, default="0", help="comma-separated deltas")
    parser.add_argument("--gate_modes", type=str, default="or", help="comma-separated: or,and,tau,none")
    parser.add_argument("--temps", type=str, default="1.0", help="comma-separated temperatures, e.g. 0.85,1.0,1.2")

    # selection objective
    parser.add_argument("--select", type=str, default="harmonic", choices=["harmonic", "new", "old", "overall"])
    parser.add_argument("--focus", type=str, default="", help="comma-separated class ids to print IoU under best")
    parser.add_argument("--save_best", type=str, default="", help="path to save best setting as json")

    CustomArgs = collections.namedtuple("CustomArgs", "flags type action target", defaults=(None, float, None, None))
    options = [
        CustomArgs(["--multiprocessing_distributed"], action="store_true", target="multiprocessing_distributed"),
        CustomArgs(["--dist_url"], type=str, target="dist_url"),
        CustomArgs(["--name"], type=str, target="name"),
        CustomArgs(["--save_dir"], type=str, target="trainer;save_dir"),
        CustomArgs(["--test"], action="store_true", target="test"),
    ]
    config = ConfigParser.from_args(parser, options)

    # inject args into config for mp.spawn
    config.config["taus"] = parse_float_list(parser.parse_args().taus) or [0.45]
    config.config["deltas"] = parse_float_list(parser.parse_args().deltas) or [0.0]
    config.config["gate_modes"] = parse_str_list(parser.parse_args().gate_modes) or ["or"]
    config.config["temps"] = parse_float_list(parser.parse_args().temps) or [1.0]
    config.config["select"] = parser.parse_args().select
    config.config["focus"] = parse_int_list(parser.parse_args().focus)
    config.config["save_best"] = parser.parse_args().save_best

    main(config)
