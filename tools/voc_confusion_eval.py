import argparse
import collections
import json
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.utils.data

import data_loader.data_loaders as module_data
import models.model as module_arch
import utils.metric as module_metric
from data_loader import VOC as VOC_NAMES
from logger.logger import Logger
from utils.parse_config import ConfigParser

torch.backends.cudnn.benchmark = True

'''
python tools/voc_confusion_eval.py \
  -c /media/disk1/media/wyh/star/saved_voc/models/overlap_5-3_phase-lambda001-k_old1-valconcatpth/step_5/config.json \
  -r /media/disk1/media/wyh/star/saved_voc/models/overlap_5-3_phase-lambda001-k_old1-valconcatpth/step_5/checkpoint-epoch60.pth \
  --device 0
'''
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

    rank = dist.get_rank() if dist.is_initialized() else 0
    logger = Logger(config.log_dir, rank=rank)
    logger.set_logger(f"voc-confusion(rank{rank})", verbosity=2)

    _set_random_seeds(config["seed"])

    task_step = config["data_loader"]["args"]["task"]["step"]
    dataset = config.init_obj("data_loader", module_data)

    test_sampler = (
        torch.utils.data.distributed.DistributedSampler(dataset.test_set, shuffle=False)
        if config["multiprocessing_distributed"]
        else None
    )
    test_loader = dataset.get_test_loader(sampler=test_sampler)

    old_classes, _ = dataset.get_task_labels(step=0)
    new_classes: List[int] = []
    for i in range(1, task_step + 1):
        cls_ids, _ = dataset.get_task_labels(step=i)
        new_classes += cls_ids
    logger.info(f"Old Classes: {old_classes}")
    logger.info(f"New Classes: {new_classes}")

    model = config.init_obj("arch", module_arch, **{"classes": dataset.get_per_task_classes()})
    if config["multiprocessing_distributed"] and (config["arch"]["args"]["norm_act"] == "bn_sync"):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # logger.info(model)

    if config["multiprocessing_distributed"]:
        device = torch.device(f"cuda:{gpu}")
        torch.cuda.set_device(device)
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[device.index])
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

    if config.resume is not None:
        _load_checkpoint(model, config.resume, logger)

    evaluator = config.init_obj(
        "evaluator",
        module_metric,
        *[dataset.n_classes + 1, list(set(old_classes + [0])), new_classes],
    )

    confusion = run_inference(model, test_loader, evaluator, device, logger)

    if dist.is_initialized():
        evaluator.sync(device)
        confusion = evaluator.confusion_matrix

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        output_path = Path(
            config.config.get(
                "output_json",
                Path(__file__).resolve().parent.parent / "docs" / "voc_confusion.json",
            )
        )
        summarize_confusion(confusion, logger, VOC_NAMES, output_path)


def run_inference(model, data_loader, evaluator, device, logger):
    model.eval()
    evaluator.reset()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            images = data["image"].to(device)
            target = data["label"].cpu().numpy()

            logit, _, _ = model(images)
            logit = torch.sigmoid(logit)
            pred = logit.argmax(dim=1) + 1

            idx = (logit > 0.5).float().sum(dim=1)
            pred[idx == 0] = 0

            pred = pred.cpu().numpy()
            evaluator.add_batch(target, pred)

            _progress(logger, batch_idx, len(data_loader))
    return evaluator.confusion_matrix


def summarize_confusion(confusion, logger, class_names, output_path: Optional[Path] = None):
    row_sums = confusion.sum(axis=1)
    summary = []
    for cls_idx, total in enumerate(row_sums):
        class_name = class_names[cls_idx] if cls_idx < len(class_names) else f"class_{cls_idx}"
        if total == 0:
            logger.info(f"[{cls_idx:2d}] {class_name}: 无样本，跳过统计")
            summary.append(
                {
                    "class_id": cls_idx,
                    "class_name": class_name,
                    "total_pixels": int(total),
                    "background_ratio": None,
                    "top_miscls": [],
                }
            )
            continue

        row = confusion[cls_idx].copy()
        background_ratio = row[0] / total

        row[cls_idx] = 0
        row[0] = 0
        top_indices = row.argsort()[::-1]
        top_entries = []
        for pred_idx in top_indices[:3]:
            count = row[pred_idx]
            if count <= 0:
                continue
            pred_name = class_names[pred_idx] if pred_idx < len(class_names) else f"class_{pred_idx}"
            ratio = float(count / total)
            top_entries.append({"pred_id": int(pred_idx), "pred_name": pred_name, "ratio": ratio})

        top_msg = (
            ", ".join(f"{e['pred_id']}:{e['pred_name']} ({e['ratio']:.2%})" for e in top_entries)
            if top_entries
            else "无误分样本"
        )
        logger.info(f"[{cls_idx:2d}] {class_name}")
        logger.info(f"  top-3 误分：{top_msg}")
        logger.info(f"  误分为 background 占比：{background_ratio:.2%}")

        summary.append(
            {
                "class_id": int(cls_idx),
                "class_name": class_name,
                "total_pixels": int(total),
                "background_ratio": float(background_ratio),
                "top_miscls": top_entries,
            }
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"classes": summary}, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存混淆统计到: {output_path}")


def _progress(logger, i, total_length):
    period = max(total_length // 5, 1)
    if i % period == 0:
        logger.info(f"[{i}/{total_length}]")


def _set_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _load_checkpoint(model, resume_path, logger):
    logger.info(f"Loading checkpoint: {resume_path} ...")
    checkpoint = torch.load(resume_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    logger.info("Checkpoint loaded.")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="VOC Confusion Matrix Evaluation")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to latest checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="indices of GPUs to enable (default: all)")

    CustomArgs = collections.namedtuple(
        "CustomArgs",
        "flags type action target default choices",
        defaults=(None, float, None, None, None, None),
    )
    options = [
        CustomArgs(["--multiprocessing_distributed"], action="store_true", target="multiprocessing_distributed"),
        CustomArgs(["--dist_url"], type=str, target="dist_url"),
        CustomArgs(["--name"], type=str, target="name"),
        CustomArgs(["--save_dir"], type=str, target="trainer;save_dir"),
        CustomArgs(["--test"], action="store_true", target="test"),
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
