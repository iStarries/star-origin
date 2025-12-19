import argparse
import collections
import random

import numpy as np
import torch
import torch.nn as nn

import models.model as module_arch
import data_loader.data_loaders as module_data
from data_loader import VOC
from logger.logger import Logger
from utils.boundary_metrics import BoundaryMetric
from utils.parse_config import ConfigParser


torch.backends.cudnn.benchmark = True


def load_checkpoint(model, checkpoint_path):
    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state.get("state_dict", state)

    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)


def build_model(config, dataset, device_ids):
    model = config.init_obj("arch", module_arch, **{"classes": dataset.get_per_task_classes()})
    if torch.cuda.is_available():
        model = nn.DataParallel(model, device_ids=device_ids)
    return model


def prepare_predictions(logit):
    logit = torch.sigmoid(logit)
    pred = logit.argmax(dim=1) + 1

    idx = (logit > 0.5).float().sum(dim=1)
    pred[idx == 0] = 0
    return pred


def evaluate_boundary(config):
    logger = Logger(config.log_dir, rank=0)
    logger.set_logger("eval-boundary", verbosity=2)

    seed = config["seed"]
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset = config.init_obj("data_loader", module_data)
    test_loader = dataset.get_test_loader()

    old_classes, _ = dataset.get_task_labels(step=0)
    boundary_width = config.config.get("boundary_width", 3)
    min_mask_pixels = config.config.get("min_mask_pixels", 50)

    logger.info(f"Evaluating on old classes (step0): {old_classes}")
    logger.info(f"Boundary width: {boundary_width}px, min mask pixels: {min_mask_pixels}")

    device_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(config, dataset, device_ids)
    model.to(device)
    load_checkpoint(model, config.resume)
    model.eval()

    metric = BoundaryMetric(target_classes=old_classes, boundary_width=boundary_width, min_mask_pixels=min_mask_pixels)

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            images = data["image"].to(device)
            labels = data["label"].to(device)

            logit, _, _ = model(images)
            pred = prepare_predictions(logit)

            metric.add_batch(labels.cpu().numpy(), pred.cpu().numpy())
            if batch_idx % 50 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")

    summary = metric.summary()

    logger.info("\n===== Boundary IoU (old 15 classes) =====")
    for cls_idx in old_classes:
        score = summary["biou_per_class"].get(cls_idx, float("nan"))
        logger.info(f"{cls_idx:2d} {VOC[cls_idx]:>12s}: {score:.4f}")
    logger.info(f"BIoU_old (macro over 15 classes): {summary['biou_old']:.4f}")

    logger.info("\n===== Boundary F-score (old 15 classes) =====")
    for cls_idx in old_classes:
        score = summary["bf_per_class"].get(cls_idx, float("nan"))
        logger.info(f"{cls_idx:2d} {VOC[cls_idx]:>12s}: {score:.4f}")
    logger.info(f"BF_old (macro over 15 classes): {summary['bf_old']:.4f}")

    return summary


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Boundary metrics for VOC 15-1 old classes")
    args.add_argument("-c", "--config", default=None, type=str, help="config file path (default: None)")
    args.add_argument("-r", "--resume", default=None, type=str, help="path to checkpoint (default: None)")
    args.add_argument("-d", "--device", default=None, type=str, help="GPU indices to enable (default: all)")

    CustomArgs = collections.namedtuple("CustomArgs", "flags type action target", defaults=(None, float, None, None))
    options = [
        CustomArgs(["--multiprocessing_distributed"], action="store_true", target="multiprocessing_distributed"),
        CustomArgs(["--dist_url"], type=str, target="dist_url"),
        CustomArgs(["--name"], type=str, target="name"),
        CustomArgs(["--save_dir"], type=str, target="trainer;save_dir"),
        CustomArgs(["--boundary_width"], type=int, target="boundary_width"),
        CustomArgs(["--min_mask_pixels"], type=int, target="min_mask_pixels"),
        CustomArgs(["--info"], type=str, target="info"),
    ]

    config = ConfigParser.from_args(args, options)
    evaluate_boundary(config)
