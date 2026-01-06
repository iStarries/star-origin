#!/usr/bin/env python3
"""
统计 Pascal VOC 21 类（含 background）在“全量 train_aug 原始标注”上的出现次数、像素数与实例数。
不依赖增量 dataloader 的过滤逻辑，直接读取原始 mask，确保所有类别都被统计。
输出：voc_stats_train.json
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from tqdm import tqdm
'''
python tools/voc_dataset_stats.py \
  --data_root /media/wyh/star/datasets/PascalVOC2012 \
  --out_dir /media/wyh/star/docs \
  --ignore_index 255
'''
# 确保仓库根目录在 sys.path（便于在项目外直接运行）
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def connected_components(binary: np.ndarray) -> List[int]:
    """返回连通域面积列表（8-连通）。优先 cv2，备用 scipy，失败则返回空列表。"""
    try:
        import cv2
        num, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
        return [int((labels == cid).sum()) for cid in range(1, num)]
    except Exception:
        try:
            from scipy import ndimage
            labels, num = ndimage.label(binary)
            return [int((labels == cid).sum()) for cid in range(1, num + 1)]
        except Exception:
            return []


def _read_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def gather_train_ids(data_root: Path) -> List[str]:
    """
    取训练相关的所有列表并去重，最大化覆盖类别：
    - train_aug.txt（若有）
    - train.txt
    - val.txt（有些稀有类只在 val 出现，避免漏计）
    """
    seg_dir = data_root / "ImageSets" / "Segmentation"
    ids = []
    for name in ["train_aug.txt", "train.txt", "val.txt"]:
        ids.extend(_read_list(seg_dir / name))
    return sorted(list(set(ids)))


def compute_stats(args):
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ignore_index = args.ignore_index

    n_classes = len(VOC_CLASSES)
    pixel_count = np.zeros(n_classes, dtype=np.int64)
    img_count = np.zeros(n_classes, dtype=np.int64)
    inst_count = np.zeros(n_classes, dtype=np.int64)
    total_valid = 0

    img_ids = gather_train_ids(data_root)

    # 标注目录：优先 Aug，其次原始
    cat_dir_aug = data_root / "SegmentationClassAug"
    cat_dir_raw = data_root / "SegmentationClass"

    for img_id in tqdm(img_ids, desc="stat-train_aug"):
        cat_path = (cat_dir_aug / f"{img_id}.png")
        if not cat_path.exists():
            cat_path = (cat_dir_raw / f"{img_id}.png")
            if not cat_path.exists():
                continue
        mask = np.array(Image.open(cat_path), dtype=np.int64)
        valid = mask != ignore_index
        if valid.sum() == 0:
            continue
        total_valid += int(valid.sum())
        for cls_id in range(n_classes):
            cls_mask = (mask == cls_id)
            if cls_mask.any():
                img_count[cls_id] += 1
                pixel_count[cls_id] += int(cls_mask.sum())
                sizes = connected_components(cls_mask.astype(np.uint8))
                inst_count[cls_id] += len(sizes)

    records = []
    for cid, cname in enumerate(VOC_CLASSES):
        records.append({
            "class_id": cid,
            "class_name": cname,
            "img_count": int(img_count[cid]),
            "pixel_count": int(pixel_count[cid]),
            "pixel_ratio": float(pixel_count[cid] / total_valid) if total_valid > 0 else 0.0,
            "inst_count": int(inst_count[cid]),
        })

    out_json = out_dir / "voc_stats_train.json"
    with open(out_json, "w") as f:
        json.dump({
            "meta": {
                "data_root": str(data_root),
                "split": "train_aug",
                "mask_dir_aug": str(cat_dir_aug),
                "mask_dir_raw": str(cat_dir_raw),
                "ignore_index": ignore_index,
                "total_valid_pixels": int(total_valid),
                "classes": VOC_CLASSES,
            },
            "records": records
        }, f, indent=2)


def parse_args():
    ap = argparse.ArgumentParser(description="VOC train_aug statistics (full 21 classes, raw masks)")
    ap.add_argument("--data_root", required=True, type=str, help="VOC root (containing ImageSets/Segmentation and SegmentationClassAug)")
    ap.add_argument("--out_dir", required=True, type=str, help="Directory to save voc_stats_train.json")
    ap.add_argument("--ignore_index", default=255, type=int)
    return ap.parse_args()


if __name__ == "__main__":
    compute_stats(parse_args())
