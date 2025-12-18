"""从 PASCAL VOC 数据集中筛选指定类别的图像及其分割真值标签。

该脚本会遍历 `Annotations` 目录，找到包含目标类别的样本，
并将对应的原始图像（JPEGImages）和语义分割标签（优先 SegmentationClassAug，
若不存在则使用 SegmentationClass）复制到输出目录，便于人工查看。

python /media/wyh/star/scripts/voc/extract_voc_subset.py \
 /media/wyh/star/datasets/PascalVOC2012 \
 /media/wyh/star/docs/air \
 --classes aeroplane
"""
from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple


DEFAULT_CLASSES = ["sofa", "chair", "aeroplane"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "筛选含指定 VOC 类别的样本，并复制图像与分割标签到目标文件夹。"
        )
    )
    parser.add_argument(
        "dataset_root",
        type=Path,
        help=(
            "VOC 数据集根目录（包含 Annotations、JPEGImages 等子目录），"
            "例如 ./datasets/PascalVOC2012"
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="保存筛选结果的目标文件夹（将生成 images/ 与 labels/ 子目录）",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=DEFAULT_CLASSES,
        help="要筛选的类别名称，默认：%(default)s",
    )
    parser.add_argument(
        "--use_aug_labels",
        action="store_true",
        help="优先使用 SegmentationClassAug 中的标签（若存在）。",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个被复制样本的 ID，便于确认运行进度。",
    )
    return parser.parse_args()


def contains_target_class(xml_path: Path, target_classes: Set[str]) -> bool:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for obj in root.iter("object"):
        name_node = obj.find("name")
        if name_node is not None and name_node.text.lower() in target_classes:
            return True
    return False


def find_label_path(
    sample_id: str, label_dirs: Tuple[Path, ...]
) -> Optional[Path]:
    candidates: Iterable[Path] = []
    candidates = [label_dir / f"{sample_id}.png" for label_dir in label_dirs]

    for path in candidates:
        if path.exists():
            return path
    return None


def prepare_label_dirs(dataset_root: Path, use_aug: bool) -> Tuple[Path, ...]:
    seg_dir = dataset_root / "SegmentationClass"
    aug_dir = dataset_root / "SegmentationClassAug"

    if not seg_dir.exists():
        raise FileNotFoundError(
            "未找到 SegmentationClass 目录，无法定位分割标签，请检查 dataset_root。"
        )

    if use_aug and aug_dir.exists():
        print("优先使用 SegmentationClassAug 目录中的增强标签。")
        return (aug_dir, seg_dir)

    if use_aug:
        print("提示：未找到 SegmentationClassAug 目录，将使用官方标签 SegmentationClass。")

    return (seg_dir,)


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root
    output_dir: Path = args.output_dir
    target_classes: Set[str] = {cls.lower() for cls in args.classes}

    annotations_dir = dataset_root / "Annotations"
    images_dir = dataset_root / "JPEGImages"

    if not annotations_dir.exists() or not images_dir.exists():
        raise FileNotFoundError(
            "未找到 VOC 数据集必要子目录，请确认 dataset_root 设置正确。"
        )

    print(
        "开始筛选 VOC 数据集："
        f" root={dataset_root.resolve()}"
        f" classes={sorted(target_classes)}"
        f" use_aug={args.use_aug_labels}"
    )

    label_dirs = prepare_label_dirs(dataset_root, args.use_aug_labels)

    target_xmls = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        if contains_target_class(xml_path, target_classes):
            target_xmls.append(xml_path)

    if not target_xmls:
        print("未在数据集中找到包含指定类别的样本。")
        return

    images_out = output_dir / "images"
    labels_out = output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    skipped_without_label = 0
    copied = 0

    for xml_path in target_xmls:
        sample_id = xml_path.stem
        image_path = images_dir / f"{sample_id}.jpg"
        label_path = find_label_path(sample_id, label_dirs)

        if label_path is None:
            skipped_without_label += 1
            continue

        if not image_path.exists():
            print(f"警告：未找到图像 {image_path}，跳过该样本。")
            continue

        shutil.copy2(image_path, images_out / image_path.name)
        shutil.copy2(label_path, labels_out / label_path.name)
        copied += 1

        if args.verbose:
            print(f"已复制：{sample_id}")

    print(f"符合条件的样本数：{len(target_xmls)}")
    print(f"成功复制的样本数：{copied}")
    if skipped_without_label:
        print(f"缺少分割标签的样本数：{skipped_without_label}")
    print(f"输出目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
