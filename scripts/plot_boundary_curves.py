#!/usr/bin/env python3
"""
Plot Boundary IoU and Boundary F-score across incremental steps for two methods.

默认会读取 ``results/boundary_voc15_1_old_classes.json``，该文件包含 step1–step5
在旧 15 类上的宏平均 BIoU 和 BF-score（STAR-Proto vs Phase-Proto）。

你也可以通过 ``--data-json`` 传入自定义结果，格式示例：
{
  "steps": [1, 2, 3, 4, 5],
  "methods": [
    {"label": "STAR-Proto", "biou": [...], "bf": [...]},
    {"label": "Phase-Proto", "biou": [...], "bf": [...]}]
}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

PREFERRED_FONTS = [
    "Noto Sans CJK SC",
    "Microsoft YaHei",
    "SimHei",
    "WenQuanYi Micro Hei",
    "Arial Unicode MS",
    "DejaVu Sans",
]

available_fonts = {Path(f).stem for f in fm.findSystemFonts()}
font_candidates = [f for f in PREFERRED_FONTS if f in available_fonts]
if font_candidates:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = font_candidates + plt.rcParams["font.sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

DEFAULT_DATA = {
    "steps": [1, 2, 3, 4, 5],
    "methods": [
        {
            "label": "STAR-Proto",
            "biou": [0.1447, 0.1552, 0.1616, 0.1729, 0.1655],
            "bf": [0.4997, 0.5156, 0.5258, 0.5412, 0.5303],
        },
        {
            "label": "Phase-Proto",
            "biou": [0.1437, 0.1536, 0.1631, 0.1730, 0.1663],
            "bf": [0.4992, 0.5135, 0.5271, 0.5421, 0.5311],
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Boundary IoU / BF-score 折线图，突出两种方法在增量步骤上的差异。"
        )
    )
    parser.add_argument(
        "--data-json",
        type=str,
        default="results/boundary_voc15_1_old_classes.json",
        help="包含 steps 与 methods 的 JSON 文件；若不存在则使用内置默认数据。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="boundary_old15_steps.png",
        help="保存图片的路径（默认 boundary_old15_steps.png）。",
    )
    parser.add_argument(
        "--title-prefix",
        type=str,
        default="VOC 15-1 Old Classes Boundary",
        help="图标题前缀。",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="输出图片的 DPI。",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="在保存后通过 matplotlib 展示图像。",
    )
    return parser.parse_args()


def load_data(path: Path) -> Dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_DATA


def _ensure_length(name: str, values: Sequence[float], steps: Sequence[int]):
    if len(values) != len(steps):
        raise ValueError(f"{name} 长度为 {len(values)}，但 steps 长度为 {len(steps)}，请检查输入数据。")


def validate_payload(payload: Dict) -> Dict:
    if "steps" not in payload or "methods" not in payload:
        raise ValueError("输入 JSON 需要包含 'steps' 和 'methods' 两个字段。")
    steps = payload["steps"]
    methods = payload["methods"]
    if not isinstance(steps, list) or not steps:
        raise ValueError("'steps' 应该是非空列表。")
    if not isinstance(methods, list) or len(methods) < 2:
        raise ValueError("'methods' 至少需要包含两组结果，方便对比。")
    validated_methods = []
    for m in methods:
        label = m.get("label") or m.get("key") or "Method"
        biou = m.get("biou")
        bf = m.get("bf")
        if biou is None or bf is None:
            raise ValueError(f"方法 {label} 缺少 'biou' 或 'bf' 字段。")
        _ensure_length(f"{label} 的 biou", biou, steps)
        _ensure_length(f"{label} 的 bf", bf, steps)
        validated_methods.append({"label": label, "biou": biou, "bf": bf})
    return {"steps": steps, "methods": validated_methods}


def plot_metric(
    ax: plt.Axes,
    steps: List[int],
    methods: List[Dict[str, Sequence[float]]],
    metric_key: str,
    metric_label: str,
    legend_loc: str = "best",
):
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for idx, method in enumerate(methods):
        values = method[metric_key]
        ax.plot(
            steps,
            values,
            marker="o",
            color=colors[idx % len(colors)],
            label=method["label"],
            linewidth=2,
        )

    if len(methods) >= 2:
        ref = methods[0][metric_key]
        alt = methods[1][metric_key]
        upper = np.maximum(ref, alt)
        lower = np.minimum(ref, alt)
        ax.fill_between(
            steps,
            lower,
            upper,
            color="#c7e9c0",
            alpha=0.35,
            label="差距区间",
        )
        for s, v1, v2, top in zip(steps, ref, alt, upper):
            diff = v2 - v1
            ax.text(
                s,
                top + 0.003,
                f"Δ {diff:+.4f}",
                fontsize=8,
                ha="center",
                va="bottom",
            )

    ax.set_xlabel("增量步骤 t")
    ax.set_ylabel(metric_label)
    ax.set_xticks(steps)
    data_min = min(min(m[metric_key]) for m in methods)
    data_max = max(max(m[metric_key]) for m in methods)
    span = data_max - data_min
    padding = max(span * 0.45, 0.01)
    ax.set_ylim(bottom=max(data_min - padding, 0), top=min(data_max + padding, 1.0))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc=legend_loc)


def plot_curves(payload: Dict, output: Path, title_prefix: str, dpi: int, show: bool):
    steps = payload["steps"]
    methods = payload["methods"]
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4))
    fig.suptitle(f"{title_prefix} (旧 15 类)", fontsize=14)

    plot_metric(ax, steps, methods, "biou", "Boundary IoU (宏平均)", legend_loc="upper left")
    plot_metric(
        ax.twinx(),
        steps,
        methods,
        "bf",
        "Boundary F-score (宏平均)",
        legend_loc="lower right",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"已保存折线图到: {output.resolve()}")
    if show:
        plt.show()


def main():
    args = parse_args()
    data_path = Path(args.data_json)
    payload = load_data(data_path)
    payload = validate_payload(payload)
    plot_curves(payload, Path(args.output), args.title_prefix, args.dpi, args.show)


if __name__ == "__main__":
    main()
