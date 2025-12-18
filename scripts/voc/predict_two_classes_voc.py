#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
import inspect
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


# ===================== 你给的路径（直接可跑） =====================
CKPT_PATH = Path("/media/wyh/star/saved_voc/models/overlap_15-1_Phase+realgrad2/step_3_20251212-230029/checkpoint-epoch60.pth")
CFG_PATH  = Path("/media/wyh/star/saved_voc/models/overlap_15-1_Phase+realgrad2/step_3_20251212-230029/config.json")

AIR_IMG_DIR  = Path("/media/wyh/star/docs/air/images")
AIR_LBL_DIR  = Path("/media/wyh/star/docs/air/labels")

SOFA_IMG_DIR = Path("/media/wyh/star/docs/sofa/images")
SOFA_LBL_DIR = Path("/media/wyh/star/docs/sofa/labels")

AIR_OUT_DIR  = Path("/media/wyh/star/docs/air/preds")
SOFA_OUT_DIR = Path("/media/wyh/star/docs/sofa/preds")

DEVICE = "cuda"  # 没有GPU会自动fallback到cpu
MAX_IMAGES_EACH = None  # 例如设成 50 只跑前50张，None=全跑
SAVE_COMPARE = True     # 是否输出简单对比图（原图/GT二值/Pred二值）


# ===================== VOC palette（P模式标签图） =====================
def voc_palette():
    palette = [0] * (256 * 3)
    for j in range(256):
        lab = j
        for i in range(8):
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            lab >>= 3
    return palette


VOC_CLASSES = [
    "background",
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair",
    "cow","diningtable","dog","horse","motorbike","person","pottedplant",
    "sheep","sofa","train","tvmonitor"
]
VOC_ID = {name: i for i, name in enumerate(VOC_CLASSES)}  # aeroplane=1, sofa=18


# ===================== 一些通用工具 =====================
def _safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _filter_kwargs(fn, kwargs: dict) -> dict:
    """只保留构造函数/函数签名里支持的kwargs，避免TypeError。"""
    try:
        sig = inspect.signature(fn)
        allowed = set(sig.parameters.keys())
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return kwargs

def _to_tensor(img: Image.Image, mean, std) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = arr.transpose(2, 0, 1)  # CHW
    t = torch.from_numpy(arr)
    mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (t - mean) / std

def _pad_to_stride(x: torch.Tensor, stride: int = 32):
    """把输入pad到stride的倍数，返回(padded, pad_info)"""
    _, _, h, w = x.shape
    ph = (stride - (h % stride)) % stride
    pw = (stride - (w % stride)) % stride
    if ph == 0 and pw == 0:
        return x, (0, 0, 0, 0)
    # pad顺序: (left, right, top, bottom)
    x = F.pad(x, (0, pw, 0, ph), mode="constant", value=0.0)
    return x, (0, pw, 0, ph)

def _unpad_mask(mask: np.ndarray, pad_info):
    left, right, top, bottom = pad_info
    if right == 0 and bottom == 0 and left == 0 and top == 0:
        return mask
    h, w = mask.shape
    return mask[: (h - bottom), : (w - right)]


def _save_palette_png(mask: np.ndarray, out_path: Path):
    """mask: HxW uint8/uint16/uint32 都行，会转uint8保存（<=255）"""
    mask_u8 = mask.astype(np.uint8)
    im = Image.fromarray(mask_u8, mode="P")
    im.putpalette(voc_palette())
    im.save(out_path)

def _save_binary_png(mask01: np.ndarray, out_path: Path):
    """0/1二值图，保存成P模式方便看。"""
    mask_u8 = (mask01.astype(np.uint8) * 255)
    im = Image.fromarray(mask_u8, mode="L")
    im.save(out_path)

def _save_overlay(rgb: Image.Image, mask01: np.ndarray, out_path: Path, alpha=0.55):
    rgb_np = np.array(rgb).astype(np.float32)
    color = np.zeros_like(rgb_np)
    color[mask01 == 1] = np.array([255, 0, 0], dtype=np.float32)  # 红色标注目标
    out = (rgb_np * (1 - alpha) + color * alpha).clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(out_path)

def _hstack_images(imgs):
    """PIL Image list -> 横向拼接"""
    widths = [im.width for im in imgs]
    heights = [im.height for im in imgs]
    out = Image.new("RGB", (sum(widths), max(heights)))
    x = 0
    for im in imgs:
        out.paste(im, (x, 0))
        x += im.width
    return out


# ===================== 关键：从你的项目里把模型建出来 =====================
import importlib.util

def build_model_from_config(cfg: dict, project_root: Path) -> torch.nn.Module:
    arch_type = cfg.get("arch", {}).get("type", None)
    arch_args = cfg.get("arch", {}).get("args", {}) or {}
    if arch_type is None:
        raise RuntimeError("config.json 里没找到 arch.type")

    # 有些实现需要 num_classes：VOC 默认 21（含背景）
    extra_try_kwargs = [
        {},
        {"num_classes": 21},
        {"n_classes": 21},
        {"classes": 21},
        {"nclass": 21},
    ]

    # 1) 先尝试常见模块名（如果你工程里正好有）
    candidates = [
        "models",
        "networks",
        "network",
        "src.models",
        "src.model",
    ]

    last_err = None
    for modname in candidates:
        try:
            m = __import__(modname, fromlist=["*"])
            if hasattr(m, arch_type):
                ModelCls = getattr(m, arch_type)
                for ext in extra_try_kwargs:
                    kwargs = dict(arch_args)
                    kwargs.update(ext)
                    kwargs = _filter_kwargs(ModelCls.__init__, kwargs)
                    try:
                        model = ModelCls(**kwargs)
                        print(f"[INFO] Built model: {modname}.{arch_type}({kwargs})")
                        return model
                    except TypeError as e:
                        last_err = e
        except Exception as e:
            last_err = e

    # 2) 自动在工程目录里找 “class DeepLabV3” 定义文件，然后从文件路径加载
    skip_dirs = {
        ".git", "__pycache__", ".idea", ".vscode",
        "docs", "saved_voc", "outputs", "runs", "wandb",
        "data", "datasets", "VOCdevkit",
    }

    def should_skip(p: Path) -> bool:
        return any(part in skip_dirs for part in p.parts)

    hits = []
    for py in project_root.rglob("*.py"):
        if should_skip(py):
            continue
        try:
            txt = py.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # 精准一点：优先找类定义
        if f"class {arch_type}" in txt:
            hits.append(py)
            break

    if not hits:
        # 退一步：找 “DeepLabV3(” 字样（可能是导入别处的注册，但也能帮助定位）
        for py in project_root.rglob("*.py"):
            if should_skip(py):
                continue
            try:
                txt = py.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if f"{arch_type}(" in txt:
                hits.append(py)
                break

    if not hits:
        raise RuntimeError(
            f"还是没找到 {arch_type} 的定义/引用文件。最后一次错误：{repr(last_err)}\n"
            "你可以在工程根目录跑：\n"
            "  grep -R \"class DeepLabV3\" -n .\n"
            "找到文件后我再帮你把 import 写死进去。"
        )

    py = hits[0]
    print(f"[INFO] Auto-found {arch_type} in: {py}")

    # 从文件路径加载模块
    spec = importlib.util.spec_from_file_location("autoload_deeplabv3", str(py))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法从文件加载模块：{py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if not hasattr(mod, arch_type):
        raise RuntimeError(
            f"在文件 {py} 里没找到类 {arch_type}。"
            f"（可能它在别的文件里被 import 进来，建议用 grep 找真正定义处）"
        )

    ModelCls = getattr(mod, arch_type)
    for ext in extra_try_kwargs:
        kwargs = dict(arch_args)
        kwargs.update(ext)
        kwargs = _filter_kwargs(ModelCls.__init__, kwargs)
        try:
            model = ModelCls(**kwargs)
            print(f"[INFO] Built model from file: {py.name}.{arch_type}({kwargs})")
            return model
        except TypeError as e:
            last_err = e
            continue

    raise RuntimeError(
        f"找到了 {arch_type} 但实例化失败。最后一次错误：{repr(last_err)}\n"
        f"命中的文件：{py}\n"
        "通常是构造函数参数名不匹配：你把报错贴出来，我就能根据 __init__ 需要的参数把 kwargs 对齐。"
    )

def load_checkpoint(model: torch.nn.Module, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 兼容不同保存格式
    state = None
    for k in ["state_dict", "model_state_dict", "model", "net", "network"]:
        if isinstance(ckpt, dict) and k in ckpt:
            state = ckpt[k]
            break
    if state is None:
        state = ckpt  # 可能直接就是state_dict

    # 兼容 DataParallel 的 module.
    new_state = {}
    if isinstance(state, dict):
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
    else:
        raise RuntimeError("checkpoint里没找到可用的state_dict格式")

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[INFO] Loaded ckpt: {ckpt_path.name}")
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")


# ===================== 推理与保存 =====================
def infer_one_dir(
    model: torch.nn.Module,
    img_dir: Path,
    lbl_dir: Path,
    out_dir: Path,
    target_name: str,
    mean, std,
    device: torch.device,
    max_images=None
):
    assert target_name in ("aeroplane", "sofa")
    target_voc_id = VOC_ID[target_name]

    _safe_mkdir(out_dir)

    img_paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    if max_images is not None:
        img_paths = img_paths[:max_images]

    print(f"[INFO] Inference on {target_name}: {len(img_paths)} images")
    model.eval()

    with torch.no_grad():
        for ip in img_paths:
            rgb = Image.open(ip).convert("RGB")
            x = _to_tensor(rgb, mean, std).unsqueeze(0).to(device)

            # pad到stride倍数，避免某些deeplab实现要求输入可整除
            x_pad, pad_info = _pad_to_stride(x, stride=32)

            out = model(x_pad)
            # 兼容 dict/tuple 输出
            if isinstance(out, dict):
                if "logits" in out:
                    logits = out["logits"]
                elif "out" in out:
                    logits = out["out"]
                else:
                    logits = next(iter(out.values()))
            elif isinstance(out, (list, tuple)):
                logits = out[0]
            else:
                logits = out

            # 插值回pad后的尺寸（有些模型输出会小一截）
            if logits.shape[-2:] != x_pad.shape[-2:]:
                logits = F.interpolate(logits, size=x_pad.shape[-2:], mode="bilinear", align_corners=False)

            pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)  # HxW（pad后）
            pred = _unpad_mask(pred, pad_info)  # HxW（原图尺寸）

            # --- 1) 保存全类别预测（VOC调色板风格） ---
            # 注意：这里直接保存“模型输出的类别id”。如果你的模型输出就是VOC 0..20，这张图就和GT完全同风格。
            stem = ip.stem
            _save_palette_png(pred, out_dir / f"{stem}_pred_voc.png")

            # --- 2) 保存当前类的二值预测 ---
            pred_bin = (pred == target_voc_id).astype(np.uint8)
            _save_binary_png(pred_bin, out_dir / f"{stem}_pred_binary.png")

            # --- 3) overlay ---
            _save_overlay(rgb, pred_bin, out_dir / f"{stem}_overlay.jpg")

            # --- 4) 可选：简单对比图（原图 / GT二值 / Pred二值） ---
            if SAVE_COMPARE:
                gt_path = lbl_dir / f"{stem}.png"
                if gt_path.exists():
                    gt = np.array(Image.open(gt_path))
                    gt_bin = (gt == target_voc_id).astype(np.uint8)

                    # 转成可视化RGB
                    gt_vis = Image.fromarray((gt_bin * 255).astype(np.uint8)).convert("RGB")
                    pr_vis = Image.fromarray((pred_bin * 255).astype(np.uint8)).convert("RGB")
                    cmp = _hstack_images([rgb, gt_vis, pr_vis])
                    cmp.save(out_dir / f"{stem}_compare.jpg")

            print(f"[OK] {target_name}: {ip.name} -> saved")


def main():
    # 把项目根目录塞进sys.path，保证能import到你的models代码
    # /media/wyh/star/scripts/voc/predict_two_classes_voc.py -> parents[2] = /media/wyh/star
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

    if not CFG_PATH.exists():
        raise FileNotFoundError(f"config not found: {CFG_PATH}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"ckpt not found: {CKPT_PATH}")

    cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))

    # 常见的VOC/imagenet归一化（config里没给就用这个）
    mean = cfg.get("mean", [0.485, 0.456, 0.406])
    std  = cfg.get("std",  [0.229, 0.224, 0.225])

    device = torch.device(DEVICE if (DEVICE.startswith("cuda") and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] device = {device}")

    # build model
    model = build_model_from_config(cfg, project_root).to(device)

    # load checkpoint
    load_checkpoint(model, CKPT_PATH)

    # sanity check dirs
    for p in [AIR_IMG_DIR, AIR_LBL_DIR, SOFA_IMG_DIR, SOFA_LBL_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"dir not found: {p}")

    # infer aeroplane set
    infer_one_dir(
        model=model,
        img_dir=AIR_IMG_DIR,
        lbl_dir=AIR_LBL_DIR,
        out_dir=AIR_OUT_DIR,
        target_name="aeroplane",
        mean=mean, std=std,
        device=device,
        max_images=MAX_IMAGES_EACH
    )

    # infer sofa set
    infer_one_dir(
        model=model,
        img_dir=SOFA_IMG_DIR,
        lbl_dir=SOFA_LBL_DIR,
        out_dir=SOFA_OUT_DIR,
        target_name="sofa",
        mean=mean, std=std,
        device=device,
        max_images=MAX_IMAGES_EACH
    )

    print("[DONE] All predictions saved.")
    print(f" - aeroplane preds: {AIR_OUT_DIR}")
    print(f" - sofa preds:      {SOFA_OUT_DIR}")


if __name__ == "__main__":
    main()
