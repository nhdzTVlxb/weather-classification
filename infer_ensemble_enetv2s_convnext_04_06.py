import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

# ============================================================
# Ensemble infer: EfficientNetV2-S + ConvNeXt-Tiny
# Weight: 0.4 * EfficientNetV2-S + 0.6 * ConvNeXt-Tiny
# Input: X is a BGR image read by cv2
# Output: class name string
# ============================================================

IM_SIZE = 384
DEFAULT_LABELS = ["cloudy", "rainy", "snowy", "sunny"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()

# 你当前最优配置：resize + 多尺度 TTA，不做水平翻转
PREPROCESS_MODE = "resize"  # "resize" or "crop"
USE_MS_TTA = True
MS_TTA_SIZES = [352, 384, 416]
USE_FLIP_TTA = False

# 集成权重
ENET_WEIGHT = 0.4
CV_WEIGHT = 0.6

# 两个权重文件名：按你说的命名
ENET_CKPT_NAME = "best_f1_enetv2_s.pth"
CV_CKPT_NAME = "best_f1_cv.pth"

# 如果平台要求文件放在 results 或 code_mo，也能自动找
ENET_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, ENET_CKPT_NAME),
    os.path.join(BASE_DIR, "results", ENET_CKPT_NAME),
    os.path.join(BASE_DIR, "code_mo", ENET_CKPT_NAME),
    os.path.join("/home/wdy/cyp/weather/enetv2_s_ms_tta_v2", ENET_CKPT_NAME),
    os.path.join("/home/wdy/cyp/weather/enetv2_s_ms_tta_v2", "best_f1.pth"),
    os.path.join("/home/cyp/speedsci/weather/enetv2_s_ms_tta_v2", ENET_CKPT_NAME),
    os.path.join("/home/cyp/speedsci/weather/enetv2_s_ms_tta_v2", "best_f1.pth"),
]

CV_CANDIDATE_PATHS = [
    os.path.join(BASE_DIR, CV_CKPT_NAME),
    os.path.join(BASE_DIR, "results", CV_CKPT_NAME),
    os.path.join(BASE_DIR, "code_mo", CV_CKPT_NAME),
    os.path.join("/home/wdy/cyp/weather/convnext_tiny_ms_tta_v2", CV_CKPT_NAME),
    os.path.join("/home/wdy/cyp/weather/convnext_tiny_ms_tta_v2", "best_f1.pth"),
    os.path.join("/home/cyp/speedsci/weather/convnext_tiny_ms_tta_v2", CV_CKPT_NAME),
    os.path.join("/home/cyp/speedsci/weather/convnext_tiny_ms_tta_v2", "best_f1.pth"),
]


def _load_checkpoint(candidate_paths, tag):
    last_error = None
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        try:
            ckpt = torch.load(path, map_location=DEVICE)
            print(f"Loaded {tag} checkpoint: {path}")
            return ckpt, path
        except Exception as e:
            last_error = e
    raise RuntimeError(f"No valid {tag} checkpoint found. Last error: {last_error}")


def _clean_state_dict(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v
    return new_state


def _extract_ckpt(ckpt):
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        class_to_idx = ckpt.get("class_to_idx", None)
        args = ckpt.get("args", {}) or {}
    else:
        state_dict = ckpt
        class_to_idx = None
        args = {}

    if class_to_idx:
        idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
        labels = [idx_to_class[i] for i in range(len(idx_to_class))]
    else:
        labels = DEFAULT_LABELS

    return _clean_state_dict(state_dict), labels, args


def _build_enetv2_s(num_classes=4, dropout=0.35):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def _build_convnext_tiny(num_classes=4, dropout=0.30):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def _load_enet_model():
    ckpt, _ = _load_checkpoint(ENET_CANDIDATE_PATHS, "EfficientNetV2-S")
    state_dict, labels, args = _extract_ckpt(ckpt)
    dropout = float(args.get("dropout", 0.35))

    model = _build_enetv2_s(num_classes=len(labels), dropout=dropout).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, labels


def _load_cv_model():
    ckpt, _ = _load_checkpoint(CV_CANDIDATE_PATHS, "ConvNeXt-Tiny")
    state_dict, labels, args = _extract_ckpt(ckpt)
    dropout = float(args.get("dropout", 0.30))

    model = _build_convnext_tiny(num_classes=len(labels), dropout=dropout).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, labels


# 加载模型。提交平台通常会 import 一次，然后反复调用 predict(X)
enet_model, enet_labels = _load_enet_model()
cv_model, cv_labels = _load_cv_model()

# 输出标签顺序优先使用 enet 的 class_to_idx；正常情况下两者都是 cloudy/rainy/snowy/sunny
label = enet_labels

if set(enet_labels) != set(cv_labels):
    raise RuntimeError(f"Label mismatch: enet={enet_labels}, convnext={cv_labels}")


def _resize_full_rgb(img_rgb, size):
    return cv2.resize(img_rgb, (size, size), interpolation=cv2.INTER_LINEAR)


def _resize_center_crop_rgb(img_rgb, size):
    h, w = img_rgb.shape[:2]
    short_side = int(size * 1.14)
    if h < w:
        new_h = short_side
        new_w = int(w * short_side / h)
    else:
        new_w = short_side
        new_h = int(h * short_side / w)
    img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top = max((new_h - size) // 2, 0)
    left = max((new_w - size) // 2, 0)
    return img_rgb[top: top + size, left: left + size]


def _to_tensor(img_rgb):
    img = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
    return torch.from_numpy(img).float().to(DEVICE)


def _preprocess_bgr_image(X, size=384, hflip=False):
    img_rgb = cv2.cvtColor(X, cv2.COLOR_BGR2RGB)

    if PREPROCESS_MODE == "resize":
        img_rgb = _resize_full_rgb(img_rgb, size)
    else:
        img_rgb = _resize_center_crop_rgb(img_rgb, size)

    if hflip:
        img_rgb = np.ascontiguousarray(img_rgb[:, ::-1, :])

    return _to_tensor(img_rgb)


@torch.no_grad()
def _predict_prob_single_model(model, X):
    sizes = MS_TTA_SIZES if USE_MS_TTA else [IM_SIZE]
    logits_sum = None
    count = 0

    for size in sizes:
        tensor = _preprocess_bgr_image(X, size=size, hflip=False)
        logits = model(tensor)
        logits_sum = logits if logits_sum is None else logits_sum + logits
        count += 1

        if USE_FLIP_TTA:
            tensor_flip = _preprocess_bgr_image(X, size=size, hflip=True)
            logits_sum = logits_sum + model(tensor_flip)
            count += 1

    logits_avg = logits_sum / count
    prob = torch.softmax(logits_avg.float(), dim=1)[0]
    return prob


def _align_prob(prob, src_labels, dst_labels):
    """Align probability order from src_labels to dst_labels."""
    if src_labels == dst_labels:
        return prob
    src_index = {name: i for i, name in enumerate(src_labels)}
    aligned = torch.zeros(len(dst_labels), device=prob.device, dtype=prob.dtype)
    for j, name in enumerate(dst_labels):
        aligned[j] = prob[src_index[name]]
    return aligned


@torch.no_grad()
def predict(X):
    """
    X: BGR image, usually loaded by cv2.imread or platform input.
    return: one of [cloudy, rainy, snowy, sunny]
    """
    enet_prob = _predict_prob_single_model(enet_model, X)
    cv_prob = _predict_prob_single_model(cv_model, X)

    enet_prob = _align_prob(enet_prob, enet_labels, label)
    cv_prob = _align_prob(cv_prob, cv_labels, label)

    final_prob = ENET_WEIGHT * enet_prob + CV_WEIGHT * cv_prob
    pred_idx = int(torch.argmax(final_prob).item())
    return label[pred_idx]
