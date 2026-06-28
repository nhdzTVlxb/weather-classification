#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import ImageFile
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

BACKBONE_NAME = "convnext_tiny"
DEFAULT_OUTPUT_DIR = "/home/wdy/cyp/weather/convnext_tiny_ms_tta_v2"
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-4
DEFAULT_BACKBONE_LR_MULT = 0.20
DEFAULT_DROPOUT = 0.30
DEFAULT_FREEZE_EPOCHS = 5
DEFAULT_EPOCHS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/wdy/cyp/weather/datanew")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=384)

    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--backbone-lr-mult", type=float, default=DEFAULT_BACKBONE_LR_MULT)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=DEFAULT_FREEZE_EPOCHS)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--use-ema", type=int, default=1)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--use-amp", type=int, default=1)

    # 正确的 TTA：默认多尺度 resize，不做水平翻转。
    # none: 单尺度；ms: 多尺度；flip: 单尺度+翻转；ms_flip: 多尺度+翻转。
    parser.add_argument("--val-tta", choices=["none", "ms", "flip", "ms_flip"], default="ms")
    parser.add_argument("--tta-sizes", default="352,384,416")
    parser.add_argument("--val-preprocess", choices=["resize", "crop"], default="resize")

    parser.add_argument("--use-class-weight", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    return parser.parse_args()


def parse_tta_sizes(s: str):
    sizes = []
    for x in s.split(','):
        x = x.strip()
        if x:
            sizes.append(int(x))
    return sizes or [384]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(image_size):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02)
        ], p=0.6),
        transforms.RandomRotation(degrees=8),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.08, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
    ])


def build_eval_transform(size, preprocess="resize", hflip=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ops = []
    if preprocess == "resize":
        ops.append(transforms.Resize((size, size)))
    else:
        ops.extend([transforms.Resize(int(size * 1.14)), transforms.CenterCrop(size)])
    if hflip:
        ops.append(transforms.Lambda(lambda img: img.transpose(0)))  # PIL.Image.FLIP_LEFT_RIGHT == 0
    ops.extend([transforms.ToTensor(), normalize])
    return transforms.Compose(ops)


def val_collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.long)


def build_model(num_classes, dropout):
    weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    model = models.convnext_tiny(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = nn.Sequential(
        model.classifier[0],
        model.classifier[1],
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model, trainable):
    for p in model.features.parameters():
        p.requires_grad_(trainable)
    for p in model.classifier.parameters():
        p.requires_grad_(True)


def make_loader(dataset, batch_size, shuffle, workers, use_cuda, collate_fn=None):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": use_cuda,
        "collate_fn": collate_fn,
    }
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)


def get_class_weights(dataset, device):
    counts = np.bincount(dataset.targets, minlength=len(dataset.classes)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        esd = self.ema.state_dict()
        for k, ema_v in esd.items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))
            else:
                ema_v.copy_(model_v)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, ema, use_amp, grad_clip, freeze_backbone):
    model.train()
    if freeze_backbone:
        model.features.eval()

    total_loss = 0.0
    y_true, y_pred = [], []

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred)


@torch.no_grad()
def validate(model, loader, criterion, device, class_names, use_amp, val_tta, tta_sizes, val_preprocess):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    use_ms = val_tta in {"ms", "ms_flip"}
    use_flip = val_tta in {"flip", "ms_flip"}
    sizes = tta_sizes if use_ms else [tta_sizes[len(tta_sizes) // 2]]

    transform_cache = {}
    def get_tfm(size, flip):
        key = (size, flip)
        if key not in transform_cache:
            transform_cache[key] = build_eval_transform(size, preprocess=val_preprocess, hflip=flip)
        return transform_cache[key]

    for pil_images, labels in tqdm(loader, desc="val", leave=False):
        labels = labels.to(device, non_blocking=True)
        logits_sum = None
        count = 0

        for size in sizes:
            batch = torch.stack([get_tfm(size, False)(img) for img in pil_images], dim=0)
            batch = batch.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(batch)
            logits_sum = logits if logits_sum is None else logits_sum + logits
            count += 1

            if use_flip:
                batch_flip = torch.stack([get_tfm(size, True)(img) for img in pil_images], dim=0)
                batch_flip = batch_flip.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                    logits_flip = model(batch_flip)
                logits_sum += logits_flip
                count += 1

        logits_avg = logits_sum / count
        loss = criterion(logits_avg, labels)

        total_loss += loss.item() * labels.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(logits_avg.argmax(dim=1).detach().cpu().tolist())

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    return {
        "loss": total_loss / len(loader.dataset),
        "acc": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_checkpoint(path, model, epoch, metric_name, metric_value, class_to_idx, args):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "class_to_idx": class_to_idx,
        "args": vars(args),
        "backbone": BACKBONE_NAME,
    }, path)


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_amp = bool(args.use_amp) and use_cuda
    tta_sizes = parse_tta_sizes(args.tta_sizes)

    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Backbone: {BACKBONE_NAME}")
    print(f"Val preprocess={args.val_preprocess}, val_tta={args.val_tta}, tta_sizes={tta_sizes}")

    train_set = datasets.ImageFolder(train_dir, transform=build_train_transform(args.image_size))
    val_set = datasets.ImageFolder(val_dir, transform=None)
    class_names = train_set.classes

    print(f"Classes: {train_set.class_to_idx}")
    print(f"Dataset sizes: train={len(train_set)} val={len(val_set)}")
    print(f"Image size={args.image_size}, batch={args.batch_size}, EMA={args.ema_decay}")

    (output_dir / "class_to_idx.json").write_text(
        json.dumps(train_set.class_to_idx, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers, use_cuda)
    val_loader = make_loader(val_set, args.batch_size, False, args.num_workers, use_cuda, collate_fn=val_collate_fn)

    model = build_model(len(class_names), args.dropout).to(device).to(memory_format=torch.channels_last)

    if args.freeze_backbone_epochs > 0:
        set_backbone_trainable(model, False)
        print(f"Freeze features for first {args.freeze_backbone_epochs} epochs.")
    else:
        set_backbone_trainable(model, True)

    weight = get_class_weights(train_set, device) if bool(args.use_class_weight) else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)

    backbone_params, head_params = [], []
    for name, param in model.named_parameters():
        if name.startswith("classifier."):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
        {"params": head_params, "lr": args.lr},
    ], weight_decay=args.weight_decay)

    if args.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs),
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - args.warmup_epochs)),
            ],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = ModelEMA(model, decay=args.ema_decay) if bool(args.use_ema) else None

    best_acc, best_f1 = -math.inf, -math.inf
    bad_epochs = 0
    history = []
    backbone_unfrozen = args.freeze_backbone_epochs <= 0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        freeze_backbone = epoch <= args.freeze_backbone_epochs
        if freeze_backbone:
            set_backbone_trainable(model, False)
        elif not backbone_unfrozen:
            set_backbone_trainable(model, True)
            backbone_unfrozen = True
            print(f"Epoch {epoch}: unfreeze features.")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, ema, use_amp, args.grad_clip, freeze_backbone
        )
        eval_model = ema.ema if ema is not None else model
        val_metrics = validate(
            eval_model, val_loader, criterion, device, class_names, use_amp,
            args.val_tta, tta_sizes, args.val_preprocess,
        )

        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr_backbone": lr_backbone,
            "lr_head": lr_head,
            "seconds": time.time() - start,
            "ema": ema is not None,
            "val_tta": args.val_tta,
            "tta_sizes": tta_sizes,
            "val_preprocess": args.val_preprocess,
            "freeze_backbone": freeze_backbone,
        }
        history.append(row)
        (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        print(
            f"Epoch {epoch:03d}/{args.epochs} freeze={freeze_backbone} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_acc']:.4f} "
            f"val_f1={row['val_macro_f1']:.4f} "
            f"lr_b={lr_backbone:.2e} lr_h={lr_head:.2e} "
            f"time={row['seconds']:.1f}s"
        )

        improved = False
        if row["val_acc"] > best_acc:
            best_acc = row["val_acc"]
            improved = True
            save_checkpoint(output_dir / "best_acc.pth", eval_model, epoch, "val_acc", best_acc, train_set.class_to_idx, args)
            (output_dir / "best_acc_report.json").write_text(json.dumps(val_metrics["classification_report"], ensure_ascii=False, indent=2), encoding="utf-8")
            (output_dir / "best_acc_confusion_matrix.json").write_text(json.dumps(val_metrics["confusion_matrix"], ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved best ACC: {best_acc:.4f}")

        if row["val_macro_f1"] > best_f1:
            best_f1 = row["val_macro_f1"]
            improved = True
            save_checkpoint(output_dir / "best_f1.pth", eval_model, epoch, "val_macro_f1", best_f1, train_set.class_to_idx, args)
            (output_dir / "best_f1_report.json").write_text(json.dumps(val_metrics["classification_report"], ensure_ascii=False, indent=2), encoding="utf-8")
            (output_dir / "best_f1_confusion_matrix.json").write_text(json.dumps(val_metrics["confusion_matrix"], ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Saved best F1: {best_f1:.4f}")

        if improved:
            bad_epochs = 0
        else:
            bad_epochs += 1
            print(f"No improvement: {bad_epochs}/{args.patience}")

        if bad_epochs >= args.patience:
            print(f"Early stopping: no ACC/F1 improvement for {args.patience} epochs.")
            break

    print("-" * 60)
    print(f"Best validation ACC: {best_acc:.4f}")
    print(f"Best validation Macro-F1: {best_f1:.4f}")
    print(f"Best ACC checkpoint: {output_dir / 'best_acc.pth'}")
    print(f"Best F1 checkpoint: {output_dir / 'best_f1.pth'}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()
