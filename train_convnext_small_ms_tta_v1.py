#!/usr/bin/env python3
import os
import math
import time
import random
import argparse
from copy import deepcopy

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import timm


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConvNeXtSmallWeather(nn.Module):
    def __init__(
        self,
        model_name="convnext_small.fb_in22k_ft_in1k",
        num_classes=4,
        dropout=0.35,
        pretrained=True,
        drop_path_rate=0.10,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=drop_path_rate,
        )

        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)


class ModelEMA:
    def __init__(self, model, decay=0.9995):
        self.ema = deepcopy(model).eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.copy_(v * self.decay + msd[k].detach() * (1.0 - self.decay))
            else:
                v.copy_(msd[k])


def build_transforms(image_size=384, val_preprocess="resize"):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.20,
                hue=0.03,
            )
        ], p=0.7),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])

    if val_preprocess == "crop":
        val_tf = transforms.Compose([
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        val_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return train_tf, val_tf


def build_loader(args):
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")

    train_tf, val_tf = build_transforms(
        image_size=args.image_size,
        val_preprocess=args.val_preprocess,
    )

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader, train_set.class_to_idx


def macro_f1_score(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1s = []

    for c in range(num_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)

        f1 = 2 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)

    return float(np.mean(f1s))


def accuracy_score(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


def cosine_lr(optimizer, base_lr, min_lr, epoch, total_epochs, warmup_epochs):
    if epoch < warmup_epochs:
        lr = base_lr * float(epoch + 1) / float(max(1, warmup_epochs))
    else:
        t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))

    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)

    return lr


def train_one_epoch(model, loader, criterion, optimizer, scaler, ema, device, args):
    model.train()

    total_loss = 0.0
    total_num = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_num += bs

    return total_loss / max(1, total_num)


@torch.no_grad()
def validate_single(model, loader, criterion, device, args):
    model.eval()

    total_loss = 0.0
    total_num = 0

    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, targets)

        preds = torch.argmax(logits, dim=1)

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_num += bs

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    f1 = macro_f1_score(all_targets, all_preds, args.num_classes)

    return total_loss / max(1, total_num), acc, f1


@torch.no_grad()
def validate_ms_tta(model, loader, criterion, device, args):
    model.eval()

    if args.val_tta == "none":
        return validate_single(model, loader, criterion, device, args)

    all_targets = []
    all_preds = []

    total_loss = 0.0
    total_num = 0

    for images, targets in loader:
        targets = targets.to(device, non_blocking=True)

        logits_sum = None
        count = 0

        for size in args.tta_sizes:
            imgs = torch.nn.functional.interpolate(
                images,
                size=(size, size),
                mode="bilinear",
                align_corners=False,
            ).to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(imgs)

            logits_sum = logits if logits_sum is None else logits_sum + logits
            count += 1

        logits_avg = logits_sum / float(count)

        loss = criterion(logits_avg, targets)
        preds = torch.argmax(logits_avg, dim=1)

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_num += bs

        all_targets.extend(targets.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    f1 = macro_f1_score(all_targets, all_preds, args.num_classes)

    return total_loss / max(1, total_num), acc, f1


def save_checkpoint(path, model, class_to_idx, args, epoch, val_acc, val_f1):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "args": vars(args),
        "epoch": epoch,
        "val_acc": val_acc,
        "val_f1": val_f1,
        "model_name": args.model_name,
    }

    torch.save(ckpt, path)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model-name", type=str, default="convnext_small.fb_in22k_ft_in1k")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-epochs", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--drop-path", type=float, default=0.10)
    parser.add_argument("--label-smoothing", type=float, default=0.05)

    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--ema-decay", type=float, default=0.9995)

    parser.add_argument("--val-preprocess", type=str, default="resize", choices=["resize", "crop"])
    parser.add_argument("--val-tta", type=str, default="ms", choices=["none", "ms"])
    parser.add_argument("--tta-sizes", type=str, default="352,384,416")

    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--no-amp", action="store_true")

    args = parser.parse_args()

    args.amp = (not args.no_amp) and torch.cuda.is_available()
    args.tta_sizes = [int(x) for x in args.tta_sizes.split(",") if x.strip()]

    seed_everything(args.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        device = torch.device(f"cuda:{args.gpu}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader, val_loader, class_to_idx = build_loader(args)
    args.num_classes = len(class_to_idx)

    print("Device:", device)
    print("Classes:", class_to_idx)
    print("Train size:", len(train_loader.dataset))
    print("Val size:", len(val_loader.dataset))
    print("Model:", args.model_name)
    print("Val preprocess:", args.val_preprocess)
    print("Val TTA:", args.val_tta, args.tta_sizes if args.val_tta == "ms" else "")

    model = ConvNeXtSmallWeather(
        model_name=args.model_name,
        num_classes=args.num_classes,
        dropout=args.dropout,
        pretrained=True,
        drop_path_rate=args.drop_path,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr_scale": 1.0,
            },
            {
                "params": model.classifier.parameters(),
                "lr_scale": 3.0,
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    best_acc = 0.0
    best_f1 = 0.0
    best_epoch = -1
    no_improve = 0

    for epoch in range(args.epochs):
        start = time.time()

        lr = cosine_lr(
            optimizer,
            base_lr=args.lr,
            min_lr=args.min_lr,
            epoch=epoch,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
        )

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            device=device,
            args=args,
        )

        eval_model = ema.ema if ema is not None else model

        val_loss, val_acc, val_f1 = validate_ms_tta(
            model=eval_model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            args=args,
        )

        improved = False

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                os.path.join(args.output_dir, "best_acc.pth"),
                eval_model,
                class_to_idx,
                args,
                epoch,
                val_acc,
                val_f1,
            )
            improved = True

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                os.path.join(args.output_dir, "best_f1.pth"),
                eval_model,
                class_to_idx,
                args,
                epoch,
                val_acc,
                val_f1,
            )
            improved = True
        else:
            no_improve += 1

        save_checkpoint(
            os.path.join(args.output_dir, "last.pth"),
            eval_model,
            class_to_idx,
            args,
            epoch,
            val_acc,
            val_f1,
        )

        elapsed = time.time() - start

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"lr={lr:.2e} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"acc={val_acc:.5f} | "
            f"f1={val_f1:.5f} | "
            f"best_f1={best_f1:.5f} | "
            f"time={elapsed:.1f}s"
        )

        if no_improve >= args.patience:
            print(f"Early stopping. Best epoch={best_epoch + 1}, best_f1={best_f1:.5f}")
            break

    print("Training done.")
    print("Best ACC:", best_acc)
    print("Best F1:", best_f1)


if __name__ == "__main__":
    main()
