import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import SpeakerGenderDataset, collate_fn
from model import GenderClassifier


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Mission 1: 신고자 성별 분류 학습 스크립트")
    p.add_argument("--train_audio_dir", type=str, required=True)
    p.add_argument("--train_label_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="../checkpoints")
    p.add_argument("--pretrained_model", type=str, default="facebook/wav2vec2-base")
    p.add_argument("--sample_rate", type=int, default=16000)
    p.add_argument("--max_duration_sec", type=float, default=6.0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_freeze_feature_extractor", action="store_true")
    p.add_argument("--region_filter", type=str, default="서울")
    p.add_argument("--time_unit", type=str, default="ms", choices=["ms", "s"])
    return p.parse_args()


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_values, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] 데이터셋 인덱싱 중 (서울 데이터만 필터링)...")
    dataset_train_mode = SpeakerGenderDataset(
        audio_dir=args.train_audio_dir,
        label_dir=args.train_label_dir,
        sample_rate=args.sample_rate,
        max_duration_sec=args.max_duration_sec,
        region_filter=args.region_filter,
        time_unit=args.time_unit,
        mode="train",
    )
    dataset_eval_mode = SpeakerGenderDataset(
        audio_dir=args.train_audio_dir,
        label_dir=args.train_label_dir,
        sample_rate=args.sample_rate,
        max_duration_sec=args.max_duration_sec,
        region_filter=args.region_filter,
        time_unit=args.time_unit,
        mode="val",
    )

    labels = [s["label"] for s in dataset_train_mode.samples]
    indices = list(range(len(dataset_train_mode)))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_ratio, random_state=args.seed, stratify=labels
    )
    print(f"[INFO] train={len(train_idx)}개, 내부 val={len(val_idx)}개")

    train_subset = Subset(dataset_train_mode, train_idx)
    val_subset = Subset(dataset_eval_mode, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    model = GenderClassifier(
        pretrained_model_name=args.pretrained_model,
        freeze_feature_extractor=not args.no_freeze_feature_extractor,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = -1.0
    best_ckpt_path = output_dir / "best_model.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_correct, running_count = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"[Epoch {epoch}/{args.epochs}]")
        for batch in pbar:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_values, attention_mask=attention_mask)
            loss = criterion(logits, labels_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * labels_t.size(0)
            preds = logits.argmax(dim=-1)
            running_correct += (preds == labels_t).sum().item()
            running_count += labels_t.size(0)
            pbar.set_postfix(
                loss=f"{running_loss / running_count:.4f}",
                acc=f"{running_correct / running_count:.4f}",
            )

        scheduler.step()
        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "pretrained_model_name": args.pretrained_model,
                        "sample_rate": args.sample_rate,
                        "max_duration_sec": args.max_duration_sec,
                        "time_unit": args.time_unit,
                        "num_classes": 2,
                    },
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )
            print(f"[INFO] 베스트 체크포인트 갱신 (val_acc={val_acc:.4f}) -> {best_ckpt_path}")

    with open(output_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"[DONE] 최고 내부 val_acc={best_val_acc:.4f}. 체크포인트: {best_ckpt_path}")


if __name__ == "__main__":
    main()
