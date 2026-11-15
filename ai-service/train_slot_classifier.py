"""Train a separate MobileNetV3-small or EfficientNet-B0 slot classifier."""

import argparse
from collections import Counter
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from slot_classifier import build_classifier_model, preprocess_crop


CLASS_NAMES = ["VACANT", "OCCUPIED"]


class LabeledSlotCrops(Dataset):
    def __init__(self, entries, image_size):
        self.entries = entries
        self.image_size = image_size

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, label = self.entries[index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Unreadable training crop: {path}")
        return preprocess_crop(image, self.image_size), label


def split_samples(data_dir, seed):
    generator = random.Random(seed)
    train_entries, validation_entries = [], []
    counts = {}
    for label_index, label in enumerate(("vacant", "occupied")):
        files = sorted((data_dir / label).glob("*.jpg"))
        files += sorted((data_dir / label).glob("*.jpeg"))
        if len(files) < 5:
            raise ValueError(
                f"Need at least 5 manually labeled {label.upper()} crops; found {len(files)}. "
                "Use collect_slot_samples.py first."
            )
        generator.shuffle(files)
        validation_count = max(1, round(len(files) * 0.2))
        validation_entries.extend((path, label_index) for path in files[:validation_count])
        train_entries.extend((path, label_index) for path in files[validation_count:])
        counts[label] = len(files)
    generator.shuffle(train_entries)
    generator.shuffle(validation_entries)
    return train_entries, validation_entries, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).with_name("dataset") / "slot_samples")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("models") / "slot_classifier.pt")
    parser.add_argument("--architecture", choices=("mobilenet_v3_small", "efficientnet_b0"),
                        default="mobilenet_v3_small")
    parser.add_argument("--pretrained", action="store_true",
                        help="Download torchvision ImageNet weights for transfer learning.")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.image_size < 32:
        parser.error("epochs, batch-size, and image-size must be positive")
    if not 0.5 <= args.confidence_threshold <= 1.0:
        parser.error("confidence-threshold must be between 0.5 and 1.0")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_entries, validation_entries, counts = split_samples(args.data_dir, args.seed)
    print(f"Manual samples by class: {counts}")
    print(f"Train={len(train_entries)} validation={len(validation_entries)}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training device: {device}; architecture: {args.architecture}")

    train_loader = DataLoader(
        LabeledSlotCrops(train_entries, args.image_size),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    validation_loader = DataLoader(
        LabeledSlotCrops(validation_entries, args.image_size),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    model = build_classifier_model(
        args.architecture, num_classes=len(CLASS_NAMES), pretrained=args.pretrained
    )
    if args.freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
    model.to(device)

    train_counts = Counter(label for _, label in train_entries)
    class_weights = torch.tensor([
        len(train_entries) / (2 * train_counts[index]) for index in range(2)
    ], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
    )

    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach().cpu()) * len(labels)

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in validation_loader:
                logits = model(images.to(device))
                correct += int((logits.argmax(1).cpu() == labels).sum())
                total += len(labels)
        accuracy = correct / max(total, 1)
        mean_loss = train_loss / max(len(train_entries), 1)
        print(
            f"epoch {epoch:03d}/{args.epochs}: "
            f"train_loss={mean_loss:.4f} validation_accuracy={accuracy:.3f}"
        )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "architecture": args.architecture,
        "class_names": CLASS_NAMES,
        "image_size": args.image_size,
        "confidence_threshold": args.confidence_threshold,
        "state_dict": best_state,
        "validation_accuracy": best_accuracy,
        "manual_sample_counts": counts,
        "seed": args.seed,
    }, args.output)
    print(f"Best validation accuracy: {best_accuracy:.3f}")
    print(f"Saved slot classifier: {args.output}")
    if device == "mps":
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
