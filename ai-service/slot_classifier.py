"""Replaceable per-slot classifier, confidence gate, and UNKNOWN-aware smoother."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import torch


class SlotStatus(str, Enum):
    OCCUPIED = "OCCUPIED"
    VACANT = "VACANT"
    UNKNOWN = "UNKNOWN"


@dataclass
class SlotPrediction:
    slot_id: str
    status: SlotStatus
    confidence: float
    timestamp: str


def utc_timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def build_classifier_model(architecture, num_classes=2, pretrained=False):
    """Build a lightweight torchvision classifier; pretrained download is opt-in."""
    from torchvision import models
    from torch import nn

    if architecture == "mobilenet_v3_small":
        weights = (models.MobileNet_V3_Small_Weights.DEFAULT
                   if pretrained else None)
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported classifier architecture: {architecture}")
    return model


def preprocess_crop(image, image_size=224):
    """Convert BGR crop to ImageNet-normalized RGB tensor using contain padding."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Slot crop must be a 3-channel BGR image.")
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("Slot crop cannot be empty.")
    scale = min(image_size / width, image_size / height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    left = (image_size - new_width) // 2
    top = (image_size - new_height) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


class SlotClassifier:
    """Interface implemented by future CNNs or an external classifier."""

    def classify_batch(self, crops, timestamp=None):
        raise NotImplementedError


class UnknownSlotClassifier(SlotClassifier):
    """Safe no-model behavior: never invents an occupancy label."""

    def classify_batch(self, crops, timestamp=None):
        timestamp = timestamp or utc_timestamp()
        return {
            crop.slot_id: SlotPrediction(
                crop.slot_id, SlotStatus.UNKNOWN, 0.0, timestamp
            )
            for crop in crops
        }


class TorchSlotClassifier(SlotClassifier):
    """Load an app-local checkpoint produced by train_slot_classifier.py."""

    def __init__(self, checkpoint_path, device=None, confidence_threshold=None):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"Slot classifier checkpoint not found: {self.checkpoint_path}")
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        checkpoint = torch.load(
            self.checkpoint_path, map_location="cpu", weights_only=True
        )
        self.architecture = checkpoint["architecture"]
        self.class_names = checkpoint["class_names"]
        if self.class_names != ["VACANT", "OCCUPIED"]:
            raise ValueError("Checkpoint classes must be ordered VACANT, OCCUPIED.")
        self.image_size = int(checkpoint.get("image_size", 224))
        self.confidence_threshold = float(
            confidence_threshold if confidence_threshold is not None
            else checkpoint.get("confidence_threshold", 0.75)
        )
        if not 0.5 <= self.confidence_threshold <= 1.0:
            raise ValueError("Classifier confidence threshold must be in [0.5, 1.0].")
        self.model = build_classifier_model(self.architecture, num_classes=2)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def classify_batch(self, crops, timestamp=None):
        timestamp = timestamp or utc_timestamp()
        if not crops:
            return {}
        batch = torch.stack([
            preprocess_crop(crop.image, self.image_size) for crop in crops
        ]).to(self.device)
        probabilities = torch.softmax(self.model(batch), dim=1).detach().cpu().numpy()
        predictions = {}
        for crop, probs in zip(crops, probabilities):
            index = int(np.argmax(probs))
            confidence = float(probs[index])
            status = (SlotStatus(self.class_names[index])
                      if confidence >= self.confidence_threshold
                      else SlotStatus.UNKNOWN)
            predictions[crop.slot_id] = SlotPrediction(
                crop.slot_id, status, confidence, timestamp
            )
        return predictions


def fuse_classifier_and_vehicle(prediction, vehicle_evidence,
                                min_vehicle_confidence=0.60,
                                min_overlap_ratio=0.20):
    """Use vehicle boxes only to support UNKNOWN; never overturn a confident crop."""
    if (prediction.status != SlotStatus.UNKNOWN
            and prediction.confidence >= 0.75):
        return prediction
    supporting_confidence = [
        float(item["confidence"])
        for item in vehicle_evidence
        if float(item["confidence"]) >= min_vehicle_confidence
        and float(item["ratio"]) >= min_overlap_ratio
    ]
    if supporting_confidence:
        return SlotPrediction(
            prediction.slot_id,
            SlotStatus.OCCUPIED,
            min(max(supporting_confidence), 0.99),
            prediction.timestamp,
        )
    return SlotPrediction(
        prediction.slot_id, SlotStatus.UNKNOWN,
        prediction.confidence, prediction.timestamp,
    )


class SlotStatusSmoother:
    """Majority smoothing that keeps weak or conflicting history UNKNOWN."""

    def __init__(self, slot_ids, window_size=5, required_votes=3):
        if window_size < 1 or not 1 <= required_votes <= window_size:
            raise ValueError("Smoothing window/votes are invalid.")
        self.required_votes = required_votes
        self.history = {
            slot_id: deque(maxlen=window_size) for slot_id in slot_ids
        }

    def update(self, predictions, timestamp=None):
        timestamp = timestamp or utc_timestamp()
        output = {}
        for slot_id, history in self.history.items():
            current = predictions.get(slot_id)
            if current is None:
                current = SlotPrediction(slot_id, SlotStatus.UNKNOWN, 0.0, timestamp)
            history.append(current)
            known = [item for item in history if item.status != SlotStatus.UNKNOWN]
            counts = {
                status: sum(item.status == status for item in known)
                for status in (SlotStatus.OCCUPIED, SlotStatus.VACANT)
            }
            winning_status = max(counts, key=counts.get)
            other_status = (SlotStatus.VACANT if winning_status == SlotStatus.OCCUPIED
                            else SlotStatus.OCCUPIED)
            if (counts[winning_status] >= self.required_votes
                    and counts[winning_status] > counts[other_status]):
                votes = [item for item in known if item.status == winning_status]
                confidence = sum(item.confidence for item in votes) / len(votes)
                status = winning_status
            else:
                status, confidence = SlotStatus.UNKNOWN, 0.0
            output[slot_id] = SlotPrediction(
                slot_id, status, confidence, timestamp
            )
        return output
