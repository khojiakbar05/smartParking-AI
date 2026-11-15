
import cv2
import json
import torch
import numpy as np

from ultralytics import YOLO

# 1. YOLO model
model = YOLO("yolo11n.pt")

device = "mps" if torch.backends.mps.is_available() else "cpu"

# 2. Parking joylarini yuklash
with open("parking_slots.json", "r") as file:
    parking_data = json.load(file)

slots = parking_data["slots"]

original_width = parking_data["frame_width"]
original_height = parking_data["frame_height"]

# 3. Camera
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    raise RuntimeError("Kamera ochilmadi!")

vehicle_classes = [2, 3, 5, 7]

# Bandlik chegarasi.
# Slot maydonining kamida 20 foizi avtomobil bbox'i bilan
# kesishsa, dastlab band deb hisoblaymiz.
OCCUPANCY_THRESHOLD = 0.20

while True:
    success, frame = camera.read()

    if not success:
        break

    height, width = frame.shape[:2]

    scale_x = width / original_width
    scale_y = height / original_height

    # 4. YOLO detection
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.35,
        classes=vehicle_classes,
        device=device,
        verbose=False
    )

    boxes = results[0].boxes

    vehicle_boxes = []

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            vehicle_boxes.append([
                int(x1), int(y1), int(x2), int(y2)
            ])

    available_count = 0
    occupied_count = 0

    # 5. Har bir parking joyini tekshirish
    for slot in slots:
        points = np.array([
            [
                int(point[0] * scale_x),
                int(point[1] * scale_y)
            ]
            for point in slot["points"]
        ], dtype=np.int32)

        # Parking polygon mask
        slot_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(slot_mask, [points], 255)

        slot_area = cv2.countNonZero(slot_mask)

        occupied = False

        for x1, y1, x2, y2 in vehicle_boxes:
            # Avtomobil bbox mask
            vehicle_mask = np.zeros(
                (height, width), dtype=np.uint8
            )

            cv2.rectangle(
                vehicle_mask,
                (x1, y1),
                (x2, y2),
                255,
                -1
            )

            # Ikki mask kesishmasi
            intersection = cv2.bitwise_and(
                slot_mask,
                vehicle_mask
            )

            intersection_area = cv2.countNonZero(intersection)

            if slot_area > 0:
                overlap_ratio = intersection_area / slot_area

                if overlap_ratio >= OCCUPANCY_THRESHOLD:
                    occupied = True
                    break

        # 6. Rang va status
        if occupied:
            color = (0, 0, 255)
            status = "OCCUPIED"
            occupied_count += 1
        else:
            color = (0, 255, 0)
            status = "AVAILABLE"
            available_count += 1

        cv2.polylines(frame, [points], True, color, 3)

        center = np.mean(points, axis=0).astype(int)

        cv2.putText(
            frame,
            f"{slot['id']} {status}",
            tuple(center),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # 7. Avtomobillarni ham ko'rsatish
    for x1, y1, x2, y2 in vehicle_boxes:
        cv2.rectangle(
            frame, (x1, y1), (x2, y2),
            (255, 200, 0), 2
        )

    # 8. Umumiy statistika
    cv2.rectangle(frame, (10, 10), (390, 105), (30, 30, 30), -1)

    cv2.putText(
        frame,
        f"Available: {available_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 255, 0), 2
    )

    cv2.putText(
        frame,
        f"Occupied: {occupied_count}",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (0, 0, 255), 2
    )

    cv2.imshow("SmartParking AI - Parking Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
