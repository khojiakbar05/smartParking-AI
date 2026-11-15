
import cv2
import torch
import time
from ultralytics import YOLO

# 1. AI model
model = YOLO("yolo11n.pt")

# 2. Apple Silicon MPS mavjud bo'lsa undan foydalanamiz
device = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"AI device: {device}")

# 3. Faqat transport vositalari
# COCO class IDs:
# 2 = car
# 3 = motorcycle
# 5 = bus
# 7 = truck
vehicle_classes = [2, 3, 5, 7]

# 4. Laptop camera
camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    raise RuntimeError("Laptop kamerasi ochilmadi!")

previous_time = time.time()

print("SmartParking AI ishga tushdi. Chiqish uchun Q bosing.")

while True:
    success, frame = camera.read()

    if not success:
        print("Frame olinmadi.")
        break

    # 5. AI detection
    results = model.predict(
        frame,
        imgsz=640,
        conf=0.35,
        classes=vehicle_classes,
        device=device,
        verbose=False
    )

    # 6. Detection natijalarini tasvirga chiqarish
    annotated_frame = results[0].plot()

    # 7. Aniqlangan transport vositalari soni
    vehicle_count = len(results[0].boxes)

    # 8. FPS hisoblash
    current_time = time.time()
    fps = 1 / max(current_time - previous_time, 0.001)
    previous_time = current_time

    cv2.putText(
        annotated_frame,
        f"Vehicles: {vehicle_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.1f}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("SmartParking AI - Vehicle Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
