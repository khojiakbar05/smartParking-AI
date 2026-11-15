
import cv2
import json
import numpy as np
from camera_devices import open_selected_camera

try:
    camera = open_selected_camera()
except RuntimeError as exc:
    raise SystemExit(str(exc))

slots = []
current_points = []
window_name = "SmartParking - Slot Editor"

def mouse_callback(event, x, y, flags, param):
    global current_points, slots

    if event == cv2.EVENT_LBUTTONDOWN:
        current_points.append([x, y])

        if len(current_points) == 4:
            slot_id = f"P{len(slots) + 1:02d}"

            slots.append({
                "id": slot_id,
                "points": current_points.copy()
            })

            print(f"{slot_id} belgilandi: {current_points}")
            current_points = []

cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

print("4 ta burchakni ketma-ket bosing.")
print("S = saqlash | U = oxirgi joyni o'chirish")
print("R = joriy nuqtalarni tozalash | Q = chiqish")

while True:
    success, frame = camera.read()

    if not success:
        break

    display = frame.copy()

    # Saqlangan parking joylari
    for slot in slots:
        points = slot["points"]
        pts = np.array(points, dtype=np.int32)

        cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        center_x = int(sum(p[0] for p in points) / 4)
        center_y = int(sum(p[1] for p in points) / 4)

        cv2.putText(
            display, slot["id"], (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0), 2
        )

    # Hozir belgilanayotgan nuqtalar
    for point in current_points:
        cv2.circle(display, tuple(point), 5, (0, 0, 255), -1)

    for i in range(1, len(current_points)):
        cv2.line(
            display,
            tuple(current_points[i - 1]),
            tuple(current_points[i]),
            (0, 0, 255), 2
        )

    cv2.putText(
        display,
        f"Slots: {len(slots)} | Points: {len(current_points)}/4",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, (255, 255, 0), 2
    )

    cv2.imshow(window_name, display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        height, width = frame.shape[:2]

        data = {
            "frame_width": width,
            "frame_height": height,
            "slots": slots
        }

        with open("parking_slots.json", "w") as file:
            json.dump(data, file, indent=4)

        print(f"{len(slots)} ta parking joyi saqlandi!")

    elif key == ord("u"):
        if current_points:
            current_points.pop()
        elif slots:
            removed = slots.pop()
            print(f"{removed['id']} o'chirildi.")

    elif key == ord("r"):
        current_points = []

    elif key == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
