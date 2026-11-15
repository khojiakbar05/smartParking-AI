
import cv2
from camera_devices import open_selected_camera

try:
    camera = open_selected_camera()
except RuntimeError as exc:
    raise SystemExit(str(exc))

print("Camera muvaffaqiyatli ishga tushdi.")
print("Chiqish uchun Q tugmasini bosing.")

while True:
    success, frame = camera.read()

    if not success:
        print("Kameradan video olinmadi.")
        break

    cv2.imshow("SmartParking - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
