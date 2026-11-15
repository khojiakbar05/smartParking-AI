
import cv2

camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not camera.isOpened():
    print("Camera ochilmadi!")
    exit()

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
