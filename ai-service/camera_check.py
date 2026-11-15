
import cv2

for index in range(5):
    camera = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)

    if camera.isOpened():
        success, frame = camera.read()

        if success:
            height, width = frame.shape[:2]
            print(
                f"Camera {index}: WORKING "
                f"({width}x{height})"
            )
        else:
            print(f"Camera {index}: Opened, but no frame")

        camera.release()
    else:
        print(f"Camera {index}: Not available")
