"""Interactively identify, save, and preview an AVFoundation camera."""

import cv2

from camera_devices import (
    discover_cameras,
    open_camera,
    print_camera_result,
    save_selection,
)

IPHONE_CAMERA_INDEX = 0


def preview(index):
    capture = None
    window_name = f"SmartParking kamera preview - indeks {index}"
    try:
        capture = open_camera(index)
        print("Jonli preview ochildi. Chiqish uchun Q tugmasini bosing.")
        while True:
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                print("Kadr olinmadi; preview to'xtatildi.")
                return False
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                return True
    except (cv2.error, RuntimeError) as exc:
        print(f"Preview xatosi: {exc}")
        return False
    finally:
        if capture is not None:
            capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


def main():
    print("iPhone 16 Pro Max Continuity Camera ni tanlash")
    print("1. iPhone qulfdan chiqarilgan va Mac yaqinida bo'lsin.")
    print("2. Ikkala qurilma bir xil Apple Account va ikki bosqichli himoyadan foydalansin.")
    print("3. Mac va iPhone'da Wi-Fi hamda Bluetooth yoqilgan bo'lsin.")
    print("4. iPhone: Settings > General > AirPlay & Continuity (yoki AirPlay & Handoff) > Continuity Camera ni yoqing.")
    print("5. macOS: System Settings > Privacy & Security > Camera bo'limida Terminal/Python'ga ruxsat bering.")
    print("6. Topilmasa, iPhone'ni Mac'ga yaqin tutib, tekshiruvni qayta boshlang.\n")

    print(f"iPhone uchun kamera indeksi {IPHONE_CAMERA_INDEX} tekshirilmoqda...\n")
    result = discover_cameras(max_index=IPHONE_CAMERA_INDEX)[IPHONE_CAMERA_INDEX]
    print_camera_result(result)
    if not result.stable:
        print("0-indeksdan barqaror kadr olinmadi. iPhone/Mac sozlamalarini tekshirib qayta urinib ko'ring.")
        return 1

    if not preview(IPHONE_CAMERA_INDEX):
        return 1
    save_selection(IPHONE_CAMERA_INDEX)
    print(f"Indeks {IPHONE_CAMERA_INDEX} tanlandi va saqlandi (.camera_selection.json).")
    print("Endi camera_test.py yoki SmartParking skriptini ishga tushiring.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EOFError, KeyboardInterrupt):
        print("\nKamera tanlash bekor qilindi.")
        raise SystemExit(130)
