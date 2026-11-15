"""AVFoundation camera diagnostics based on real consecutive frame reads."""

from camera_devices import discover_cameras, print_camera_result


def main():
    print("Mac kameralarini tekshirish (AVFoundation)...")
    print("Indekslar jismoniy kameralar sonini anglatmaydi; takror bo'lishi mumkin.\n")
    results = discover_cameras()
    for result in results:
        print_camera_result(result)

    stable = [result for result in results if result.stable]
    if not stable:
        print("Barqaror kamera topilmadi. iPhone uchun Continuity Camera yoqilganini tekshiring.")
        return 1
    print(f"\nBarqaror indekslar: {', '.join(str(item.index) for item in stable)}")
    print("Qaysi biri iPhone ekanini aniqlash uchun camera_select.py orqali preview qiling.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
