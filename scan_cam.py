import cv2

print("=== Scan semua index kamera (MSMF) ===")
for i in range(5):
    cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
    if cap.isOpened():
        ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_native = cap.get(cv2.CAP_PROP_FPS)
        brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
        print(f"  Index {i}: resolusi={w}x{h}, fps_native={fps_native}, brightness={brightness}, baca_frame={ok}")
        cap.release()
    else:
        print(f"  Index {i}: TIDAK TERSEDIA")

print()
print("=== Scan semua index kamera (Default/Auto) ===")
for i in range(5):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ok, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_native = cap.get(cv2.CAP_PROP_FPS)
        print(f"  Index {i}: resolusi={w}x{h}, fps_native={fps_native}, baca_frame={ok}")
        cap.release()
    else:
        print(f"  Index {i}: TIDAK TERSEDIA")
