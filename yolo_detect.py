"""Step 2 - YOLO (Ultralytics) on CPU. Works for image, video file, or webcam (--source 0)."""
import argparse, os, time, cv2
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, help="image path, video path, or 0 for webcam")
ap.add_argument("--model", default="yolov8n.pt")
ap.add_argument("--conf", type=float, default=0.3)
ap.add_argument("--imgsz", type=int, default=640, help="use 320 for faster CPU")
a = ap.parse_args()

model = YOLO(a.model)
os.makedirs("outputs", exist_ok=True)
is_image = a.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

if is_image:
    img = cv2.imread(a.source)
    model.predict(img, device="cpu", imgsz=a.imgsz, verbose=False)   # warm-up
    t = time.time()
    res = model.predict(img, conf=a.conf, device="cpu", imgsz=a.imgsz, verbose=False)[0]
    print(f"Inference: {(time.time() - t) * 1000:.1f} ms | {len(res.boxes)} objects")
    for b in res.boxes:
        print(f"  {model.names[int(b.cls)]:<15} {float(b.conf):.2f}")
    out = res.plot()
    cv2.imwrite("outputs/yolo_result.jpg", out)
    print("Saved outputs/yolo_result.jpg")
    cv2.imshow("YOLO (press any key)", out); cv2.waitKey(0)
else:
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    prev = time.time()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=a.conf, device="cpu", imgsz=a.imgsz, verbose=False)[0]
        out = res.plot()
        now = time.time(); fps = 1 / max(now - prev, 1e-6); prev = now
        cv2.putText(out, f"YOLO CPU FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("YOLO live (q to quit)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
cv2.destroyAllWindows()
