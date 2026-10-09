"""Step 4 - compare speed of YOLO (Ultralytics) vs OpenCV DNN on your CPU."""
import argparse, time, cv2
from ultralytics import YOLO
from dnn_detect import detect

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--runs", type=int, default=20)
a = ap.parse_args()
img = cv2.imread(a.source)

yolo = YOLO("yolov8n.pt")
yolo.predict(img, device="cpu", verbose=False)
t = time.time()
for _ in range(a.runs):
    yolo.predict(img, device="cpu", verbose=False)
y = (time.time() - t) / a.runs * 1000

net = cv2.dnn.readNetFromONNX("models/yolov8n.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV); net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
detect(net, img, 640, 0.3)
t = time.time()
for _ in range(a.runs):
    detect(net, img, 640, 0.3)
d = (time.time() - t) / a.runs * 1000

print(f"YOLO (Ultralytics, CPU): {y:6.1f} ms/image ({1000 / y:5.1f} FPS)")
print(f"OpenCV DNN (CPU)       : {d:6.1f} ms/image ({1000 / d:5.1f} FPS)")
