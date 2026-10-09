"""Run ONCE. Downloads YOLOv8n and exports it to ONNX so OpenCV DNN can load it."""
import os, shutil
from ultralytics import YOLO

model = YOLO("yolov8n.pt")                      # downloads weights (~6 MB)
path = model.export(format="onnx", imgsz=640)   # creates yolov8n.onnx
os.makedirs("models", exist_ok=True)
shutil.move(path, "models/yolov8n.onnx")
print("Ready: models/yolov8n.onnx")
