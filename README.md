# Image Processing and Object Detection using YOLO and OpenCV DNN

This project performs image processing and object detection on a CPU using YOLOv8 and the OpenCV DNN module. CUDA is not required.

## Features
- Image processing with OpenCV: grayscale, Gaussian blur, histogram equalization, sharpening, Canny edge detection, Otsu thresholding, and contours
- Object detection with YOLOv8n (Ultralytics, PyTorch)
- Object detection with OpenCV DNN using an ONNX model
- Speed comparison between YOLO and OpenCV DNN

## Technologies Used
Python, OpenCV, YOLOv8 (Ultralytics), PyTorch, ONNX, OpenCV DNN, NumPy, Anaconda

## Project Files
| File | Purpose |
|---|---|
| `setup_model.py` | Downloads YOLOv8n and exports it to ONNX (run once) |
| `preprocess.py` | Image processing montage |
| `yolo_detect.py` | YOLO detection on image, video, or webcam |
| `dnn_detect.py` | OpenCV DNN detection on image, video, or webcam |
| `compare.py` | Speed comparison of YOLO vs OpenCV DNN |

## How to Run
1. Create and activate an environment:
```
   conda create -n yolo python=3.11 -y
   conda activate yolo
```
2. Install the libraries:
```
   pip install torch torchvision
   pip install -r requirements.txt
```
3. Create the ONNX model (once):
```
   python setup_model.py
```
4. Put an image in the `images` folder (for example `images/car.jpg`) and run:
```
   python preprocess.py --source images/car.jpg
   python yolo_detect.py --source images/car.jpg
   python dnn_detect.py --source images/car.jpg
   python compare.py --source images/car.jpg
```
5. Live webcam detection (press `q` to quit):
```
   python dnn_detect.py --source 0 --size 320
```

## Results
Tested on a laptop with an AMD Ryzen AI 7 350 CPU.

| Method | Time per image | FPS |
|---|---|---|
| YOLO (Ultralytics, CPU) | 22.4 ms | 44.6 |
| OpenCV DNN (CPU) | 99.8 ms | 10.0 |

Both methods detected the car in the test image with similar confidence (0.88 for YOLO, 0.89 for OpenCV DNN). Raising the confidence threshold with `--conf 0.5` removes weak false detections.

## Notes
- Model files (`.pt`, `.onnx`) are not included. Run `python setup_model.py` to create them.
- Results are saved in the `outputs` folder.
