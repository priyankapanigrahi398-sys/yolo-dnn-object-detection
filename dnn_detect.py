"""Step 3 - YOLO through OpenCV's DNN module (cv2.dnn) on CPU, using the ONNX model.
No PyTorch needed at run time. Works for image, video file, or webcam (--source 0)."""
import argparse, os, time, cv2, numpy as np

COCO = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
"fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant",
"bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
"sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
"wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot",
"hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop",
"mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock",
"vase","scissors","teddy bear","hair drier","toothbrush"]

def letterbox(img, size):
    """Resize keeping aspect ratio, pad with grey (same as YOLO training)."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    resized = cv2.resize(img, (nw, nh))
    top, left = int(round(dh - 0.1)), int(round(dw - 0.1))
    padded = cv2.copyMakeBorder(resized, top, size - nh - top, left, size - nw - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, left, top

def detect(net, img, size, conf_thr, nms_thr=0.45):
    padded, r, left, top = letterbox(img, size)
    blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (size, size), swapRB=True, crop=False)
    net.setInput(blob)
    out = net.forward()[0].T                       # (8400, 84)
    scores = out[:, 4:]
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf > conf_thr
    out, cls, conf = out[keep], cls[keep], conf[keep]
    if len(out) == 0:
        return []
    h, w = img.shape[:2]
    x = np.clip((out[:, 0] - out[:, 2] / 2 - left) / r, 0, w)
    y = np.clip((out[:, 1] - out[:, 3] / 2 - top) / r, 0, h)
    bw, bh = out[:, 2] / r, out[:, 3] / r
    boxes = np.stack([x, y, bw, bh], 1).astype(int).tolist()
    idx = cv2.dnn.NMSBoxes(boxes, conf.tolist(), conf_thr, nms_thr)
    return [(boxes[i], float(conf[i]), int(cls[i])) for i in np.array(idx).flatten()]

def draw(img, dets):
    for (x, y, bw, bh), s, c in dets:
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(img, f"{COCO[c]} {s:.2f}", (x, max(y - 6, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="image path, video path, or 0 for webcam")
    ap.add_argument("--model", default="models/yolov8n.onnx")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--size", type=int, default=640)
    a = ap.parse_args()

    if not os.path.exists(a.model):
        raise SystemExit("Model missing. Run:  python setup_model.py")
    net = cv2.dnn.readNetFromONNX(a.model)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    os.makedirs("outputs", exist_ok=True)

    if a.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        img = cv2.imread(a.source)
        detect(net, img, a.size, a.conf)                       # warm-up
        t = time.time(); dets = detect(net, img, a.size, a.conf)
        print(f"Inference: {(time.time() - t) * 1000:.1f} ms | {len(dets)} objects")
        for _, s, c in dets:
            print(f"  {COCO[c]:<15} {s:.2f}")
        out = draw(img, dets)
        cv2.imwrite("outputs/dnn_result.jpg", out)
        print("Saved outputs/dnn_result.jpg")
        cv2.imshow("OpenCV DNN (press any key)", out); cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
        prev = time.time()
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            out = draw(frame, detect(net, frame, a.size, a.conf))
            now = time.time(); fps = 1 / max(now - prev, 1e-6); prev = now
            cv2.putText(out, f"DNN CPU FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("OpenCV DNN live (q to quit)", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
    cv2.destroyAllWindows()
