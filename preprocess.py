"""Step 1 - classic image processing with OpenCV (no neural network)."""
import argparse, os, cv2, numpy as np

def label(img, text):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img = img.copy()
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
a = ap.parse_args()

img = cv2.imread(a.source)
if img is None:
    raise SystemExit("Cannot read image: " + a.source)
img = cv2.resize(img, (480, 360))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(img, (7, 7), 0)
equal = cv2.equalizeHist(gray)
sharp = cv2.filter2D(img, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
edges = cv2.Canny(gray, 100, 200)
_, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cont = img.copy()
cv2.drawContours(cont, [c for c in cnts if cv2.contourArea(c) > 300], -1, (0, 255, 0), 2)

tiles = [label(img, "original"), label(gray, "gray"), label(blur, "blur"), label(equal, "hist equalize"),
         label(sharp, "sharpen"), label(edges, "canny edges"), label(thr, "threshold"), label(cont, "contours")]
montage = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
os.makedirs("outputs", exist_ok=True)
cv2.imwrite("outputs/processing_montage.jpg", montage)
print("Saved outputs/processing_montage.jpg")
cv2.imshow("Image processing (press any key)", montage)
cv2.waitKey(0); cv2.destroyAllWindows()
