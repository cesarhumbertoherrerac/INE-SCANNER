from flask import Flask, request, send_file
import cv2
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "INE Scanner API is running"
    
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def scan_document(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, (int(image.shape[1] / ratio), 500))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return orig

    pts = screenCnt.reshape(4, 2) * ratio
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

scan = cv2.bilateralFilter(warped, 7, 50, 50)
alpha = 1.1
beta = 5
scan = cv2.convertScaleAbs(scan, alpha=alpha, beta=beta)
kernel = np.array([[0, -1, 0],
                   [-1, 4.2,-1],
                   [0, -1, 0]])
scan = cv2.filter2D(scan, -1, kernel)

    return scan

@app.route("/scan", methods=["POST"])
def scan():
    file = request.files["file"]
    processed = scan_document(file.read())

    pil_img = Image.fromarray(processed)
    img_io = io.BytesIO()
    pil_img.save(img_io, "JPEG", quality=95)
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
