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


def _candidate_quad_from_contour(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4 and cv2.isContourConvex(approx):
        return approx.reshape(4, 2).astype("float32")

    return None


def find_document_contour(resized):
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)

    canny = cv2.Canny(gray, 35, 130)
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        19,
        2,
    )
    adaptive = cv2.bitwise_not(adaptive)

    edges = cv2.bitwise_or(canny, adaptive)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_h, image_w = resized.shape[:2]
    image_area = image_h * image_w

    for contour in contours[:30]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue

        quad = _candidate_quad_from_contour(contour)
        if quad is None:
            continue

        rect = order_points(quad)
        width = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
        height = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))

        if height <= 0 or width <= 0:
            continue

        ratio = width / height
        if 1.2 <= ratio <= 2.2:
            return quad

    if contours:
        largest = contours[0]
        if cv2.contourArea(largest) >= image_area * 0.1:
            box = cv2.boxPoints(cv2.minAreaRect(largest))
            return box.astype("float32")

    return None


def apply_natural_tone(image_bgr):
    mean_b, mean_g, mean_r = cv2.mean(image_bgr)[:3]
    mean_gray = (mean_b + mean_g + mean_r) / 3.0

    gain_b = mean_gray / (mean_b + 1e-6)
    gain_g = mean_gray / (mean_g + 1e-6)
    gain_r = mean_gray / (mean_r + 1e-6)

    balanced = image_bgr.astype(np.float32)
    balanced[:, :, 0] *= gain_b
    balanced[:, :, 1] *= gain_g
    balanced[:, :, 2] *= gain_r
    balanced = np.clip(balanced, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(balanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    l = clahe.apply(l)

    result = cv2.merge((l, a, b))
    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(result, alpha=1.02, beta=1)


def scan_document(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("No se pudo decodificar la imagen enviada")

    orig = image.copy()
    target_height = 900
    ratio = image.shape[0] / float(target_height)
    resized = cv2.resize(image, (int(image.shape[1] / ratio), target_height))

    screen_cnt = find_document_contour(resized)
    if screen_cnt is None:
        return apply_natural_tone(orig)

    pts = screen_cnt * ratio
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    if max_width < 40 or max_height < 40:
        return apply_natural_tone(orig)

    destination = np.array(
        [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
        dtype="float32",
    )

    transform = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(orig, transform, (max_width, max_height))
    return apply_natural_tone(warped)


@app.route("/scan", methods=["POST"])
def scan():
    file = request.files.get("file")
    if file is None:
        return {"error": "Debes enviar un archivo en el campo 'file'"}, 400

    try:
        processed = scan_document(file.read())
    except ValueError as exc:
        return {"error": str(exc)}, 400

    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

    pil_img = Image.fromarray(processed_rgb)
    img_io = io.BytesIO()
    pil_img.save(img_io, "JPEG", quality=95)
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
