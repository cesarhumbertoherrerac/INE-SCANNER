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

    for epsilon_factor in (0.012, 0.016, 0.02, 0.03):
        approx = cv2.approxPolyDP(contour, epsilon_factor * perimeter, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype("float32")

    return None


def _quad_quality_score(quad, image_w, image_h):
    rect = order_points(quad)
    width = max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3]))
    height = max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1]))
    if width <= 0 or height <= 0:
        return -1.0

    ratio = width / height
    if ratio < 1.15 or ratio > 2.25:
        return -1.0

    area = cv2.contourArea(rect.astype(np.float32))
    image_area = float(image_w * image_h)
    if area <= image_area * 0.04:
        return -1.0

    # Qué tan rectangular es el contorno respecto a su rectángulo rotado.
    box = cv2.boxPoints(cv2.minAreaRect(rect.astype(np.float32)))
    box_area = cv2.contourArea(box)
    rectangularity = area / (box_area + 1e-6)

    # Dar preferencia a quads cercanos al centro (evita ruido del fondo).
    cx, cy = np.mean(rect, axis=0)
    dx = abs(cx - image_w / 2.0) / (image_w / 2.0)
    dy = abs(cy - image_h / 2.0) / (image_h / 2.0)
    center_score = 1.0 - min(1.0, 0.6 * dx + 0.4 * dy)

    ratio_score = max(0.0, 1.0 - abs(ratio - 1.58) / 0.9)
    area_score = min(1.0, area / (image_area * 0.95))

    return 0.42 * area_score + 0.25 * ratio_score + 0.18 * rectangularity + 0.15 * center_score


def _build_edge_map(resized_bgr):
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    canny_a = cv2.Canny(gray_clahe, 35, 120)
    canny_b = cv2.Canny(gray_clahe, 55, 170)

    grad_x = cv2.Sobel(gray_clahe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_clahe, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(grad_x, grad_y)
    mag_u8 = np.uint8(np.clip(mag / (mag.max() + 1e-6) * 255.0, 0, 255))
    _, mag_bin = cv2.threshold(mag_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    low_sat = cv2.inRange(hsv, (0, 0, 35), (179, 105, 255))

    edges = cv2.bitwise_or(canny_a, canny_b)
    edges = cv2.bitwise_or(edges, mag_bin)
    edges = cv2.bitwise_or(edges, low_sat)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges


def find_document_contour(resized):
    edges = _build_edge_map(resized)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    image_h, image_w = resized.shape[:2]
    image_area = image_h * image_w

    best_quad = None
    best_score = -1.0

    for contour in contours[:60]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.035:
            continue

        quad = _candidate_quad_from_contour(contour)
        if quad is None:
            continue

        score = _quad_quality_score(quad, image_w, image_h)
        if score > best_score:
            best_score = score
            best_quad = quad

    if best_quad is not None:
        return best_quad

    for contour in contours[:12]:
        if cv2.contourArea(contour) < image_area * 0.08:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(contour))
        return box.astype("float32")

    return None


def refine_quad_corners(resized_bgr, quad):
    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 35, 0.01)
    corners = order_points(quad).reshape(-1, 1, 2).astype(np.float32)

    cv2.cornerSubPix(gray, corners, (9, 9), (-1, -1), criteria)
    return corners.reshape(4, 2)


def tighten_crop(warped_bgr):
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    mask = gray > 20
    ys, xs = np.where(mask)
    if len(xs) > 0 and len(ys) > 0:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()

        if (x1 - x0) > w * 0.55 and (y1 - y0) > h * 0.55:
            warped_bgr = warped_bgr[max(0, y0 - 2): min(h, y1 + 3), max(0, x0 - 2): min(w, x1 + 3)]
            gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 45, 140)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    col_strength = edges.sum(axis=0).astype(np.float32)
    row_strength = edges.sum(axis=1).astype(np.float32)

    if col_strength.max() <= 0 or row_strength.max() <= 0:
        return warped_bgr

    col_thr = col_strength.max() * 0.12
    row_thr = row_strength.max() * 0.12

    x_candidates = np.where(col_strength > col_thr)[0]
    y_candidates = np.where(row_strength > row_thr)[0]

    if len(x_candidates) == 0 or len(y_candidates) == 0:
        return warped_bgr

    x_left, x_right = int(x_candidates[0]), int(x_candidates[-1])
    y_top, y_bottom = int(y_candidates[0]), int(y_candidates[-1])

    if (x_right - x_left) < w * 0.6 or (y_bottom - y_top) < h * 0.6:
        return warped_bgr

    m = 2
    return warped_bgr[max(0, y_top - m): min(h, y_bottom + m + 1), max(0, x_left - m): min(w, x_right + m + 1)]


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
    target_height = 1000
    ratio = image.shape[0] / float(target_height)
    resized = cv2.resize(image, (int(image.shape[1] / ratio), target_height))

    screen_cnt = find_document_contour(resized)
    if screen_cnt is None:
        return apply_natural_tone(orig)

    refined = refine_quad_corners(resized, screen_cnt)
    pts = refined * ratio
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
    tightened = tighten_crop(warped)
    return apply_natural_tone(tightened)


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
