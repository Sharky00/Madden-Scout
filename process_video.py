import cv2
import numpy as np

# --- ROIs (percentages) for 1280x720 ---
ROIS = {
    "prev_play":     (0.70, 0.05, 0.98, 0.23),
    "score_strip":   (0.00, 0.92, 1.00, 0.99),
    "quarter_time":  (0.46, 0.925, 0.60, 0.985),
    "play_clock":    (0.60, 0.925, 0.645, 0.985),
    "down_distance": (0.81, 0.942, 0.94, 0.982),
    "yard_line":     (0.945, 0.942, 0.985, 0.982),
    "score_left":    (0.22, 0.925, 0.34, 0.985),
    "score_right":   (0.34, 0.925, 0.46, 0.985),
}

def crop_pct(img, box):
    h, w = img.shape[:2]
    x0, y0, x1, y1 = int(box[0]*w), int(box[1]*h), int(box[2]*w), int(box[3]*h)
    return img[y0:y1, x0:x1].copy()

def match_prev_play(img_bgr, tpl):
    roi = crop_pct(img_bgr, ROIS["prev_play"])
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tgray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    if tgray.shape[0] > gray.shape[0] or tgray.shape[1] > gray.shape[1]:
        return 0.0, gray, roi  # no match if template larger
    res = cv2.matchTemplate(gray, tgray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)
    return float(max_val), gray, roi

def panel_stable(curr_gray, prev_gray, diff_thr=0.012):
    """Return True if frame-to-frame change is tiny (fade finished).
       diff_thr is mean absolute difference normalized to [0..1]."""
    if prev_gray is None or prev_gray.shape != curr_gray.shape:
        return False, 1.0
    diff = np.mean(np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32))) / 255.0
    return (diff < diff_thr), float(diff)

def sharp_enough(curr_gray, sharp_thr=150.0):
    """Reject blurry/faded frames."""
    sharp = cv2.Laplacian(curr_gray, cv2.CV_64F).var()
    return (sharp >= sharp_thr), float(sharp)

def save_hud_regions_to_dir(img_bgr, frame_dir):
    frame_dir.mkdir(parents=True, exist_ok=True)
    prev = crop_pct(img_bgr, ROIS["prev_play"])
    cv2.imwrite(str(frame_dir / "prev_play.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 92])

    score = crop_pct(img_bgr, ROIS["score_strip"])
    cv2.imwrite(str(frame_dir / "score_strip.png"), score)

    for name in ["quarter_time","play_clock","down_distance","yard_line","score_left","score_right"]:
        crop = crop_pct(img_bgr, ROIS[name])
        crop = cv2.resize(crop, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(frame_dir / f"{name}.png"), crop)


def _ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM for single-channel images."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    # Gaussian windows
    mu1 = cv2.GaussianBlur(a, (7, 7), 1.5)
    mu2 = cv2.GaussianBlur(b, (7, 7), 1.5)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(a * a, (7, 7), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (7, 7), 1.5) - mu2_sq
    sigma12   = cv2.GaussianBlur(a * b, (7, 7), 1.5) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
               (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

def yardline_similar_to_prev(img_bgr, ROI, thr: float = 0.97, lw_thr: float = 0.4) -> bool:
    """
    True if ROI (yard_line/down_distance/etc.) is visually similar to its own
    previous frame: lw_thr < SSIM <= thr. Keeps a per-ROI previous frame.
    """
    roi = crop_pct(img_bgr, ROIS[ROI])
    if roi is None or roi.size == 0:
        return False
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # per-ROI cache
    store = getattr(yardline_similar_to_prev, "_prev_gray_map", {})
    prev_gray = store.get(ROI)

    # compare to previous (if any)
    if prev_gray is None or prev_gray.shape != gray.shape:
        store[ROI] = gray
        yardline_similar_to_prev._prev_gray_map = store
        return False

    sim = _ssim_gray(gray, prev_gray)

    # update AFTER comparing
    store[ROI] = gray
    yardline_similar_to_prev._prev_gray_map = store

    return lw_thr < sim <= thr

