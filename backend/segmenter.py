"""Card grid segmentation engine.

Takes a photo of cards laid out in a grid and segments each individual card.
Uses contour detection to locate cards, then infers grid boundaries for
clean, consistent cropping.
"""

import cv2
import numpy as np
from pathlib import Path


def segment_cards(image_path: str, output_dir: str, min_card_area_ratio: float = 0.005) -> list[Path]:
    """Segment individual cards from a grid photo.

    Strategy: use contour detection to find card centers, cluster them
    into rows and columns, then crop using grid boundaries (midpoints
    between adjacent rows/cols). This gives clean crops even when
    contour shapes are irregular.

    Args:
        image_path: Path to the uploaded grid photo.
        output_dir: Directory to save cropped card images.
        min_card_area_ratio: Minimum card area as fraction of total image area.

    Returns:
        List of paths to cropped card images, ordered left-to-right, top-to-bottom.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]
    total_area = height * width
    min_card_area = total_area * min_card_area_ratio

    # --- Step 1: Find card regions via thresholding ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.erode(thresh, kernel_erode, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Step 2: Extract card center points ---
    card_centers = []
    card_sizes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_card_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        aspect = min(bw, bh) / max(bw, bh) if bw > 0 and bh > 0 else 0
        if aspect < 0.35 or aspect > 0.95:
            continue
        cx = x + bw // 2
        cy = y + bh // 2
        card_centers.append((cx, cy))
        card_sizes.append((bw, bh))

    if not card_centers:
        return []

    # Filter outlier-small detections
    if len(card_sizes) >= 3:
        areas = [w * h for w, h in card_sizes]
        median_area = float(np.median(areas))
        filtered = [(c, s) for c, s in zip(card_centers, card_sizes)
                     if s[0] * s[1] >= median_area * 0.25]
        card_centers = [c for c, _ in filtered]
        card_sizes = [s for _, s in filtered]

    if not card_centers:
        return []

    # Use median size (robust to fragments that drag down the mean)
    avg_w = int(np.median([s[0] for s in card_sizes]))
    avg_h = int(np.median([s[1] for s in card_sizes]))

    # For clustering, use the larger dimension as the card "height"
    # to handle both portrait and landscape cards
    cluster_size = max(avg_w, avg_h)

    # --- Step 3: Cluster centers into rows and columns ---
    ys = sorted([c[1] for c in card_centers])
    xs = sorted([c[0] for c in card_centers])

    row_centers = _cluster_1d(ys, cluster_size * 0.5)
    col_centers = _cluster_1d(xs, cluster_size * 0.5)

    n_rows = len(row_centers)
    n_cols = len(col_centers)

    if n_rows < 1 or n_cols < 1:
        return []

    # --- Step 4: Compute grid boundaries (midpoints between adjacent rows/cols) ---
    row_bounds = _compute_boundaries(row_centers, avg_h, 0, height)
    col_bounds = _compute_boundaries(col_centers, avg_w, 0, width)

    # --- Step 5: Crop each cell, adding padding to tighten around the card ---
    saved_paths = []
    idx = 0
    for ri in range(n_rows):
        for ci in range(n_cols):
            y1 = row_bounds[ri]
            y2 = row_bounds[ri + 1]
            x1 = col_bounds[ci]
            x2 = col_bounds[ci + 1]

            # Tighten crop: find the bright region within this cell
            cell_thresh = thresh[y1:y2, x1:x2]
            tight = _tighten_crop(cell_thresh, x1, y1, x2, y2, margin=8)
            if tight:
                x1, y1, x2, y2 = tight

            card_img = img[y1:y2, x1:x2]
            card_path = output_path / f"card_{idx:03d}.jpg"
            cv2.imwrite(str(card_path), card_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(card_path)
            idx += 1

    return saved_paths


def _compute_boundaries(centers: list[float], avg_size: int,
                        img_min: int, img_max: int) -> list[int]:
    """Compute grid cell boundaries from cluster centers.

    Boundaries are midpoints between adjacent centers, with the first
    and last boundaries extending to the image edges (clamped).
    """
    centers = sorted(centers)
    bounds = []

    # First boundary: half a card-width before the first center
    first = max(img_min, int(centers[0] - avg_size * 0.6))
    bounds.append(first)

    # Interior boundaries: midpoints between adjacent centers
    for i in range(len(centers) - 1):
        mid = int((centers[i] + centers[i + 1]) / 2)
        bounds.append(mid)

    # Last boundary: half a card-width after the last center
    last = min(img_max, int(centers[-1] + avg_size * 0.6))
    bounds.append(last)

    return bounds


def _tighten_crop(cell_thresh: np.ndarray, x1: int, y1: int,
                  x2: int, y2: int, margin: int = 8) -> tuple | None:
    """Tighten a grid cell crop to the actual card within it.

    Finds the bounding box of bright pixels in the thresholded cell
    and returns tightened coordinates with a small margin.
    """
    if cell_thresh.size == 0:
        return None

    # Find rows and cols with bright pixels
    row_sums = np.sum(cell_thresh, axis=1)
    col_sums = np.sum(cell_thresh, axis=0)

    bright_rows = np.where(row_sums > 0)[0]
    bright_cols = np.where(col_sums > 0)[0]

    if len(bright_rows) == 0 or len(bright_cols) == 0:
        return None

    # Only tighten if we'd remove significant background
    cell_h = y2 - y1
    cell_w = x2 - x1
    card_h = bright_rows[-1] - bright_rows[0]
    card_w = bright_cols[-1] - bright_cols[0]

    # Don't tighten if the card fills most of the cell already
    if card_h > cell_h * 0.9 and card_w > cell_w * 0.9:
        return None

    ty1 = max(y1, y1 + int(bright_rows[0]) - margin)
    ty2 = min(y2, y1 + int(bright_rows[-1]) + margin)
    tx1 = max(x1, x1 + int(bright_cols[0]) - margin)
    tx2 = min(x2, x1 + int(bright_cols[-1]) + margin)

    return (tx1, ty1, tx2, ty2)


def _cluster_1d(values: list[float], min_gap: float) -> list[float]:
    """Cluster sorted 1D values into groups separated by at least min_gap."""
    if not values:
        return []
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] < min_gap:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [float(np.mean(c)) for c in clusters]


def segment_grid_uniform(image_path: str, output_dir: str, rows: int, cols: int) -> list[Path]:
    """Segment cards assuming a uniform grid layout.

    Fallback when contour detection doesn't work well —
    just divide the image into equal cells.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]
    cell_h = height // rows
    cell_w = width // cols

    margin_h = int(cell_h * 0.05)
    margin_w = int(cell_w * 0.05)

    saved_paths = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            y1 = r * cell_h + margin_h
            y2 = (r + 1) * cell_h - margin_h
            x1 = c * cell_w + margin_w
            x2 = (c + 1) * cell_w - margin_w

            card_img = img[y1:y2, x1:x2]
            card_path = output_path / f"card_{idx:03d}.jpg"
            cv2.imwrite(str(card_path), card_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(card_path)
            idx += 1

    return saved_paths
