"""Card grid segmentation engine.

Takes a photo of cards laid out in a grid and segments each individual card.
Uses OpenCV contour detection to find rectangular card shapes.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image


def segment_cards(image_path: str, output_dir: str, min_card_area_ratio: float = 0.005) -> list[Path]:
    """Segment individual cards from a grid photo.

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

    # Preprocess: grayscale, blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Cards on dark background: use Otsu threshold to separate bright cards from dark bg
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Open to remove noise, then erode to separate touching cards
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Erode to create clear gaps between adjacent cards
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    thresh = cv2.erode(thresh, kernel_erode, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    card_rects = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_card_area:
            continue

        # Approximate the contour to a polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Get the bounding rect for cropping
        x, y, bw, bh = cv2.boundingRect(contour)

        # Check aspect ratio using bounding rect — more stable than minAreaRect
        # Cards are typically ~2.5x3.5 inches (ratio ~0.71)
        aspect = min(bw, bh) / max(bw, bh) if bw > 0 and bh > 0 else 0
        if aspect < 0.45 or aspect > 0.95:
            continue

        card_rects.append((x, y, bw, bh))

    # Merge overlapping/adjacent boxes that are fragments of the same card.
    # This handles cards with strong internal design lines (e.g. Score 2007)
    # that get split into top/bottom halves by contour detection.
    if len(card_rects) >= 2:
        card_rects = _merge_overlapping_rects(card_rects)

    # Filter out outlier-small detections (e.g. shadows) — must be at least
    # 25% of the median card area to be considered a real card
    if len(card_rects) >= 3:
        areas = [w * h for (_, _, w, h) in card_rects]
        median_area = float(np.median(areas))
        card_rects = [(x, y, w, h) for (x, y, w, h) in card_rects
                      if w * h >= median_area * 0.25]

    # Sort cards: top-to-bottom, then left-to-right
    if card_rects:
        card_rects = _sort_grid(card_rects)

    # Infer grid and fill missing cells for dark-bordered cards that
    # blend into the background and aren't picked up by thresholding
    if len(card_rects) >= 3:
        card_rects = _fill_missing_grid_cells(card_rects, width, height)

    # Crop and save each card
    saved_paths = []
    for i, (x, y, w, h) in enumerate(card_rects):
        # Add small padding
        pad = 5
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(width, x + w + pad)
        y2 = min(height, y + h + pad)

        card_img = img[y1:y2, x1:x2]
        card_path = output_path / f"card_{i:03d}.jpg"
        cv2.imwrite(str(card_path), card_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(card_path)

    return saved_paths


def _merge_overlapping_rects(rects: list[tuple]) -> list[tuple]:
    """Merge bounding boxes that are fragments of the same card.

    When a card's internal design creates multiple contours, we get
    two (or more) boxes for one card. Only merge when:
    - Boxes have significant horizontal overlap (>50%)
    - Boxes are very close vertically (gap < 8% of combined height)
    - At least one box looks like a fragment (aspect ratio outside
      normal card range 0.60-0.85), suggesting it's a half-card
    """
    if len(rects) < 2:
        return rects

    merged = list(rects)
    changed = True
    while changed:
        changed = False
        new_merged = []
        used = set()
        for i in range(len(merged)):
            if i in used:
                continue
            x1, y1, w1, h1 = merged[i]
            r1_left, r1_right = x1, x1 + w1
            r1_top, r1_bottom = y1, y1 + h1
            for j in range(i + 1, len(merged)):
                if j in used:
                    continue
                x2, y2, w2, h2 = merged[j]
                r2_left, r2_right = x2, x2 + w2
                r2_top, r2_bottom = y2, y2 + h2

                # At least one box must look like a fragment (not a full card)
                a1 = min(w1, h1) / max(w1, h1) if w1 > 0 and h1 > 0 else 0
                a2 = min(w2, h2) / max(w2, h2) if w2 > 0 and h2 > 0 else 0
                either_is_fragment = a1 > 0.85 or a2 > 0.85 or a1 < 0.55 or a2 < 0.55
                if not either_is_fragment:
                    continue

                # Check horizontal overlap
                overlap_x = max(0, min(r1_right, r2_right) - max(r1_left, r2_left))
                min_width = min(w1, w2)
                if min_width == 0 or overlap_x < min_width * 0.5:
                    continue

                # Vertical gap must be very small
                gap_y = max(0, max(r1_top, r2_top) - min(r1_bottom, r2_bottom))
                combined_h = max(r1_bottom, r2_bottom) - min(r1_top, r2_top)
                if combined_h == 0 or gap_y > combined_h * 0.08:
                    continue

                # Merge: union of the two boxes
                nx = min(r1_left, r2_left)
                ny = min(r1_top, r2_top)
                nr = max(r1_right, r2_right)
                nb = max(r1_bottom, r2_bottom)
                x1, y1, w1, h1 = nx, ny, nr - nx, nb - ny
                r1_left, r1_right = nx, nr
                r1_top, r1_bottom = ny, nb
                used.add(j)
                changed = True

            new_merged.append((x1, y1, w1, h1))
        merged = new_merged

    return merged


def _fill_missing_grid_cells(
    card_rects: list[tuple], img_width: int, img_height: int
) -> list[tuple]:
    """Infer grid layout from detected cards and fill gaps.

    Dark-bordered cards can blend into dark backgrounds, causing contour
    detection to miss them. This uses the detected cards to figure out
    the grid dimensions (rows/cols) and synthesizes bounding boxes for
    any empty cells.
    """
    if len(card_rects) < 3:
        return card_rects

    # Cluster card centers into rows and columns
    centers_y = sorted(set(r[1] + r[3] // 2 for r in card_rects))
    centers_x = sorted(set(r[0] + r[2] // 2 for r in card_rects))

    avg_h = int(np.mean([r[3] for r in card_rects]))
    avg_w = int(np.mean([r[2] for r in card_rects]))

    # Cluster y-centers into rows
    row_centers = _cluster_1d(centers_y, avg_h * 0.5)
    col_centers = _cluster_1d(centers_x, avg_w * 0.5)

    n_rows = len(row_centers)
    n_cols = len(col_centers)

    if n_rows < 2 or n_cols < 2:
        return card_rects

    # Check each grid cell for a detected card
    occupied = set()
    for r in card_rects:
        cx, cy = r[0] + r[2] // 2, r[1] + r[3] // 2
        ri = _nearest_idx(row_centers, cy)
        ci = _nearest_idx(col_centers, cx)
        occupied.add((ri, ci))

    # Fill missing cells
    filled = list(card_rects)
    for ri, ry in enumerate(row_centers):
        for ci, cx in enumerate(col_centers):
            if (ri, ci) not in occupied:
                x = int(cx - avg_w // 2)
                y = int(ry - avg_h // 2)
                x = max(0, min(x, img_width - avg_w))
                y = max(0, min(y, img_height - avg_h))
                filled.append((x, y, avg_w, avg_h))

    return _sort_grid(filled)


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
    return [np.mean(c) for c in clusters]


def _nearest_idx(centers: list[float], value: float) -> int:
    """Return index of the nearest center."""
    return int(np.argmin([abs(c - value) for c in centers]))


def _sort_grid(rects: list[tuple]) -> list[tuple]:
    """Sort rectangles in reading order (top-to-bottom, left-to-right).

    Groups cards into rows based on y-coordinate proximity,
    then sorts each row by x-coordinate.
    """
    if not rects:
        return rects

    # Sort by y first
    rects.sort(key=lambda r: r[1])

    # Group into rows: cards within similar y-range are same row
    rows = []
    current_row = [rects[0]]
    avg_height = np.mean([r[3] for r in rects])

    for rect in rects[1:]:
        if abs(rect[1] - current_row[0][1]) < avg_height * 0.5:
            current_row.append(rect)
        else:
            rows.append(sorted(current_row, key=lambda r: r[0]))
            current_row = [rect]
    rows.append(sorted(current_row, key=lambda r: r[0]))

    # Flatten back
    return [rect for row in rows for rect in row]


def segment_grid_uniform(image_path: str, output_dir: str, rows: int, cols: int) -> list[Path]:
    """Segment cards assuming a uniform grid layout.

    Fallback when contour detection doesn't work well —
    just divide the image into equal cells.

    Args:
        image_path: Path to the uploaded grid photo.
        output_dir: Directory to save cropped card images.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.

    Returns:
        List of paths to cropped card images.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = img.shape[:2]
    cell_h = height // rows
    cell_w = width // cols

    # Trim margins (10% of cell size)
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
