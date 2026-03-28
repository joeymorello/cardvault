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

    # Preprocess: grayscale, blur, edge detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Adaptive threshold to handle varying lighting
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
    )

    # Morphological operations to clean up edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

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

        # Cards should be roughly rectangular (4 corners)
        # But be lenient — allow 4-8 vertices for slightly curved edges
        if len(approx) < 4 or len(approx) > 12:
            continue

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Check aspect ratio — cards are typically ~2.5x3.5 inches (ratio ~0.71)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue
        aspect = min(w, h) / max(w, h)
        if aspect < 0.5 or aspect > 0.95:
            continue

        # Get the bounding rect for cropping
        x, y, bw, bh = cv2.boundingRect(contour)
        card_rects.append((x, y, bw, bh))

    # Sort cards: top-to-bottom, then left-to-right
    if card_rects:
        card_rects = _sort_grid(card_rects)

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
