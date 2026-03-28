"""Card identification via OCR text extraction.

Uses Tesseract OCR to pull visible text from card images,
then structures the extracted text into candidate fields.
"""

import json
import re
import pytesseract
import cv2
from pathlib import Path


def extract_card_text(image_path: str) -> dict:
    """Run OCR on a card image and attempt to parse structured fields.

    Returns a dict with extracted text and any fields we can auto-detect.
    """
    image_path = Path(image_path).resolve()
    if not image_path.exists():
        return {"error": f"Image not found: {image_path}"}

    img = cv2.imread(str(image_path))
    if img is None:
        return {"error": f"Could not read image: {image_path}"}

    # Preprocess for better OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Sharpen
    gray = cv2.GaussianBlur(gray, (1, 1), 0)
    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=20)

    # Run OCR
    raw_text = pytesseract.image_to_string(gray, config="--psm 6")
    raw_text_block = pytesseract.image_to_string(gray, config="--psm 3")

    # Combine both passes for best coverage
    all_text = f"{raw_text}\n{raw_text_block}"
    lines = [l.strip() for l in all_text.splitlines() if l.strip()]
    unique_lines = list(dict.fromkeys(lines))  # dedupe preserving order

    # Try to auto-detect fields from OCR text
    result = {
        "raw_ocr_text": "\n".join(unique_lines),
        "player_name": _guess_player_name(unique_lines),
        "year": _guess_year(all_text),
        "card_number": _guess_card_number(all_text),
        "manufacturer": _guess_manufacturer(all_text),
        "sport": None,
        "card_set": None,
        "condition": None,
        "condition_notes": None,
        "rarity": None,
        "estimated_price_low": None,
        "estimated_price_high": None,
        "ai_confidence": None,
    }

    return result


def _guess_player_name(lines: list[str]) -> str | None:
    """Guess player name — typically the most prominent text on the card.

    Heuristic: look for lines that are mostly alphabetic, title-case,
    and 2-4 words long (typical name format).
    """
    candidates = []
    for line in lines:
        # Skip very short or very long lines
        if len(line) < 3 or len(line) > 40:
            continue
        # Skip lines that are mostly numbers
        if sum(c.isdigit() for c in line) > len(line) * 0.4:
            continue
        # Skip known non-name patterns
        lower = line.lower()
        if any(kw in lower for kw in ["topps", "bowman", "fleer", "upper deck", "donruss",
                                       "score", "copyright", "printed", "card", "no.", "#",
                                       "www.", ".com", "tm", "®"]):
            continue
        words = line.split()
        if 1 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
            candidates.append(line)

    return candidates[0] if candidates else None


def _guess_year(text: str) -> int | None:
    """Look for a 4-digit year between 1900-2030."""
    years = re.findall(r'\b(19[0-9]{2}|20[0-2][0-9])\b', text)
    if years:
        # Return the earliest year found (most likely production year)
        return min(int(y) for y in years)
    return None


def _guess_card_number(text: str) -> str | None:
    """Look for card number patterns like #123, No. 45, etc."""
    patterns = [
        r'#\s*(\d+)',
        r'[Nn]o\.?\s*(\d+)',
        r'\b(\d{1,4})\s*of\s*\d+',
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None


def _guess_manufacturer(text: str) -> str | None:
    """Check for known card manufacturers in the text."""
    manufacturers = {
        "topps": "Topps",
        "bowman": "Bowman",
        "fleer": "Fleer",
        "upper deck": "Upper Deck",
        "donruss": "Donruss",
        "score": "Score",
        "pinnacle": "Pinnacle",
        "pacific": "Pacific",
        "skybox": "SkyBox",
        "hoops": "Hoops",
        "panini": "Panini",
        "leaf": "Leaf",
    }
    lower = text.lower()
    for key, name in manufacturers.items():
        if key in lower:
            return name
    return None
