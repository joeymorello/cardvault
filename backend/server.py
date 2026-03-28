"""CardVault API server."""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from database import init_db, insert_batch, insert_card, update_card_identification
from database import get_cards, get_card, get_batches, get_stats
from segmenter import segment_cards, segment_grid_uniform
from identifier import identify_card_with_claude

BASE_DIR = Path(__file__).parent.parent
UPLOADS_DIR = BASE_DIR / "uploads"
CARDS_DIR = BASE_DIR / "cards"
FRONTEND_DIR = BASE_DIR / "frontend"

UPLOADS_DIR.mkdir(exist_ok=True)
CARDS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="CardVault", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve card images
app.mount("/images/cards", StaticFiles(directory=str(CARDS_DIR)), name="card_images")
app.mount("/images/uploads", StaticFiles(directory=str(UPLOADS_DIR)), name="upload_images")


@app.on_event("startup")
def startup():
    init_db()


@app.get("/api/stats")
def api_stats():
    return get_stats()


@app.get("/api/batches")
def api_batches():
    return get_batches()


@app.get("/api/cards")
def api_cards(
    status: Optional[str] = None,
    sport: Optional[str] = None,
    player_name: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    rarity: Optional[str] = None,
    price_min: Optional[float] = None,
    price_max: Optional[float] = None,
    sort_by: str = "id",
    sort_dir: str = "desc",
    limit: int = Query(default=50, le=200),
    offset: int = 0,
):
    filters = {
        "status": status,
        "sport": sport,
        "player_name": player_name,
        "year_min": year_min,
        "year_max": year_max,
        "rarity": rarity,
        "price_min": price_min,
        "price_max": price_max,
    }
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}
    cards, total = get_cards(filters, sort_by, sort_dir, limit, offset)
    return {"cards": cards, "total": total, "limit": limit, "offset": offset}


@app.get("/api/cards/{card_id}")
def api_card(card_id: int):
    card = get_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return card


@app.post("/api/upload")
async def api_upload(
    file: UploadFile = File(...),
    mode: str = Query(default="auto", description="Segmentation mode: auto or grid"),
    rows: Optional[int] = Query(default=None, description="Grid rows (for grid mode)"),
    cols: Optional[int] = Query(default=None, description="Grid cols (for grid mode)"),
):
    """Upload a photo of cards in a grid layout."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file
    batch_id_str = uuid.uuid4().hex[:12]
    ext = Path(file.filename or "photo.jpg").suffix or ".jpg"
    upload_filename = f"batch_{batch_id_str}{ext}"
    upload_path = UPLOADS_DIR / upload_filename

    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Segment cards from the image
    card_output_dir = CARDS_DIR / batch_id_str
    try:
        if mode == "grid" and rows and cols:
            card_paths = segment_grid_uniform(str(upload_path), str(card_output_dir), rows, cols)
            grid_rows, grid_cols = rows, cols
        else:
            card_paths = segment_cards(str(upload_path), str(card_output_dir))
            grid_rows, grid_cols = None, None
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {e}")

    if not card_paths:
        raise HTTPException(
            status_code=422,
            detail="No cards detected. Try grid mode with explicit rows/cols.",
        )

    # Store batch and cards in DB
    batch_id = insert_batch(
        filename=file.filename or upload_filename,
        upload_path=str(upload_path),
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        card_count=len(card_paths),
    )

    card_ids = []
    for i, card_path in enumerate(card_paths):
        # Store relative path for serving
        rel_path = str(card_path.relative_to(CARDS_DIR))
        card_id = insert_card(batch_id, rel_path, i)
        card_ids.append(card_id)

    return {
        "batch_id": batch_id,
        "cards_detected": len(card_paths),
        "card_ids": card_ids,
    }


@app.post("/api/cards/{card_id}/identify")
def api_identify_card(card_id: int):
    """Run AI identification on a single card."""
    card = get_card(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")

    image_path = CARDS_DIR / card["image_path"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Card image not found")

    result = identify_card_with_claude(str(image_path))
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    update_card_identification(card_id, result)
    return get_card(card_id)


@app.post("/api/batches/{batch_id}/identify")
def api_identify_batch(batch_id: int):
    """Run AI identification on all pending cards in a batch."""
    cards, _ = get_cards({"status": "pending"}, limit=200)
    batch_cards = [c for c in cards if c["batch_id"] == batch_id]

    if not batch_cards:
        return {"message": "No pending cards in this batch", "identified": 0}

    identified = 0
    errors = []
    for card in batch_cards:
        image_path = CARDS_DIR / card["image_path"]
        if not image_path.exists():
            errors.append({"card_id": card["id"], "error": "Image not found"})
            continue

        result = identify_card_with_claude(str(image_path))
        if "error" in result:
            errors.append({"card_id": card["id"], "error": result["error"]})
            continue

        update_card_identification(card["id"], result)
        identified += 1

    return {"identified": identified, "errors": errors, "total": len(batch_cards)}


# Serve frontend
@app.get("/")
async def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/{path:path}")
async def serve_static(path: str):
    file_path = FRONTEND_DIR / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    return FileResponse(str(FRONTEND_DIR / "index.html"))
