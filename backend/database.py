"""CardVault database layer using SQLite."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

DB_PATH = Path(__file__).parent.parent / "cardvault.db"


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS batches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            upload_path TEXT NOT NULL,
            grid_rows INTEGER,
            grid_cols INTEGER,
            card_count INTEGER DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id INTEGER NOT NULL REFERENCES batches(id),
            image_path TEXT NOT NULL,
            grid_position INTEGER,

            -- Identification fields (filled by AI agent)
            player_name TEXT,
            year INTEGER,
            card_set TEXT,
            card_number TEXT,
            manufacturer TEXT,
            sport TEXT,

            -- Condition & value (filled by research agent)
            condition TEXT,
            condition_notes TEXT,
            rarity TEXT,
            estimated_price_low REAL,
            estimated_price_high REAL,
            price_currency TEXT DEFAULT 'USD',
            last_price_update TEXT,

            -- Metadata
            status TEXT NOT NULL DEFAULT 'pending',
            ai_confidence REAL,
            raw_ai_response TEXT,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_cards_status ON cards(status);
        CREATE INDEX IF NOT EXISTS idx_cards_player ON cards(player_name);
        CREATE INDEX IF NOT EXISTS idx_cards_sport ON cards(sport);
        CREATE INDEX IF NOT EXISTS idx_cards_year ON cards(year);
        CREATE INDEX IF NOT EXISTS idx_cards_rarity ON cards(rarity);
        CREATE INDEX IF NOT EXISTS idx_cards_price ON cards(estimated_price_high);
    """)
    conn.commit()
    conn.close()


def insert_batch(filename: str, upload_path: str, grid_rows: int, grid_cols: int, card_count: int) -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO batches (filename, upload_path, grid_rows, grid_cols, card_count) VALUES (?, ?, ?, ?, ?)",
        (filename, upload_path, grid_rows, grid_cols, card_count),
    )
    batch_id = cur.lastrowid
    conn.commit()
    conn.close()
    return batch_id


def insert_card(batch_id: int, image_path: str, grid_position: int) -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO cards (batch_id, image_path, grid_position) VALUES (?, ?, ?)",
        (batch_id, image_path, grid_position),
    )
    card_id = cur.lastrowid
    conn.commit()
    conn.close()
    return card_id


def update_card_identification(card_id: int, data: dict):
    conn = get_db()
    conn.execute(
        """UPDATE cards SET
            player_name = ?, year = ?, card_set = ?, card_number = ?,
            manufacturer = ?, sport = ?, condition = ?, condition_notes = ?,
            rarity = ?, estimated_price_low = ?, estimated_price_high = ?,
            ai_confidence = ?, raw_ai_response = ?,
            status = 'identified', updated_at = datetime('now'),
            last_price_update = datetime('now')
        WHERE id = ?""",
        (
            data.get("player_name"),
            data.get("year"),
            data.get("card_set"),
            data.get("card_number"),
            data.get("manufacturer"),
            data.get("sport"),
            data.get("condition"),
            data.get("condition_notes"),
            data.get("rarity"),
            data.get("estimated_price_low"),
            data.get("estimated_price_high"),
            data.get("ai_confidence"),
            json.dumps(data) if data else None,
            card_id,
        ),
    )
    conn.commit()
    conn.close()


def get_cards(filters: dict = None, sort_by: str = "id", sort_dir: str = "desc", limit: int = 50, offset: int = 0):
    conn = get_db()
    query = "SELECT * FROM cards WHERE 1=1"
    params = []

    if filters:
        if filters.get("status"):
            query += " AND status = ?"
            params.append(filters["status"])
        if filters.get("sport"):
            query += " AND sport = ?"
            params.append(filters["sport"])
        if filters.get("player_name"):
            query += " AND player_name LIKE ?"
            params.append(f"%{filters['player_name']}%")
        if filters.get("year_min"):
            query += " AND year >= ?"
            params.append(filters["year_min"])
        if filters.get("year_max"):
            query += " AND year <= ?"
            params.append(filters["year_max"])
        if filters.get("rarity"):
            query += " AND rarity = ?"
            params.append(filters["rarity"])
        if filters.get("price_min"):
            query += " AND estimated_price_high >= ?"
            params.append(filters["price_min"])
        if filters.get("price_max"):
            query += " AND estimated_price_low <= ?"
            params.append(filters["price_max"])

    allowed_sorts = {"id", "player_name", "year", "estimated_price_high", "rarity", "created_at"}
    if sort_by not in allowed_sorts:
        sort_by = "id"
    sort_dir = "ASC" if sort_dir.lower() == "asc" else "DESC"
    query += f" ORDER BY {sort_by} {sort_dir} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    rows = conn.execute(query, params).fetchall()
    total = conn.execute(
        query.split("ORDER BY")[0].replace("SELECT *", "SELECT COUNT(*)"),
        params[:-2],
    ).fetchone()[0]
    conn.close()
    return [dict(r) for r in rows], total


def get_card(card_id: int):
    conn = get_db()
    row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_batches():
    conn = get_db()
    rows = conn.execute("SELECT * FROM batches ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    conn = get_db()
    stats = {}
    stats["total_cards"] = conn.execute("SELECT COUNT(*) FROM cards").fetchone()[0]
    stats["identified"] = conn.execute("SELECT COUNT(*) FROM cards WHERE status = 'identified'").fetchone()[0]
    stats["pending"] = conn.execute("SELECT COUNT(*) FROM cards WHERE status = 'pending'").fetchone()[0]
    stats["total_batches"] = conn.execute("SELECT COUNT(*) FROM batches").fetchone()[0]
    row = conn.execute("SELECT SUM(estimated_price_low), SUM(estimated_price_high) FROM cards WHERE status = 'identified'").fetchone()
    stats["total_value_low"] = row[0] or 0
    stats["total_value_high"] = row[1] or 0
    conn.close()
    return stats
