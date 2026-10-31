"""The scanned-card collection, kept in SQLite (cards.db next to this file).

The scanner adds each identified card with add_card(); scanning a card that
is already in the collection raises its quantity instead of adding a row. A
stamped copy (Play! Pokemon, 30th Celebration ...) is its own row.
card_app.py reads it back with search().

    python card_db.py refresh-images   # re-point saved cards at Limitless pictures
"""
from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime

DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id          INTEGER PRIMARY KEY,
    tcgdex_id   TEXT NOT NULL,            -- e.g. "sv04.5-091"
    stamp       TEXT NOT NULL DEFAULT '', -- e.g. "Play! Pokemon stamp (Prize Pack)"
    name        TEXT NOT NULL,
    card_type   TEXT NOT NULL,            -- Pokemon, Item, Supporter, Basic Energy ...
    set_name    TEXT NOT NULL DEFAULT '',
    set_code    TEXT NOT NULL DEFAULT '', -- printed code, e.g. "PAF"
    number      TEXT NOT NULL DEFAULT '', -- as listed, e.g. "091/91"
    number_sort INTEGER NOT NULL DEFAULT 0,
    rarity      TEXT NOT NULL DEFAULT '',
    energy      TEXT NOT NULL DEFAULT '', -- energy icons, energy cards only
    link        TEXT NOT NULL DEFAULT '',
    image       TEXT NOT NULL DEFAULT '',
    quantity    INTEGER NOT NULL DEFAULT 1,
    first_scanned TEXT NOT NULL,
    last_scanned  TEXT NOT NULL,
    UNIQUE (tcgdex_id, stamp)             -- a stamped copy is its own row
);
"""

# Columns the app may sort by, mapped to SQL. Kept as a whitelist because the
# sort column cannot be passed as a query parameter.
SORTS = {
    "name": "name COLLATE NOCASE",
    "type": "card_type COLLATE NOCASE",
    "set": "set_name COLLATE NOCASE",
    "number": "set_name COLLATE NOCASE, number_sort",
    "rarity": "rarity COLLATE NOCASE",
    "quantity": "quantity",
    "scanned": "last_scanned",
    "stamp": "stamp COLLATE NOCASE",
}


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, timeout=10)
    conn.row_factory = sqlite3.Row
    # WAL lets the app read while the scanner writes.
    conn.execute("PRAGMA journal_mode=WAL")
    _migrate(conn)
    conn.executescript(SCHEMA)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Bring a collection saved before stamps were tracked up to date.

    That table made tcgdex_id unique on its own, so a stamped copy could not
    sit beside a plain one. SQLite cannot change a constraint in place, so
    the table is rebuilt with every existing card kept (as unstamped).
    """
    cols = [r[1] for r in conn.execute("PRAGMA table_info(cards)")]
    if not cols or "stamp" in cols:
        return
    with conn:
        conn.execute("ALTER TABLE cards RENAME TO cards_old")
        conn.executescript(SCHEMA)
        keep = ", ".join(c for c in cols if c != "id")
        conn.execute("INSERT INTO cards ({0}) SELECT {0} FROM cards_old".format(keep))
        conn.execute("DROP TABLE cards_old")


def _number_sort(number: str) -> int:
    m = re.search(r"\d+", number or "")
    return int(m.group()) if m else 0


def add_card(card: dict, card_type: str, set_code: str = "", energy: str = "",
             copies: int = 1, stamp: str = "") -> int:
    """Save `copies` of one identified card (a card dict from card_search);
    returns its quantity afterwards. A card already saved with the same stamp
    (or both unstamped) gets the copies added to its count; a different stamp
    is a separate row."""
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        row = conn.execute("SELECT quantity FROM cards WHERE tcgdex_id = ? AND stamp = ?",
                           (card["id"], stamp)).fetchone()
        if row:
            conn.execute("UPDATE cards SET quantity = quantity + ?, last_scanned = ? "
                         "WHERE tcgdex_id = ? AND stamp = ?", (copies, now, card["id"], stamp))
            return row["quantity"] + copies
        conn.execute(
            "INSERT INTO cards (tcgdex_id, stamp, name, card_type, set_name, set_code, number,"
            " number_sort, rarity, energy, link, image, quantity, first_scanned,"
            " last_scanned) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (card["id"], stamp, card.get("name", ""), card_type, card.get("set", ""),
             set_code, card.get("number", ""), _number_sort(card.get("number", "")),
             card.get("rarity", ""), energy, card.get("link", ""), card.get("image", ""),
             copies, now, now))
        return copies


def search(name: str = "", card_type: str = "", set_name: str = "",
           sort: str = "name", descending: bool = False, stamp: str = "") -> list[dict]:
    """Cards matching every filter given; name is a case-insensitive
    'contains', type, set and stamp are exact ("none" = unstamped only)."""
    where, args = [], []
    if stamp == "none":
        where.append("stamp = ''")
    elif stamp:
        where.append("stamp = ?")
        args.append(stamp)
    if name:
        where.append("name LIKE ?")
        args.append("%{}%".format(name))
    if card_type:
        where.append("card_type = ?")
        args.append(card_type)
    if set_name:
        where.append("set_name = ?")
        args.append(set_name)
    order = SORTS.get(sort, SORTS["name"])
    if descending:
        order = ", ".join(part + " DESC" for part in order.split(", "))
    sql = "SELECT * FROM cards"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY " + order + ", name COLLATE NOCASE"
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args)]


def choices() -> dict:
    """The distinct types and sets present, for the app's dropdowns."""
    with connect() as conn:
        types = [r[0] for r in conn.execute(
            "SELECT DISTINCT card_type FROM cards ORDER BY card_type COLLATE NOCASE")]
        sets = [r[0] for r in conn.execute(
            "SELECT DISTINCT set_name FROM cards WHERE set_name != '' "
            "ORDER BY set_name COLLATE NOCASE")]
        stamps = [r[0] for r in conn.execute(
            "SELECT DISTINCT stamp FROM cards WHERE stamp != '' "
            "ORDER BY stamp COLLATE NOCASE")]
        totals = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(quantity), 0) FROM cards").fetchone()
    return {"types": types, "sets": sets, "stamps": stamps,
            "unique": totals[0], "total": totals[1]}


def refresh_images() -> int:
    """Point every saved card at its Limitless picture; returns how many changed."""
    from card_search import _limitless_page   # network code lives there

    with connect() as conn:
        rows = conn.execute("SELECT id, link, image FROM cards "
                            "WHERE link LIKE '%limitlesstcg.com%'").fetchall()
    changed = 0
    for row in rows:
        ok, image = _limitless_page(row["link"])
        if ok and image and image != row["image"]:
            with connect() as conn:
                conn.execute("UPDATE cards SET image = ? WHERE id = ?", (image, row["id"]))
            changed += 1
    return changed


def set_quantity(row_id: int, quantity: int) -> None:
    """Change how many copies are held; 0 removes the card."""
    with connect() as conn:
        if quantity <= 0:
            conn.execute("DELETE FROM cards WHERE id = ?", (row_id,))
        else:
            conn.execute("UPDATE cards SET quantity = ? WHERE id = ?", (quantity, row_id))


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["refresh-images"]:
        print("updated {} card pictures".format(refresh_images()))
    else:
        print(__doc__)
