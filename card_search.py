"""Look a card up online by its name and collector number.

Uses the free TCGdex API (https://tcgdex.dev), which needs no key.

    python card_search.py "Ultra Ball" 091/091
"""
from __future__ import annotations

import difflib
import json
import os
import re
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html import unescape as html_unescape

from energy_icons import describe as describe_energy

# Try IPv4 addresses before IPv6. This network has no working IPv6, and the
# card-image server (Limitless's CDN) lists IPv6 addresses first: each
# download waited ~16 s for IPv6 to time out before IPv4 connected in 0.15 s.
# IPv6 stays as a fallback. Applies to every web request in this process.
_getaddrinfo = socket.getaddrinfo


def _ipv4_first(*args, **kwargs):
    return sorted(_getaddrinfo(*args, **kwargs), key=lambda info: info[0] != socket.AF_INET)


socket.getaddrinfo = _ipv4_first

API = "https://api.tcgdex.net/v2/en"
HEADERS = {"User-Agent": "PokemonReader/1.0"}
TIMEOUT = 15
MAX_DETAIL_FETCHES = 8      # parallel requests
MAX_CANDIDATES = 24         # printings fetched in full to compare set totals
MAX_LISTED = 15             # printings printed when only the name is known


# --- web cache ----------------------------------------------------------------------
# Every answer from TCGdex and every Limitless page lookup is kept on disk for
# a week, so a card seen before is found without touching the network. Card
# data does not change; a week only lets new sets and fixes through.
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_cache.json")
CACHE_MAX_AGE = 7 * 24 * 3600
_cache: dict | None = None
_cache_lock = threading.Lock()


def _cache_get(key: str) -> tuple[bool, object]:
    global _cache
    with _cache_lock:
        if _cache is None:
            try:
                with open(CACHE_FILE, encoding="utf-8") as fh:
                    _cache = json.load(fh)
            except (OSError, ValueError):
                _cache = {}
        entry = _cache.get(key)
    if entry and time.time() - entry[0] < CACHE_MAX_AGE:
        return True, entry[1]
    return False, None


def _cache_put(key: str, value) -> None:
    with _cache_lock:
        if _cache is None:
            return
        _cache[key] = [time.time(), value]
        try:
            tmp = CACHE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(_cache, fh)
            os.replace(tmp, CACHE_FILE)
        except OSError:
            pass


def _get(path: str, params: dict | None = None, cache: bool = True):
    url = API + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    if cache:
        hit, value = _cache_get("api:" + url)
        if hit:
            return value
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        value = json.load(resp)
    if cache:
        _cache_put("api:" + url, value)
    return value


# TCG Pocket is a phone game; its cards (sets A1, A2a, P-A, B1 ...) are never
# the physical card in front of the camera.
POCKET_ID = re.compile(r"^(?:[AB]\d|P-A)")


ENERGY_TYPES = ("Grass", "Fire", "Water", "Lightning", "Psychic", "Fighting",
                "Darkness", "Metal", "Fairy")
BASIC_ENERGY_NAME = re.compile(r"(basic\s*)?energy", re.IGNORECASE)


def _query_names(query: str) -> list:
    try:
        return _get("/cards", {"name": query}) or []
    except Exception:
        return []


def _fetch_card(card_id: str) -> dict | None:
    for _attempt in range(2):           # TCGdex drops the odd request
        try:
            return _get("/cards/" + urllib.parse.quote(card_id))
        except Exception:
            pass
    return None


def _one_digit_off(read: str, official: int) -> bool:
    """Whether a read total differs from the real one by one digit."""
    real = str(official).zfill(len(read))
    return len(real) == len(read) and sum(a != b for a, b in zip(read, real)) == 1


def _same_number(local_id: str, number: str) -> bool:
    """'091' matches '91' and '091'; 'TG12' matches only 'TG12'."""
    a, b = str(local_id).upper(), number.upper()
    if a.isdigit() and b.isdigit():
        return int(a) == int(b)
    return a == b


# --- printed set codes ("PAF", "MEE") -------------------------------------------
# TCGdex lists each set's printed code. Fetching them takes one request per
# set, so they are cached next to this file and refreshed when a week old.
SET_CODES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "set_codes.json")
SET_CODES_MAX_AGE = 7 * 24 * 3600
_set_codes: dict[str, str] | None = None
_set_codes_lock = threading.Lock()


def set_codes() -> dict[str, str]:
    """{set id: printed code}, e.g. {"sv04.5": "PAF", "mee": "MEE"}."""
    global _set_codes
    with _set_codes_lock:
        if _set_codes is not None:
            return _set_codes
        try:
            if time.time() - os.path.getmtime(SET_CODES_FILE) < SET_CODES_MAX_AGE:
                with open(SET_CODES_FILE, encoding="utf-8") as fh:
                    _set_codes = json.load(fh)
                    return _set_codes
        except (OSError, ValueError):
            pass

        def code_of(set_id: str) -> tuple[str, str]:
            for _attempt in range(2):
                try:
                    # Not cached: these carry every set's full card list, and
                    # the codes themselves are saved in SET_CODES_FILE.
                    d = _get("/sets/" + urllib.parse.quote(set_id), cache=False)
                    return set_id, ((d.get("abbreviation") or {}).get("official") or "")
                except Exception:
                    pass
            return set_id, ""

        try:
            ids = [s["id"] for s in _get("/sets", cache=False)]
        except Exception:
            return {}                       # offline: no codes this run, retry later
        with ThreadPoolExecutor(max_workers=MAX_DETAIL_FETCHES) as pool:
            _set_codes = {sid: code for sid, code in pool.map(code_of, ids) if code}
        try:
            with open(SET_CODES_FILE, "w", encoding="utf-8") as fh:
                json.dump(_set_codes, fh, indent=0, sort_keys=True)
        except OSError:
            pass
        return _set_codes


def _code_score(read: str, code: str) -> float:
    """How well an OCR read ("MEEBI", "PAFO") fits a printed code ("MEE").

    The read often carries extra letters from the "EN" badge beside the code,
    so the best-fitting stretch of it is compared, not the whole string.
    """
    # O/0 and I/l/1 look alike on the badge ("3OC" for 30C, "SVl" for SVI);
    # fold them together on both sides, so letter codes like OBF and PAL
    # still match themselves.
    fold = str.maketrans("OIL", "011")
    read, code = read.upper().translate(fold), code.upper().translate(fold)
    if not read or not code:
        return 0.0
    if len(read) <= len(code):
        return difflib.SequenceMatcher(None, read, code).ratio()
    return max(difflib.SequenceMatcher(None, read[i:i + len(code)], code).ratio()
               for i in range(len(read) - len(code) + 1))


def best_set_code(read: str, wait: bool = True) -> str:
    """The real set code closest to an OCR read, for display; '' if none fits.

    Only three-letter codes: those are what cards print on the badge (the
    two-letter ones belong to old sets with no badge). With wait=False it
    returns '' rather than block while the code list is still downloading.
    """
    if not read:
        return ""
    loaded = set_codes() if wait else (_set_codes or {})
    codes = {c for c in loaded.values() if len(c) == 3}
    if not read or not codes:
        return ""
    score, code = max((_code_score(read, c), c) for c in codes)
    return code if score >= 0.67 else ""


# --- stamps ------------------------------------------------------------------------
# The stamps the scanner recognises, by the key it reports. A stamped card is
# listed under its plain printing; the stamp says which version it is (and,
# for the anniversary Pikachu on a card from another set, that it is a
# Classic Collection reprint - see below).
STAMPS = {
    "30": "30th Celebration stamp",
    "25": "25th Celebrations stamp",
    "?": "anniversary stamp",
    "play": "Play! Pokemon stamp (Prize Pack)",
}
ANNIVERSARY_STAMPS = ("30", "25", "?")


# --- anniversary Classic Collection reprints ------------------------------------
# Reprints that keep the original number but add a 25/30 logo. TCGdex's ids
# for them do not match the printed card, so they are found through
# Limitless, whose page for the original lists its reprints.
CLASSIC_SETS = {"cel25cc", "30th-c"}
ANNIVERSARY_SETS = {"cel25", "30th"} | CLASSIC_SETS
ANNIVERSARY_CODES = {"30": ("30C",), "25": ("CEL",), "?": ("30C", "CEL")}
LIMITLESS = "https://limitlesstcg.com"
TITLE = re.compile(r"<title>\s*(.*?)\s+-\s+(.*?)\s+\(([A-Z0-9]+)\)\s+#(\S+)\s+[–-]", re.S)


def _limitless_info(url: str) -> dict:
    """What one Limitless card page says, read once and cached:
    {"ok": page exists, "image": its card picture, "title": [name, set, code,
    number] or None, "reprints": {code: [paths]} for the anniversary sets}.

    The picture's address differs by era (Limitless's own CDN for newer sets,
    pokemontcg.io for old ones), so it is read from the page's og:image
    rather than built. A failed connection is not cached.
    """
    hit, info = _cache_get("limitless:" + url)
    if hit:
        return info
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            html = resp.read(400_000).decode("utf-8", "ignore")
    except urllib.error.HTTPError:
        info = {"ok": False, "image": "", "title": None, "reprints": {}}
        _cache_put("limitless:" + url, info)    # 404: Limitless names it differently
        return info
    except Exception:
        return {"ok": True, "image": "", "title": None, "reprints": {}}  # offline / slow
    image = OG_IMAGE.search(html)
    title = TITLE.search(html)
    info = {"ok": True,
            "image": (image.group(1) or image.group(2)) if image else "",
            "title": [html_unescape(g) for g in title.groups()] if title else None,
            "reprints": {code: list(dict.fromkeys(re.findall(
                r'href="(/cards/{}/[A-Za-z0-9_]+)"'.format(code), html)))
                for code in ("30C", "CEL")}}
    _cache_put("limitless:" + url, info)
    return info


def _anniversary_reprint(original: dict, anniversary: str) -> dict | None:
    """The Classic Collection reprint of `original` matching the logo read."""
    link = original.get("link", "")
    if "limitlesstcg.com/cards/" not in link:
        return None
    reprints = _limitless_info(link)["reprints"]
    for code in ANNIVERSARY_CODES.get(anniversary, ()):
        for path in reprints.get(code, []):
            page = _limitless_info(LIMITLESS + path)
            if not page["title"]:
                continue
            name, set_name, set_code, number = page["title"]
            return {"id": "limitless:{}/{}".format(set_code, number), "name": name,
                    "set": set_name, "number": number, "code": set_code,
                    "rarity": "Classic Collection reprint of {} {}".format(
                        original.get("set", ""), original.get("number", "")).strip(),
                    "link": LIMITLESS + path, "image": page["image"]}
    return None


def _link(card: dict) -> str:
    """A web page for the card.

    Limitless TCG has a page for every card with a printed set code, at
    /cards/<code>/<number> - including printings TCGdex has no picture for,
    such as basic energies. Older sets without a code fall back to TCGdex's
    picture of the card.
    """
    set_id, _, local_id = str(card.get("id", "")).rpartition("-")
    code = set_codes().get(set_id, "")
    local_id = str(card.get("localId") or local_id)
    # TCGdex numbers the Classic Collection reprints its own way (Palkia LV.X
    # is 30th-c-022, Limitless 30C/CC12), and 30C/22 is a different card.
    if set_id in CLASSIC_SETS:
        return _image(card)
    if code and local_id:
        number = local_id.lstrip("0") or "0" if local_id.isdigit() else local_id
        return "https://limitlesstcg.com/cards/{}/{}".format(code, number)
    return _image(card)


def _image(card: dict) -> str:
    image = card.get("image")
    return image + "/high.png" if image else ""


OG_IMAGE = re.compile(r'property="og:image"\s+content="([^"]+)"|og:image"\s+content="([^"]+)"')


def _limitless_page(url: str) -> tuple[bool, str]:
    """(page exists, its card picture)."""
    info = _limitless_info(url)
    return info["ok"], info["image"]


def _check_links(cards: list[dict]) -> None:
    """Use Limitless's picture of each card; where Limitless has no page
    (some promos are named differently there), fall back to TCGdex's."""
    pages = [c for c in cards if "limitlesstcg.com" in c.get("link", "")]
    with ThreadPoolExecutor(max_workers=MAX_DETAIL_FETCHES) as pool:
        for c, (ok, image) in zip(pages, pool.map(_limitless_page,
                                                  [c["link"] for c in pages])):
            if not ok:
                c["link"] = c.get("image", "")
            elif image:
                c["image"] = image


def _summary(card: dict) -> dict:
    s = card.get("set") or {}
    total = (s.get("cardCount") or {}).get("official")
    return {
        "id": card.get("id", ""),
        "name": card.get("name", ""),
        "set": s.get("name", ""),
        "number": "{}/{}".format(card.get("localId", "?"), total) if total
                  else str(card.get("localId", "?")),
        "rarity": card.get("rarity", ""),
        "link": _link(card),
        "image": _image(card),
    }


def search_card(name: str, number: str = "", energy: list[str] | None = None,
                set_code: str = "", anniversary: str = "") -> dict:
    """Find the printing that matches `name` and, if given, `number`.

    `energy` is the list of energy icons read off an energy card, e.g.
    ["Grass"] or ["Colorless", "Colorless"]. It names a basic energy's type
    (which its printed name leaves out) and is reported with the result.

    `set_code` is the raw OCR read of the set badge ("MEEBI"). It is only
    compared against the real codes of the printings that already match, so a
    blurry read still picks between, say, MEE and SVE.

    `anniversary` is the stamp the scanner found (a STAMPS key). It is
    reported with the result; an anniversary stamp ("30", "25", "?") on a
    card from another set also swaps the match for its Classic Collection
    reprint, which keeps the original's printed number.

    Returns {"found": bool, "text": short line for the screen, ...}.
    """
    result = _search(name.strip(), number, energy or [], set_code)
    if result.get("found") and set_code:
        result["cards"], result["set_code"] = _narrow_by_code(result["cards"], set_code)
    # The badge names a set, but nothing found is from it: TCGdex is missing
    # that printing (it lists MEE 1-8, not the 30th-stamped MEE 9-16). Ask
    # Limitless for that set and number directly.
    code = best_set_code(set_code)
    if code and number and not result.get("set_code"):
        card = _limitless_card(code, number, name.strip(), energy or [])
        if card:
            result = {"found": True, "text": "{} - {}".format(card["name"], number),
                      "cards": [card], "set_code": code}
    if set_code and not result.get("set_code"):
        result["set_code"] = best_set_code(set_code)   # what the badge said, for display
    if result.get("found"):
        _check_links(result["cards"][:MAX_LISTED])     # only the ones printed
    stamp = anniversary
    if stamp:
        result["stamp"] = STAMPS.get(stamp, stamp)
    # Cards of the anniversary sets themselves (30th Celebration, Celebrations)
    # carry the stamp too, and so do stamped energies (MEE 9-16); only any
    # other card with an anniversary stamp is a Classic Collection reprint.
    if (result.get("found") and stamp in ANNIVERSARY_STAMPS
            and not re.search(r"Energy$", name.strip(), re.I)
            and not any(c["id"].rpartition("-")[0] in ANNIVERSARY_SETS
                        for c in result["cards"])):
        originals = [c for c in result["cards"] if c["id"].rpartition("-")[0] not in CLASSIC_SETS]
        reprints = [r for r in (_anniversary_reprint(c, anniversary) for c in originals[:3]) if r]
        if reprints:
            result["cards"] = reprints
        else:
            result["stamp"] += " - no Classic Collection reprint found, showing the original"
    if energy:
        result["energy"] = describe_energy(energy)
    return result


def _limitless_card(code: str, number: str, name: str, energy: list[str]) -> dict | None:
    """The card at Limitless /cards/<code>/<number>, if its name fits.

    For a basic energy (read as just "Basic Energy") the icon's type must be
    in the name instead, so a misread number cannot land on another energy.
    """
    num = number.partition("/")[0]
    num = num.lstrip("0") or "0" if num.isdigit() else num
    url = "{}/cards/{}/{}".format(LIMITLESS, code, num)
    page = _limitless_info(url)
    if not page["ok"] or not page["title"]:
        return None
    found_name, set_name, found_code, found_number = page["title"]
    if BASIC_ENERGY_NAME.fullmatch(name):
        typed = [t for t in energy if t in ENERGY_TYPES]
        fits = bool(typed) and typed[0].lower() in found_name.lower()
    else:
        fits = difflib.SequenceMatcher(None, name.lower(), found_name.lower()).ratio() >= 0.5
    if not fits:
        return None
    return {"id": "limitless:{}/{}".format(found_code, found_number), "name": found_name,
            "set": set_name, "number": number, "code": found_code, "rarity": "",
            "link": url, "image": page["image"]}


def _narrow_by_code(cards: list[dict], read: str) -> tuple[list[dict], str]:
    """Keep the printings whose set code best fits the badge read."""
    codes = set_codes()
    scored = []
    for c in cards:
        code = codes.get(c["id"].rsplit("-", 1)[0], "")
        scored.append((_code_score(read, code), code, c))
    best = max(s for s, _, _ in scored)
    if best < 0.67:
        return cards, ""                    # the read fits none of them: keep all
    keep = [(code, c) for s, code, c in scored if s == best]
    return [c for _, c in keep], keep[0][0]


def _sets_for_code(read: str) -> list[str]:
    """Set ids whose printed code best fits a badge read ('' -> none)."""
    if not read:
        return []
    scored = [(_code_score(read, code), sid) for sid, code in set_codes().items()
              if len(code) == 3]
    if not scored:
        return []
    best = max(s for s, _ in scored)
    return [sid for s, sid in scored if s == best] if best >= 0.67 else []


def _by_set_and_number(name: str, number: str, code_read: str) -> list:
    """Find a card by set code and number when its name did not match.

    OCR can lose part of a name - an energy icon printed inside it
    ("Telepathic <P> Energy" is "Telepathic Psychic Energy"), or a first
    letter ("let Energy" for Jet Energy). The badge and number still pin the
    card down; a loose name check guards against a misread number.
    """
    num = number.partition("/")[0]
    found = []
    for set_id in _sets_for_code(code_read):
        try:
            cards = _get("/sets/" + urllib.parse.quote(set_id)).get("cards") or []
        except Exception:
            continue
        for c in cards:
            if (_same_number(c.get("localId", ""), num)
                    and difflib.SequenceMatcher(None, name.lower(),
                                                c.get("name", "").lower()).ratio() >= 0.5):
                found.append(dict(c, id=c.get("id") or "{}-{}".format(set_id, c["localId"])))
    return found


def _search(name: str, number: str, energy: list[str], set_code: str = "") -> dict:
    if not name:
        return {"found": False, "text": "no name to search"}

    results = []
    if BASIC_ENERGY_NAME.fullmatch(name):
        # A basic energy's type is only an icon, so the name reads as just
        # "Energy". The icon says which type; without it, gather every basic
        # energy and let the number decide.
        types = [t for t in energy if t in ENERGY_TYPES][:1] or list(ENERGY_TYPES)
        if not number and len(types) > 1:
            return {"found": False, "text": "basic energy - no number or icon read to tell which"}
        queries = ["eq:{} Energy".format(t) for t in types]
        queries += ["eq:Basic {} Energy".format(t) for t in types]
        with ThreadPoolExecutor(max_workers=MAX_DETAIL_FETCHES) as pool:
            for found in pool.map(_query_names, queries):
                results += found
    else:
        tries = ["eq:" + name]
        if re.search(r"pokemon", name, re.I):
            # OCR usually drops the accent; TCGdex spells it "Pokémon".
            tries.append("eq:" + re.sub(r"pokemon", "Pokémon", name, flags=re.I))
        if re.search(r"\sLV\.X$", name, re.I):
            # TCGdex lists LV.X cards under the plain name ("Palkia"), with
            # "LV.X" only in the rarity.
            tries.append("eq:" + re.sub(r"\s*LV\.X$", "", name, flags=re.I))
        typed = [t for t in energy if t in ENERGY_TYPES]
        if typed and re.search(r"\bEnergy$", name, re.I):
            # An energy icon printed inside the name is lost to OCR:
            # "Telepathic <P> Energy" is listed as "Telepathic Psychic Energy".
            stem = re.sub(r"\s*Energy$", "", name, flags=re.I)
            tries.append("eq:{} {} Energy".format(stem, typed[0]).strip())
        # The exact-name variants go out together and the first (in the order
        # above) that finds anything wins; the slow "contains" search is only
        # tried when none of them did.
        with ThreadPoolExecutor(max_workers=len(tries)) as pool:
            for found in pool.map(_query_names, tries):
                if found:
                    results = found
                    break
        if not results:
            results = _query_names(name)    # contains, any case
    results = [c for c in results if not POCKET_ID.match(c.get("id", ""))]

    num = number.partition("/")[0]
    if number and not any(_same_number(c.get("localId", ""), num) for c in results):
        # The name missed (or matched other printings only): try the badge.
        results = _by_set_and_number(name, number, set_code) or results
    if (number and not BASIC_ENERGY_NAME.fullmatch(name)
            and not any(_same_number(c.get("localId", ""), num) for c in results)):
        # Still no printing with this number - and older cards have no set
        # code to go on. TCGdex may list the card under a longer name: a
        # symbol OCR drops ("Diancie ◇" for Prism Star) or a suffix. Widen to
        # every name containing the one read; the number and set total pick.
        wider = [c for c in _query_names(name) if not POCKET_ID.match(c.get("id", ""))]
        if any(_same_number(c.get("localId", ""), num) for c in wider):
            results = wider
    if not results:
        return {"found": False, "text": "not found: " + name}

    if number:
        # With a number, only printings matching both name and number count -
        # never fall back to other printings of the name.
        num, _, total = number.partition("/")
        candidates = [c for c in results if _same_number(c.get("localId", ""), num)]
        # Printed exactly as read ("008", not "8") narrows it to the same era.
        exact = [c for c in candidates if str(c.get("localId", "")).upper() == num.upper()]
        candidates = exact or candidates
        if not candidates:
            return {"found": False, "text": "no {} numbered {}".format(name, number)}
        candidates = candidates[:MAX_CANDIDATES]
        with ThreadPoolExecutor(max_workers=MAX_DETAIL_FETCHES) as pool:
            # Load each candidate's Limitless page alongside its TCGdex details
            # (the link is known from the listing alone); the link check after
            # the search then finds them already cached.
            pages = [_link(c) for c in candidates]
            for page in pages:
                if "limitlesstcg.com" in page:
                    pool.submit(_limitless_info, page)
            fetched = list(pool.map(_fetch_card, [c["id"] for c in candidates]))
        cards, near = [], []
        for listed, detail in zip(candidates, fetched):
            if detail is None:
                # Lookup failed: name and number still match, set unconfirmed.
                cards.append({"id": listed.get("id", ""), "name": listed.get("name", ""),
                              "set": "", "number": str(listed.get("localId", "?")),
                              "rarity": "", "link": _link(listed),
                              "image": _image(listed)})
                continue
            # Many sets share a number; the printed total must match too.
            official = (detail.get("set") or {}).get("cardCount", {}).get("official")
            if total.isdigit() and official is not None and official != int(total):
                if _one_digit_off(total, official):
                    near.append(_summary(detail))
                continue
            cards.append(_summary(detail))
        if not cards and near:
            # No exact total, but one off by a single misread digit (131 read
            # as 135): that is the card.
            cards = near
        if not cards:
            return {"found": False, "text": "no {} numbered {}".format(name, number)}
        return {"found": True, "text": "{} - {}".format(cards[0]["name"], number),
                "cards": cards}

    # No number read: list every printing of the name.
    cards = [{"id": c.get("id", ""), "name": c.get("name", ""), "set": "",
              "number": str(c.get("localId", "?")), "rarity": "", "link": _link(c),
              "image": _image(c)}
             for c in results]
    return {"found": True, "text": "{} - {} printings".format(cards[0]["name"], len(cards)),
            "cards": cards}


def format_result(result: dict, name: str, number: str) -> str:
    """The lines printed in the terminal for one search."""
    lines = ["", "=" * 60,
             "Searched: {}{}".format(name, "  #" + number if number else "  (no number)")]
    if result.get("energy"):
        lines.append("  Energy: " + result["energy"])
    if result.get("set_code"):
        lines.append("  Set code: " + result["set_code"])
    if result.get("stamp"):
        lines.append("  Stamp: " + result["stamp"])
    if not result.get("found"):
        lines.append("  " + result.get("text", "not found"))
        return "\n".join(lines)
    cards = result["cards"]
    if not number:
        lines.append("  {} printings of this name:".format(len(cards)))
    elif len(cards) > 1:
        lines.append("  {} printings match this name and number:".format(len(cards)))
    for c in cards[:MAX_LISTED]:
        where = c["set"] or c["id"].rsplit("-", 1)[0]
        lines.append("  {}  |  {}  {}{}".format(
            c["name"], where, c["number"], "  |  " + c["rarity"] if c["rarity"] else ""))
        if c["link"]:
            lines.append("    " + c["link"])
    if len(cards) > MAX_LISTED:
        lines.append("  ... and {} more".format(len(cards) - MAX_LISTED))
    return "\n".join(lines)


if __name__ == "__main__":
    # python card_search.py NAME [NUMBER] [ENERGY,ENERGY...] [SETCODE]
    n = sys.argv[2] if len(sys.argv) > 2 else ""
    e = [x for x in sys.argv[3].split(",") if x] if len(sys.argv) > 3 else []
    code = sys.argv[4] if len(sys.argv) > 4 else ""
    print(format_result(search_card(sys.argv[1], n, e, code), sys.argv[1], n))
