"""Find the four corners of a Pokemon card and draw a border around them.

    python card_border.py                 # live camera
    python card_border.py --image a.jpg   # one photo, writes card_border_out.jpg

Press q or Esc to quit the live view.
"""
from __future__ import annotations

import argparse
import queue
import re
from collections import Counter
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

import card_db
from card_search import STAMPS, best_set_code, format_result, search_card, set_codes
from energy_icons import describe as describe_energy, find_icons

# A real card is 63mm x 88mm, so short/long side = 0.716 whichever way up it is.
CARD_ASPECT = 63.0 / 88.0
ASPECT_TOLERANCE = 0.13      # accepts 0.59 - 0.85, enough for a tilted card

MIN_AREA_FRAC = 0.05         # ignore anything smaller than 5% of the frame
MAX_AREA_FRAC = 0.97         # ...or basically the whole frame
MIN_RECTANGULARITY = 0.80    # contour area vs its own min-area rectangle
HULL_RECTANGULARITY = 0.95   # ...far stricter for a convex-hull candidate
HULL_MAX_AREA_FRAC = 0.50
MAX_CORNER_SKEW = 20.0       # degrees a corner may be off square
CONTAINER_MAX_SKEW = 10.0    # ...stricter for a shape taking over from the winner
HULL_MAX_SKEW = 10.0         # ...and for a convex-hull candidate

WORK_WIDTH = 720             # detection runs downscaled; corners are scaled back
GRADIENT_PERCENTILE = 95     # Canny thresholds come from edge strength
MIN_CANNY_HIGH = 18

GREEN = (0, 230, 0)
AMBER = (60, 170, 255)


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left.

    The top-left has the smallest x+y and the bottom-right the largest, while
    the top-right has the smallest y-x and the bottom-left the largest.
    """
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    out = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    out[0] = pts[np.argmin(s)]
    out[1] = pts[np.argmin(d)]
    out[2] = pts[np.argmax(s)]
    out[3] = pts[np.argmax(d)]
    return out


def aspect_of(quad: np.ndarray) -> float:
    """Short side / long side, so the card's orientation does not matter."""
    tl, tr, br, bl = quad
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if w < 1 or h < 1:
        return 0.0
    return min(w, h) / max(w, h)


# The candidates the last find_corners() call weighed (score, area, quad,
# from-hull, aspect score), for diagnosing a wrong pick.
last_candidates: list = []


def max_corner_skew(quad: np.ndarray) -> float:
    """How far (degrees) the quad's least square corner is from 90."""
    worst = 0.0
    for i in range(4):
        a = quad[i - 1] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-6))
        worst = max(worst, abs(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) - 90.0))
    return worst


def auto_canny(gray: np.ndarray) -> np.ndarray:
    """Canny with thresholds taken from this frame's gradient distribution.

    The usual recipe bases them on the brightness median, which fails on the
    very common case of a pale card lying on a pale desk: a median of 160 asks
    for a high threshold of 212, while the card's edge against light wood is
    only ~25 grey levels, so the outline is discarded and nothing is found.
    Thresholding on how strong the edges actually are keeps the border whatever
    the contrast between card and surface.
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    hi = float(np.percentile(cv2.magnitude(gx, gy), GRADIENT_PERCENTILE))
    hi = max(MIN_CANNY_HIGH, min(255.0, hi))
    return cv2.Canny(gray, int(0.4 * hi), int(hi))


def binary_views(small: np.ndarray) -> list[np.ndarray]:
    """Two ways of separating the card from the background.

    Edges work when the card contrasts with the surface; saturation covers the
    case where it does not, because a printed card is far more colourful than
    wood, paper or a desk mat.
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    # Bilateral keeps the card's border crisp while flattening the artwork.
    gray = cv2.bilateralFilter(gray, 7, 60, 60)
    views = [auto_canny(gray)]

    sat = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
    if float(sat.std()) > 8.0:
        sat = cv2.GaussianBlur(sat, (5, 5), 0)
        _, mask = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        views.append(mask)

    kernel = np.ones((3, 3), np.uint8)
    # Close gaps so a border broken by glare still forms one loop.
    views = [cv2.erode(cv2.dilate(v, kernel, iterations=2), kernel) for v in views]

    # Brightness covers a pale card on a dark, busy surface (a dotted mat),
    # where the edge outline fuses with the texture and saturation only picks
    # out the artwork. Opening strips the texture's specks off the card blob.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    views.append(cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)))
    return views


def find_corners(frame: np.ndarray) -> np.ndarray | None:
    """Return the card's 4 corners in frame coordinates, or None.

    Ordered top-left, top-right, bottom-right, bottom-left.
    """
    if frame is None or frame.size == 0:
        return None

    h, w = frame.shape[:2]
    scale = WORK_WIDTH / float(w) if w > WORK_WIDTH else 1.0
    small = (cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
             if scale != 1.0 else frame)
    area_small = float(small.shape[0] * small.shape[1])

    # score, area, quad, and whether it came from a convex hull
    candidates: list[tuple[float, float, np.ndarray, bool]] = []
    for view in binary_views(small):
        contours, _ = cv2.findContours(view, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        top = sorted(contours, key=cv2.contourArea, reverse=True)[:12]
        # Each outline's convex hull is a candidate too: a card's dark
        # artwork can carve into its outline (seen on a sleeved card), which
        # then fails the rectangle test although its hull is the card.
        shapes = [(c, False) for c in top]
        for c in top:
            hull = cv2.convexHull(c)
            if cv2.contourArea(c) < 0.85 * cv2.contourArea(hull):
                shapes.append((hull, True))
        for cnt, is_hull in sorted(shapes, key=lambda s: cv2.contourArea(s[0]), reverse=True):
            area = cv2.contourArea(cnt)
            if area < area_small * MIN_AREA_FRAC:
                break        # sorted descending, everything after is smaller
            if area > area_small * MAX_AREA_FRAC:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            quad = None
            # A card's corners are rounded, so try a few fit tolerances.
            for eps in (0.02, 0.03, 0.045, 0.06):
                approx = cv2.approxPolyDP(cnt, eps * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = approx.reshape(4, 2).astype(np.float32)
                    break
            if quad is None:
                # No clean quad: fall back to the rotated bounding box, which
                # still recovers a card whose edge is partly lost to glare.
                quad = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)

            quad = order_corners(quad)
            rect = cv2.minAreaRect(quad)[1]
            rect_area = rect[0] * rect[1]
            rectangularity = area / rect_area if rect_area > 0 else 0.0
            if rectangularity < MIN_RECTANGULARITY:
                continue
            # A hull qualifies only as a near-perfect rectangle of card size:
            # real card hulls measured 0.98, hulls of clutter 0.91 or less.
            if is_hull and (rectangularity < HULL_RECTANGULARITY
                            or area > area_small * HULL_MAX_AREA_FRAC):
                continue

            aspect = aspect_of(quad)
            if abs(aspect - CARD_ASPECT) > ASPECT_TOLERANCE:
                continue
            # Seen from above, a card's corners are near right angles. A box
            # edge in perspective can still average out to card proportions,
            # but as a lopsided four-sided shape - reject those. Hulls are
            # held to a tighter limit (real card hulls measured 2-6 degrees,
            # a hull around the card plus clutter 14).
            skew = max_corner_skew(quad)
            if skew > (HULL_MAX_SKEW if is_hull else MAX_CORNER_SKEW):
                continue
            # Something running off the edge of the picture (a box at the
            # side of the desk) is not a whole card - and a card cut off
            # could not be read anyway.
            on_edge = sum(1 for x, y in quad if x <= 3 or y <= 3
                          or x >= small.shape[1] - 4 or y >= small.shape[0] - 4)
            if on_edge >= 2:
                continue

            # Prefer card-like proportions, then a healthy share of the frame.
            aspect_score = max(0.0, 1.0 - abs(aspect - CARD_ASPECT) / ASPECT_TOLERANCE)
            fill_score = min(1.0, (area / area_small) / 0.35)
            score = 0.6 * aspect_score + 0.2 * fill_score + 0.2 * rectangularity

            candidates.append((score, area, quad, is_hull, aspect_score))

    global last_candidates
    last_candidates = [(s, a / scale ** 2, q / scale, h, asp) for s, a, q, h, asp in candidates]
    # Hulls are a fallback for when no real outline qualifies: a hull also
    # forms around the card merged with clutter beside it, and being bigger it
    # would outscore the card's own outline.
    if any(not c[3] for c in candidates):
        candidates = [c for c in candidates if not c[3]]
    if not candidates:
        return None
    _score, best_area, best, _, best_aspect = max(candidates, key=lambda c: c[0])
    # The artwork window is itself a card-shaped rectangle. If the winner sits
    # inside a larger candidate, that larger one is the card around it - as
    # long as it is at least as card-shaped. Only real outlines may take over
    # this way, and only square-cornered ones: the card merged with clutter
    # beside it (a tablet, a box) also contains the card, but is a worse shape.
    cx, cy = best.mean(axis=0)
    for _s, area, quad, is_hull, aspect_score in candidates:
        if (not is_hull and area > best_area * 1.3
                and aspect_score >= best_aspect - 0.1
                and max_corner_skew(quad) <= CONTAINER_MAX_SKEW
                and cv2.pointPolygonTest(quad.reshape(-1, 1, 2), (float(cx), float(cy)),
                                         False) > 0):
            best, best_area, best_aspect = quad, area, aspect_score
    return best / scale


# --- card type ---------------------------------------------------------------
# The flattened card, and the slice of it that carries the type wording.
CARD_W, CARD_H = 734, 1024
# The type is read from the top-left tag alone (the name is read separately,
# from the same band stretched to NAME_X1). Sized from real cards: the longest tag ("Pokemon Tool") reaches about 40%
# across, and every tag sits inside the top 12%.
TYPE_X0, TYPE_X1 = 0.00, 0.45
TYPE_Y0, TYPE_Y1 = 0.00, 0.12

TYPE_COLOR = (255, 170, 0)

# The blue box drawn on screen is the tag box stretched across the card, so it
# also takes in the name (and, on a Pokemon, the HP, which is filtered out).
NAME_X1 = 0.97

NAME_COLOR = (255, 255, 255)
SEARCH_COLOR = (0, 230, 0)
# Search once this many reads in a row (each ~0.5 s) agree on type, name,
# number and energy - quicker than a fixed wait, and just as safe against a
# card still being moved into place.
READS_TO_AGREE = 2

# The collector number, printed along the bottom-left edge after the set code
# ("PAF EN 091/091"). Measured on a flattened card: the line sits at 94-96%
# down and 7-31% across; the box stops short of the item-rule panel at 34%.
SET_X0, SET_X1 = 0.03, 0.34
SET_Y0, SET_Y1 = 0.92, 0.99

SET_COLOR = (255, 0, 200)

# Older cards (Diamond & Pearl and before) print the number at the bottom
# RIGHT instead, beside the set symbol ("106/106"). Read only when the
# bottom-left box has no number.
NUM_R_X0, NUM_R_X1 = 0.60, 0.985
NUM_R_Y0, NUM_R_Y1 = 0.935, 0.995
# Top of the taller last-resort boxes, for a card sitting inset in a sleeve.
NUM_TALL_Y0 = 0.87

# The word in that corner, mapped to the card type. Stage wording means it is
# a Pokemon; every other tag names its own type.
STAGE_WORDS = ("BASIC", "STAGE1", "STAGE2", "STAGE", "VSTAR", "VMAX", "RESTORED",
               "LEVELUP", "BREAK", "MEGA")

SPECIAL_ENERGY_KEYS = ("SPECIALENERGY", "SPECLALENERGY", "SPECIALENERG", "SPEClALENERGY")
BASIC_ENERGY_KEYS = ("BASICENERGY", "BASICENERG")

TAG_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("Pokemon Tool", ("POKEMONTOOL", "POKMONTOOL", "POKEMONTOOI", "PKMNTOOL",
                      "POKEMONTO0L")),
    ("Supporter", ("SUPPORTER", "SUPPORTE", "SUPP0RTER")),
    ("Stadium", ("STADIUM", "STADIU", "STADIOM")),
    ("Item", ("ITEM", "ITEN", "LTEM")),
    # Scarlet & Violet tags say which kind of energy; checked before plain
    # "Energy", which they both contain.
    ("Special Energy", SPECIAL_ENERGY_KEYS),
    ("Basic Energy", BASIC_ENERGY_KEYS),
    ("Energy", ("ENERGY", "ENERG", "EMERGY")),
    ("Trainer", ("TRAINER", "TRAINE", "TRAlNER")),
]


def classify_type(texts: list[str]) -> str:
    """Name the card type from the single word in the card's top-left corner."""
    squashed = "".join(ch for ch in " ".join(texts).upper() if ch.isalnum())
    if not squashed:
        return "Unknown"

    # Named tags are tested before the stage words, and the order matters: a
    # basic energy card's tag reads "Basic Energy", which contains "BASIC", so
    # checking stage words first would call every basic energy a Pokemon.
    for name, keys in TAG_WORDS:
        if any(k in squashed for k in keys):
            return name

    # Nothing named it, so a stage word means it is a Pokemon.
    if any(word in squashed for word in STAGE_WORDS):
        return "Pokemon"
    return "Unknown"


# Only real number prefixes (Trainer Gallery, Galarian Gallery, ...), so stray
# set-code letters next to the digits are not swallowed ("PF091/091").
NUMBER_RE = re.compile(r"((?:TG|GG|SV|RC|SH)?\d{1,3})\s*/\s*((?:TG|GG|SV|RC|SH)?\d{1,3})")


SLASH_AS_DIGIT_RE = re.compile(r"(?<!\d)(\d{1,3})[71LI|](\d{3})(?!\d)")

# Basic energies print the number alone, with no set total ("MEE 008").
BARE_NUMBER_RE = re.compile(r"(?<!\d)(\d{3})(?!\d)")


def parse_number(texts: list[str], allow_bare: bool = False) -> str:
    """Pull the collector number ("091/091") out of the bottom-left OCR text.

    With `allow_bare`, a lone three-digit number ("008") is accepted too.
    """
    joined = " ".join(texts).upper()
    # OCR reads a zero as the letter O now and then ("2O1/198").
    joined = re.sub(r"(?<=\d)O|O(?=\d)", "0", joined)
    m = NUMBER_RE.search(joined)
    if m:
        return "{}/{}".format(m.group(1), m.group(2))
    # Older cards print the slash in italics and OCR reads it as 7, 1 or l:
    # "1067106" is 106/106. Accepted only when the total is three digits and
    # the number is not far past it (secret rares run a little over).
    for m in SLASH_AS_DIGIT_RE.finditer(joined):
        num, total = int(m.group(1)), int(m.group(2))
        if total >= 10 and 0 < num <= total * 1.5:
            return "{}/{}".format(m.group(1), m.group(2))
    if allow_bare:
        m = BARE_NUMBER_RE.search(joined)
        if m:
            return m.group(1)
    return ""


def _is_header_noise(text: str) -> bool:
    """Text in the top band that is not part of the name."""
    squashed = "".join(ch for ch in text.upper() if ch.isalnum())
    if len(squashed) < 2 or squashed.isdigit():
        return True
    # "Energy" in mixed case is part of a name ("Double Turbo Energy"); only
    # the all-caps ENERGY banner is decoration.
    if squashed == "ENERGY" and not text.isupper():
        return False
    # HP, also as OCR mangles it on older cards: "#P120", "MP120", "WP120".
    if re.fullmatch(r"[A-Z]{0,2}\d+|HP|TERA|TRAINER|TRAINE", squashed):
        return True
    if "EVOLVES" in squashed or squashed.startswith("STAGE"):
        return True
    if squashed in STAGE_WORDS:
        return True
    return any(squashed in keys for _name, keys in TAG_WORDS)


def has_hp(boxes: list[tuple[str, float, float, float]]) -> bool:
    """Whether the top band shows an HP value - which only Pokemon have.

    Older cards (Diamond & Pearl and before) print no stage tag in the top
    left, so this is how they are recognised. The HP sits in the right half
    and is a multiple of ten; OCR often mangles the "HP" itself ("#P120").
    """
    for text, _h, x, _y in boxes:
        if x < CARD_W * 0.45:
            continue
        # "HP" reads as anything two letters long: "MP120", "WP120", "#P120".
        m = re.fullmatch(r"[A-Z]{0,2}(\d{2,3})", "".join(ch for ch in text.upper()
                                                         if ch.isalnum()))
        if m and 30 <= int(m.group(1)) <= 400 and int(m.group(1)) % 10 == 0:
            return True
    return False


def parse_name(boxes: list[tuple[str, float, float, float]], card_type: str) -> str:
    """Pick the name out of the top band's OCR boxes (text, height, x, y).

    The name is printed in the largest type in that band; the tag, HP label,
    "Evolves from ..." line and TRAINER banner are all smaller or filtered.
    """
    # A scrap of two or three letters out in the right half is the HP label
    # read on its own ("HB"), never a name.
    kept = [b for b in boxes if not _is_header_noise(b[0])
            and not (b[2] > CARD_W * 0.55
                     and len("".join(ch for ch in b[0] if ch.isalnum())) <= 3)]
    if not kept:
        return ""
    tallest = max(kept, key=lambda b: b[1])
    t_h, t_bottom = tallest[1], tallest[3] + tallest[1]
    # The name's own words, plus a smaller prefix printed on the same baseline
    # ("Alolan" before "Exeggutor", "Team Rocket's" before a name).
    line = [b for b in kept
            if (b[1] >= 0.7 * t_h and abs(b[3] - tallest[3]) < 0.6 * t_h)
            or (b[1] >= 0.4 * t_h and abs(b[3] + b[1] - t_bottom) < 0.35 * t_h)]
    name = " ".join(b[0] for b in sorted(line, key=lambda b: b[2])).strip()
    # Drop symbols OCR turns into odd characters - the gender sign after
    # "Nidoran" comes out as a CJK glyph, which the window shows as "???" and
    # which breaks the search. Latin letters (é in Pokémon) are kept; the
    # search tells Nidoran♀ from Nidoran♂ by the number.
    name = " ".join("".join(ch for ch in name if ord(ch) < 0x250).split())
    if card_type == "Pokemon":
        # Diamond & Pearl era LV.X, printed small: "Palkiaw.X", "PalkiaLV.X".
        name = re.sub(r"\s*(?:L\s*[VW]|[VW])\s*\.?\s*X$", " LV.X", name, flags=re.IGNORECASE)
        # OCR glues the rule-box suffix on, in any case: "CharizardeX" ->
        # "Charizard ex". Modern ex is lower case; old EX and GX/V are upper.
        def split_suffix(m: re.Match) -> str:
            suffix = m.group(1)
            return " " + (suffix if suffix == "EX" else
                          "ex" if suffix.lower() == "ex" else suffix.upper())
        name = re.sub(r"(?<=[a-z])(ex|gx|vmax|vstar|v)$", split_suffix, name,
                      flags=re.IGNORECASE)
    if card_type.endswith("Energy"):
        # "MistEnergy" -> "Mist Energy"; a stray letter before it goes too
        # ("Double TurboE Energy" -> "Double Turbo Energy").
        name = re.sub(r"(?<=[a-z])E?\s*Energy$", " Energy", name)
    return name


def rectify_matrix(corners: np.ndarray) -> np.ndarray:
    """Transform taking the card's corners onto a flat CARD_W x CARD_H image."""
    src = corners.astype(np.float32)
    tl, tr, br, bl = src
    width = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0
    height = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0
    if width > height:
        src = np.roll(src, -1, axis=0)   # card is on its side
    dst = np.array([[0, 0], [CARD_W - 1, 0], [CARD_W - 1, CARD_H - 1], [0, CARD_H - 1]],
                   dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def rectify(frame: np.ndarray, corners: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(frame, rectify_matrix(corners), (CARD_W, CARD_H),
                               flags=cv2.INTER_CUBIC)


def flat_to_frame(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Map points on the flattened card back onto the camera frame."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.linalg.inv(matrix)).reshape(-1, 2)


def region_on_frame(corners: np.ndarray, fx0: float, fx1: float,
                    fy0: float, fy1: float) -> np.ndarray:
    """The four points of a box on the flat card, in camera-frame coordinates."""
    x0, x1 = CARD_W * fx0, CARD_W * fx1
    y0, y1 = CARD_H * fy0, CARD_H * fy1
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    return flat_to_frame(box, rectify_matrix(corners))


def type_region_on_frame(corners: np.ndarray) -> np.ndarray:
    """The blue tag-and-name box across the top, in camera-frame coordinates."""
    return region_on_frame(corners, TYPE_X0, NAME_X1, TYPE_Y0, TYPE_Y1)


def set_region_on_frame(corners: np.ndarray) -> np.ndarray:
    """The four points of the bottom-left set/number box, in frame coordinates."""
    return region_on_frame(corners, SET_X0, SET_X1, SET_Y0, SET_Y1)


class TypeReader:
    """OCRs the type band. Lazily loads the OCR model on first use."""

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR
        # By default the detector scales an image's SHORT side up to 736px, so
        # a 712x122 band became ~4300x736 and took 1.2s. The bands are already
        # upscaled as needed, so only cap the long side.
        self._ocr = RapidOCR(det_limit_type="max", det_limit_side_len=2000)
        # A more lenient reader for the collector number. The tiny italic
        # "008" on a basic energy reads at 47-59% confidence, and the default
        # 50% cut-off threw it away, leaving nothing. Only used as a fallback.
        self._ocr_loose = RapidOCR(det_limit_type="max", det_limit_side_len=2000,
                                   det_box_thresh=0.3, det_thresh=0.2, text_score=0.3)
        self.badge_box = None       # where the last read found the set badge
        self.stamp_box = None       # ...and the anniversary stamp
        # One throwaway inference, so the first real read is not slow.
        warm = np.full((60, 220, 3), 255, np.uint8)
        cv2.putText(warm, "warm", (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        self._ocr(warm)
        self._ocr_loose(warm)

    @staticmethod
    def enhance(band: np.ndarray, upscale: float = 3.0) -> np.ndarray:
        """Make small, low-contrast type wording legible to OCR.

        Ordered deliberately. A hazy webcam frame can have a standard deviation
        under 10 (a healthy one is 40-70), and CLAHE alone does not rescue that
        - measured on a real frame it lifted std to 18 and OCR still returned
        nothing. Stretching the histogram to the full range first, then
        sharpening, then CLAHE, took the same frame to std 45+ and OCR went
        from zero text to reading the card.
        """
        grey = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        grey = cv2.normalize(grey, None, 0, 255, cv2.NORM_MINMAX)
        out = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

        # Counteract the softness that comes with a hazy or out-of-focus frame.
        out = cv2.addWeighted(out, 2.5, cv2.GaussianBlur(out, (0, 0), 3), -1.5, 0)
        out = cv2.resize(out, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def read(self, flat: np.ndarray) -> tuple[str, list[str]]:
        band = flat[int(CARD_H * TYPE_Y0):int(CARD_H * TYPE_Y1),
                    int(CARD_W * TYPE_X0):int(CARD_W * TYPE_X1)]
        texts: list[str] = []
        # Two passes at different magnifications: small wording like "Special
        # Energy" and large banner text do not peak at the same scale.
        for upscale in (5.0, 3.5):
            result, _ = self._ocr(self.enhance(band, upscale))
            texts += [str(item[1]).strip() for item in (result or [])]
            if classify_type(texts) != "Unknown":
                break
        return classify_type(texts), texts

    def read_number(self, flat: np.ndarray, allow_bare: bool = False,
                    region: tuple[float, float, float, float] | None = None,
                    tries: slice = slice(None)) -> tuple[str, list[str]]:
        """Read the collector number from the bottom-left line, or from
        `region` (x0, x1, y0, y1 as fractions of the card).

        `allow_bare` accepts a number with no set total, as basic energies
        print it ("008"). `tries` picks which of the read passes to run, so
        the caller can try both corners quickly before the slower passes.
        """
        x0, x1, y0, y1 = region or (SET_X0, SET_X1, SET_Y0, SET_Y1)
        band = flat[int(CARD_H * y0):int(CARD_H * y1), int(CARD_W * x0):int(CARD_W * x1)]

        def clahe(img: np.ndarray) -> np.ndarray:
            g = cv2.createCLAHE(3.0, (4, 4)).apply(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

        # Plain upscaling first, not enhance(): on this tiny dark-on-light
        # print the sharpen + CLAHE pass smears the slash and digits ("091/091"
        # came out as "09110910"), while a straight resize reads it cleanly.
        # The lenient reader and a gentle CLAHE come last, for faint print.
        passes = ((self._ocr, 3.0, None), (self._ocr, 4.5, None),
                  (self._ocr_loose, 3.0, None), (self._ocr_loose, 3.0, clahe))
        texts: list[str] = []
        number = ""
        for ocr, upscale, prep in passes[tries]:
            big = cv2.resize(band, None, fx=upscale, fy=upscale,
                             interpolation=cv2.INTER_CUBIC)
            result, _ = ocr(prep(big) if prep else big, use_cls=False)
            texts += [str(item[1]).strip() for item in (result or [])]
            number = parse_number(texts, allow_bare)
            if number:
                break
        return number, texts

    def read_set_code(self, flat: np.ndarray) -> str:
        """Raw OCR of the set-code badge ("MEEBI" for "MEE EN"), or ''.

        The badge is the black box with white letters left of the number. It
        is found as the darkest box-shaped blob in the strip, flipped to dark
        text on white, and read. The read is rough at camera resolution; the
        search snaps it to the nearest real set code.
        """
        y0, x0 = int(CARD_H * SET_Y0), int(CARD_W * SET_X0)
        band = flat[y0:int(CARD_H * SET_Y1), x0:int(CARD_W * SET_X1)]
        v = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)[:, :, 2]
        dark = (v < np.median(v) * 0.6).astype(np.uint8) * 255
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        boxes = [s for s in stats[1:] if 12 <= s[3] <= 40 and 1.4 * s[3] <= s[2] <= 4.5 * s[3]]
        self.badge_box = None
        if not boxes:
            return ""
        x, y, w, h, _ = max(boxes, key=lambda s: s[4])
        self.badge_box = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)   # for the camera view
        inv = 255 - cv2.cvtColor(band[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        big = cv2.resize(inv, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        big = cv2.copyMakeBorder(big, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        result, _ = self._ocr(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR),
                              use_det=False, use_cls=False)
        text = str(result[0][0]) if result else ""
        # Digits stay: some codes have them ("30C" is 30th Celebration). This
        # is read from the badge box alone, so the collector number beside it
        # never gets in.
        return "".join(ch for ch in text.upper() if ch.isalnum())

    def read_anniversary(self, flat: np.ndarray) -> str:
        """'30' or '25' if the card carries an anniversary stamp, else ''.

        Two stamps, both a Pikachu head with the years on it:
        - yellow with red years, at the bottom right of the artwork. Classic
          Collection reprints carry it and keep the original card's number -
          Palkia LV.X still says 106/106. '?' means it is there but its
          digits did not read.
        - rainbow with blue years, lower down on full-art cards such as the
          stamped Mega Evolution energies (MEE 9-16).
        Where it was found is kept in self.stamp_box, for the camera view.
        """
        return self.read_stamp(flat)

    def read_stamp(self, flat: np.ndarray) -> str:
        """Which stamp the card carries, as a STAMPS key, or '' for none.

        Every card is checked. Each known stamp has its own finder, tried in
        turn; where the stamp was found is kept in self.stamp_box.
        """
        self.stamp_box = None
        for finder in (self._play_stamp, self._yellow_stamp, self._outlined_stamp,
                       self._rainbow_stamp):
            found = finder(flat)
            if found:
                return found
        return ""

    def _play_stamp(self, flat: np.ndarray) -> str:
        """The Play! Pokémon logo of Prize Pack cards: a dark-bordered box,
        red on the right, black at the bottom left, a white Poké Ball and
        trainer across it. No other card part has all three colours together
        in that layout."""
        x0, y0 = int(CARD_W * 0.55), int(CARD_H * 0.30)
        region = flat[y0:int(CARD_H * 0.97), x0:]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2]
        black = value < 90
        red = (cv2.inRange(hsv, (0, 110, 80), (10, 255, 255))
               | cv2.inRange(hsv, (165, 110, 80), (180, 255, 255))) > 0
        white = cv2.inRange(hsv, (0, 0, 170), (180, 60, 255)) > 0
        dark = cv2.morphologyEx(black.astype(np.uint8) * 255, cv2.MORPH_CLOSE,
                                np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(dark)
        for x, y, w, h, _area in stats[1:]:
            if not (70 <= w <= 200 and 55 <= h <= 170 and 1.0 <= w / h <= 1.7):
                continue
            box = (slice(y, y + h), slice(x, x + w))
            if not (red[box].mean() >= 0.12 and white[box].mean() >= 0.25
                    and black[box].mean() >= 0.12):
                continue
            right = (slice(y, y + h), slice(x + w // 2, x + w))
            bottom_left = (slice(y + h // 2, y + h), slice(x, x + w // 2))
            if red[right].mean() > black[right].mean() and \
                    black[bottom_left].mean() > red[bottom_left].mean():
                self.stamp_box = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
                return "play"
        return ""

    def _read_years(self, sheet: np.ndarray) -> str:
        """Read the digits off a black-on-white sheet cut from a stamp."""
        big = cv2.resize(sheet, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        big = cv2.copyMakeBorder(big, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        result, _ = self._ocr_loose(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR), use_cls=False)
        return "".join(ch for item in (result or []) for ch in str(item[1]) if ch.isdigit())

    def _outlined_stamp(self, flat: np.ndarray) -> str:
        """The rainbow stamp, found by its white outline.

        Whatever the card's background - green Grass, blue Water, dark
        Darkness - the stamp is ringed in white, so a head-sized area enclosed
        by white is the stamp. The years are read from its lower half, where
        they are blue/dark on the warm bands. Several ways of lifting them out
        are tried, since each suits some backgrounds better.
        """
        x0, y0 = int(CARD_W * 0.55), int(CARD_H * 0.30)
        region = flat[y0:int(CARD_H * 0.95), x0:]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        white = cv2.dilate(cv2.inRange(hsv, (0, 0, 175), (180, 70, 255)),
                           np.ones((3, 3), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(white),
                                                               connectivity=4)
        value = hsv[:, :, 2]
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            # Head-sized only - the open background is far bigger. (The ears
            # can reach the card's edge, so touching it is allowed.)
            if not (80 <= w <= 180 and 60 <= h <= 170 and area >= 3000):
                continue
            inside = labels == i
            lower = inside.copy()
            lower[:y + int(h * 0.45)] = False
            warm = (cv2.inRange(hsv, (0, 90, 90), (38, 255, 255)) > 0)[lower].mean()
            if warm < 0.15:
                continue                    # no yellow-to-red bottom: not the stamp
            blue = cv2.inRange(hsv, (90, 40, 0), (140, 255, 210)) > 0
            dark = value < np.median(value[inside]) - 50
            partial = ""
            for digits in (blue, dark, blue | dark):
                sheet = np.full(value.shape, 255, np.uint8)
                sheet[digits & lower] = 0
                read = self._read_years(sheet[y:y + h, x:x + w])
                if read in ("25", "30"):
                    self.stamp_box = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
                    return read
                if read and read in "3025":
                    partial = read
            if partial:
                # An outlined rainbow head with some of the years on it: a
                # stamp, but which anniversary is not certain.
                self.stamp_box = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
                return "?"
        return ""

    def _rainbow_stamp(self, flat: np.ndarray) -> str:
        """The rainbow stamp: its blue and green top half stands out from any
        background, and it only counts if the years on it can be read."""
        x0, y0 = int(CARD_W * 0.55), int(CARD_H * 0.30)
        region = flat[y0:int(CARD_H * 0.95), x0:]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        cool = cv2.inRange(hsv, (40, 90, 60), (130, 255, 255))     # green + blue
        cool = cv2.morphologyEx(cool, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(cool)
        heads = [s for s in stats[1:] if 70 <= s[2] <= 200 and 30 <= s[3] <= 160
                 and s[4] >= 1500]
        for x, y, w, _h, _area in sorted(heads, key=lambda s: -s[4])[:3]:
            # The green/blue part is the top of the head (ears down to the
            # middle); the whole head is about as tall as it is wide.
            y1 = min(region.shape[0], y + int(w * 1.05))
            box = hsv[y:y1, x:x + w]
            lower = box[int(box.shape[0] * 0.45):]
            warm = (cv2.inRange(lower, (0, 90, 90), (38, 255, 255)) > 0).mean()
            if warm < 0.15:
                continue                    # no yellow-to-red bottom: not the stamp
            # The years are blue on the warm half: lift them out as black on
            # white for the reader.
            digits = cv2.inRange(lower, (90, 40, 0), (140, 255, 200))
            sheet = np.full(digits.shape, 255, np.uint8)
            sheet[digits > 0] = 0
            big = cv2.resize(sheet, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            big = cv2.copyMakeBorder(big, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
            result, _ = self._ocr_loose(cv2.cvtColor(big, cv2.COLOR_GRAY2BGR), use_cls=False)
            read = "".join(ch for item in (result or []) for ch in str(item[1]) if ch.isdigit())
            if read in ("25", "30"):
                self.stamp_box = (x0 + x, y0 + y, x0 + x + w, y0 + y1)
                return read
        return ""

    def _yellow_stamp(self, flat: np.ndarray) -> str:
        x0, y0 = int(CARD_W * 0.60), int(CARD_H * 0.30)
        region = flat[y0:int(CARD_H * 0.60), x0:]
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        yellow = cv2.inRange(hsv, (18, 110, 170), (35, 255, 255))
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(yellow)
        for x, y, w, h, area in sorted(stats[1:], key=lambda s: -s[4])[:3]:
            # The logo is a head-sized blob (~125x108 on the flat card); a
            # yellow background (Lightning cards) is far bigger.
            if not (70 <= w <= 180 and 60 <= h <= 160 and 0.9 <= w / h <= 1.5
                    and 0.4 <= area / (w * h) <= 0.8):
                continue
            box = hsv[y:y + h, x:x + w]
            red = (cv2.inRange(box, (0, 90, 60), (10, 255, 255))
                   | cv2.inRange(box, (165, 90, 60), (180, 255, 255)))
            if not 0.04 <= (red > 0).sum() / area <= 0.35:
                continue                    # no red years on it
            lower = region[y + h // 2:y + h, x:x + w]
            big = cv2.resize(lower, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
            big = cv2.copyMakeBorder(big, 12, 12, 12, 12, cv2.BORDER_CONSTANT,
                                     value=(255, 255, 255))
            result, _ = self._ocr(big, use_det=False, use_cls=False)
            digits = "".join(ch for ch in (result[0][0] if result else "") if ch.isdigit())
            self.stamp_box = (x0 + x, y0 + y, x0 + x + w, y0 + y + h)
            return digits if digits in ("25", "30") else "?"
        return ""

    def top_boxes(self, flat: np.ndarray) -> list[tuple[str, float, float, float]]:
        """OCR the full-width top band: (text, height, x, y) per box, flat px."""
        band = flat[int(CARD_H * TYPE_Y0):int(CARD_H * TYPE_Y1),
                    int(CARD_W * TYPE_X0):int(CARD_W * NAME_X1)]
        upscale = 2.0
        big = cv2.resize(band, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        result, _ = self._ocr(big, use_cls=False)
        boxes = []
        for box, text, _conf in result or []:
            pts = np.asarray(box, dtype=np.float32) / upscale
            boxes.append((str(text).strip(), float(pts[:, 1].max() - pts[:, 1].min()),
                          float(pts[:, 0].min()), float(pts[:, 1].min())))
        return boxes

    def read_name(self, flat: np.ndarray, card_type: str) -> tuple[str, list[str]]:
        """Read the card name from the full-width top band."""
        boxes = self.top_boxes(flat)
        return parse_name(boxes, card_type), [b[0] for b in boxes]

    def read_card(self, flat: np.ndarray
                  ) -> tuple[str, str, str, tuple[str, ...], str, str, dict]:
        """Type, collector number, name, (energy cards only) the energy icons,
        the raw set-code read, any anniversary stamp, and where the badge and
        stamp sit - in one job for the worker."""
        boxes = self.top_boxes(flat)
        # The tag usually reads fine in this same pass; only the words in the
        # tag's own slice count, so a name like "Energy Search" cannot.
        tag_texts = [b[0] for b in boxes if b[2] < CARD_W * TYPE_X1]
        card_type = classify_type(tag_texts)
        if card_type == "Unknown" and has_hp(boxes):
            card_type = "Pokemon"                 # older card: no stage tag, but HP
        if card_type == "Unknown":
            card_type, _ = self.read(flat)        # slower, enhanced pass
        if card_type == "Energy":
            # Sword & Shield cards put only "ENERGY" in the tag and print
            # "Special Energy" / "Basic Energy" over on the right.
            band = "".join(ch for ch in " ".join(b[0] for b in boxes).upper()
                           if ch.isalnum())
            if any(k in band for k in SPECIAL_ENERGY_KEYS):
                card_type = "Special Energy"
            elif any(k in band for k in BASIC_ENERGY_KEYS):
                card_type = "Basic Energy"
        # One quick read of each corner first - most cards are done here - and
        # only then the slower passes, bottom-left before bottom-right.
        bare = card_type in ("Basic Energy", "Energy")
        right = (NUM_R_X0, NUM_R_X1, NUM_R_Y0, NUM_R_Y1)
        # Last resort: taller boxes. A card in a sleeve is found by the
        # sleeve's outline, so the card sits inset and its number rides above
        # the usual boxes.
        tall_left = (SET_X0, SET_X1, NUM_TALL_Y0, SET_Y1)
        tall_right = (NUM_R_X0, NUM_R_X1, NUM_TALL_Y0, NUM_R_Y1)
        number = ""
        for region, tries in ((None, slice(0, 1)), (right, slice(0, 1)),
                              (None, slice(1, None)), (right, slice(1, None)),
                              (tall_left, slice(0, 3)), (tall_right, slice(0, 3))):
            number, _ = self.read_number(flat, allow_bare=bare and region is None,
                                         region=region, tries=tries)
            if number:
                break
        energy = tuple(find_icons(flat)) if card_type.endswith("Energy") else ()
        name = parse_name(boxes, card_type)
        if not name and card_type == "Basic Energy":
            # "Basic <icon> Energy" reads like the tag and is filtered out;
            # the icon carries the type, so the generic name is enough.
            name = "Basic Energy"
        anniversary = self.read_stamp(flat)
        code_read = self.read_set_code(flat)
        # Where the badge and stamp were found (flat-card pixels), to box them
        # on the camera view.
        marks = {"badge": self.badge_box, "stamp": self.stamp_box}
        return card_type, number, name, energy, code_read, anniversary, marks


class CornerSmoother:
    """Steady the outline across frames.

    The detector re-decides from scratch every frame, and on a marginal frame a
    different binary view or a different corner-fit tolerance can win, moving
    the corners by ~17px even while the card sits perfectly still. Averaging
    successive estimates removes that flicker.

    A large jump is treated as a genuinely new position and snapped to
    immediately, so the border still follows a card you actually move. Brief
    dropouts are held rather than blanked, so one bad frame does not make the
    border disappear.
    """

    def __init__(self, alpha: float = 0.35, snap_distance: float = 70.0,
                 hold_frames: int = 3):
        self.alpha = alpha                  # 0 = frozen, 1 = no smoothing
        self.snap_distance = snap_distance  # px before we assume a real move
        self.hold_frames = hold_frames
        self.corners: np.ndarray | None = None
        self.missing = 0

    def update(self, measured: np.ndarray | None) -> np.ndarray | None:
        if measured is None:
            self.missing += 1
            if self.missing > self.hold_frames:
                self.corners = None
            return self.corners

        self.missing = 0
        if self.corners is None:
            self.corners = measured.copy()
            return self.corners

        drift = float(np.mean(np.linalg.norm(measured - self.corners, axis=1)))
        if drift > self.snap_distance:
            self.corners = measured.copy()          # card was moved
        else:
            self.corners = (1.0 - self.alpha) * self.corners + self.alpha * measured
        return self.corners


def draw_label(frame: np.ndarray, text: str, top: int, color, scale: float) -> int:
    """Write `text` boxed against the right edge, starting at `top`.

    Returns the y just below the box, so labels can be stacked.
    """
    thick = max(2, int(round(scale * 2)))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = frame.shape[1] - tw - 30
    y = top + th + 20
    cv2.rectangle(frame, (x - 18, y - th - 20), (x + tw + 18, y + 20), (20, 20, 20), -1)
    cv2.rectangle(frame, (x - 18, y - th - 20), (x + tw + 18, y + 20), color, 3)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)
    return y + 20


def savable_card(result: dict) -> tuple[dict | None, str]:
    """The one card a search result can be saved as, or (None, why not).

    Only a single, definite match is saved - if several printings fit, saving
    one of them would be a guess.
    """
    if not result.get("found"):
        return None, "  (not saved - no match)"
    cards = result["cards"]
    if len(cards) != 1:
        return None, "  (not saved - {} printings match; can't tell which)".format(len(cards))
    return cards[0], ""


def save_card(card: dict, card_type: str, energy, stamp: str = "", copies: int = 1) -> str:
    """Add `copies` of a card to the collection database; returns a status line.

    A stamped copy is kept apart from plain copies of the same card.
    """
    code = card.get("code") or set_codes().get(card["id"].rpartition("-")[0], "")
    qty = card_db.add_card(card, card_type, code, describe_energy(list(energy)), copies,
                           stamp=STAMPS.get(stamp, ""))
    return "  Added {}{} - now {} in the collection".format(
        copies, " ({})".format(STAMPS[stamp]) if stamp in STAMPS else "", qty)


def save_result(result: dict, card_type: str, energy, stamp: str = "",
                copies: int = 1) -> str:
    """Save a search result straight away (one copy unless told otherwise)."""
    card, why_not = savable_card(result)
    return save_card(card, card_type, energy, stamp, copies) if card else why_not


def parse_copies(line: str) -> int | None:
    """What was typed at the copies prompt: Enter alone is 1, a number is
    that many (0 skips the card), anything else is None."""
    line = line.strip()
    if not line:
        return 1
    return int(line) if line.isdigit() and int(line) <= 999 else None


def terminal_lines(lines: "queue.Queue[str]") -> None:
    """Feed each line typed in the terminal into `lines`. input() blocks, so
    this runs on its own thread for the whole session."""
    while True:
        try:
            lines.put(input())
        except EOFError:
            return


_preview_cache: dict[str, np.ndarray] = {}


def load_preview(url: str) -> np.ndarray | None:
    """The card picture at `url`, decoded straight from memory - nothing is
    written to disk. Kept for the session so a repeat card shows at once."""
    if not url:
        return None
    if url in _preview_cache:
        return _preview_cache[url]
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = np.frombuffer(resp.read(), np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if image is not None:
        _preview_cache[url] = image
    return image


def draw_preview(frame: np.ndarray, image: np.ndarray, caption: str) -> None:
    """Show the found card's picture in the bottom-right corner of the view."""
    height = int(frame.shape[0] * 0.45)
    width = int(image.shape[1] * height / image.shape[0])
    margin = 20
    x0 = frame.shape[1] - width - margin
    y0 = frame.shape[0] - height - margin
    if x0 < 0 or y0 < 40:
        return
    frame[y0:y0 + height, x0:x0 + width] = cv2.resize(image, (width, height),
                                                      interpolation=cv2.INTER_AREA)
    cv2.rectangle(frame, (x0 - 3, y0 - 3), (x0 + width + 3, y0 + height + 3), GREEN, 3)
    (tw, th), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    # The caption is often wider than the picture: end it at the picture's
    # right edge so it never runs off the frame.
    right = x0 + width + 3
    left = max(0, min(x0 - 3, right - tw - 12))
    cv2.rectangle(frame, (left, y0 - th - 16), (right, y0 - 3), (20, 20, 20), -1)
    cv2.putText(frame, caption, (left + 6, y0 - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2,
                cv2.LINE_AA)


PROMPT_COLOR = (0, 230, 255)


def draw_prompt(frame: np.ndarray, card_name: str, typed: str, asking: bool) -> None:
    """The copy-count question, boxed at the bottom of the camera view.

    asking=False is the plain "press Enter" box shown when nothing can be
    saved (no match, or several printings matched).
    """
    if asking:
        lines = [("{}  -  how many copies?".format(card_name), 0.9),
                 ("Copies: {}_".format(typed), 1.6),
                 ("type a number, Enter to save  (Enter alone = 1, 0 = skip)", 0.6)]
    else:
        lines = [("Not saved - see the terminal", 0.9),
                 ("Press Enter for the next card", 1.1)]
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, s, 2)[0] for t, s in lines]
    width = max(w for w, _ in sizes) + 48
    height = sum(h + 22 for _, h in sizes) + 24
    x0 = max(10, (frame.shape[1] - width) // 2)
    y0 = frame.shape[0] - height - 30
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (20, 20, 20), -1)
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), PROMPT_COLOR, 3)
    y = y0 + 12
    for (text, scale), (w, h) in zip(lines, sizes):
        y += h + 22
        cv2.putText(frame, text, (x0 + (width - w) // 2, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, PROMPT_COLOR if scale > 1 else (230, 230, 230),
                    3 if scale > 1 else 2, cv2.LINE_AA)


def name_label(name: str, energy) -> str:
    """The name, with an energy card's icons after it: 'Mist Energy [Colorless]'."""
    kinds = describe_energy(list(energy))
    return "{} [{}]".format(name, kinds) if name and kinds else name


def draw_big_type(frame: np.ndarray, card_type: str, set_text: str = "",
                  name: str = "", search_text: str = "") -> None:
    """Stack the type, number, name and search result in the top-right corner."""
    top = 14
    for text, color, scale in ((card_type, TYPE_COLOR, 1.9), (set_text, SET_COLOR, 1.3),
                               (name, NAME_COLOR, 1.1), (search_text, SEARCH_COLOR, 0.8)):
        if text:
            top = draw_label(frame, text, top, color, scale) + 12


BADGE_COLOR = (255, 255, 0)
STAMP_COLOR = (0, 200, 255)


def draw_marks(out: np.ndarray, corners: np.ndarray, marks: dict | None,
               badge_label: str, stamp_label: str) -> None:
    """Box the set badge and the anniversary stamp where they were found."""
    if not marks:
        return
    matrix = rectify_matrix(corners)
    for key, color, label in (("badge", BADGE_COLOR, badge_label),
                              ("stamp", STAMP_COLOR, stamp_label)):
        box = marks.get(key)
        if not box:
            continue
        x0, y0, x1, y1 = (float(v) for v in box)
        pad = 4.0
        pts = flat_to_frame(np.array([[x0 - pad, y0 - pad], [x1 + pad, y0 - pad],
                                      [x1 + pad, y1 + pad], [x0 - pad, y1 + pad]]),
                            matrix).astype(np.int32)
        cv2.polylines(out, [pts], True, color, 3, cv2.LINE_AA)
        if label:
            top = pts[np.argmin(pts[:, 1])]
            cv2.putText(out, label, (int(pts[:, 0].min()), int(top[1]) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def draw_border(frame: np.ndarray, corners: np.ndarray | None,
                card_type: str = "", set_text: str = "", name: str = "",
                search_text: str = "", marks: dict | None = None,
                badge_label: str = "", stamp_label: str = "") -> np.ndarray:
    """Draw the border, the four corners, the type band and the type name."""
    out = frame.copy()
    if corners is None:
        cv2.putText(out, "no card found", (20, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, AMBER, 2, cv2.LINE_AA)
        return out

    pts = corners.astype(np.int32)
    cv2.polylines(out, [pts], True, GREEN, 3, cv2.LINE_AA)
    for (x, y), corner in zip(pts, ("TL", "TR", "BR", "BL")):
        cv2.circle(out, (int(x), int(y)), 9, GREEN, -1, cv2.LINE_AA)
        cv2.putText(out, "{} ({},{})".format(corner, int(x), int(y)),
                    (int(x) + 14, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, GREEN, 2, cv2.LINE_AA)
    # The band the card type is read from, drawn in perspective on the card.
    band = type_region_on_frame(corners).astype(np.int32)
    cv2.polylines(out, [band], True, TYPE_COLOR, 3, cv2.LINE_AA)
    cv2.putText(out, "tag + name", (int(band[0][0]) + 8, int(band[0][1]) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, TYPE_COLOR, 2, cv2.LINE_AA)
    # The box the collector number is read from.
    box = set_region_on_frame(corners).astype(np.int32)
    cv2.polylines(out, [box], True, SET_COLOR, 3, cv2.LINE_AA)
    cv2.putText(out, "number", (int(box[3][0]) + 8, int(box[3][1]) + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, SET_COLOR, 2, cv2.LINE_AA)

    cv2.putText(out, "card found - aspect {:.3f}".format(aspect_of(corners)),
                (20, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2, cv2.LINE_AA)
    draw_marks(out, corners, marks, badge_label, stamp_label)
    draw_big_type(out, card_type, set_text, name, search_text)
    return out


def stamp_text(stamp: str) -> str:
    """The stamp's name for the camera view, or ''."""
    return STAMPS.get(stamp, "")


def find_camera_by_name(name: str) -> int:
    """Index of the Media Foundation camera whose name contains `name`."""
    from cv2_enumerate_cameras import enumerate_cameras
    cams = enumerate_cameras(cv2.CAP_MSMF)
    for cam in cams:
        if name.lower() in cam.name.lower():
            return cam.index
    raise RuntimeError("no camera named {!r} (found: {})".format(
        name, ", ".join(c.name for c in cams) or "none"))


def open_named_camera(name: str):
    # DroidCam only registers as a Media Foundation device, not DirectShow.
    # Its picture settings live in the phone app, so none are forced here.
    index = find_camera_by_name(name)
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if not cap.isOpened():
        raise RuntimeError("could not open camera {!r}".format(name))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    return cap


class LatestFrame:
    """Read a network stream on its own thread and keep only the newest frame.

    A stream queues frames while the main loop is busy with detection and OCR,
    so reading them in order falls further and further behind the phone. This
    drains the stream continuously and hands back whatever arrived last.
    """

    def __init__(self, cap):
        self.cap = cap
        self.frame = None
        self.ok = True
        self.fresh = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.ok:
            ok, frame = self.cap.read()
            if not ok:
                self.ok = False
            else:
                self.frame = frame
            self.fresh.set()

    def read(self):
        # Wait for a frame newer than the last one returned.
        if not self.fresh.wait(timeout=5.0):
            return False, None
        self.fresh.clear()
        return self.ok and self.frame is not None, self.frame

    def get(self, prop):
        return self.cap.get(prop)

    def release(self):
        self.ok = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def open_stream(url: str):
    # The DroidCam phone app serves its picture straight over Wi-Fi. It allows
    # one viewer at a time, so the DroidCam PC client must be closed.
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(
            "could not open {} - is the DroidCam app open on the phone, on the "
            "same Wi-Fi, and the DroidCam PC client closed?".format(url))
    return LatestFrame(cap)


def open_camera(index: int, exposure: float | None, brightness: float | None):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("could not open camera {}".format(index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Always start from the camera's defaults. DirectShow stores these on the
    # device itself, so a value left behind by any other program persists into
    # this session. A raised brightness is the nasty one: it lifts the black
    # level instead of adding light, which washes the whole picture out to flat
    # grey and destroys the contrast OCR depends on.
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
    cap.set(cv2.CAP_PROP_CONTRAST, 32)
    cap.set(cv2.CAP_PROP_GAMMA, 120)
    cap.set(cv2.CAP_PROP_SATURATION, 64)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)       # auto
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)

    if exposure is not None:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)   # manual, on DirectShow
        cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if brightness is not None:
        cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
    return cap


def exposure_arg(value: str):
    """Accept a number, or the word "auto" to leave the camera on automatic."""
    if str(value).strip().lower() in ("auto", "none", ""):
        return None
    return float(value)


def main() -> int:
    ap = argparse.ArgumentParser(description="Outline a Pokemon card's corners.")
    ap.add_argument("--image", help="run on one photo instead of the camera")
    # 1080p: at 720p the set-code badge ("MEE") is ~25px wide and unreadable.
    ap.add_argument("--camera", default="http://192.168.1.163:4747/video?1920x1080",
                    help="stream URL (default: DroidCam phone over Wi-Fi), a "
                         "camera name, or a number for a DirectShow index")
    ap.add_argument("--exposure", type=exposure_arg, default=1.0,
                    help="manual exposure (default 1). Pass --exposure auto to "
                         "let the camera choose.")
    ap.add_argument("--brightness", type=float, default=None, help="0-255")
    ap.add_argument("--raw", action="store_true",
                    help="disable corner smoothing (show the raw per-frame fit)")
    args = ap.parse_args()
    if isinstance(args.exposure, str) or args.exposure is None:
        args.exposure = None

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print("could not read {}".format(args.image))
            return 1
        corners = find_corners(frame)
        card_type, set_text, name, energy = "", "", "", ()
        if corners is not None:
            reader = TypeReader()
            flat = rectify(frame, corners)
            (card_type, set_text, name, energy, code_read, anniversary,
             marks) = reader.read_card(flat)
            print("type   : {}".format(card_type))
            print("number : {}".format(set_text or "?"))
            print("set    : {}   (badge read: {!r})".format(
                best_set_code(code_read) or "?", code_read))
            print("name   : {}".format(name or "?"))
            if card_type.endswith("Energy"):
                print("energy : {}".format(describe_energy(list(energy)) or "?"))
            if anniversary:
                print("stamp  : {}".format(stamp_text(anniversary)))
            if card_type not in ("", "Unknown") and name:
                print(format_result(search_card(name, set_text, list(energy), code_read,
                                                anniversary), name, set_text))
        else:
            marks, code_read, anniversary = None, "", ""
        cv2.imwrite("card_border_out.jpg",
                    draw_border(frame, corners, card_type, set_text,
                                name_label(name, energy), "", marks,
                                best_set_code(code_read), stamp_text(anniversary)))
        print("corners: {}".format(
            "none found" if corners is None else corners.astype(int).tolist()))
        print("wrote card_border_out.jpg")
        return 0

    if "://" in args.camera:
        cap = open_stream(args.camera)
    elif args.camera.isdigit():
        cap = open_camera(int(args.camera), args.exposure, args.brightness)
    else:
        cap = open_named_camera(args.camera)
    window = "Card border  (q to quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1200, 700)
    print("camera {} at {}x{} - press q to quit".format(
        args.camera, int(cap.get(3)), int(cap.get(4))))
    smoother = CornerSmoother() if not args.raw else None

    print("loading OCR model ...")
    type_reader = TypeReader()
    pool = ThreadPoolExecutor(max_workers=1)
    job = None
    card_type, set_text, name, energy = "", "", "", ()
    marks = None                # where the badge / stamp were last seen on the card
    # Set-code badge reads, voted on while the card holds still: a single read
    # at camera resolution is rough, the most common one is reliable.
    code_votes: Counter[str] = Counter()
    # Anniversary-logo reads, voted the same way ("" counts too, so one stray
    # yellow blob cannot make a card a reprint).
    logo_votes: Counter[str] = Counter()
    # Collector-number reads, voted too: one frame misreading 106/106 as
    # 108/108 must not replace the number the other frames agree on.
    number_votes: Counter[str] = Counter()
    # Fetch the set-code list now (a minute the first time, then cached), so
    # the first search does not wait for it.
    threading.Thread(target=set_codes, daemon=True).start()
    # Online lookup: runs once READS_TO_AGREE reads in a row agree, and never
    # while the type is Unknown.
    search_pool = ThreadPoolExecutor(max_workers=1)
    search_job = None
    search_for = None           # (type, name, number, energy) the job is searching
    search_stamp = ""           # ...and the stamp seen on that card
    search_text = ""            # short status on screen; results go to the terminal
    last_info, searched_info = None, None
    prev_read, agreeing = None, 0   # the last read, and how many in a row matched it
    # After a result is printed, nothing more is searched until the terminal
    # answers: how many copies to save (Enter alone = 1), or just Enter when
    # there was nothing to save. Lines typed in the terminal arrive here.
    typed: queue.Queue[str] = queue.Queue()
    threading.Thread(target=terminal_lines, args=(typed,), daemon=True).start()
    waiting_for_enter = False
    pending = None              # (card, type, energy, stamp) waiting for a copy count
    copy_buffer = ""            # digits typed on the camera window so far
    # Picture of the found card, shown in the corner until the next card.
    preview_pool = ThreadPoolExecutor(max_workers=1)
    preview_job, preview, preview_caption = None, None, ""
    print("ready")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed")
                break
            corners = find_corners(frame)
            if smoother is not None:
                corners = smoother.update(corners)

            if waiting_for_enter and not typed.empty():
                line = typed.get()
                copy_buffer = ""
                copies = parse_copies(line) if pending else 1
                if copies is None:
                    print("  Type a number, or just Enter for 1: ", end="", flush=True)
                else:
                    if pending and copies > 0:
                        print(save_card(*pending, copies=copies))
                    elif pending:
                        print("  (not saved - 0 copies)")
                    pending = None
                    waiting_for_enter = False
                    search_text = ""
                    agreeing = 0                      # the next card needs fresh reads
                    preview, preview_job = None, None
                    print("ready for the next card")

            # Read the type on a worker thread, so the preview stays smooth.

            if corners is None:
                card_type, set_text, name, energy = "", "", "", ()
                code_votes.clear()
                logo_votes.clear()
                number_votes.clear()
                prev_read, agreeing = None, 0
                marks = None
                if not waiting_for_enter:
                    search_text = ""
                    searched_info = None
                    preview, preview_job = None, None
            else:
                if job is not None and job.done():
                    try:
                        (card_type, number, new_name, new_energy, code_read,
                         logo, new_marks) = job.result()
                        # Keep the last place a badge / stamp was seen.
                        marks = {k: new_marks.get(k) or (marks or {}).get(k)
                                 for k in ("badge", "stamp")}
                        # Keep the last good read through a frame that blurs.
                        if number:
                            number_votes[number] += 1
                            set_text = number_votes.most_common(1)[0][0]
                        name = new_name or name
                        energy = ((new_energy or energy) if card_type.endswith("Energy")
                                  else ())
                        if code_read:
                            code_votes[code_read] += 1
                        logo_votes[logo] += 1
                        read = (card_type, name, set_text, energy)
                        agreeing = agreeing + 1 if read == prev_read else 1
                        prev_read = read
                    except Exception as exc:
                        print("card read failed: {}".format(exc))
                    job = None
                if job is None:
                    job = pool.submit(type_reader.read_card, rectify(frame, corners))

            info = (card_type, name, set_text, energy)
            if info != last_info:
                last_info = info
                code_votes.clear()                # a different card's badge
                logo_votes.clear()
            code_read = code_votes.most_common(1)[0][0] if code_votes else ""
            # A definite stamp read - the years on a Pikachu ("25"/"30"), or
            # the Play! Pokémon logo - has never come from anything else on a
            # card, so one such read wins. The vaguer "?" (Pikachu seen, years
            # not read) needs half the reads.
            definite = [v for v in ("30", "25", "play") if logo_votes[v]]
            if definite:
                anniversary = max(definite, key=lambda v: logo_votes[v])
            elif logo_votes["?"] * 2 >= sum(logo_votes.values()) > 0:
                anniversary = "?"
            else:
                anniversary = ""
            if (not waiting_for_enter
                    and card_type not in ("", "Unknown") and name
                    and agreeing >= READS_TO_AGREE
                    and info != searched_info
                    and (search_job is None or search_job.done())):
                searched_info = info
                # Kept with the job: searched_info is cleared if the card is
                # taken away before the search comes back.
                search_for = info
                search_stamp = anniversary
                search_text = "searching online..."
                search_job = search_pool.submit(search_card, name, set_text, list(energy),
                                                code_read, anniversary)
            if search_job is not None and search_job.done():
                s_type, s_name, s_number, s_energy = search_for
                card = None
                try:
                    result = search_job.result()
                except Exception as exc:
                    result = {"found": False, "text": "search failed: {}".format(exc)}
                search_job = None
                if not result.get("found"):
                    # Often an early misread (74/131 read as 7/131 while the
                    # camera focuses). Keep reading instead of stopping: a
                    # changed read searches again by itself.
                    print("  no match for {} {} - still reading...".format(s_name, s_number))
                    search_text = "no match yet - still reading"
                else:
                    print(format_result(result, s_name, s_number))
                    card, why_not = savable_card(result)
                    if not card:
                        print(why_not)
                    # Fetch the found card's picture for the corner preview -
                    # only for a single match, as a guess would mislead.
                    if card:
                        preview_caption = "Found: {} - {} {}".format(
                            card.get("name", ""), card.get("set", ""),
                            card.get("number", "")).strip(" -")
                        preview_job = preview_pool.submit(load_preview, card.get("image", ""))
                    while not typed.empty():           # drop anything typed earlier
                        typed.get()
                    copy_buffer = ""
                    if card:
                        pending = (card, s_type, s_energy, search_stamp)
                        print("\nHow many copies? Type it on the camera window, or here "
                              "(Enter = 1): ", end="", flush=True)
                    else:
                        print("\nPress Enter for the next card...")
                    search_text = ""
                    waiting_for_enter = True

            # Never wait on the set-code list here: it may still be downloading.
            code = best_set_code(code_read, wait=False)
            stamp = stamp_text(anniversary)
            number_label = " ".join(p for p in (code, set_text) if p) if set_text else ""
            if stamp and number_label:
                number_label += "  - " + stamp
            # The stamp box only while the reads agree there is a stamp.
            shown = dict(marks or {}, stamp=(marks or {}).get("stamp") if stamp else None)
            view = draw_border(frame, corners, card_type, number_label,
                               name_label(name, energy), search_text,
                               shown, code or "set badge", stamp)
            if preview_job is not None and preview_job.done():
                preview = preview_job.result()
                preview_job = None
            if preview is not None:
                draw_preview(view, preview, preview_caption)
            if waiting_for_enter:
                draw_prompt(view, pending[0]["name"] if pending else "", copy_buffer,
                            asking=pending is not None)
            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            # The copy count is typed straight onto the camera window: digits,
            # Backspace to fix, Enter to save (nothing typed = 1). It goes
            # through the same queue as a line typed in the terminal.
            if waiting_for_enter:
                if pending and chr(key).isdigit() and len(copy_buffer) < 3:
                    copy_buffer += chr(key)
                elif key in (8, 127):
                    copy_buffer = copy_buffer[:-1]
                elif key == 13:
                    print(copy_buffer or "(Enter)")
                    typed.put(copy_buffer)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pool.shutdown(wait=False)
        search_pool.shutdown(wait=False)
        preview_pool.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
