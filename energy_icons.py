"""Recognise the energy-type icons along the right end of a card's top band.

An energy card shows what it provides as round icons: on Scarlet & Violet
cards at the bottom right of the top band, on Sword & Shield cards at the top
right. Each type has its own colour and symbol - green leaf (Grass), red flame
(Fire), blue drop (Water), yellow bolt (Lightning), purple eye (Psychic),
orange fist (Fighting), black ball (Darkness), grey triangle (Metal), pink
star (Fairy), white star (Colorless) and the rainbow "any type" icon.

Circles are found with a Hough transform, then each is compared with
reference icons cut from official card images (energy_icons.npz). Circles that
match nothing closely - digits, letters - are dropped.

    python energy_icons.py build <folder of Type__id.png cards>   # rebuild refs
    python energy_icons.py test <folder>                          # leave-one-out
    python energy_icons.py add Grass <flattened camera frame.png> # add a camera sample
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

CARD_W, CARD_H = 734, 1024          # the flattened card (matches card_border)
REGION_X0, REGION_X1 = 0.55, 0.99   # right end of the top band
REGION_Y1 = 0.13
MIN_R, MAX_R = 18, 27               # icon radius on the flat card, px
PATCH = 32
MAX_DISTANCE = 22.0                 # Lab distance above which it is not an icon...
LOOSE_DISTANCE = 45.0               # ...unless its 3 nearest references agree

REF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "energy_icons.npz")

_mask = np.zeros((PATCH, PATCH), np.uint8)
cv2.circle(_mask, (PATCH // 2, PATCH // 2), int(PATCH * 0.45), 255, -1)
_MASK = _mask > 0
_refs: tuple[np.ndarray, list[str]] | None = None


def _region(flat: np.ndarray) -> tuple[np.ndarray, int]:
    x0 = int(CARD_W * REGION_X0)
    return flat[0:int(CARD_H * REGION_Y1), x0:int(CARD_W * REGION_X1)], x0


def _circles(region: np.ndarray) -> list[tuple[float, float, float]]:
    gray = cv2.medianBlur(cv2.cvtColor(region, cv2.COLOR_BGR2GRAY), 3)
    found = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=24, param1=120,
                             param2=22, minRadius=MIN_R, maxRadius=MAX_R)
    return [] if found is None else [tuple(c) for c in found[0]]


def _patch(region: np.ndarray, x: float, y: float, r: float) -> np.ndarray | None:
    x0, y0, x1, y1 = int(x - r), int(y - r), int(x + r), int(y + r)
    if x0 < 0 or y0 < 0 or x1 > region.shape[1] or y1 > region.shape[0]:
        return None
    crop = cv2.resize(region[y0:y1, x0:x1], (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    # A little blur so the match is on colour and overall shape, not on
    # sharpness, which varies between the camera and the reference scans.
    crop = cv2.GaussianBlur(crop, (3, 3), 0)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    # Lightness is weighted down: camera exposure moves it far more than hue.
    d = (a - b)[_MASK]
    d[:, 0] *= 0.5
    return float(np.sqrt((d ** 2).sum(axis=1)).mean())


def _load_refs() -> tuple[np.ndarray, list[str]]:
    global _refs
    if _refs is None:
        data = np.load(REF_FILE)
        _refs = (data["patches"], [str(t) for t in data["types"]])
    return _refs


def classify_patch(patch: np.ndarray, refs=None) -> tuple[str, float]:
    patches, types = refs or _load_refs()
    dists = [_distance(patch, p) for p in patches]
    i = int(np.argmin(dists))
    return types[i], dists[i]


def _accepted(patch: np.ndarray, refs=None) -> str:
    """The icon's type, or '' if it is not convincingly an icon.

    A close match is enough. Camera colours drift from the references
    (lighting, holo foil), so a looser match is also taken when its three
    nearest references all agree on the type - a digit or letter that merely
    looks round scatters across types instead.
    """
    patches, types = refs or _load_refs()
    ranked = sorted((_distance(patch, p), t) for p, t in zip(patches, types))
    best, kind = ranked[0]
    if best <= MAX_DISTANCE:
        return kind
    if best <= LOOSE_DISTANCE and len({t for _, t in ranked[:3]}) == 1:
        return kind
    return ""


def find_icons(flat: np.ndarray, refs=None) -> list[str]:
    """Energy types shown along the band's right end, left to right."""
    region, _x0 = _region(flat)
    hits = []
    for x, y, r in _circles(region):
        patch = _patch(region, x, y, r)
        if patch is None:
            continue
        kind = _accepted(patch, refs)
        if kind:
            hits.append((x, kind))
    return [kind for _x, kind in sorted(hits)]


def describe(icons: list[str]) -> str:
    """'Colorless x2', 'Grass', 'Rainbow (any type) x3' ..."""
    if not icons:
        return ""
    parts = []
    for kind in dict.fromkeys(icons):
        label = "Rainbow (any type)" if kind == "Rainbow" else kind
        n = icons.count(kind)
        parts.append(label + (" x{}".format(n) if n > 1 else ""))
    return ", ".join(parts)


# --- building the references ---------------------------------------------------
def _flat_of(path: str) -> np.ndarray:
    """An official scan softened to roughly camera sharpness."""
    img = cv2.imread(path)
    img = cv2.resize(img, (CARD_W // 2, CARD_H // 2), interpolation=cv2.INTER_AREA)
    return cv2.resize(img, (CARD_W, CARD_H))


def _reference_patch(path: str) -> np.ndarray | None:
    """The type icon of a labelled card: the rightmost large circle."""
    region, _ = _region(_flat_of(path))
    circles = [c for c in _circles(region) if c[2] >= 19]
    if not circles:
        return None
    x, y, r = max(circles, key=lambda c: c[0])
    return _patch(region, x, y, r)


def _labelled(folder: str) -> list[tuple[str, str]]:
    return [(f.split("__")[0], os.path.join(folder, f))
            for f in sorted(os.listdir(folder)) if "__" in f and f.endswith(".png")]


def build(folder: str) -> None:
    patches, types = [], []
    for kind, path in _labelled(folder):
        if kind == "Dragon":
            continue                        # never an energy type
        p = _reference_patch(path)
        if p is None:
            print("no icon found:", path)
            continue
        patches.append(p)
        types.append(kind)
    np.savez_compressed(REF_FILE, patches=np.array(patches), types=np.array(types))
    print("saved {} references to {}".format(len(patches), REF_FILE))


def add(kind: str, path: str) -> None:
    """Add the icon of one camera frame (a flattened card image, 734x1024) to
    the references as `kind`. Camera colours differ from the official scans
    the references started from - a camera shot of Grass sat closer to the
    scans' Darkness than to their Grass - so real camera samples fix that."""
    flat = cv2.imread(path)
    if flat is None or flat.shape[:2] != (CARD_H, CARD_W):
        raise SystemExit("need a flattened card image ({}x{}): {}".format(CARD_W, CARD_H, path))
    region, _ = _region(flat)
    circles = [c for c in _circles(region) if c[2] >= 19]
    if not circles:
        raise SystemExit("no icon found in " + path)
    x, y, r = max(circles, key=lambda c: c[0])
    patches, types = _load_refs()
    patches = np.concatenate([patches, _patch(region, x, y, r)[None]])
    np.savez_compressed(REF_FILE, patches=patches, types=np.array(types + [kind]))
    print("added {} from {} ({} references now)".format(kind, path, len(types) + 1))


def test(folder: str) -> None:
    items = [(k, _reference_patch(p), p) for k, p in _labelled(folder) if k != "Dragon"]
    items = [i for i in items if i[1] is not None]
    wrong = 0
    for i, (kind, patch, path) in enumerate(items):
        others = [it for j, it in enumerate(items) if j != i]
        refs = (np.array([o[1] for o in others]), [o[0] for o in others])
        got, dist = classify_patch(patch, refs)
        ok = got == kind
        wrong += not ok
        print("{:4} {:10} -> {:10} {:5.1f}  {}".format("ok" if ok else "BAD", kind, got,
                                                       dist, os.path.basename(path)))
    print("{} / {} correct".format(len(items) - wrong, len(items)))


if __name__ == "__main__":
    if sys.argv[1] == "add":
        add(sys.argv[2], sys.argv[3])
    else:
        {"build": build, "test": test}[sys.argv[1]](sys.argv[2])
