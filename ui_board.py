"""Emotion Rush screen renderer: the himpunan notice board.

Every screen is a royal-blue felt board with paper strips, pins, tape and stamps.
Pieces use only three angles (-6 for strips and labels, -3 for photos and headers,
+4 for papers and notes) and each piece overlaps a neighbour, so the board reads as
one composition. The direction contract lives in .impeccable/surfaces/.

Pillow draws the type with real fonts once per screen; OpenCV composes the live parts
(camera print, timer, score) every frame. All coordinates are on a 1600x900 canvas.
"""

import functools
import math
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 1600, 900
_HERE = os.path.dirname(os.path.abspath(__file__))
_FONTS = os.path.join(_HERE, "assets", "fonts")

# Palette (RGB). Stabilo colours only ever mark the item that matters right now.
BOARD = (27, 35, 201)
INK = (17, 17, 17)
PAPER = (255, 255, 255)
YELLOW = (255, 225, 74)
PINK = (255, 61, 139)
ORANGE = (255, 138, 0)
GREEN = (52, 199, 89)
TAPE = (240, 228, 192)
MUTED = (119, 119, 119)
CAM_EMPTY = (46, 49, 56)

STRIP, PHOTO, NOTE = -6, -3, 4

COURSE_COLORS = {
    "happy": YELLOW, "surprise": YELLOW, "angry": ORANGE, "fear": ORANGE,
    "sad": PINK, "disgust": GREEN, "neutral": GREEN,
}


def bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


# ─── Type ───────────────────────────────────────────────────

@functools.lru_cache(maxsize=None)
def archivo(size, weight=900, width=62):
    font = ImageFont.truetype(os.path.join(_FONTS, "Archivo-Variable.ttf"), int(size))
    font.set_variation_by_axes([weight, width])
    return font


@functools.lru_cache(maxsize=None)
def roboto(size, weight=400):
    font = ImageFont.truetype(os.path.join(_FONTS, "Roboto-Variable.ttf"), int(size))
    font.set_variation_by_axes([weight, 100])
    return font


def _wrap(text, font, max_w):
    lines, current = [], ""
    for word in text.split():
        trial = (current + " " + word).strip()
        if current and font.getlength(trial) > max_w:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines or [""]


def _text_box(rows, bg, pad, min_w=0, align="left", min_h=0):
    """rows: (text, font, fill, line_height_px). Caps sit centred in each line box."""
    widths = [font.getlength(text) for text, font, _, _ in rows]
    w = int(max(max(widths, default=0) + pad[0] + pad[2], min_w))
    h = int(max(sum(lh for *_, lh in rows) + pad[1] + pad[3], min_h))
    img = Image.new("RGBA", (w, h), (bg + (255,)) if bg else (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    y = pad[1] + (h - pad[1] - pad[3] - sum(lh for *_, lh in rows)) / 2
    for (text, font, fill, lh), tw in zip(rows, widths):
        baseline = y + lh / 2 + font.size * 0.35
        if align == "center":
            x = (w - tw) / 2
        elif align == "right":
            x = w - pad[2] - tw
        else:
            x = pad[0]
        draw.text((x, baseline), text, font=font, fill=fill, anchor="ls")
        y += lh
    return img


def label(text, size, bg=PAPER, fg=INK, width=80, weight=900, max_w=None, lh=1.0,
          pad=(22, 6, 22, 10), align="left"):
    font = archivo(size, weight, width)
    rows = _wrap(text, font, max_w - pad[0] - pad[2]) if max_w else [text]
    return _text_box([(r, font, fg, size * lh) for r in rows], bg, pad, align=align)


def strip(text, size, bg=PAPER, fg=INK):
    return label(text, size, bg, fg, width=62, lh=0.92, pad=(22, 2, 22, 8))


def B(text, size, weight=700, width=85, lh=1.3, fill=INK):
    """One block of a paper: wrapped to the paper width."""
    return (text, size, weight, width, lh, fill)


def paper(blocks, w, bg=PAPER, pad=(20, 18, 20, 18), align="left", min_h=0):
    inner = w - pad[0] - pad[2]
    rows = []
    for text, size, weight, width, lh, fill in blocks:
        font = archivo(size, weight, width)
        for line in _wrap(text, font, inner):
            rows.append((line, font, fill, size * lh))
    return _text_box(rows, bg, pad, min_w=w, align=align, min_h=min_h)


def stamp(text, size, border=8, pad=(22, 4, 22, 8), alpha=235):
    font = archivo(size, 900, 70)
    tw = font.getlength(text)
    w = int(tw + pad[0] + pad[2] + 2 * border)
    h = int(size * 0.95 + pad[1] + pad[3] + 2 * border)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    colour = PINK + (alpha,)
    half = border / 2
    draw.rectangle([half, half, w - half - 1, h - half - 1], outline=colour, width=border)
    draw.text((w / 2, h / 2 + size * 0.36), text, font=font, fill=colour, anchor="ms")
    return img


@functools.lru_cache(maxsize=None)
def pin_img(kind):
    img = Image.new("RGBA", (34, 38), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    if kind == "off":
        draw.ellipse([4, 4, 30, 30], fill=(15, 21, 128, 255), outline=(58, 66, 224, 255), width=3)
    else:
        colour, dark = {"done": (YELLOW, (168, 143, 18)), "on": (PINK, (158, 29, 85))}[kind]
        draw.ellipse([4, 7, 30, 33], fill=dark + (255,))
        draw.ellipse([4, 4, 30, 30], fill=colour + (255,))
    return img


def tape_img(w=150, h=42):
    return Image.new("RGBA", (w, h), TAPE + (219,))


# ─── Placement and blending ─────────────────────────────────

def _rot(img, deg):
    if not deg:
        return img
    return img.convert("RGBa").rotate(-deg, resample=Image.BICUBIC, expand=True).convert("RGBA")


def _shadow(img, blur=7, dy=6, opacity=0.38, margin=22):
    w, h = img.size
    mask = Image.new("L", (w + 2 * margin, h + 2 * margin), 0)
    mask.paste(img.getchannel("A"), (margin, margin + dy))
    mask = mask.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: int(v * opacity))
    out = Image.new("RGBA", mask.size, (5, 8, 60, 0))
    out.putalpha(mask)
    out.alpha_composite(img, (margin, margin))
    return out


def _comp(canvas, img, x, y):
    cw, ch = canvas.size
    left, top = max(0, -x), max(0, -y)
    right, bottom = min(img.width, cw - x), min(img.height, ch - y)
    if right > left and bottom > top:
        canvas.alpha_composite(img.crop((left, top, right, bottom)), (x + left, y + top))


class Sprite:
    __slots__ = ("bgr", "a", "x", "y", "box", "anim", "delay")

    def __init__(self, img, x, y, box=None, anim=None, delay=0):
        arr = np.asarray(img, dtype=np.uint8)
        self.bgr = arr[:, :, 2::-1].astype(np.float32)
        self.a = arr[:, :, 3:4].astype(np.float32) / 255.0
        self.x, self.y, self.box, self.anim, self.delay = x, y, box, anim, delay


def place(img, left, top, deg=0, shadow=True, center_x=None, anim=None):
    if center_x is not None:
        left = center_x - img.width / 2
    cx, cy = left + img.width / 2, top + img.height / 2
    rotated = _rot(img, deg)
    box = (int(cx - rotated.width / 2), int(cy - rotated.height / 2), rotated.width, rotated.height)
    if shadow:
        rotated = _shadow(rotated)
    x, y = int(round(cx - rotated.width / 2)), int(round(cy - rotated.height / 2))
    return Sprite(rotated, x, y, box, anim)


def _blend(out, sp, dx=0, dy=0, scale=1.0):
    colour, alpha = sp.bgr, sp.a
    x, y = sp.x + dx, sp.y + dy
    if scale != 1.0:
        h, w = alpha.shape[:2]
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        colour = cv2.resize(colour, (nw, nh), interpolation=cv2.INTER_LINEAR)
        alpha = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LINEAR)[:, :, None]
        x += (w - nw) // 2
        y += (h - nh) // 2
    h, w = alpha.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return
    a = alpha[y0 - y:y1 - y, x0 - x:x1 - x]
    region = out[y0:y1, x0:x1].astype(np.float32)
    region += (colour[y0 - y:y1 - y, x0 - x:x1 - x] - region) * a
    out[y0:y1, x0:x1] = region.astype(np.uint8)


def _ease_back(p):
    c1 = 1.70158
    return 1 + (c1 + 1) * (p - 1) ** 3 + c1 * (p - 1) ** 2


def _blend_anim(out, sp, t):
    if sp.anim == "drop":
        p = (t - sp.delay * 0.035) / 0.28
        if p <= 0:
            return
        if p < 1:
            _blend(out, sp, dy=int(-70 * (1 - _ease_back(p))))
            return
    elif sp.anim == "slam":
        p = (t - 0.18) / 0.24
        if p <= 0:
            return
        if p < 1:
            _blend(out, sp, scale=1 + 0.7 * (1 - _ease_back(p)))
            return
    _blend(out, sp)


def slam_scale(t, start=0.0, length=0.2, amount=0.5):
    p = (t - start) / length
    if p <= 0:
        return 1 + amount
    if p >= 1:
        return 1.0
    return 1 + amount * (1 - _ease_back(p))


# ─── Camera print ───────────────────────────────────────────

def fit_feed(frame, iw, ih, boxes=None, colour=None, tint=0.0):
    """Cover-fit a camera frame into a print; draw face boxes in the same mapping."""
    if frame is None:
        return np.full((ih, iw, 3), bgr(CAM_EMPTY), np.uint8)
    sh, sw = frame.shape[:2]
    s = max(iw / sw, ih / sh)
    rw, rh = max(iw, int(round(sw * s))), max(ih, int(round(sh * s)))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA if s < 0.6 else cv2.INTER_LINEAR)
    ox, oy = (rw - iw) // 2, (rh - ih) // 2
    feed = resized[oy:oy + ih, ox:ox + iw].copy()
    for x, y, w, h in boxes or ():
        cv2.rectangle(feed, (int(x * s - ox), int(y * s - oy)),
                      (int((x + w) * s - ox), int((y + h) * s - oy)), bgr(colour or GREEN), 4, cv2.LINE_AA)
    if tint:
        feed = cv2.addWeighted(feed, 1 - tint, np.full_like(feed, bgr(BOARD)), tint, 0)
    return feed


class CamSlot:
    def __init__(self, left, top, iw, ih, deg, pad, bottom, captions, cap_size, center_x=None, cap_width=75):
        self.iw, self.ih, self.pad, self.cap_width = iw, ih, pad, cap_width
        ow, oh = iw + 2 * pad, ih + pad + bottom
        if center_x is not None:
            left = center_x - ow / 2
        self.ow, self.oh = ow, oh
        self.frames = {c: self._frame(c, cap_size, bottom) for c in captions}
        cx, cy = left + ow / 2, top + oh / 2
        if deg:
            matrix = cv2.getRotationMatrix2D((ow / 2, oh / 2), -deg, 1.0)
            rad = math.radians(deg)
            cs, sn = abs(math.cos(rad)), abs(math.sin(rad))
            rw, rh = int(math.ceil(ow * cs + oh * sn)), int(math.ceil(ow * sn + oh * cs))
            matrix[0, 2] += rw / 2 - ow / 2
            matrix[1, 2] += rh / 2 - oh / 2
            self.matrix = matrix
            self.mask = cv2.warpAffine(np.ones((oh, ow), np.float32), matrix, (rw, rh),
                                       flags=cv2.INTER_LINEAR)[:, :, None]
        else:
            rw, rh = ow, oh
            self.matrix, self.mask = None, None
        self.size = (rw, rh)
        self.x, self.y = int(round(cx - rw / 2)), int(round(cy - rh / 2))
        self.box = (self.x, self.y, rw, rh)
        self.inside = self.x >= 0 and self.y >= 0 and self.x + rw <= W and self.y + rh <= H
        if self.mask is not None:
            # Most of a rotated print is opaque; only its anti-aliased edge needs blending.
            m = self.mask[:, :, 0]
            self.full = (m >= 0.999)[:, :, None]
            self.edge = np.nonzero((m > 0.001) & (m < 0.999))
            self.edge_a = m[self.edge][:, None]
        self.static = None
        self.static_caption = captions[0]

    def _frame(self, caption, cap_size, bottom):
        img = Image.new("RGB", (self.ow, self.oh), PAPER)
        if caption:
            draw = ImageDraw.Draw(img)
            font = archivo(cap_size, 800, self.cap_width)
            draw.text((self.pad + 6, self.oh - bottom / 2 + cap_size * 0.35), caption,
                      font=font, fill=INK, anchor="ls")
        return np.asarray(img)[:, :, ::-1].copy()

    def shadow(self):
        mask = self.mask[:, :, 0] if self.mask is not None else np.ones((self.oh, self.ow), np.float32)
        margin = 30
        layer = np.zeros((mask.shape[0] + 2 * margin, mask.shape[1] + 2 * margin), np.uint8)
        layer[margin + 8:margin + 8 + mask.shape[0], margin:margin + mask.shape[1]] = (mask * 255).astype(np.uint8)
        layer = Image.fromarray(layer).filter(ImageFilter.GaussianBlur(11)).point(lambda v: int(v * 0.45))
        out = Image.new("RGBA", layer.size, (5, 8, 60, 0))
        out.putalpha(layer)
        return out, self.x - margin, self.y - margin

    def draw(self, out, feed, caption=None):
        frame = self.frames.get(caption)
        if frame is None:
            frame = next(iter(self.frames.values()))
        img = frame.copy()
        p = self.pad
        img[p:p + self.ih, p:p + self.iw] = feed
        x, y = self.x, self.y
        if self.matrix is None:
            out[y:y + self.oh, x:x + self.ow] = img
            return
        warped = cv2.warpAffine(img, self.matrix, self.size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        rw, rh = self.size
        if self.inside:
            region = out[y:y + rh, x:x + rw]
            np.copyto(region, warped, where=self.full)
            ey, ex = self.edge
            region[ey, ex] = (region[ey, ex] * (1 - self.edge_a) + warped[ey, ex] * self.edge_a).astype(np.uint8)
            return
        x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + rw, W), min(y + rh, H)
        a = self.mask[y0 - y:y1 - y, x0 - x:x1 - x]
        region = out[y0:y1, x0:x1].astype(np.float32)
        region += (warped[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) - region) * a
        out[y0:y1, x0:x1] = region.astype(np.uint8)


# ─── Board builder and scene cache ──────────────────────────

@functools.lru_cache(maxsize=1)
def _felt():
    img = np.empty((H, W, 3), np.float32)
    img[:] = BOARD
    for y in range(0, H, 5):
        img[y:y + 2] = img[y:y + 2] * (1 - 0.028) + 255 * 0.028
    for x in range(0, W, 5):
        img[:, x:x + 2] = img[:, x:x + 2] * (1 - 0.05) + np.array([0, 0, 40], np.float32) * 0.05
    return img.astype(np.uint8)


class Scene:
    def __init__(self, base, over, slots, buttons):
        self.base, self.over, self.slots, self.buttons = base, over, slots, buttons
        # Once the entrance animation is over, the papers are baked into `settled` and only
        # the ones lying on a live photo are blended again each frame.
        self.settled, self.reblend = None, None
        self.settle_time = 0.8 + 0.035 * len(over)


def _overlaps(sp, box):
    h, w = sp.a.shape[:2]
    bx, by, bw, bh = box
    return sp.x < bx + bw and bx < sp.x + w and sp.y < by + bh and by < sp.y + h


class Board:
    def __init__(self, plain=None):
        if plain is not None:
            self.img = Image.new("RGBA", (W, H), plain + (255,))
        else:
            self.img = Image.fromarray(_felt()).convert("RGBA")
        self.over, self.slots, self.buttons = [], {}, {}

    def ghost(self, left, top, w, h, deg):
        layer = _rot(Image.new("RGBA", (w, h), (255, 255, 255, 18)), deg)
        _comp(self.img, layer, int(left + w / 2 - layer.width / 2), int(top + h / 2 - layer.height / 2))

    def staple(self, x, y, deg):
        _comp(self.img, _rot(Image.new("RGBA", (16, 3), (255, 255, 255, 56)), deg), x, y)

    def under(self, img, left, top, deg=0, shadow=True, center_x=None, button=None):
        sp = place(img, left, top, deg, shadow, center_x)
        arr = np.concatenate([sp.bgr[:, :, ::-1], sp.a * 255], axis=2).astype(np.uint8)
        _comp(self.img, Image.fromarray(arr, "RGBA"), sp.x, sp.y)
        if button:
            self.buttons[button] = sp.box
        return sp.box

    def add(self, img, left, top, deg=0, shadow=True, center_x=None, button=None, anim="drop"):
        sp = place(img, left, top, deg, shadow, center_x, anim)
        sp.delay = len(self.over)
        self.over.append(sp)
        if button:
            self.buttons[button] = sp.box
        return sp.box

    def pins(self, states, top=22, center_x=W // 2, gap=30, vertical_left=None):
        size = 34
        for i, kind in enumerate(states):
            if vertical_left is not None:
                self.add(pin_img(kind), vertical_left, top + i * (size + gap), shadow=True)
            else:
                total = len(states) * size + (len(states) - 1) * gap
                self.add(pin_img(kind), center_x - total / 2 + i * (size + gap), top, shadow=True)

    def photo(self, name, left, top, iw, ih, deg=PHOTO, pad=16, bottom=56, captions=(None,),
              cap_size=26, center_x=None, static=None, cap_width=75):
        slot = CamSlot(left, top, iw, ih, deg, pad, bottom, tuple(captions), cap_size, center_x, cap_width)
        sh, sx, sy = slot.shadow()
        _comp(self.img, sh, sx, sy)
        slot.static = static
        self.slots[name] = slot
        return slot

    def scene(self):
        base = np.asarray(self.img.convert("RGB"))[:, :, ::-1].copy()
        for slot in self.slots.values():
            if slot.static is not None:
                slot.draw(base, slot.static, slot.static_caption)
        return Scene(base, self.over, self.slots, self.buttons)


_scenes = {}
_last_buttons = {}


def _scene(key, build):
    scene = _scenes.get(key)
    if scene is None:
        if len(_scenes) > 24:
            _scenes.clear()
        scene = _scenes[key] = build()
    return scene


def compose(scene, t, feeds=None, captions=None, dynamic=()):
    global _last_buttons
    live = [(name, slot) for name, slot in scene.slots.items()
            if slot.static is None and feeds and name in feeds]
    if t >= scene.settle_time:
        if scene.settled is None:
            settled = scene.base.copy()
            for sp in scene.over:
                _blend(settled, sp)
            boxes = [slot.box for slot in scene.slots.values() if slot.static is None]
            scene.reblend = [sp for sp in scene.over if any(_overlaps(sp, b) for b in boxes)]
            scene.settled = settled
        out = scene.settled.copy()
        for name, slot in live:
            slot.draw(out, feeds[name], (captions or {}).get(name))
        for sp in scene.reblend:
            _blend(out, sp)
    else:
        out = scene.base.copy()
        for name, slot in live:
            slot.draw(out, feeds[name], (captions or {}).get(name))
        for sp in scene.over:
            _blend_anim(out, sp, t)
    for item in dynamic:
        if item is None:
            continue
        if isinstance(item, tuple):
            _blend(out, item[0], scale=item[1])
        else:
            _blend(out, item)
    _last_buttons = dict(scene.buttons)
    return out


def hit_test(x, y):
    """Name of the button under (x, y) on the last rendered screen, or None."""
    for name, (bx, by, bw, bh) in reversed(list(_last_buttons.items())):
        if bx <= x < bx + bw and by <= y < by + bh:
            return name
    return None


def fmt(value, digits=1):
    return f"{value:.{digits}f}".replace(".", ",")


# ─── Shared pieces ──────────────────────────────────────────

FOV_ASPECT = 293 / 238


@functools.lru_cache(maxsize=256)
def _count_note(count, left, top, colour=GREEN):
    img = paper([B("Muka kebaca", 24), B(str(count), 80, 900, 62, 1.0)], 250, bg=colour)
    return place(img, left, top, NOTE)


@functools.lru_cache(maxsize=64)
def _digit_paper(text, left, top):
    img = Image.new("RGBA", (330, 400), PAPER + (255,))
    draw = ImageDraw.Draw(img)
    draw.text((165, 200), text, font=archivo(360, 900, 62), fill=INK, anchor="mm")
    return place(img, left, top, NOTE)


def _course_list(courses, done, active):
    """Papers listing the seven courses; finished ones struck through."""
    font = archivo(22, 700, 80)
    lh = 22 * 1.65
    w = 390
    h = int(len(courses) * lh + 40)
    img = Image.new("RGBA", (w, h), PAPER + (255,))
    draw = ImageDraw.Draw(img)
    for i, (name, colour) in enumerate(courses):
        y = 20 + i * lh
        base = y + lh / 2 + 22 * 0.35
        tw = font.getlength(name)
        if i == active:
            draw.rectangle([14, y + 5, 20 + tw + 6, y + lh - 5], fill=colour + (255,))
        fill = MUTED if i in done else INK
        draw.text((20, base), name, font=font, fill=fill, anchor="ls")
        if i in done:
            draw.line([18, base - 7, 22 + tw, base - 7], fill=MUTED, width=2)
    return img


# ─── Screens ────────────────────────────────────────────────

def render_idle(frame, t, played_today):
    def build():
        b = Board()
        b.ghost(20, 620, 330, 260, NOTE)
        b.ghost(1280, 640, 290, 230, STRIP)
        b.ghost(560, 800, 480, 140, PHOTO)
        b.staple(180, 700, NOTE)
        b.staple(1420, 720, STRIP)
        b.photo("cam", 0, 196, 700, 568, captions=("Masuk frame, maksimal bertiga.",), center_x=W // 2)
        b.add(tape_img(), 422, 186, -40, shadow=False)
        b.add(tape_img(), 1032, 156, 32, shadow=False)
        b.add(stamp("NILAI LANGSUNG KELUAR", 30), 830, 690, STRIP)
        b.add(label("MADING HIMPUNAN EKSPRESI · ITB INSIGHT 2026", 34, INK, PAPER), 0, 140, PHOTO, center_x=W // 2)
        b.pins(["off"] * 7, top=92)
        b.add(strip("RUSH", 150, INK, PAPER), 130, 370, STRIP)
        b.add(strip("EMOTION", 104), 70, 270, STRIP)
        b.add(paper([B("GIMANA MAINNYA?", 30, 900, 70, 1.25),
                     B("1. Pasang muka sesuai soal", 24), B("2. AI langsung ngasih nilai", 24),
                     B("3. Pulang bawa strip foto", 24)], 340), 70, 530, NOTE)
        b.add(pin_img("on"), 226, 522)
        b.add(paper([B("Udah main hari ini", 24), B(str(played_today), 96, 900, 62, 1.05)], 250), 1250, 520, NOTE)
        b.add(label("KELUAR", 36), 1230, 462, STRIP, button="exit")
        b.add(label("CEK KAMERA", 44), 1190, 390, STRIP, button="calibrate")
        b.add(label("GAS MAIN", 80, YELLOW), 1140, 272, STRIP, button="start")
        b.add(pin_img("on"), 1408, 262)
        return b.scene()

    scene = _scene(("idle", played_today), build)
    slot = scene.slots["cam"]
    return compose(scene, t, {"cam": fit_feed(frame, slot.iw, slot.ih)})


def render_calibration(frame, t, boxes, count):
    def build():
        b = Board()
        b.ghost(30, 640, 300, 230, NOTE)
        b.ghost(1290, 650, 280, 220, STRIP)
        b.photo("cam", 0, 168, 900, 506, captions=("Geser kamera sampai semua muka kebaca.",), center_x=W // 2)
        b.add(label("CEK KAMERA", 44, INK, PAPER), 0, 112, PHOTO, center_x=W // 2)
        b.add(paper([B("TIPS", 30, 900, 70, 1.25),
                     B("Lampu dari depan, jangan dari belakang.", 24),
                     B("Muka jangan ketutupan rambut atau tangan.", 24)], 300), 1240, 300, NOTE)
        b.add(label("BALIK KE MENU", 44), 0, 790, STRIP, center_x=W // 2, button="back")
        return b.scene()

    scene = _scene(("calibration",), build)
    slot = scene.slots["cam"]
    feed = fit_feed(frame, slot.iw, slot.ih, boxes, GREEN)
    return compose(scene, t, {"cam": feed}, dynamic=[_count_note(count, 80, 330)])


def _countdown_scene(key, title, title_colour, kicker, caption_lines, side_fn, pin_states):
    """side_fn builds the left paper; it only runs when the scene is first built."""
    def build():
        b = Board()
        b.ghost(1300, 700, 270, 200, STRIP)
        b.ghost(30, 720, 300, 180, NOTE)
        b.pins(pin_states, top=22)
        b.photo("cam", 0, 190, 625, 508, captions=caption_lines, center_x=W // 2)
        box = b.add(label(title, 74, title_colour), 0, 100, PHOTO, center_x=W // 2)
        b.add(label(kicker, 30, INK, PAPER), box[0] + 40, 66, STRIP)
        b.add(side_fn(), 95, 290, NOTE)
        return b.scene()
    return _scene(key, build)


@functools.lru_cache(maxsize=1)
def _ready_label():
    # Dynamic so it lands on top of the digit paper it is tucked under.
    return place(label("SIAP-SIAP", 44, YELLOW), 1150, 640, STRIP)


def render_preparation(frame, t, remaining, boxes, count):
    def side():
        return paper([B("BIAR NILAINYA MANTAP", 28, 900, 70, 1.25), B("Lampu dari depan", 24),
                      B("Muka jangan ketutupan", 24), B("Maksimal bertiga", 24)], 390)
    scene = _countdown_scene(("prep",), "SEMESTER EKSPRESI DIMULAI", PINK, "SIAP-SIAP DULU",
                             ("Atur posisi dulu, bentar lagi mulai.",), side, ["off"] * 7)
    slot = scene.slots["cam"]
    digit = str(max(1, math.ceil(remaining)))
    frac = 1 - (remaining - math.floor(remaining)) if remaining > 0 else 1
    return compose(scene, t, {"cam": fit_feed(frame, slot.iw, slot.ih, boxes, GREEN)},
                   dynamic=[_count_note(count, 140, 575), (_digit_paper(digit, 1110, 250), slam_scale(frac)), _ready_label()])


def render_countdown(frame, t, remaining, index, courses, boxes, count):
    key, name, instruction = courses[index][:3]
    colour = COURSE_COLORS.get(key, YELLOW)
    names = [(c[1], COURSE_COLORS.get(c[0], YELLOW)) for c in courses]
    pin_states = ["done"] * index + ["on"] + ["off"] * (len(courses) - index - 1)
    scene = _countdown_scene(("countdown", index, tuple(c[1] for c in courses)), name.upper(), colour,
                             f"MATKUL {index + 1} / {len(courses)}", (instruction,),
                             lambda: _course_list(names, set(range(index)), index), pin_states)
    slot = scene.slots["cam"]
    digit = str(max(1, math.ceil(remaining)))
    frac = 1 - (remaining - math.floor(remaining)) if remaining > 0 else 1
    return compose(scene, t, {"cam": fit_feed(frame, slot.iw, slot.ih, boxes, colour)},
                   dynamic=[_count_note(count, 140, 575), (_digit_paper(digit, 1110, 250), slam_scale(frac)), _ready_label()])


@functools.lru_cache(maxsize=256)
def _timer_paper(text):
    img = paper([B(text, 120, 900, 62, 1.0), B("detik", 24)], 230, align="center")
    return place(img, 1180, 140, NOTE)


@functools.lru_cache(maxsize=128)
def _record_paper(value):
    img = paper([B("Rekor", 24), B(str(value), 96, 900, 62, 1.05)], 220, align="center")
    return place(img, 200, 540, NOTE)


@functools.lru_cache(maxsize=128)
def _live_bar(value, colour):
    w, h = 900, 72
    img = Image.new("RGBA", (w, h), PAPER + (255,))
    draw = ImageDraw.Draw(img)
    font = archivo(26, 800, 80)
    draw.text((22, h / 2 + 26 * 0.35), "NILAI LIVE", font=font, fill=INK, anchor="ls")
    x0, x1 = 22 + font.getlength("NILAI LIVE") + 20, w - 110
    draw.rectangle([x0, 19, x0 + (x1 - x0) * value / 100, h - 19], fill=colour + (255,))
    draw.text((w - 22, h / 2 + 48 * 0.35), str(value), font=archivo(48, 900, 62), fill=INK, anchor="rs")
    return place(img, 0, 744, PHOTO, center_x=W // 2)


@functools.lru_cache(maxsize=4)
def _all_out():
    return place(strip("ALL OUT!", 150, YELLOW), 0, 560, STRIP, center_x=W // 2)


def render_playing(frame, t, remaining, total, index, courses, score, best, boxes):
    key, name = courses[index][:2]
    colour = COURSE_COLORS.get(key, YELLOW)

    def build():
        b = Board()
        b.ghost(20, 150, 200, 300, NOTE)
        b.ghost(1380, 560, 200, 300, STRIP)
        b.photo("cam", 0, 110, 780, 634, deg=0, bottom=16, center_x=W // 2)
        b.pins(["done"] * index + ["on"] + ["off"] * (len(courses) - index - 1), top=170, vertical_left=389)
        b.add(label(f"{index + 1}/{len(courses)} · {name.upper()}", 56, colour), 0, 52, PHOTO, center_x=W // 2)
        b.add(label("FOTO TERBAIK OTOMATIS TERSIMPAN", 34, max_w=280), 1196, 330, STRIP)
        return b.scene()

    scene = _scene(("playing", index, name), build)
    slot = scene.slots["cam"]
    dynamic = [_record_paper(int(best)), _timer_paper(fmt(remaining)), _live_bar(int(score), colour)]
    if remaining < 1.5:
        dynamic.append((_all_out(), slam_scale(1.5 - remaining, 0.0, 0.22, 0.6)))
    return compose(scene, t, {"cam": fit_feed(frame, slot.iw, slot.ih, boxes, colour)}, dynamic=dynamic)


@functools.lru_cache(maxsize=64)
def _next_label(text):
    return place(label(text, 34), 470, 724, PHOTO)


def render_result(t, remaining, index, total, result, course_key, is_last):
    colour = COURSE_COLORS.get(course_key, YELLOW)

    def build():
        b = Board()
        b.ghost(30, 660, 330, 220, NOTE)
        b.ghost(1300, 670, 270, 210, STRIP)
        b.pins(["done"] * (index + 1) + ["off"] * (total - index - 1), top=62)
        feed = fit_feed(result.get("frame"), 601, 488)
        b.photo("cam", 494, 140, 601, 488, captions=(result["challenge"],), static=feed)
        b.add(tape_img(), 466, 136, -40, shadow=False)
        b.add(label(result["challenge"].upper(), 44, colour, max_w=470), 50, 250, STRIP)
        b.add(label(f"MATKUL {index + 1}/{total} KELAR", 30, INK, PAPER), 70, 196, STRIP)
        passed = result["passed"]
        b.add(label("LULUS" if passed else "NGULANG", 44, GREEN if passed else PINK), 130, 420, STRIP)
        faces = result.get("face_count", 0)
        if faces > 1:
            note = f"Nilainya dirata-rata dari {faces} muka di foto."
        elif faces == 1:
            note = "Nilainya dari 1 muka di foto."
        else:
            note = "Mukanya nggak kebaca, jadi nilainya 0."
        b.add(paper([B(note, 24)], 330), 80, 486, NOTE)
        b.add(paper([B("Nilai", 24), B(str(result["pct_int"]), 170, 900, 62, 0.95)], 300), 1100, 170, NOTE)
        b.add(paper([B(result["comment"], 26, 700, 85, 1.25)], 340, bg=YELLOW), 1150, 420, NOTE)
        b.add(stamp(result["grade"], 220, border=14, pad=(30, 0, 30, 10)), 720, 380, STRIP, shadow=False, anim="slam")
        return b.scene()

    scene = _scene(("result", index, result["challenge"], result["pct_int"]), build)
    what = "Transkrip keluar" if is_last else "Matkul berikutnya"
    return compose(scene, t, dynamic=[_next_label(f"{what} {fmt(max(0.0, remaining))} detik lagi")])


# ─── SIX transcript (pinned reference: the real SIX ITB tables) ─

SIX_BLUE = (48, 113, 169)
SIX_BTN = (51, 122, 183)
SIX_BTN_EDGE = (46, 109, 164)
SIX_INFO = (217, 237, 247)
SIX_INFO_EDGE = (188, 232, 241)
SIX_STRIPE = (245, 245, 245)
SIX_TOTAL = (252, 248, 227)
SIX_TEXT = (51, 51, 51)
SIX_MUTED = (153, 153, 153)
SIX_SCORE = (169, 68, 66)
SIX_CHECK = (60, 118, 61)
SIX_BORDER = (221, 221, 221)


def render_transcript(results, nr, predicate, best_frame, codes):
    def build():
        b = Board(plain=(255, 255, 255))
        img = b.img
        draw = ImageDraw.Draw(img)
        x0, x1, top = 50, 1250, 50
        draw.rounded_rectangle([x0, top, x1, top + 70], radius=6, fill=SIX_BLUE)
        draw.rectangle([x0, top + 40, x1, top + 70], fill=SIX_BLUE)
        draw.text((x0 + 26, top + 35 + 11), "Transkrip Ekspresi", font=roboto(32), fill=PAPER, anchor="ls")
        y = top + 70
        draw.rectangle([x0, y, x1, y + 52], fill=SIX_INFO, outline=SIX_INFO_EDGE)
        # Calendar glyph, drawn rather than borrowed from an icon font.
        cx, cy = x0 + 22, y + 15
        draw.rectangle([cx, cy + 4, cx + 22, cy + 24], outline=SIX_TEXT, width=2)
        draw.line([cx, cy + 10, cx + 22, cy + 10], fill=SIX_TEXT, width=2)
        draw.line([cx + 6, cy, cx + 6, cy + 6], fill=SIX_TEXT, width=2)
        draw.line([cx + 16, cy, cx + 16, cy + 6], fill=SIX_TEXT, width=2)
        draw.text((cx + 36, y + 26 + 9), "Semester Ganjil 2026/2027 · ITB Insight", font=roboto(24, 700),
                  fill=SIX_TEXT, anchor="ls")
        y += 52
        cols = [x0 + 18, x0 + 230, x0 + 760, x0 + 860, x0 + 960, x0 + 1080]
        heads = ["Kode", "Mata Kuliah", "SKS", "Nilai", "Skor", "Lulus"]
        for cx_, head in zip(cols, heads):
            draw.text((cx_, y + 40), head, font=roboto(24, 700), fill=SIX_TEXT, anchor="ls")
        y += 54
        draw.line([x0, y, x1, y], fill=SIX_BORDER, width=3)
        row_h = 56
        for i, res in enumerate(results):
            if i % 2 == 1:
                draw.rectangle([x0, y, x1, y + row_h], fill=SIX_STRIPE)
            draw.line([x0, y, x1, y], fill=(230, 230, 230), width=1)
            base = y + row_h / 2 + 9
            num = f"{i + 1} "
            draw.text((cols[0], base), num, font=roboto(24, 700), fill=SIX_TEXT, anchor="ls")
            code_x = cols[0] + roboto(24, 700).getlength(num)
            code, suffix = codes[i]
            draw.text((code_x, base), code, font=roboto(24), fill=SIX_TEXT, anchor="ls")
            draw.text((code_x + roboto(24).getlength(code), base), suffix, font=roboto(24), fill=SIX_MUTED,
                      anchor="ls")
            draw.text((cols[1], base), res["challenge"], font=roboto(24), fill=SIX_TEXT, anchor="ls")
            draw.text((cols[2], base), "3", font=roboto(24), fill=SIX_TEXT, anchor="ls")
            draw.text((cols[3], base), res["grade"], font=roboto(24), fill=SIX_TEXT, anchor="ls")
            draw.text((cols[4], base), f"{res['pct_int']}%", font=roboto(24), fill=SIX_SCORE, anchor="ls")
            if res["passed"]:
                ck = cols[5] + 10
                draw.line([(ck, base - 10), (ck + 8, base - 2), (ck + 24, base - 20)], fill=SIX_CHECK, width=4)
            else:
                draw.text((cols[5] + 10, base), "–", font=roboto(24), fill=SIX_MUTED, anchor="ls")
            y += row_h
        draw.rectangle([x0, y, x1, y + row_h], fill=SIX_TOTAL)
        draw.text((cols[2], y + row_h / 2 + 9), str(3 * len(results)), font=roboto(24, 700), fill=SIX_TEXT,
                  anchor="ls")
        draw.text((cols[3], y + row_h / 2 + 9), f"NR {nr:.2f}", font=roboto(24, 700), fill=SIX_TEXT, anchor="ls")
        y += row_h
        draw.rounded_rectangle([x0, top, x1, y], radius=6, outline=SIX_BLUE, width=2)
        y += 40
        draw.text((x0, y + 30), f"Predikat: {predicate}", font=roboto(28, 700), fill=SIX_TEXT, anchor="ls")
        draw.text((x0, y + 70), "atau tekan Spasi", font=roboto(22), fill=SIX_MUTED, anchor="ls")
        btn = (x1 - 380, y, x1, y + 64)
        draw.rounded_rectangle(btn, radius=6, fill=SIX_BTN, outline=SIX_BTN_EDGE, width=2)
        draw.text(((btn[0] + btn[2]) / 2, y + 32 + 10), "Pilih foto buat strip", font=roboto(28), fill=PAPER,
                  anchor="ms")
        b.buttons["next"] = (btn[0], btn[1], btn[2] - btn[0], btn[3] - btn[1])
        # Passport photo: the player's best frame, like the photo beside the SIX table.
        px0, py0, pw, ph = 1290, 50, 260, 346
        draw.rounded_rectangle([px0, py0, px0 + pw, py0 + ph], radius=6, fill=PAPER, outline=SIX_BORDER, width=2)
        if best_frame is not None:
            face = fit_feed(best_frame, pw - 20, ph - 20)
            img.paste(Image.fromarray(face[:, :, ::-1]), (px0 + 10, py0 + 10))
        draw.text((px0 + pw / 2, py0 + ph + 34), "Foto terbaik", font=roboto(22), fill=SIX_MUTED, anchor="ms")
        return b.scene()

    key = ("six", tuple((r["challenge"], r["pct_int"]) for r in results))
    return compose(_scene(key, build), 10.0)


# ─── Photo selection, preview, email, done ──────────────────

_SEL_POS = [(80, 140), (450, 140), (820, 140), (1190, 140), (265, 452), (635, 452), (1005, 452)]
_SEL_IW, _SEL_IH = 280, 227


def render_selection(t, results, selected, message):
    def build():
        b = Board()
        b.ghost(40, 700, 300, 190, NOTE)
        for i, res in enumerate(results):
            left, top = _SEL_POS[i]
            chosen = i in selected
            feed = fit_feed(res.get("frame"), _SEL_IW, _SEL_IH, tint=0.0 if chosen else 0.45)
            slot = b.photo(f"p{i}", left, top, _SEL_IW, _SEL_IH, pad=11, bottom=44,
                           captions=(f"{i + 1} · {res['challenge']}",), cap_size=21, cap_width=62, static=feed)
            b.buttons[f"card{i}"] = slot.box
        b.add(label("PILIH 6 FOTO BUAT STRIP", 64, PINK), 0, 34, PHOTO, center_x=W // 2)
        if not message:
            b.add(label("Klik fotonya buat milih atau batal", 26, INK, PAPER, weight=800, width=85), 0, 800, STRIP,
                  center_x=W // 2 + 60)
        for i, res in enumerate(results):
            left, top = _SEL_POS[i]
            b.add(label(f"{res['grade']} · {res['pct_int']}", 26), left - 12, top + 200, STRIP)
            if i in selected:
                b.add(pin_img("on"), left + 140, top - 8)
                b.add(label("MASUK", 30, YELLOW), left + 190, top + 8, STRIP)
        b.add(paper([B("Dipilih", 24), B(f"{len(selected)}/6", 80, 900, 62, 1.0)], 220), 70, 762, NOTE)
        if len(selected) == 6:
            b.add(label("LANJUT", 72, YELLOW), 1280, 770, STRIP, button="continue")
        else:
            b.add(label(f"KURANG {6 - len(selected)} FOTO" if len(selected) < 6 else "KEBANYAKAN", 40),
                  1240, 790, STRIP, button="continue")
        if message:
            b.add(paper([B(message, 24)], 520, bg=PINK), 360, 790, NOTE)
        return b.scene()

    key = ("select", tuple(r["challenge"] for r in results), tuple(sorted(selected)), message)
    return compose(_scene(key, build), t)


def render_preview(t, sheet, gif_frame, gif_caption):
    def build():
        b = Board()
        b.ghost(1290, 40, 270, 200, NOTE)
        h = 800
        w = int(sheet.shape[1] * h / sheet.shape[0])
        strip_img = Image.fromarray(cv2.resize(sheet, (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1]).convert("RGBA")
        b.under(strip_img, 200, 52, PHOTO)
        b.add(pin_img("on"), 200 + w / 2 - 17, 40)
        b.photo("gif", 760, 160, 560, 455, captions=(None,))
        b.add(label("CEK HASILNYA DULU", 64, PINK), 760, 66, PHOTO)
        b.add(label("SIMPAN", 72, YELLOW), 790, 700, STRIP, button="approve")
        b.add(label("GANTI FOTO", 44), 1120, 724, STRIP, button="change")
        b.add(label("Y = simpan · N = ganti", 24, INK, PAPER, weight=800, width=85), 1150, 806, STRIP)
        return b.scene()

    scene = _scene(("preview", id(sheet)), build)
    slot = scene.slots["gif"]
    if gif_caption not in slot.frames:
        slot.frames[gif_caption] = slot._frame(gif_caption, 26, 56)
    return compose(scene, t, {"gif": fit_feed(gif_frame, slot.iw, slot.ih)}, {"gif": gif_caption})


@functools.lru_cache(maxsize=128)
def _email_form(text, caret, error):
    w, h = 880, 300
    img = Image.new("RGBA", (w, h), PAPER + (255,))
    draw = ImageDraw.Draw(img)
    draw.text((36, 60), "Email kamu", font=archivo(28, 800, 85), fill=INK, anchor="ls")
    shown = text if len(text) <= 40 else "…" + text[-39:]
    font = archivo(46, 700, 85)
    if text:
        draw.text((36, 142), shown + ("_" if caret else ""), font=font, fill=INK, anchor="ls")
    else:
        draw.text((36, 142), "nama@email.com", font=font, fill=(150, 150, 150), anchor="ls")
    draw.line([36, 160, w - 36, 160], fill=PINK if error else INK, width=3)
    font_small = archivo(24, 700, 85)
    for i, line in enumerate(_wrap("Email cuma dipakai buat nama folder hasil foto di PC ini.", font_small, w - 72)):
        draw.text((36, 214 + i * 32), line, font=font_small, fill=INK, anchor="ls")
    return place(img, 0, 236, PHOTO, center_x=W // 2)


@functools.lru_cache(maxsize=16)
def _error_note(text):
    return place(paper([B(text, 24)], 520, bg=PINK), 380, 590, NOTE)


def render_email(t, text, caret, error):
    def build():
        b = Board()
        b.ghost(40, 640, 320, 240, NOTE)
        b.ghost(1260, 80, 300, 220, STRIP)
        b.add(label("SIMPAN HASILMU", 72, PINK), 0, 120, PHOTO, center_x=W // 2)
        b.add(label("SIMPAN", 64, YELLOW), 1000, 598, STRIP, button="save")
        b.add(label("Enter = simpan · Esc = balik", 26, INK, PAPER, weight=800, width=85), 990, 716, STRIP)
        return b.scene()

    scene = _scene(("email",), build)
    return compose(scene, t, dynamic=[_email_form(text, caret, bool(error)), _error_note(error) if error else None])


def render_done(t, nr, predicate, saved, email):
    def build():
        b = Board()
        b.ghost(40, 600, 320, 260, NOTE)
        b.ghost(1250, 80, 300, 240, STRIP)
        b.add(strip("MAKASIH UDAH MAIN!", 110), 0, 110, STRIP, center_x=W // 2)
        b.add(paper([B("NR kamu", 26), B(fmt(nr, 2), 170, 900, 62, 0.95), B(predicate, 28, 800, 80, 1.25)],
                    520, align="center"), 0, 290, NOTE, center_x=W // 2)
        if saved:
            status = f"Di akhir acara hasilnya akan dikirim ke {email}. Stay tune!"
        else:
            status = "Fotonya nggak disimpan."
        b.add(paper([B(status, 24)], 460), 140, 520, PHOTO)
        b.add(stamp("TERSIMPAN" if saved else "NGGAK DISIMPAN", 70, border=10), 960, 262, STRIP,
              shadow=False, anim="slam")
        b.add(label("MAIN LAGI · R", 56, YELLOW), 520, 720, STRIP, button="again")
        b.add(label("KE MENU · Q", 40), 920, 744, STRIP, button="menu")
        return b.scene()

    return compose(_scene(("done", round(nr, 2), predicate, saved, email), build), t)
