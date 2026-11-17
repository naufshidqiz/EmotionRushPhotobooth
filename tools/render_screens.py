"""Render every Emotion Rush screen to PNG without a webcam, for design review.

Usage (from the project folder):
    venv\\Scripts\\python.exe tools\\render_screens.py [output_folder]

Uses a synthetic camera image, so no player photos are involved.
"""

import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ui_board as ui  # noqa: E402

COURSES = [
    ("happy", "Senang Keberlanjutan", "Senyum paling lebar, gas!"),
    ("angry", "Sistem Manajemen Emosi", "Keluarin muka paling kesel kamu."),
    ("sad", "Pembelajaran Mesin untuk Sedih", "Keluarin muka paling galau kamu."),
    ("neutral", "Dinamika Netral", "Lempeng aja, tanpa ekspresi."),
    ("fear", "Pemrosesan Takut", "Pasang muka paling ketakutan."),
    ("disgust", "Mekanika Jijik", "Muka jijik, kayak nyium kaus kaki seminggu."),
    ("surprise", "Medan ElektroKagettika", "Kaget! Mata melotot, mulut nganga."),
]


def fake_camera(seed=0, w=1280, h=720):
    """A neutral stand-in for the webcam: gradient, grid and two face-sized ellipses."""
    y, x = np.mgrid[0:h, 0:w]
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = (60 + 60 * x / w + seed * 9) % 255
    img[..., 1] = 70 + 50 * y / h
    img[..., 2] = 90
    for gx in range(0, w, 80):
        cv2.line(img, (gx, 0), (gx, h), (110, 110, 120), 1)
    for cx in (520, 760):
        cv2.ellipse(img, (cx, 380), (95, 125), 0, 0, 360, (150, 170, 200), -1)
    return img


def fov(frame):
    h, w = frame.shape[:2]
    tw = min(w, int(round(h * ui.FOV_ASPECT)))
    left = (w - tw) // 2
    return frame[:, left:left + tw].copy()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join("design", "renders")
    os.makedirs(out, exist_ok=True)
    cam = fake_camera()
    crop = fov(cam)
    boxes = [(170, 230, 190, 190), (420, 240, 180, 180)]
    t = 5.0
    results = []
    for i, (key, name, _) in enumerate(COURSES):
        pct = [92, 81, 74, 88, 55, 63, 70][i]
        grade = {92: "A", 81: "AB", 74: "B", 88: "A", 55: "BC", 63: "B", 70: "B"}[pct]
        results.append({"challenge": name, "emo_key": key, "pct_int": pct, "grade": grade, "passed": True,
                        "frame": fov(fake_camera(i + 1)), "face_count": 2,
                        "comment": "Bagus! Tapi muka kamu masih nahan-nahan. KELUARIN semua!"})
    sheet = np.full((1672, 941, 3), 230, np.uint8)
    template = os.path.join("assets", "photo_strip_template.png")
    if os.path.exists(template):
        sheet = cv2.imread(template)

    shots = {
        "01-idle": lambda: ui.render_idle(cam, t, 128),
        "02-calibration": lambda: ui.render_calibration(cam, t, [(500, 260, 200, 200), (760, 270, 190, 190)], 2),
        "03-preparation": lambda: ui.render_preparation(crop, t, 3.4, boxes, 2),
        "04-countdown": lambda: ui.render_countdown(crop, t, 2.6, 2, COURSES, boxes, 2),
        "05-playing": lambda: ui.render_playing(crop, t, 4.2, 5.0, 2, COURSES, 68, 74, boxes),
        "05b-playing-allout": lambda: ui.render_playing(crop, t, 1.0, 5.0, 2, COURSES, 81, 81, boxes),
        "06-result": lambda: ui.render_result(t, 2.4, 2, 7, results[1], "angry", False),
        "07-transcript": lambda: ui.render_transcript(results, 3.14, "Sangat memuaskan.", results[0]["frame"],
                                                      [(f"EK{1101 + i}", "-26") for i in range(7)]),
        "08-selection": lambda: ui.render_selection(t, results, {0, 1, 2, 3}, ""),
        "08b-selection-full": lambda: ui.render_selection(t, results, {0, 1, 2, 3, 5, 6},
                                                          "Udah 6, batalin satu dulu kalau mau ganti."),
        "09-preview": lambda: ui.render_preview(t, sheet, results[2]["frame"], "Pembelajaran Mesin untuk Sedih · 3/7"),
        "10-email": lambda: ui.render_email(t, "player@example.com", True, ""),
        "10b-email-error": lambda: ui.render_email(t, "player@", False,
                                                   "Emailnya belum bener. Cek lagi, terus tekan Enter."),
        "11-done": lambda: ui.render_done(t, 3.14, "Sangat memuaskan.", True, "player@example.com"),
    }
    for name, fn in shots.items():
        start = time.perf_counter()
        img = fn()
        first = time.perf_counter() - start
        start = time.perf_counter()
        fn()
        again = time.perf_counter() - start
        cv2.imwrite(os.path.join(out, f"{name}.png"), img)
        print(f"{name:22s} first {first * 1000:6.0f} ms   cached {again * 1000:5.1f} ms")


if __name__ == "__main__":
    main()
