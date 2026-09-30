"""
EMO RUSH!!! — Emotion Challenge Game v4
=========================================
Alur:
  START MENU (start / calibrate / exit)
  PREPARATION (5 detik "Get Ready", kamera aktif)
  → COUNTDOWN (3 detik, tampilkan emosi yang harus dibuat)
  → GAME (5 detik, deteksi & simpan frame terbaik)
  → RESULT per-emosi (2 detik, skor + komentar)
  → LOOP 7 emosi
  → FINAL RESULT (total score + grade ala SIX ITB)
  → PHOTO SELECTION (choose six of seven photos)
  → PHOTO PREVIEW (save or discard the photo strip)

Install:
    pip install fer opencv-python tensorflow setuptools

Jalankan:
    python emotion_challenge_v4.py
"""

import cv2
import os
import time
import random
import threading
import re
from importlib import resources
import numpy as np
from fer.fer import FER

# ─── Konfigurasi ────────────────────────────────────────────
CHALLENGE_DURATION  = 5      # detik fase GAME
PREP_DURATION       = 5      # detik fase PREPARATION
COUNTDOWN_DURATION  = 3      # detik fase COUNTDOWN
RESULT_DURATION     = 3      # detik fase RESULT per emosi
PHOTO_FOLDER        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_photo")
PHOTO_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "photo_strip_template.png"
)
MAX_PLAYERS         = 3
FONT                = cv2.FONT_HERSHEY_SIMPLEX
DETECTION_WIDTH     = 640
# The uploaded strip's six photo openings are about 293 x 238 px. Keep the
# camera capture at that aspect ratio so the printed photos need no crop.
PHOTO_FRAME_ASPECT  = 293 / 238
DETECTION_RESULT_TIMEOUT = 1.0  # Clear stale face data if inference stops returning results.
EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+$"
)

# Urutan emosi tetap (shuffle setiap ronde baru)
CHALLENGES = [
    ("happy",    "HAPPY",    "Senyum selebar mungkin!",       "😁"),
    ("sad",      "SAD",      "Tampilkan wajah sedihmu!",       "😢"),
    ("angry",    "ANGRY",    "Tunjukkan ekspresi marahmu!",    "😡"),
    ("fear",     "FEAR",     "Tampilkan wajah ketakutan!",     "😨"),
    ("disgust",  "DISGUST",  "Buat ekspresi jijik!",           "🤢"),
    ("surprise", "SURPRISE", "Buat ekspresi terkejut!",        "😲"),
    ("neutral",  "NEUTRAL",  "Wajah datar, tanpa ekspresi!",   "😐"),
]

EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0  ),
    "fear":     (150, 0,   150),
    "happy":    (0,   210, 60 ),
    "sad":      (210, 90,  0  ),
    "surprise": (0,   210, 210),
    "neutral":  (160, 160, 160),
}

# Komentar per-emosi (bikin penasaran main lagi)
COMMENTS = {
    (95, 100): ("A+", (255, 215, 0),   [
        "DEWA EKSPRESI! Muka kamu bisa jadi template meme!",
        "Lebay tapi SEMPURNA! Oscar untuk kamu!",
        "TOTALITAS ABIS! Bahkan AI speechless liat kamu!",
    ]),
    (85,  94): ("A",  (0, 220, 120),   [
        "Hampir sempurna! Dikit lagi bisa kalahkan Anya Geraldine!",
        "Keren banget! Coba lebih 'gila' lagi di ronde berikutnya!",
        "Talent show material nih! 5 poin lagi buat A+!",
    ]),
    (75,  84): ("B+", (0, 200, 255),   [
        "Bagus! Tapi muka kamu masih nahan-nahan. KELUARIN semua!",
        "Potential terlihat! Next round buang gengsinya ya!",
        "B+ itu udah keren, tapi A+ itu LEGENDARIS. Coba lagi!",
    ]),
    (50,  74): ("B",  (255, 180, 0),   [
        "Lumayan! Tapi kayaknya masih 'jaim' nih... berani ga?",
        "Separuh jalan! Ronde berikutnya TOTALITAS atau nggak sama sekali!",
        "Wajah kamu bilang B, tapi jiwa kamu bilang A+. Buktiin!",
    ]),
    (30,  49): ("C",  (200, 120, 0),   [
        "Hmm... kamera udah siap, kamu yang belum. Gas lagi!",
        "C bukan berarti Cukup, tapi Coba lagi dengan lebih gila!",
        "Muka kamu datar banget, bahkan AI bingung. Ekspresif dikit dong!",
    ]),
    (0,   29): ("D",  (80,  80,  200), [
        "D untuk 'Dare to try again?'! Masa muka gitu aja kalah sama AI!",
        "AI hampir kasihan... hampir. Ayo buktiin kamu bisa lebih baik!",
        "Nggak apa-apa, semua legend pernah mulai dari D. Next round ALL OUT!",
    ]),
}

# Grade akhir (ala SIX ITB)
FINAL_GRADES = [
    (95, 100, "A+", (255, 215, 0),   4.00, "Summa Cum Laude Ekspresi!"),
    (85,  94, "A",  (0, 220, 120),   4.00, "Magna Cum Laude Ekspresi!"),
    (75,  84, "B+", (0, 200, 255),   3.50, "Ekspresimu di atas rata-rata!"),
    (50,  74, "B",  (255, 180, 0),   3.00, "Lumayan, tapi masih bisa lebih!"),
    (30,  49, "C",  (200, 120, 0),   2.00, "Cukup, tapi jangan puas!"),
    (0,   29, "D",  (80,  80,  200), 1.00, "Perlu banyak latihan ekspresi!"),
]


# ─── Helpers ────────────────────────────────────────────────

def get_comment(pct_int):
    for (lo, hi), (grade, color, msgs) in COMMENTS.items():
        if lo <= pct_int <= hi:
            return grade, color, random.choice(msgs)
    return "D", (80, 80, 200), "Coba lagi!"


def get_final_grade(avg_pct):
    for lo, hi, grade, color, gpa, desc in FINAL_GRADES:
        if lo <= avg_pct <= hi:
            return grade, color, gpa, desc
    return "D", (80, 80, 200), 1.00, "Perlu banyak latihan ekspresi!"


def draw_bg(img, x1, y1, x2, y2, color=(20, 20, 20), alpha=0.65):
    ov = img.copy()
    cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def draw_progress_bar(img, x, y, w, h, pct, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
    filled = int(w * min(pct, 1.0))
    if filled > 0:
        # Gradient efek sederhana (terang di ujung)
        cv2.rectangle(img, (x, y), (x + filled, y + h), color, -1)
        bright = tuple(min(255, c + 60) for c in color)
        cv2.rectangle(img, (x + filled - 4, y), (x + filled, y + h), bright, -1)
    cv2.putText(img, f"{int(pct * 100)}%",
                (x + w + 10, y + h - 2), FONT, 0.55, color, 1, cv2.LINE_AA)


def draw_timer_arc(img, cx, cy, radius, remaining, total, color):
    angle = int(360 * (remaining / total))
    cv2.circle(img, (cx, cy), radius, (50, 50, 50), 6)
    if angle > 0:
        cv2.ellipse(img, (cx, cy), (radius, radius), -90, 0, -(360 - angle), color, 6)
    cv2.putText(img, f"{remaining:.1f}",
                (cx - 24, cy + 10), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "detik",
                (cx - 20, cy + 30), FONT, 0.38, (160, 160, 160), 1, cv2.LINE_AA)


def wrap_text(img, text, x, y, max_w, font_scale, color, thickness=1, line_gap=28):
    """Render teks dengan word-wrap sederhana."""
    words = text.split()
    line  = ""
    cy    = y
    for word in words:
        test = line + (" " if line else "") + word
        (tw, _), _ = cv2.getTextSize(test, FONT, font_scale, thickness)
        if tw > max_w and line:
            cv2.putText(img, line, (x, cy), FONT, font_scale, color, thickness, cv2.LINE_AA)
            cy  += line_gap
            line = word
        else:
            line = test
    if line:
        cv2.putText(img, line, (x, cy), FONT, font_scale, color, thickness, cv2.LINE_AA)
    return cy + line_gap


# ─── State ──────────────────────────────────────────────────

class GameState:
    def __init__(self):
        self.challenges   = CHALLENGES.copy()
        random.shuffle(self.challenges)
        self.current_idx  = 0
        self.total        = len(self.challenges)

        # PHASE: start_menu | calibration | preparation | countdown | playing | result_per | final_result |
        #        photo_selection | photo_preview | email_entry | done
        self.phase        = "start_menu"
        self.phase_start  = time.time()

        # Per challenge
        self.face_results  = []
        self.face_boxes    = []
        self.exit_requested = False
        self.current_score = 0.0
        self.current_face_count = 0
        self.detection_at  = 0.0
        self.play_generation = 0
        self.best_frame    = None
        self.best_pct      = 0.0
        self.best_face_count = 0

        # One best group frame per challenge; kept in memory until save/discard.
        self.all_results   = []
        self.selected_photo_indices = set()
        self.photo_selection_message = ""
        self.photo_selection_size = (1280, 720)
        self.contact_sheet = None
        self.email_input = ""
        self.email_error = ""
        self.saved_email = None
        self.saved_folder_path = None
        self.gif_preview_index = 0
        self.gif_preview_updated = 0.0
        self.photo_saved   = False
        self.saved_photo_path = None
        self.saved_gif_path = None

    @property
    def current_challenge(self):
        if self.current_idx < self.total:
            return self.challenges[self.current_idx]
        return None

    def transition(self, phase):
        self.phase       = phase
        self.phase_start = time.time()

    def start_playing(self):
        self.best_frame  = None
        self.best_pct    = 0.0
        self.face_results = []
        self.face_boxes = []
        self.current_score = 0.0
        self.current_face_count = 0
        self.detection_at = 0.0
        self.play_generation += 1
        self.best_face_count = 0
        self.transition("playing")

    def finish_challenge(self, last_frame):
        ch = self.challenges[self.current_idx]
        frame_to_save = self.best_frame if self.best_frame is not None else last_frame.copy()

        pct_int                   = int(self.best_pct * 100)
        grade, gcolor, comment    = get_comment(pct_int)

        self.all_results.append({
            "challenge": ch[1],
            "emo_key":   ch[0],
            "pct":       self.best_pct,
            "pct_int":   pct_int,
            "frame":     frame_to_save.copy(),
            "face_count": self.best_face_count,
            "grade":     grade,
            "comment":   comment,
            "color":     gcolor,
        })

        self.current_idx += 1
        self.transition("result_per")

    def after_result_per(self):
        if self.current_idx >= self.total:
            self.transition("final_result")
        else:
            self.transition("countdown")

class LatestFrameEmotionWorker:
    """Run FER away from the UI loop, keeping only the newest pending frame."""

    def __init__(self, detector):
        self.detector = detector
        self._condition = threading.Condition()
        self._latest = None
        self._result = None
        self._result_id = 0
        self._stopping = False
        self._thread = threading.Thread(
            target=self._run, name="emotion-inference", daemon=True
        )
        self._thread.start()

    def submit(self, frame, context):
        # The pending frame is replaced each UI iteration so inference never builds a backlog.
        with self._condition:
            self._latest = (frame.copy(), context)
            self._condition.notify()

    def latest_result(self, after_id):
        with self._condition:
            if self._result is None or self._result_id <= after_id:
                return after_id, None
            return self._result_id, self._result

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join()

    def _run(self):
        while True:
            with self._condition:
                while self._latest is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                frame, context = self._latest
                self._latest = None

            try:
                height, width = frame.shape[:2]
                target_height = max(1, int(height * DETECTION_WIDTH / width))
                small_frame = cv2.resize(frame, (DETECTION_WIDTH, target_height))
                face_rectangles = self.detector.find_faces(small_frame)
                face_rectangles = list(face_rectangles) if face_rectangles is not None else []
                face_rectangles = sorted(
                    face_rectangles,
                    key=lambda box: max(0, box[2]) * max(0, box[3]),
                    reverse=True,
                )
                if len(context) > 3 and context[3] == "calibration":
                    detections = [{"box": box, "emotions": {}} for box in face_rectangles]
                else:
                    face_rectangles = face_rectangles[:MAX_PLAYERS]
                    detections = self.detector.detect_emotions(
                        small_frame, face_rectangles=face_rectangles
                    )
                if detections:
                    scale_x = width / DETECTION_WIDTH
                    scale_y = height / target_height
                    result = {
                        "context": context,
                        "faces": [
                            {
                                "emotions": face["emotions"],
                                "box": (
                                    int(face["box"][0] * scale_x),
                                    int(face["box"][1] * scale_y),
                                    int(face["box"][2] * scale_x),
                                    int(face["box"][3] * scale_y),
                                ),
                            }
                            for face in detections
                        ],
                        "frame": frame,
                        "status": "faces",
                    }
                else:
                    result = {"context": context, "status": "no_face"}
            except Exception as exc:
                print(f"[ERROR] Emotion inference failed: {type(exc).__name__}: {exc}")
                result = {"context": context, "status": "error"}

            with self._condition:
                self._result = result
                self._result_id += 1


# ─── Phase Renderers ────────────────────────────────────────

def start_menu_button_layout(width, height):
    button_w, button_h, gap = 300, 58, 18
    left = (width - button_w) // 2
    top = height // 2 - 5
    return [
        (left, top, button_w, button_h),
        (left, top + button_h + gap, button_w, button_h),
        (left, top + 2 * (button_h + gap), button_w, button_h),
    ]


def calibration_back_button_layout(width, height):
    button_w, button_h = 230, 48
    return ((width - button_w) // 2, height - button_h - 18, button_w, button_h)


def render_start_menu(img, state):
    """Show the entry menu over the live camera preview."""
    H, W = img.shape[:2]
    img[:] = cv2.addWeighted(img, 0.30, np.zeros_like(img), 0.70, 0)
    title = "EMO RUSH!!!"
    (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_TRIPLEX, 1.8, 3)
    cv2.putText(img, title, ((W - title_w) // 2, H // 2 - 88),
                cv2.FONT_HERSHEY_TRIPLEX, 1.8, (255, 225, 40), 3, cv2.LINE_AA)
    subtitle = "EMOTION CHALLENGE"
    (subtitle_w, _), _ = cv2.getTextSize(subtitle, FONT, 0.62, 1)
    cv2.putText(img, subtitle, ((W - subtitle_w) // 2, H // 2 - 52),
                FONT, 0.62, (245, 245, 245), 1, cv2.LINE_AA)

    labels = (("START GAME", (45, 145, 80)),
              ("CALIBRATE CAMERA", (165, 115, 35)),
              ("EXIT", (80, 80, 170)))
    for rect, (label, color) in zip(start_menu_button_layout(W, H), labels):
        x, y, width, height = rect
        cv2.rectangle(img, (x, y), (x + width, y + height), color, -1)
        (label_w, _), _ = cv2.getTextSize(label, FONT, 0.65, 2)
        cv2.putText(img, label, (x + (width - label_w) // 2, y + 38),
                    FONT, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "F = Toggle Fullscreen    Q = Exit", (W // 2 - 142, H - 12), FONT, 0.42,
                (220, 220, 220), 1, cv2.LINE_AA)


def render_calibration(img, state):
    """Show live face boxes, current face count, and a return-to-menu button."""
    H, W = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (W, 90), (22, 29, 42), -1)
    cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
    cv2.putText(img, "CAMERA CALIBRATION", (25, 38), FONT, 0.78,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"FACES DETECTED: {state.current_face_count}", (25, 72),
                FONT, 0.62, (80, 235, 130), 2, cv2.LINE_AA)
    cv2.putText(img, "Adjust camera position and lighting. Press ESC to return.",
                (W - 460, 38), FONT, 0.48, (230, 230, 235), 1, cv2.LINE_AA)
    bx, by, bw, bh = calibration_back_button_layout(W, H)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (58, 66, 82), -1)
    cv2.putText(img, "BACK TO MENU", (bx + 42, by + 31), FONT, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)


def render_preparation(img, state):
    """5 detik GET READY — kamera aktif, pemain posisikan wajah."""
    H, W    = img.shape[:2]
    elapsed = time.time() - state.phase_start
    remain  = max(0.0, PREP_DURATION - elapsed)

    # Overlay semi-transparan
    draw_bg(img, 0, 0, W, H, alpha=0.45)

    # Judul game
    title = "EMO RUSH!!!"
    (tw, th), _ = cv2.getTextSize(title, FONT, 2.2, 4)
    cv2.putText(img, title, ((W - tw) // 2, H // 2 - 130),
                FONT, 2.2, (255, 215, 0), 4, cv2.LINE_AA)

    # GET READY
    cv2.putText(img, "GET READY",
                (W // 2 - 130, H // 2 - 55), FONT, 1.4, (0, 220, 255), 3, cv2.LINE_AA)

    # Countdown bulat besar
    num_str = str(int(remain) + 1) if remain > 0 else "GO!"
    (nw, nh), _ = cv2.getTextSize(num_str, FONT, 3.5, 6)
    cv2.putText(img, num_str, ((W - nw) // 2, H // 2 + 60),
                FONT, 3.5, (255, 220, 0), 6, cv2.LINE_AA)

    # Instruksi
    cv2.putText(img, "Posisikan wajah kamu di depan kamera!",
                (W // 2 - 215, H // 2 + 110), FONT, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(img, "Pastikan pencahayaan cukup dan wajah terlihat jelas",
                (W // 2 - 245, H // 2 + 140), FONT, 0.5, (130, 130, 130), 1, cv2.LINE_AA)

    # Progress bar bawah
    bar_w = W - 80
    bar_x = 40
    bar_y = H - 40
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + 12), (50, 50, 50), -1)
    filled = int(bar_w * (1 - remain / PREP_DURATION))
    cv2.rectangle(img, (bar_x, bar_y), (bar_x + filled, bar_y + 12), (0, 220, 255), -1)

    if remain <= 0:
        state.transition("countdown")


def render_countdown(img, state):
    """3 detik nama emosi + instruksi."""
    H, W    = img.shape[:2]
    elapsed = time.time() - state.phase_start
    remain  = max(0.0, COUNTDOWN_DURATION - elapsed)
    ch      = state.current_challenge

    draw_bg(img, 0, 0, W, H, alpha=0.60)

    if ch:
        color = EMOTION_COLORS.get(ch[0], (200, 200, 200))

        # Challenge counter
        cv2.putText(img, f"Challenge  {state.current_idx + 1}  /  {state.total}",
                    (W // 2 - 120, H // 2 - 155), FONT, 0.65, (160, 160, 160), 1, cv2.LINE_AA)

        # Nama emosi besar
        (ew, _), _ = cv2.getTextSize(ch[1], FONT, 2.0, 4)
        cv2.putText(img, ch[1], ((W - ew) // 2, H // 2 - 70),
                    FONT, 2.0, color, 4, cv2.LINE_AA)

        # Garis bawah dekorasi
        lx = (W - ew) // 2
        cv2.line(img, (lx, H // 2 - 55), (lx + ew, H // 2 - 55), color, 2)

        # Instruksi
        (iw, _), _ = cv2.getTextSize(ch[2], FONT, 0.7, 1)
        cv2.putText(img, ch[2], ((W - iw) // 2, H // 2 - 15),
                    FONT, 0.7, (210, 210, 210), 1, cv2.LINE_AA)

    # Angka countdown
    num = max(1, int(remain) + 1)
    (nw, _), _ = cv2.getTextSize(str(num), FONT, 4.5, 7)
    cv2.putText(img, str(num), ((W - nw) // 2, H // 2 + 90),
                FONT, 4.5, (255, 220, 0), 7, cv2.LINE_AA)
    cv2.putText(img, "BERSIAP...", (W // 2 - 80, H // 2 + 145),
                FONT, 0.7, (180, 180, 180), 1, cv2.LINE_AA)

    if remain <= 0:
        state.start_playing()


def render_playing(img, state, camera_frame=None):
    """5 detik pemain berekspresi."""
    H, W      = img.shape[:2]
    elapsed   = time.time() - state.phase_start
    remaining = max(0.0, CHALLENGE_DURATION - elapsed)
    ch        = state.current_challenge
    color     = EMOTION_COLORS.get(ch[0], (200, 200, 200))
    pct       = state.current_score

    # ── Top bar ──
    draw_bg(img, 0, 0, W, 110)
    cv2.putText(img, f"Challenge {state.current_idx + 1}/{state.total}  —  {ch[1]}",
                (15, 38), FONT, 0.9, color, 2, cv2.LINE_AA)
    cv2.putText(img, ch[2], (15, 72), FONT, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    draw_timer_arc(img, W - 75, 58, 44, remaining, CHALLENGE_DURATION, color)

    # Best score
    cv2.putText(img, f"Best: {int(state.best_pct * 100)}%",
                (W - 165, 120), FONT, 0.55, (255, 220, 0), 1, cv2.LINE_AA)

    # ── Pesan motivasi saat detik terakhir ──
    if remaining < 1.5:
        msg = "ALL OUT! EKSPRESI TOTALITAS!"
        (mw, _), _ = cv2.getTextSize(msg, FONT, 0.8, 2)
        draw_bg(img, (W - mw) // 2 - 10, H // 2 - 30,
                (W + mw) // 2 + 10, H // 2 + 10, alpha=0.7)
        cv2.putText(img, msg, ((W - mw) // 2, H // 2 + 5),
                    FONT, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

    # ── Bottom bar ──
    draw_bg(img, 0, H - 80, W, H)
    cv2.putText(img, f"{ch[1]} group average:",
                (15, H - 55), FONT, 0.48, (160, 160, 160), 1, cv2.LINE_AA)
    draw_progress_bar(img, 15, H - 45, W - 130, 22, pct, color)
    cv2.putText(img, "Foto terbaik tersimpan otomatis",
                (15, H - 12), FONT, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

    if remaining <= 0:
        state.finish_challenge(camera_frame if camera_frame is not None else img)


def render_result_per(img, state):
    """Tampilkan skor + komentar per emosi selama RESULT_DURATION detik."""
    H, W    = img.shape[:2]
    elapsed = time.time() - state.phase_start
    remain  = max(0.0, RESULT_DURATION - elapsed)

    if not state.all_results:
        return

    res   = state.all_results[-1]
    color = res["color"]

    # Background penuh
    overlay = np.zeros((H, W, 3), dtype=np.uint8)
    overlay[:] = (15, 15, 15)

    # Warna aksen samping
    cv2.rectangle(overlay, (0, 0), (8, H), color, -1)
    cv2.rectangle(overlay, (W - 8, 0), (W, H), color, -1)

    # Header
    cv2.putText(overlay, f"Challenge {state.current_idx} / {state.total}  SELESAI!",
                (W // 2 - 190, 55), FONT, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

    # Nama emosi
    (ew, _), _ = cv2.getTextSize(res["challenge"], FONT, 2.0, 4)
    cv2.putText(overlay, res["challenge"], ((W - ew) // 2, 120),
                FONT, 2.0, color, 4, cv2.LINE_AA)

    # Score besar
    score_str = f"{res['pct_int']}%"
    (sw, _), _ = cv2.getTextSize(score_str, FONT, 3.5, 6)
    cv2.putText(overlay, score_str, ((W - sw) // 2, 210),
                FONT, 3.5, (255, 255, 255), 6, cv2.LINE_AA)

    # Grade badge
    gw, gh = 120, 60
    gx = (W - gw) // 2
    gy = 230
    cv2.rectangle(overlay, (gx, gy), (gx + gw, gy + gh), color, -1)
    (lw, lh), _ = cv2.getTextSize(res["grade"], FONT, 1.4, 3)
    cv2.putText(overlay, res["grade"],
                (gx + (gw - lw) // 2, gy + (gh + lh) // 2),
                FONT, 1.4, (0, 0, 0), 3, cv2.LINE_AA)

    # Komentar
    wrap_text(overlay, res["comment"],
              W // 2 - 280, 330, 560, 0.65, (210, 210, 210), 1, 30)

    # Progress bar emosi (visual recap)
    bar_y = H - 110
    draw_bg(overlay, 40, bar_y - 10, W - 40, bar_y + 35, alpha=0.0)
    cv2.putText(overlay, "SCORE:", (50, bar_y + 18),
                FONT, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
    draw_progress_bar(overlay, 130, bar_y, W - 220, 24, res["pct"], color)

    # Countdown lanjut
    cv2.putText(overlay, f"Lanjut dalam  {remain:.1f}s ...",
                (W // 2 - 110, H - 30), FONT, 0.55, (100, 100, 100), 1, cv2.LINE_AA)

    img[:] = overlay[:]

    if remain <= 0:
        state.after_result_per()


def render_final_result(img, state):
    """Tampilan nilai akhir ala SIX ITB."""
    H, W = img.shape[:2]

    avg     = sum(r["pct"] for r in state.all_results) / len(state.all_results) if state.all_results else 0
    avg_pct = int(avg * 100)
    grade, gcolor, gpa, desc = get_final_grade(avg_pct)

    # Layar putih/krem ala SIX ITB
    bg = np.ones((H, W, 3), dtype=np.uint8) * 245
    bg[:60, :] = (12, 45, 100)   # header biru ITB

    # ── Header ──
    cv2.putText(bg, "SISTEM INFORMASI EKSPRESI — ITB",
                (W // 2 - 230, 38), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

    # ── Title ──
    cv2.putText(bg, "TRANSKRIP NILAI EMOSI", (W // 2 - 175, 95),
                FONT, 0.85, (12, 45, 100), 2, cv2.LINE_AA)
    cv2.line(bg, (60, 105), (W - 60, 105), (12, 45, 100), 1)

    # ── Tabel nilai per emosi ──
    row_h  = 38
    tbl_x  = 80
    tbl_y  = 125
    headers = ["No", "MATA KULIAH EKSPRESI", "SKS", "NILAI", "%"]
    col_x   = [tbl_x, tbl_x + 45, tbl_x + 330, tbl_x + 385, tbl_x + 445]

    # Header tabel
    cv2.rectangle(bg, (tbl_x - 5, tbl_y - 22),
                  (W - tbl_x + 5, tbl_y + 3), (12, 45, 100), -1)
    for i, h in enumerate(headers):
        cv2.putText(bg, h, (col_x[i], tbl_y),
                    FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Baris per emosi
    for i, r in enumerate(state.all_results):
        ry = tbl_y + (i + 1) * row_h
        row_color = (230, 235, 245) if i % 2 == 0 else (245, 245, 245)
        cv2.rectangle(bg, (tbl_x - 5, ry - 22),
                      (W - tbl_x + 5, ry + 8), row_color, -1)

        ecol = EMOTION_COLORS.get(r["emo_key"], (0, 0, 0))
        cv2.putText(bg, str(i + 1),             (col_x[0], ry), FONT, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(bg, r["challenge"],         (col_x[1], ry), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
        cv2.putText(bg, "3",                    (col_x[2], ry), FONT, 0.5, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(bg, r["grade"],             (col_x[3], ry), FONT, 0.6, ecol,          1, cv2.LINE_AA)
        cv2.putText(bg, f"{r['pct_int']}%",    (col_x[4], ry), FONT, 0.5, (60, 60, 60),  1, cv2.LINE_AA)

    # ── Garis footer tabel ──
    fy = tbl_y + (len(state.all_results) + 1) * row_h - 15
    cv2.line(bg, (tbl_x - 5, fy), (W - tbl_x + 5, fy), (12, 45, 100), 2)

    # ── IPK / Rata-rata ──
    ry2 = fy + 35
    cv2.putText(bg, f"Indeks Prestasi Ekspresi (IPE): {gpa:.2f}",
                (tbl_x, ry2), FONT, 0.65, (12, 45, 100), 2, cv2.LINE_AA)
    cv2.putText(bg, f"Grade Akhir: {grade}   ({avg_pct}%)",
                (tbl_x, ry2 + 35), FONT, 0.7, gcolor, 2, cv2.LINE_AA)
    cv2.putText(bg, desc,
                (tbl_x, ry2 + 65), FONT, 0.55, (60, 60, 60), 1, cv2.LINE_AA)

    # ── Tombol lanjut ──
    btn_y = H - 60
    cv2.rectangle(bg, (W // 2 - 160, btn_y - 30), (W // 2 + 160, btn_y + 10),
                  (12, 45, 100), -1)
    cv2.putText(bg, "Tekan  SPASI  untuk pilih 6 foto",
                (W // 2 - 155, btn_y - 5), FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

    img[:] = bg[:]


def cover_resize(image, target_w, target_h):
    """Resize and center-crop an image to completely fill a photo tile."""
    height, width = image.shape[:2]
    scale = max(target_w / width, target_h / height)
    resized = cv2.resize(image, (int(width * scale), int(height * scale)))
    top = (resized.shape[0] - target_h) // 2
    left = (resized.shape[1] - target_w) // 2
    return resized[top:top + target_h, left:left + target_w]


def fit_resize(image, target_w, target_h, background=(28, 30, 36)):
    """Fit an image inside a preview box without cutting off any of its FOV."""
    height, width = image.shape[:2]
    scale = min(target_w / width, target_h / height)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), background, dtype=np.uint8)
    x = (target_w - resized_w) // 2
    y = (target_h - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized
    return canvas


def photo_fov(frame):
    """Center-crop to the template opening aspect before detection and saving."""
    height, width = frame.shape[:2]
    target_width = min(width, int(round(height * PHOTO_FRAME_ASPECT)))
    left = (width - target_width) // 2
    return frame[:, left:left + target_width].copy(), left


def draw_photo_fov_guide(img):
    """Shade the part of the camera image that will not appear in the strip."""
    height, width = img.shape[:2]
    target_width = min(width, int(round(height * PHOTO_FRAME_ASPECT)))
    left = (width - target_width) // 2
    right = left + target_width
    shade = img.copy()
    shade[:, :left] = (0, 0, 0)
    shade[:, right:] = (0, 0, 0)
    cv2.addWeighted(shade, 0.58, img, 0.42, 0, img)
    cv2.rectangle(img, (left, 0), (right - 1, height - 1), (255, 255, 255), 2)
    cv2.putText(img, "PHOTO AREA", (left + 12, height - 14), FONT, 0.48,
                (255, 255, 255), 1, cv2.LINE_AA)
    return left


def create_contact_sheet(results):
    """Place six selected photos into the supplied six-frame PNG template."""
    if len(results) != 6:
        raise ValueError(f"The photo template needs exactly 6 selected photos; got {len(results)}")

    page = cv2.imread(PHOTO_TEMPLATE_PATH, cv2.IMREAD_COLOR)
    if page is None:
        raise FileNotFoundError(f"Photo strip template could not be opened: {PHOTO_TEMPLATE_PATH}")
    if page.shape[:2] != (1672, 941):
        raise ValueError(f"Expected a 941x1672 photo template; got {page.shape[1]}x{page.shape[0]}")

    # Interior rectangles stay inside the original white frame borders.
    slots = [
        (95, 320, 293, 238), (95, 596, 293, 238), (93, 873, 295, 238),
        (563, 553, 284, 238), (563, 823, 284, 238), (562, 1101, 287, 238),
    ]
    for result, (x, y, width, height) in zip(results, slots):
        # Captured frames already match the template opening ratio; resize only
        # removes pixel rounding differences between the raster frames.
        photo = cv2.resize(result["frame"], (width, height), interpolation=cv2.INTER_AREA)
        radius = 15
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (radius, 0), (width - radius - 1, height - 1), 255, -1)
        cv2.rectangle(mask, (0, radius), (width - 1, height - radius - 1), 255, -1)
        for cx, cy in ((radius, radius), (width - radius - 1, radius),
                       (radius, height - radius - 1), (width - radius - 1, height - radius - 1)):
            cv2.circle(mask, (cx, cy), radius, 255, -1, cv2.LINE_AA)
        roi = page[y:y + height, x:x + width]
        roi[mask > 0] = photo[mask > 0]
    return page


def photo_selection_layout(width, height):
    """Return the seven thumbnail hit boxes and the Continue button rectangle."""
    margin, gap, columns = 30, 18, 4
    card_w = (width - 2 * margin - gap * (columns - 1)) // columns
    card_h, top = 220, 115
    row_gap = 14
    cards = []
    for index in range(7):
        row, col = divmod(index, columns)
        x = margin + col * (card_w + gap)
        y = top + row * (card_h + row_gap)
        cards.append((x, y, card_w, card_h))
    button_w, button_h = 240, 54
    button = ((width - button_w) // 2, height - button_h - 18, button_w, button_h)
    return cards, button


def render_photo_selection(img, state):
    """Show seven clickable thumbnails; the player must select exactly six."""
    H, W = img.shape[:2]
    img[:] = cv2.addWeighted(img, 0.28, np.zeros_like(img), 0.72, 0)
    heading = "CHOOSE 6 PHOTOS FOR YOUR TEMPLATE"
    (tw, _), _ = cv2.getTextSize(heading, FONT, 0.9, 2)
    cv2.putText(img, heading, ((W - tw) // 2, 44), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"{len(state.selected_photo_indices)} / 6 selected — click a photo to toggle it",
                (W // 2 - 220, 82), FONT, 0.56, (225, 230, 240), 1, cv2.LINE_AA)

    cards, button = photo_selection_layout(W, H)
    for index, (result, (x, y, card_w, card_h)) in enumerate(zip(state.all_results, cards)):
        selected = index in state.selected_photo_indices
        fill = (52, 68, 62) if selected else (42, 45, 54)
        border = (70, 220, 130) if selected else (150, 155, 165)
        cv2.rectangle(img, (x, y), (x + card_w, y + card_h), fill, -1)
        cv2.rectangle(img, (x, y), (x + card_w, y + card_h), border, 3 if selected else 1)
        thumb_w, thumb_h = card_w - 20, 160
        thumb = fit_resize(result["frame"], thumb_w, thumb_h, fill)
        img[y + 10:y + 10 + thumb_h, x + 10:x + 10 + thumb_w] = thumb
        marker = "X" if selected else "+"
        color = (70, 220, 130) if selected else (235, 235, 235)
        cv2.putText(img, marker, (x + card_w - 32, y + 31), FONT, 0.8, color, 2, cv2.LINE_AA)
        label = f"{index + 1}. {result['challenge']}  {result['pct_int']}%"
        cv2.putText(img, label, (x + 10, y + 181), FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    bx, by, bw, bh = button
    enabled = len(state.selected_photo_indices) == 6
    button_color = (55, 145, 85) if enabled else (90, 94, 103)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), button_color, -1)
    label = "CONTINUE" if enabled else "SELECT SIX PHOTOS"
    (label_w, _), _ = cv2.getTextSize(label, FONT, 0.62, 2)
    cv2.putText(img, label, (bx + (bw - label_w) // 2, by + 35),
                FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if state.photo_selection_message:
        (message_w, _), _ = cv2.getTextSize(state.photo_selection_message, FONT, 0.46, 1)
        cv2.putText(img, state.photo_selection_message, ((W - message_w) // 2, by - 9),
                    FONT, 0.46, (180, 220, 255), 1, cv2.LINE_AA)


def photo_preview_button_layout(width, height):
    button_w, button_h, gap = 240, 52, 28
    total_w = button_w * 2 + gap
    left = (width - total_w) // 2
    y = height - button_h - 16
    return (left, y, button_w, button_h), (left + button_w + gap, y, button_w, button_h)


def email_form_layout(width, height):
    field = (width // 2 - 360, height // 2 - 45, 720, 58)
    button = (width // 2 - 140, height // 2 + 82, 280, 54)
    return field, button


def submit_email_entry(state):
    """Validate the email-shaped folder name, then save the local photo set."""
    email = state.email_input.strip()
    if not EMAIL_PATTERN.fullmatch(email) or len(email) > 254:
        state.email_error = "Enter a valid email address, then press Enter."
        return False
    if save_photo_session(state, email):
        state.saved_email = email
        state.transition("done")
        return True
    state.email_error = "Could not save the files. Check the console, then retry."
    return False


def handle_photo_selection_mouse(event, x, y, flags, state):
    """Handle thumbnail selection, preview approval, and local save clicks."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    width, height = state.photo_selection_size
    if state.phase == "start_menu":
        start_rects = start_menu_button_layout(width, height)
        if _inside_button(x, y, start_rects[0]):
            state.transition("preparation")
        elif _inside_button(x, y, start_rects[1]):
            state.face_boxes = []
            state.current_face_count = 0
            state.detection_at = 0.0
            state.transition("calibration")
        elif _inside_button(x, y, start_rects[2]):
            state.exit_requested = True
        return

    if state.phase == "calibration":
        if _inside_button(x, y, calibration_back_button_layout(width, height)):
            state.face_boxes = []
            state.current_face_count = 0
            state.transition("start_menu")
        return

    if state.phase == "photo_selection":
        cards, button = photo_selection_layout(width, height)
        for index, (left, top, card_w, card_h) in enumerate(cards):
            if left <= x < left + card_w and top <= y < top + card_h:
                state.photo_selection_message = ""
                if index in state.selected_photo_indices:
                    state.selected_photo_indices.remove(index)
                elif len(state.selected_photo_indices) < 6:
                    state.selected_photo_indices.add(index)
                else:
                    state.photo_selection_message = "Six are selected already. Deselect one first."
                return

        bx, by, bw, bh = button
        if bx <= x < bx + bw and by <= y < by + bh:
            if len(state.selected_photo_indices) != 6:
                state.photo_selection_message = "Select exactly six photos to continue."
                return
            selected_results = [
                state.all_results[i] for i in range(len(state.all_results))
                if i in state.selected_photo_indices
            ]
            try:
                state.contact_sheet = create_contact_sheet(selected_results)
            except (FileNotFoundError, ValueError, cv2.error) as exc:
                state.photo_selection_message = f"Could not create template preview: {exc}"
                return
            state.photo_selection_message = ""
            state.gif_preview_index = 0
            state.gif_preview_updated = time.monotonic()
            state.transition("photo_preview")
            return

    if state.phase == "photo_preview":
        approve_button, change_button = photo_preview_button_layout(width, height)
        ax, ay, aw, ah = approve_button
        if ax <= x < ax + aw and ay <= y < ay + ah:
            state.email_input = ""
            state.email_error = ""
            state.transition("email_entry")
            return
        cx, cy, cw, ch = change_button
        if cx <= x < cx + cw and cy <= y < cy + ch:
            state.transition("photo_selection")
            return

    if state.phase == "email_entry":
        _, (bx, by, bw, bh) = email_form_layout(width, height)
        if bx <= x < bx + bw and by <= y < by + bh:
            submit_email_entry(state)


def _inside_button(x, y, rect):
    left, top, width, height = rect
    return left <= x < left + width and top <= y < top + height


def save_photo_session(state, email):
    """Save only the strip PNG and looping GIF in a folder named for the email."""
    normalized_email = email.strip().lower()
    if not EMAIL_PATTERN.fullmatch(normalized_email) or len(normalized_email) > 254:
        print("[ERROR] Invalid email folder name.")
        return False
    user_folder = os.path.join(PHOTO_FOLDER, normalized_email)
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{time.time_ns() % 1_000_000:06d}"
    written = []
    try:
        os.makedirs(user_folder, exist_ok=True)
        sheet_path = os.path.join(user_folder, f"{stamp}_EMO_RUSH_PHOTO_STRIP.png")
        if not cv2.imwrite(sheet_path, state.contact_sheet):
            raise OSError(f"Could not write photo strip: {sheet_path}")
        written.append(sheet_path)

        from PIL import Image

        gif_frames = []
        for result in state.all_results:
            frame = result["frame"]
            height, width = frame.shape[:2]
            gif_width = min(720, width)
            gif_height = max(1, round(height * gif_width / width))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(
                Image.fromarray(rgb).resize(
                    (gif_width, gif_height), Image.Resampling.LANCZOS
                )
            )

        gif_path = os.path.join(user_folder, f"{stamp}_EMO_RUSH_LOOP.gif")
        written.append(gif_path)
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=650,
            loop=0,
            optimize=True,
            disposal=2,
        )
    except Exception as exc:
        for path in written:
            try:
                os.remove(path)
            except OSError:
                pass
        print(f"[ERROR] Could not save the photo set: {exc}")
        return False

    state.photo_saved = True
    state.saved_photo_path = sheet_path
    state.saved_gif_path = gif_path
    state.saved_folder_path = user_folder
    print(
        f"[SAVED] Photo strip and looping GIF -> {user_folder}/"
    )
    discard_photo_images(state)
    return True


def discard_photo_images(state):
    """Release in-memory photo data after a save or discard decision."""
    state.contact_sheet = None
    for result in state.all_results:
        result.pop("frame", None)


def render_photo_preview(img, state):
    """Show the strip beside the looping GIF and offer approve or change choices."""
    H, W = img.shape[:2]
    img[:] = (24, 27, 34)
    title = "REVIEW YOUR PHOTO SET"
    (title_w, _), _ = cv2.getTextSize(title, FONT, 0.8, 2)
    cv2.putText(img, title, ((W - title_w) // 2, 34), FONT, 0.8, (245, 245, 245), 2, cv2.LINE_AA)

    cv2.putText(img, "PHOTO STRIP", (175, 72), FONT, 0.55, (215, 220, 230), 1, cv2.LINE_AA)
    if state.contact_sheet is not None:
        sheet_h, sheet_w = state.contact_sheet.shape[:2]
        scale = min(450 / sheet_w, (H - 170) / sheet_h)
        preview_w, preview_h = int(sheet_w * scale), int(sheet_h * scale)
        preview = cv2.resize(state.contact_sheet, (preview_w, preview_h))
        ox, oy = 45, 92
        img[oy:oy + preview_h, ox:ox + preview_w] = preview

    cv2.putText(img, "LOOPING GIF PREVIEW", (W - 590, 115),
                FONT, 0.55, (215, 220, 230), 1, cv2.LINE_AA)
    if state.all_results:
        now = time.monotonic()
        if now - state.gif_preview_updated >= 0.65:
            state.gif_preview_index = (state.gif_preview_index + 1) % len(state.all_results)
            state.gif_preview_updated = now
        current = state.all_results[state.gif_preview_index]
        gif_w, gif_h = 650, 366
        gif_frame = fit_resize(current["frame"], gif_w, gif_h)
        gx = W - gif_w - 35
        gy = 195
        cv2.rectangle(img, (gx - 3, gy - 3), (gx + gif_w + 3, gy + gif_h + 3), (245, 245, 245), 2)
        img[gy:gy + gif_h, gx:gx + gif_w] = gif_frame
        caption = f"{current['challenge']}  {current['pct_int']}%   ({state.gif_preview_index + 1}/{len(state.all_results)})"
        (caption_w, _), _ = cv2.getTextSize(caption, FONT, 0.55, 1)
        cv2.putText(img, caption, (gx + (gif_w - caption_w) // 2, gy + gif_h + 28),
                    FONT, 0.55, (230, 230, 235), 1, cv2.LINE_AA)

    approve, change = photo_preview_button_layout(W, H)
    ax, ay, aw, ah = approve
    cx, cy, cw, ch = change
    cv2.rectangle(img, (ax, ay), (ax + aw, ay + ah), (40, 145, 83), -1)
    cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), (95, 93, 104), -1)
    cv2.putText(img, "APPROVE", (ax + 70, ay + 34), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "CHANGE PHOTOS", (cx + 28, cy + 34), FONT, 0.58, (255, 255, 255), 2, cv2.LINE_AA)


def render_email_entry(img, state):
    """Collect an email-shaped local folder name; no email is sent."""
    H, W = img.shape[:2]
    img[:] = cv2.addWeighted(img, 0.2, np.zeros_like(img), 0.8, 0)
    title = "SAVE YOUR PHOTO SET"
    (title_w, _), _ = cv2.getTextSize(title, FONT, 0.95, 2)
    cv2.putText(img, title, ((W - title_w) // 2, H // 2 - 120),
                FONT, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
    note = "Enter your email address. It will be used as the local folder name; nothing will be emailed."
    (note_w, _), _ = cv2.getTextSize(note, FONT, 0.46, 1)
    cv2.putText(img, note, ((W - note_w) // 2, H // 2 - 78),
                FONT, 0.46, (220, 225, 235), 1, cv2.LINE_AA)

    field, button = email_form_layout(W, H)
    fx, fy, fw, fh = field
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (245, 245, 245), -1)
    cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (115, 175, 240), 2)
    if state.email_input:
        typed = state.email_input[-58:]
        if len(state.email_input) > 58:
            typed = "..." + typed
        if int(time.monotonic() * 2) % 2 == 0:
            typed += "_"
        text_color = (25, 35, 50)
    else:
        typed, text_color = "name@example.com", (130, 135, 145)
    cv2.putText(img, typed, (fx + 16, fy + 38), FONT, 0.7, text_color, 1, cv2.LINE_AA)

    bx, by, bw, bh = button
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (45, 145, 85), -1)
    cv2.putText(img, "SAVE FILES", (bx + 73, by + 34), FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if state.email_error:
        (error_w, _), _ = cv2.getTextSize(state.email_error, FONT, 0.48, 1)
        cv2.putText(img, state.email_error, ((W - error_w) // 2, by - 14),
                    FONT, 0.48, (100, 170, 255), 1, cv2.LINE_AA)
    cv2.putText(img, "Press Enter to save  |  Esc to return to preview",
                (W // 2 - 190, by + bh + 38), FONT, 0.48, (210, 215, 225), 1, cv2.LINE_AA)


def render_done(img, state):
    """Layar akhir setelah semua foto dipilih."""
    H, W = img.shape[:2]
    draw_bg(img, 0, 0, W, H, alpha=0.88)

    cv2.putText(img, "TERIMA KASIH SUDAH MAIN!",
                (W // 2 - 230, H // 2 - 80), FONT, 1.0, (255, 215, 0), 2, cv2.LINE_AA)

    avg     = sum(r["pct"] for r in state.all_results) / len(state.all_results) if state.all_results else 0
    avg_pct = int(avg * 100)
    grade, gcolor, _, _ = get_final_grade(avg_pct)

    cv2.putText(img, f"Final Score: {avg_pct}%   Grade: {grade}",
                (W // 2 - 200, H // 2 - 20), FONT, 0.9, gcolor, 2, cv2.LINE_AA)

    if state.photo_saved:
        status = f"Saved locally for {state.saved_email} (no email sent)"
        status_color = (0, 180, 80)
    else:
        status = "Photos discarded"
        status_color = (140, 140, 140)
    (status_w, _), _ = cv2.getTextSize(status, FONT, 0.55, 1)
    status_scale = min(0.55, (W - 40) / max(1, status_w) * 0.55)
    (status_w, _), _ = cv2.getTextSize(status, FONT, status_scale, 1)
    cv2.putText(img, status, ((W - status_w) // 2, H // 2 + 45),
                FONT, status_scale, status_color, 1, cv2.LINE_AA)
    cv2.putText(img, "R = Main Lagi    |    Q = Menu Awal",
                (W // 2 - 165, H // 2 + 130), FONT, 0.65, (140, 140, 140), 1, cv2.LINE_AA)


# ─── Main ───────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  EMO RUSH!!! — Emotion Challenge Game v4")
    print("  Alur: START MENU -> PREP -> COUNTDOWN -> GAME -> FINAL -> PHOTO SELECTION")
    print("  Q = keluar  |  R = main lagi (di layar done)")
    print("=" * 60)

    try:
        cascade_file = resources.files("fer").joinpath(
            "data", "haarcascade_frontalface_default.xml"
        )
        cascade_path = str(cascade_file)
        if not cascade_file.is_file() or cv2.CascadeClassifier(cascade_path).empty():
            raise FileNotFoundError(
                f"FER face cascade is missing or invalid: {cascade_path}"
            )
        detector = FER(cascade_file=cascade_path, mtcnn=False)
    except Exception as exc:
        print(f"[ERROR] Emotion model could not be initialized: {type(exc).__name__}: {exc}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam tidak bisa diakses!")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    worker = LatestFrameEmotionWorker(detector)
    state = GameState()
    game_generation = 0
    last_result_id = 0
    window_name = "EMO RUSH!!!"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    fullscreen = True
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("EMO RUSH!!!", handle_photo_selection_mouse, state)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam stopped returning frames.")
                break

            display = frame.copy()
            photo_left = 0
            photo_frame, photo_left = photo_fov(frame)

            state.photo_selection_size = (display.shape[1], display.shape[0])

            if state.phase in ("playing", "calibration"):
                context = (game_generation, state.play_generation, state.current_idx, state.phase)
                worker.submit(photo_frame if state.phase == "playing" else frame, context)

                last_result_id, inference = worker.latest_result(last_result_id)
                if inference is not None and inference["context"] == context:
                    if inference["status"] == "faces":
                        state.face_results = inference["faces"]
                        state.face_boxes = [face["box"] for face in state.face_results]
                        state.current_face_count = len(state.face_results)
                        state.detection_at = time.monotonic()
                        if state.phase == "playing":
                            ch = state.current_challenge
                            scores = [
                                float(face["emotions"].get(ch[0], 0.0))
                                for face in state.face_results
                            ]
                            state.current_score = sum(scores) / len(scores)
                            if state.current_score > state.best_pct:
                                state.best_pct = state.current_score
                                state.best_frame = inference["frame"]
                                state.best_face_count = state.current_face_count
                    elif state.phase == "calibration":
                        state.face_results = []
                        state.face_boxes = []
                        state.current_face_count = 0
                        state.detection_at = 0.0

                # Brief misses keep the last result; sustained misses expire after the timeout.
                if (state.face_boxes and state.detection_at
                        and time.monotonic() - state.detection_at > DETECTION_RESULT_TIMEOUT):
                    state.face_results = []
                    state.face_boxes = []
                    state.current_score = 0.0
                    state.current_face_count = 0

            # ── Render per phase ──
            if state.phase in ("preparation", "countdown", "playing"):
                draw_photo_fov_guide(display)
            if   state.phase == "start_menu":  render_start_menu(display, state)
            elif state.phase == "calibration": render_calibration(display, state)
            elif state.phase == "preparation": render_preparation(display, state)
            elif state.phase == "countdown":   render_countdown(display, state)
            elif state.phase == "playing":     render_playing(display, state, photo_frame)
            elif state.phase == "result_per":  render_result_per(display, state)
            elif state.phase == "final_result":render_final_result(display, state)
            elif state.phase == "photo_selection":
                render_photo_selection(display, state)
            elif state.phase == "photo_preview": render_photo_preview(display, state)
            elif state.phase == "email_entry": render_email_entry(display, state)
            elif state.phase == "done":        render_done(display, state)

            # Draw every detected player's box after the game UI.
            if (state.phase in ("playing", "calibration") and state.face_boxes
                    and state.detection_at
                    and time.monotonic() - state.detection_at <= DETECTION_RESULT_TIMEOUT):
                if state.phase == "playing":
                    challenge = state.current_challenge
                    color = EMOTION_COLORS.get(challenge[0], (255, 255, 255))
                else:
                    color = (70, 235, 125)
                for x, y, box_w, box_h in state.face_boxes:
                    if state.phase == "playing":
                        x += photo_left
                    cv2.rectangle(display, (x, y), (x + box_w, y + box_h), color, 2)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, mode)
                if not fullscreen:
                    cv2.resizeWindow(window_name, frame.shape[1], frame.shape[0])

            if state.phase == "start_menu" and key in (10, 13, ord(' ')):
                state.transition("preparation")

            if state.phase == "calibration" and key == 27:
                state.face_boxes = []
                state.current_face_count = 0
                state.transition("start_menu")

            if state.phase == "final_result":
                if key == ord(' ') or key == 13:
                    state.selected_photo_indices.clear()
                    state.photo_selection_message = ""
                    state.transition("photo_selection")

            if state.phase == "photo_selection" and key == 27:
                state.transition("final_result")

            if state.phase == "photo_preview":
                if key in (ord('y'), ord('Y')):
                    state.email_input = ""
                    state.email_error = ""
                    state.transition("email_entry")
                elif key in (ord('n'), ord('N')):
                    state.transition("photo_selection")

            if state.phase == "email_entry":
                if key in (10, 13):
                    submit_email_entry(state)
                elif key == 27:
                    state.transition("photo_preview")
                elif key in (8, 127):
                    state.email_input = state.email_input[:-1]
                    state.email_error = ""
                elif 32 <= key < 127 and chr(key) in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._%+-@":
                    if len(state.email_input) < 254:
                        state.email_input += chr(key)
                        state.email_error = ""

            if state.exit_requested:
                break

            if state.phase in ("final_result", "photo_selection", "photo_preview", "done") and key in (ord('q'), ord('Q')):
                state = GameState()
                cv2.setMouseCallback("EMO RUSH!!!", handle_photo_selection_mouse, state)
                game_generation += 1
                continue
            elif state.phase == "done" and key in (ord('r'), ord('R')):
                state = GameState()
                state.transition("preparation")
                cv2.setMouseCallback("EMO RUSH!!!", handle_photo_selection_mouse, state)
                game_generation += 1

            if key in (ord('q'), ord('Q')) and state.phase != "email_entry":
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()

    # Terminal summary
    if state.all_results:
        print("\n── LAPORAN AKHIR ──")
        for r in state.all_results:
            print(f"  {r['challenge']:<10}  {r['pct_int']}%  Grade: {r['grade']}  ({r['face_count']} faces)")
        avg_pct = int(sum(r["pct"] for r in state.all_results) / len(state.all_results) * 100)
        grade, _, gpa, _ = get_final_grade(avg_pct)
        print(f"\n  Final Score: {avg_pct}%  |  Grade: {grade}  |  IPE: {gpa:.2f}")
    if state.photo_saved:
        print(f"\nPhoto strip saved: {state.saved_photo_path}")
        print(f"Looping GIF saved: {state.saved_gif_path}")
    elif state.all_results:
        print("\nPhotos discarded or not saved.")


if __name__ == "__main__":
    main()
