"""
Emotion Rush — Emotion Challenge Game
=====================================
Flow:
  START MENU (gas main / cek kamera / keluar)
  PREPARATION (5 s get ready, faces are counted)
  → COUNTDOWN (3 s, shows the course to act out)
  → GAME (5 s, scores every frame and keeps the best one)
  → RESULT per course (3 s, grade stamp and comment)
  → repeat for all 7 courses
  → FINAL RESULT (SIX-style transcript with NR)
  → PHOTO SELECTION (choose six of seven photos)
  → PHOTO PREVIEW (strip + GIF, save or change)
  → EMAIL ENTRY (local folder name) → DONE

Screens are drawn by ui_board.py (the himpunan notice board).

Run:
    venv\\Scripts\\python.exe DRAFT0_EC.py
"""

import glob
import os
import random
import re
import threading
import time
from importlib import resources

import cv2
import numpy as np
from fer.fer import FER

import ui_board as ui

# ─── Configuration ──────────────────────────────────────────
CHALLENGE_DURATION  = 5      # seconds of the GAME phase
PREP_DURATION       = 5      # seconds of the PREPARATION phase
COUNTDOWN_DURATION  = 3      # seconds of the COUNTDOWN phase
RESULT_DURATION     = 3      # seconds of each RESULT screen
PHOTO_FOLDER        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hasil_photo")
PHOTO_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "photo_strip_template.png"
)
MAX_PLAYERS         = 3
DETECTION_WIDTH     = 640
# The strip's six photo openings are about 293 x 238 px. Keep the camera capture
# at that aspect ratio so the printed photos need no crop.
PHOTO_FRAME_ASPECT  = 293 / 238
DETECTION_RESULT_TIMEOUT = 1.0  # Clear stale face data if inference stops returning results.
WINDOW_NAME = "Emotion Rush"
# Phases that only count faces (no emotion scoring).
FACE_ONLY_PHASES = ("calibration", "preparation", "countdown")
EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9_%+-]+(?:\.[A-Za-z0-9_%+-]+)*@"
    r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+$"
)

# (FER emotion key, course name, instruction, course code). Order is shuffled each game.
CHALLENGES = [
    ("happy",    "Senang Keberlanjutan",           "Senyum paling lebar, gas!",                    "EK1101"),
    ("angry",    "Sistem Manajemen Emosi",         "Keluarin muka paling kesel kamu.",             "EK1102"),
    ("sad",      "Pembelajaran Mesin untuk Sedih", "Keluarin muka paling galau kamu.",             "EK1103"),
    ("neutral",  "Dinamika Netral",                "Lempeng aja, tanpa ekspresi.",                 "EK1104"),
    ("fear",     "Pemrosesan Takut",               "Pasang muka paling ketakutan.",                "EK1105"),
    ("disgust",  "Mekanika Jijik",                 "Muka jijik, kayak nyium kaus kaki seminggu.",  "EK1106"),
    ("surprise", "Medan ElektroKagettika",         "Kaget! Mata melotot, mulut nganga.",           "EK1107"),
]
CODE_SUFFIX = "-26"

# ITB grade scale: (minimum score %, grade, grade points).
GRADES = [
    (85, "A", 4.0),
    (75, "AB", 3.5),
    (60, "B", 3.0),
    (50, "BC", 2.5),
    (30, "C", 2.0),
    (15, "D", 1.0),
    (0,  "E", 0.0),
]
PASSING_GRADES = {"A", "AB", "B", "BC", "C"}

# Per-grade comments (to make people want another round).
COMMENTS = {
    "A": [
        "DEWA EKSPRESI! Muka kamu bisa jadi template meme!",
        "Lebay tapi SEMPURNA! Oscar untuk kamu!",
        "TOTALITAS ABIS! Bahkan AI speechless liat kamu!",
        "Hampir sempurna! Dikit lagi bisa kalahkan Anya Geraldine!",
        "Keren banget! Coba lebih 'gila' lagi di ronde berikutnya!",
        "Talent show material nih!",
    ],
    "AB": [
        "Bagus! Tapi muka kamu masih nahan-nahan. KELUARIN semua!",
        "Potential terlihat! Next round buang gengsinya ya!",
        "AB itu udah keren, tapi A itu LEGENDARIS. Coba lagi!",
    ],
    "B": [
        "Lumayan! Tapi kayaknya masih 'jaim' nih... berani ga?",
        "Separuh jalan! Ronde berikutnya TOTALITAS atau nggak sama sekali!",
        "Wajah kamu bilang {grade}, tapi jiwa kamu bilang A. Buktiin!",
    ],
    "C": [
        "Hmm... kamera udah siap, kamu yang belum. Gas lagi!",
        "C bukan berarti Cukup, tapi Coba lagi dengan lebih gila!",
        "Muka kamu datar banget, bahkan AI bingung. Ekspresif dikit dong!",
    ],
    "D": [
        "D untuk 'Dare to try again?'! Masa muka gitu aja kalah sama AI!",
        "AI hampir kasihan... hampir. Ayo buktiin kamu bisa lebih baik!",
        "Nggak apa-apa, semua legend pernah mulai dari D. Next round ALL OUT!",
    ],
    "E": [
        "AI hampir kasihan... hampir. Ayo buktiin kamu bisa lebih baik!",
        "Nggak apa-apa, semua legend pernah mulai dari bawah. Next round ALL OUT!",
        "Mukanya belum kebaca nih. Maju dikit, terus ALL OUT!",
    ],
}
COMMENTS["BC"] = COMMENTS["B"]

# Final standing by NR (mean grade points).
PREDICATES = [
    (3.51, "Dengan pujian. Muka kamu cum laude!"),
    (3.01, "Sangat memuaskan."),
    (2.76, "Memuaskan."),
    (2.00, "Lulus, tapi tipis."),
    (0.00, "Belum lulus. Ulang semester depan, ya."),
]


# ─── Helpers ────────────────────────────────────────────────

def grade_for(pct_int):
    for minimum, grade, points in GRADES:
        if pct_int >= minimum:
            return grade, points
    return "E", 0.0


def get_comment(grade):
    return random.choice(COMMENTS[grade]).replace("{grade}", grade)


def final_standing(results):
    """Return (NR, predicate) for a finished game."""
    if not results:
        return 0.0, PREDICATES[-1][1]
    nr = sum(r["points"] for r in results) / len(results)
    for minimum, text in PREDICATES:
        if nr >= minimum:
            return nr, text
    return nr, PREDICATES[-1][1]


def count_saved_today():
    """Photo strips saved today, across every email folder."""
    today = time.strftime("%Y%m%d")
    return len(glob.glob(os.path.join(PHOTO_FOLDER, "*", f"{today}_*_PHOTO_STRIP.png")))


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
        self.restart_requested = None  # "again" or "menu", set by the done screen buttons
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

    def clear_faces(self):
        self.face_results = []
        self.face_boxes = []
        self.current_face_count = 0
        self.detection_at = 0.0

    def start_playing(self):
        self.best_frame  = None
        self.best_pct    = 0.0
        self.clear_faces()
        self.current_score = 0.0
        self.play_generation += 1
        self.best_face_count = 0
        self.transition("playing")

    def finish_challenge(self, last_frame):
        ch = self.challenges[self.current_idx]
        frame_to_save = self.best_frame if self.best_frame is not None else last_frame.copy()
        pct_int = int(self.best_pct * 100)
        grade, points = grade_for(pct_int)
        self.all_results.append({
            "challenge": ch[1],
            "emo_key":   ch[0],
            "code":      ch[3],
            "pct":       self.best_pct,
            "pct_int":   pct_int,
            "frame":     frame_to_save.copy(),
            "face_count": self.best_face_count,
            "grade":     grade,
            "points":    points,
            "passed":    grade in PASSING_GRADES,
            "comment":   get_comment(grade),
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
                if len(context) > 3 and context[3] in FACE_ONLY_PHASES:
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


# ─── Photos ─────────────────────────────────────────────────

def photo_fov(frame):
    """Center-crop to the template opening aspect before detection and saving."""
    height, width = frame.shape[:2]
    target_width = min(width, int(round(height * PHOTO_FRAME_ASPECT)))
    left = (width - target_width) // 2
    return frame[:, left:left + target_width].copy(), left


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
        sheet_path = os.path.join(user_folder, f"{stamp}_EMOTION_RUSH_PHOTO_STRIP.png")
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

        gif_path = os.path.join(user_folder, f"{stamp}_EMOTION_RUSH_LOOP.gif")
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
    print(f"[SAVED] Photo strip and looping GIF -> {user_folder}/")
    discard_photo_images(state)
    return True


def discard_photo_images(state):
    """Release in-memory photo data after a save or discard decision."""
    state.contact_sheet = None
    for result in state.all_results:
        result.pop("frame", None)


# ─── Actions shared by mouse and keyboard ───────────────────

def open_calibration(state):
    state.clear_faces()
    state.transition("calibration")


def back_to_menu_from_calibration(state):
    state.clear_faces()
    state.transition("start_menu")


def open_selection(state):
    state.selected_photo_indices.clear()
    state.photo_selection_message = ""
    state.transition("photo_selection")


def toggle_photo(state, index):
    state.photo_selection_message = ""
    if index in state.selected_photo_indices:
        state.selected_photo_indices.remove(index)
    elif len(state.selected_photo_indices) < 6:
        state.selected_photo_indices.add(index)
    else:
        state.photo_selection_message = "Udah 6, batalin satu dulu kalau mau ganti."


def confirm_selection(state):
    if len(state.selected_photo_indices) != 6:
        state.photo_selection_message = "Pilih pas 6 foto dulu ya."
        return
    selected = [state.all_results[i] for i in sorted(state.selected_photo_indices)]
    try:
        state.contact_sheet = create_contact_sheet(selected)
    except (FileNotFoundError, ValueError, cv2.error) as exc:
        print(f"[ERROR] Could not create the strip preview: {exc}")
        state.photo_selection_message = "Template strip nggak bisa dibuka. Cek folder assets."
        return
    state.photo_selection_message = ""
    state.gif_preview_index = 0
    state.gif_preview_updated = time.monotonic()
    state.transition("photo_preview")


def open_email_entry(state):
    state.email_input = ""
    state.email_error = ""
    state.transition("email_entry")


def submit_email_entry(state):
    """Validate the email-shaped folder name, then save the local photo set."""
    email = state.email_input.strip()
    if not EMAIL_PATTERN.fullmatch(email) or len(email) > 254:
        state.email_error = "Emailnya belum bener. Cek lagi, terus tekan Enter."
        return False
    if save_photo_session(state, email):
        state.saved_email = email.lower()
        state.transition("done")
        return True
    state.email_error = "Gagal nyimpen file. Cek console, terus coba lagi."
    return False


def handle_mouse(event, x, y, flags, state):
    """Route clicks to the button under the pointer on the current screen."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    name = ui.hit_test(x, y)
    if name is None:
        return
    phase = state.phase
    if phase == "start_menu":
        if name == "start":
            state.transition("preparation")
        elif name == "calibrate":
            open_calibration(state)
        elif name == "exit":
            state.exit_requested = True
    elif phase == "calibration" and name == "back":
        back_to_menu_from_calibration(state)
    elif phase == "final_result" and name == "next":
        open_selection(state)
    elif phase == "photo_selection":
        if name.startswith("card"):
            toggle_photo(state, int(name[4:]))
        elif name == "continue":
            confirm_selection(state)
    elif phase == "photo_preview":
        if name == "approve":
            open_email_entry(state)
        elif name == "change":
            state.transition("photo_selection")
    elif phase == "email_entry" and name == "save":
        submit_email_entry(state)
    elif phase == "done" and name in ("again", "menu"):
        state.restart_requested = name


# ─── Frame loop pieces ──────────────────────────────────────

def advance_timers(state, photo_frame):
    """Move timed phases forward."""
    elapsed = time.time() - state.phase_start
    if state.phase == "preparation" and elapsed >= PREP_DURATION:
        state.transition("countdown")
    elif state.phase == "countdown" and elapsed >= COUNTDOWN_DURATION:
        state.start_playing()
    elif state.phase == "playing" and elapsed >= CHALLENGE_DURATION:
        state.finish_challenge(photo_frame)
    elif state.phase == "result_per" and elapsed >= RESULT_DURATION:
        state.after_result_per()


def live_boxes(state):
    if (state.face_boxes and state.detection_at
            and time.monotonic() - state.detection_at <= DETECTION_RESULT_TIMEOUT):
        return state.face_boxes
    return []


def render(state, frame, photo_frame, played_today):
    t = time.time() - state.phase_start
    phase = state.phase
    boxes = live_boxes(state)
    count = len(boxes)
    courses = state.challenges

    if phase == "start_menu":
        return ui.render_idle(photo_frame, t, played_today)
    if phase == "calibration":
        return ui.render_calibration(frame, t, boxes, count)
    if phase == "preparation":
        return ui.render_preparation(photo_frame, t, max(0.0, PREP_DURATION - t), boxes, count)
    if phase == "countdown":
        return ui.render_countdown(photo_frame, t, max(0.0, COUNTDOWN_DURATION - t), state.current_idx,
                                   courses, boxes, count)
    if phase == "playing":
        return ui.render_playing(photo_frame, t, max(0.0, CHALLENGE_DURATION - t), CHALLENGE_DURATION,
                                 state.current_idx, courses, state.current_score * 100, state.best_pct * 100,
                                 boxes)
    if phase == "result_per":
        result = state.all_results[-1]
        return ui.render_result(t, max(0.0, RESULT_DURATION - t), state.current_idx - 1, state.total, result,
                                result["emo_key"], state.current_idx >= state.total)
    if phase == "final_result":
        nr, predicate = final_standing(state.all_results)
        best = max(state.all_results, key=lambda r: r["pct"])
        codes = [(r["code"], CODE_SUFFIX) for r in state.all_results]
        return ui.render_transcript(state.all_results, nr, predicate, best.get("frame"), codes)
    if phase == "photo_selection":
        return ui.render_selection(t, state.all_results, state.selected_photo_indices,
                                   state.photo_selection_message)
    if phase == "photo_preview":
        now = time.monotonic()
        if now - state.gif_preview_updated >= 0.65:
            state.gif_preview_index = (state.gif_preview_index + 1) % len(state.all_results)
            state.gif_preview_updated = now
        current = state.all_results[state.gif_preview_index]
        caption = f"{current['challenge']} · {state.gif_preview_index + 1}/{len(state.all_results)}"
        return ui.render_preview(t, state.contact_sheet, current.get("frame"), caption)
    if phase == "email_entry":
        caret = int(time.monotonic() * 2) % 2 == 0
        return ui.render_email(t, state.email_input, caret, state.email_error)
    nr, predicate = final_standing(state.all_results)
    return ui.render_done(t, nr, predicate, state.photo_saved, state.saved_email or "")


# ─── Main ───────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Emotion Rush - Emotion Challenge Game")
    print("  F = fullscreen  |  Q = keluar  |  R = main lagi (di layar akhir)")
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
    played_today = count_saved_today()
    counted_generation = -1
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    fullscreen = True
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(WINDOW_NAME, handle_mouse, state)

    def new_game(start_playing):
        nonlocal state, game_generation
        state = GameState()
        if start_playing:
            state.transition("preparation")
        cv2.setMouseCallback(WINDOW_NAME, handle_mouse, state)
        game_generation += 1

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Webcam stopped returning frames.")
                break

            photo_frame, _ = photo_fov(frame)

            if state.phase == "playing" or state.phase in FACE_ONLY_PHASES:
                context = (game_generation, state.play_generation, state.current_idx, state.phase)
                worker.submit(frame if state.phase == "calibration" else photo_frame, context)

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
                    elif state.phase in FACE_ONLY_PHASES:
                        state.clear_faces()

                # Brief misses keep the last result; sustained misses expire after the timeout.
                if (state.face_boxes and state.detection_at
                        and time.monotonic() - state.detection_at > DETECTION_RESULT_TIMEOUT):
                    state.clear_faces()
                    state.current_score = 0.0

            advance_timers(state, photo_frame)
            display = render(state, frame, photo_frame, played_today)
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('f'), ord('F')) and state.phase != "email_entry":
                fullscreen = not fullscreen
                mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, mode)
                if not fullscreen:
                    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

            if state.phase == "start_menu" and key in (10, 13, ord(' ')):
                state.transition("preparation")

            elif state.phase == "calibration" and key == 27:
                back_to_menu_from_calibration(state)

            elif state.phase == "final_result" and key in (13, ord(' ')):
                open_selection(state)

            elif state.phase == "photo_selection" and key == 27:
                state.transition("final_result")

            elif state.phase == "photo_preview":
                if key in (ord('y'), ord('Y')):
                    open_email_entry(state)
                elif key in (ord('n'), ord('N')):
                    state.transition("photo_selection")

            elif state.phase == "email_entry":
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

            if state.phase == "done" and state.photo_saved and counted_generation != game_generation:
                played_today = count_saved_today()
                counted_generation = game_generation

            if state.exit_requested:
                break

            if state.restart_requested == "again" or (state.phase == "done" and key in (ord('r'), ord('R'))):
                new_game(start_playing=True)
                continue
            if state.restart_requested == "menu" or (
                    state.phase in ("final_result", "photo_selection", "photo_preview", "done")
                    and key in (ord('q'), ord('Q'))):
                new_game(start_playing=False)
                continue

            if key in (ord('q'), ord('Q')) and state.phase != "email_entry":
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()

    # Terminal summary
    if state.all_results:
        print("\n── TRANSKRIP ──")
        for r in state.all_results:
            print(f"  {r['challenge']:<32}  {r['pct_int']:>3}%  Nilai: {r['grade']:<2}  ({r['face_count']} muka)")
        nr, predicate = final_standing(state.all_results)
        print(f"\n  NR: {nr:.2f}  |  {predicate}")
    if state.photo_saved:
        print(f"\nPhoto strip saved: {state.saved_photo_path}")
        print(f"Looping GIF saved: {state.saved_gif_path}")
    elif state.all_results:
        print("\nPhotos discarded or not saved.")


if __name__ == "__main__":
    main()
