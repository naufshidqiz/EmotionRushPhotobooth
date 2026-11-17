# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Python 3.11 desktop app on Windows 10/11: one fullscreen OpenCV window (`DRAFT0_EC.py`), webcam via OpenCV, emotion scoring with FER on the CPU. Not a browser app. Platform is recorded as `web` only because Impeccable has no desktop value; HTML mockups are used for design review and then ported to the OpenCV renderer.

Rendering decision (user, 2026-11-17): stay on OpenCV and draw text with Pillow using real TTF fonts instead of the built-in Hershey fonts. Pygame and a browser kiosk were offered and declined.

## Users

Visitors to the EMO RUSH booth at ITB Insight 2026 (PlayTech-ShowTech), playing alone or in groups of up to three faces in front of one webcam. They walk up, play one short session, and leave with a photo strip and a looping GIF.

## Product Purpose

An emotion-recognition photobooth game. Players act out seven expressions; each is scored by FER, the session is graded like an ITB academic transcript, and the best frame of each expression becomes a photo. Players pick six of the seven photos for the ITB Insight 2026 strip template, review the strip and GIF, and save them.

Success at the booth: a session that is fun, finishes quickly, needs little help from an operator, and produces a strip people want to keep.

## Positioning

The session is graded as a parody of SIX, ITB's academic information system: each expression is a "mata kuliah", the result is a transcript with SKS and a grade point ("NR"), and the course names are puns on real Teknik Fisika courses. This joke is specific to an ITB audience.

## Operating Context

- Booth hardware: a landscape monitor or TV, mouse, keyboard, and a USB or built-in webcam (720p minimum, 1080p recommended).
- Session flow: start menu (start, calibrate camera, exit) → 5 s get-ready → for each of 7 expressions: 3 s countdown, 5 s capture, 3 s result → transcript → choose 6 of 7 photos → review strip and GIF → type an email → done.
- The email names the local folder `hasil_photo/<email>/`. The organisers email each player their strip and GIF after the event; the game itself sends nothing.
- Fullscreen by default; F toggles fullscreen, Q exits.

## Capabilities and Constraints

- Up to three faces are scored per frame; the expression score is their average.
- Captured photos are center-cropped to the strip opening ratio 293:238 so the strip needs no further crop.
- Strip template is a fixed 941×1672 PNG with six openings at fixed pixel positions.
- The game must stay responsive on a CPU-only Windows PC; emotion inference runs on a worker thread.

Expression courses (user, 2026-11-17):

| Emotion key | Mata kuliah |
|---|---|
| happy | Senang Keberlanjutan |
| angry | Sistem Manajemen Emosi |
| sad | Pembelajaran Mesin untuk Sedih |
| neutral | Dinamika Netral |
| fear | Pemrosesan Takut |
| disgust | Mekanika Jijik |
| surprise | Medan ElektroKagettika |

Open decisions:
- Grade scale: the code uses A+/A/B+/B/C/D; switching to ITB's A/AB/B/BC/C/D/E was proposed, not yet confirmed.
- On-screen language: copy currently mixes Indonesian and English.
- Course codes and SKS per course.

## Brand Commitments

- Name: Emotion Rush (user, 2026-11-17). Older code and the supplied strip artwork still say "EMO RUSH"; the strip reads "EMO RUSH PLAYTECH-SHOWTECH" and "ITB INSIGHT 2026".
- The strip template `assets/photo_strip_template.png` is a supplied asset and is not redesigned.
- The transcript screen follows the real SIX ITB interface, from screenshots the user supplied on 2026-11-17 (binding reference).
- Voice: playful Indonesian with casual student slang, as in the existing per-grade comments.
- The UI must not look AI-generated; directions are grounded in real references, not invented moods.

## Evidence on Hand

- `assets/photo_strip_template.png` (ITB Insight 2026 strip artwork).
- User-supplied SIX ITB screenshots (in conversation, not stored in the repo; they contain personal data and must not be copied into the game).
- No player testimonials, booth photos, or measured play data exist yet.

## Product Principles

1. Keep the line moving: every screen tells the player what to do next without an operator.
2. The joke works by being exact: the SIX parody earns its laugh from fidelity to the real system, not from decoration.
3. The photo is the prize: every step serves a strip the player wants to keep.
4. Player photos stay on this PC and are never sent anywhere.
