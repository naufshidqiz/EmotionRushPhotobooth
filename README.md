# EMO RUSH Photobooth

A Windows webcam game that detects facial expressions, scores seven challenges, and creates a photo strip plus an animated GIF. Players choose six photos for the supplied strip template. The game does not send email; approved results are saved locally in a folder named for the email entered by the player.

## PC recommendations

| Part | Recommended | Minimum practical |
| --- | --- | --- |
| Operating system | Windows 11, 64-bit | Windows 10, 64-bit |
| Processor | Recent Intel Core i5 or AMD Ryzen 5, 4 or more cores | 64-bit dual or quad core; slower emotion detection is expected |
| Memory | 16 GB RAM | 8 GB RAM |
| Graphics | Integrated graphics is fine | No dedicated GPU required |
| Camera | USB or built-in 720p webcam; 1080p recommended | Webcam recognized by Windows |
| Display | 1920 × 1080 recommended | 1366 × 768 |
| Free storage | 10 GB for Python libraries, virtual environment, and photo output | 5 GB |
| Internet | Needed to clone the repository and install Python packages | Needed during setup |

These are practical recommendations for running the app and installing its machine-learning libraries. On Windows, this setup uses TensorFlow on the CPU; a gaming GPU is not required. Current TensorFlow releases no longer support native-Windows GPU execution after version 2.10. See [TensorFlow's Windows installation notes](https://www.tensorflow.org/install/pip#windows-native).

## Setup on Windows

### 1. Install prerequisites

Install:

- [Git for Windows](https://git-scm.com/download/win)
- [64-bit Python 3.11](https://www.python.org/downloads/windows/) (choose the latest Python 3.11.x installer)

In the Python installer, enable the Python launcher (`py`). You do not need to install CUDA or configure a GPU.

### 2. Clone the project

Open PowerShell or Windows Terminal and run:

```powershell
git clone https://github.com/naufshidqiz/EmotionRushPhotobooth.git
cd EmotionRushPhotobooth
```

### 3. Create the environment and install the libraries

Run these commands from the project folder:

```powershell
py -3.11 -m venv venv
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install -r requirements.txt
```

TensorFlow is a large download, so setup may take a few minutes. The `venv` folder is local to each computer and is not included in Git.

### 4. Start the game

```powershell
run_game.bat
```

Alternatively:

```powershell
venv\Scripts\python.exe DRAFT0_EC.py
```

The game opens fullscreen. Press **F** to switch between fullscreen and a normal window. Allow camera access if Windows asks. If the camera is busy in another program, close that program and start the game again.

## Playing and saving

1. Select **Start Game**. Use **Calibrate Camera** first if you want to check the camera, lighting, face boxes, or detected face count.
2. Complete the seven expression challenges.
3. Select six captured photos for the strip, review the strip and GIF, and approve or change the selection.
4. Enter an email address to name the local result folder. The program does not send an email.

The strip and GIF are saved under:

```text
hasil_photo/<email>/
```

For example, `hasil_photo/player@example.com/` contains the approved PNG strip and animated GIF. Captured photos and generated output are ignored by Git so personal images are not pushed when you commit project changes.

## Troubleshooting

- **`py` is not recognized:** reinstall 64-bit Python and enable the Python launcher, then reopen the terminal.
- **Python 3.11 is not found:** install Python 3.11, then rerun the environment setup commands.
- **Camera cannot open:** check Windows **Settings → Privacy & security → Camera**, allow desktop apps to use the camera, and close apps already using it.
- **A package installation fails:** confirm the terminal has internet access and that you have several gigabytes of free disk space. Remove the incomplete `venv` folder and repeat Step 3.
- **The app exits with an OpenCV or TensorFlow error:** activate/setup the project environment again using the exact commands above; avoid installing packages into a different Python installation.

## Project files

```text
DRAFT0_EC.py                   Game
run_game.bat                   Windows launcher
requirements.txt               Python dependencies
assets/photo_strip_template.png Photo-strip artwork
hasil_photo/                   Local results (not tracked by Git)
```
