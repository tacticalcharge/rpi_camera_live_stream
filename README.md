# Raspberry Pi Camera Live Stream + Motion Recording

This project streams a USB webcam live to a local web page, records clips on motion, and sends ntfy alerts.

## What it does
- Live MJPEG stream at `http://<pi-ip>:5000/`
- Motion detection with OpenCV
- Recording starts on motion and stops after 30s of no motion
- Sends a notification via ntfy on motion start (cooldown 60s)

## Setup (Raspberry Pi OS 64-bit)

1) System packages
```
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv
```

2) Create venv and install Python deps
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3) Configure env
```
cp .env.example .env
```
Edit `.env` if needed (camera index, motion area, ntfy topic, etc.).

4) Run
```
python3 app.py
```

Open `http://<pi-ip>:5000/` on your phone (same Wi‑Fi).

## Ntfy notes
Default topic is `Cat Surveilance`. If you want a different topic, change `NTFY_TOPIC` in `.env`.
If you add auth, set `NTFY_TOKEN` to a valid token.

## Recording files
Recordings are saved to `./recordings/` by default.

## Troubleshooting
- If the stream is black, try changing `CAMERA_INDEX` (0, 1, 2…)
- If recordings are too sensitive, raise `MOTION_AREA`
- If CPU is high, lower `FRAME_WIDTH/HEIGHT` or `STREAM_FPS`
