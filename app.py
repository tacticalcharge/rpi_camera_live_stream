import os
import time
import threading
import logging
from datetime import datetime
from urllib.parse import quote

import cv2
import requests
from flask import Flask, Response, render_template_string, send_from_directory, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def _get_env(name, cast=str, default=None):
    value = os.getenv(name, None)
    if value is None:
        return default
    try:
        return cast(value)
    except ValueError:
        return default


CAMERA_INDEX = _get_env("CAMERA_INDEX", int, 0)
FRAME_WIDTH = _get_env("FRAME_WIDTH", int, 0)
FRAME_HEIGHT = _get_env("FRAME_HEIGHT", int, 0)
FPS = _get_env("FPS", int, 20)
STREAM_FPS = _get_env("STREAM_FPS", int, 12)
MOTION_AREA = _get_env("MOTION_AREA", int, 5000)
NO_MOTION_SECONDS = _get_env("NO_MOTION_SECONDS", int, 30)
RECORDINGS_DIR = _get_env("RECORDINGS_DIR", str, "recordings")
RECORD_FOURCC = _get_env("RECORD_FOURCC", str, "mp4v")
RECORD_EXT = _get_env("RECORD_EXT", str, ".mp4")
NOTIFY_COOLDOWN_SECONDS = _get_env("NOTIFY_COOLDOWN_SECONDS", int, 60)
NTFY_ENABLED = _get_env("NTFY_ENABLED", lambda v: v.lower() == "true", True)
NTFY_BASE_URL = _get_env("NTFY_BASE_URL", str, "https://ntfy.sh")
NTFY_TOPIC = _get_env("NTFY_TOPIC", str, "Cat Surveilance")
NTFY_TOKEN = _get_env("NTFY_TOKEN", str, "")
SITE_URL = _get_env("SITE_URL", str, "http://192.168.68.113:5000")

os.makedirs(RECORDINGS_DIR, exist_ok=True)


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if FRAME_WIDTH and FRAME_WIDTH > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        if FRAME_HEIGHT and FRAME_HEIGHT > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        self.lock = threading.Lock()
        self.last_frame = None
        self.last_frame_time = 0.0
        self.motion_detected = False
        self.last_motion_time = None
        self.recording = False
        self.writer = None
        self.recording_path = None
        self.recording_filename = None
        self.last_notify_time = 0.0
        self.last_motion_state = False

        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _notify(self, message, file_path=None, filename=None, ignore_cooldown=False):
        if not NTFY_ENABLED:
            return
        if not ignore_cooldown:
            now = time.time()
            if now - self.last_notify_time < NOTIFY_COOLDOWN_SECONDS:
                return
            self.last_notify_time = now
        try:
            topic = quote(NTFY_TOPIC)
            url = f"{NTFY_BASE_URL.rstrip('/')}/{topic}"
            headers = {"Title": "Motion detected"}
            if message:
                headers["Message"] = message
            if NTFY_TOKEN:
                headers["Authorization"] = f"Bearer {NTFY_TOKEN}"
            if file_path:
                if filename:
                    headers["Filename"] = filename
                with open(file_path, "rb") as handle:
                    requests.post(url, data=handle, headers=headers, timeout=10)
            else:
                requests.post(url, data=(message or "").encode("utf-8"), headers=headers, timeout=5)
        except Exception:
            pass

    def _start_recording(self, frame_shape):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{timestamp}{RECORD_EXT}"
        path = os.path.join(RECORDINGS_DIR, filename)
        fourcc = cv2.VideoWriter_fourcc(*RECORD_FOURCC)
        height, width = frame_shape[:2]
        self.writer = cv2.VideoWriter(path, fourcc, FPS, (width, height))
        self.recording = True
        self.recording_path = path
        self.recording_filename = filename
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] recording started: {path}")

    def _stop_recording(self, notify=False):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.recording = False
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] recording stopped")
        if notify and self.recording_path:
            link_message = f"Motion recorded. Live view: {SITE_URL}"
            self._notify(link_message)
            self._notify(
                "Motion video attached.",
                file_path=self.recording_path,
                filename=self.recording_filename,
                ignore_cooldown=True,
            )
        self.recording_path = None
        self.recording_filename = None

    def _detect_motion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        mask = self.bg.apply(gray)
        _, thresh = cv2.threshold(mask, 244, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = False
        boxes = []
        for c in contours:
            if cv2.contourArea(c) < MOTION_AREA:
                continue
            motion = True
            (x, y, w, h) = cv2.boundingRect(c)
            boxes.append((x, y, w, h))
        return motion, boxes

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            motion, boxes = self._detect_motion(frame)
            now = time.time()

            if motion:
                self.last_motion_time = now
                if not self.last_motion_state:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] motion detected")
                if not self.recording:
                    self._start_recording(frame.shape)
                if self.writer is not None:
                    self.writer.write(frame)
            else:
                if self.last_motion_state:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] motion no longer detected")
                if self.recording and self.last_motion_time is not None:
                    if now - self.last_motion_time >= NO_MOTION_SECONDS:
                        self._stop_recording(notify=True)
                if self.recording and self.writer is not None:
                    self.writer.write(frame)

            display = frame.copy()
            if motion:
                for (x, y, w, h) in boxes:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(display, "MOTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            with self.lock:
                self.last_frame = display
                self.last_frame_time = now
                self.motion_detected = motion
                self.last_motion_state = motion

        self.cap.release()
        self._stop_recording(notify=True)

    def get_frame(self):
        with self.lock:
            return self.last_frame, self.last_frame_time, self.motion_detected


camera = Camera()


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Raspberry Pi Camera</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: Arial, sans-serif; background: #111; color: #eee; margin: 0; }
    header { padding: 12px 16px; background: #222; display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    main { padding: 12px 16px 20px; display: flex; flex-direction: column; align-items: center; }
    .frame-wrap { width: min(100%, 1100px); }
    .frame {
      width: 100%;
      height: auto;
      border: 2px solid #444;
      border-radius: 6px;
      display: block;
      background: #000;
    }
    .status { margin-top: 8px; font-size: 14px; color: #bbb; }
    .btn { background: #2d2d2d; color: #eee; border: 1px solid #444; padding: 8px 12px; border-radius: 6px; cursor: pointer; }
    .btn:hover { background: #3a3a3a; }
    .recordings { width: min(100%, 1100px); margin-top: 18px; }
    .recordings h2 { margin: 0 0 8px; font-size: 18px; }
    .recordings-list { display: flex; flex-direction: column; gap: 8px; }
    .rec-btn { text-align: left; width: 100%; }
    .rec-meta { font-size: 12px; color: #aaa; margin-left: 6px; }
    .player { width: min(100%, 1100px); margin-top: 16px; display: none; }
    .player video { width: 100%; max-height: 70vh; background: #000; border: 2px solid #333; border-radius: 6px; }
    .player-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 6px; }
    .player-title { font-size: 14px; color: #bbb; }
  </style>
</head>
<body>
  <header>
    <h1>Raspberry Pi Camera</h1>
    <button class="btn" id="fs-btn" type="button">Full screen</button>
  </header>
  <main>
    <div class="frame-wrap">
      <img class="frame" id="cam" src="/stream" />
    </div>
    <div class="status" id="status">Loading…</div>
    <section class="player" id="player">
      <div class="player-header">
        <div class="player-title" id="player-title">Playing</div>
        <button class="btn" id="player-close" type="button">Close</button>
      </div>
      <video id="player-video" controls></video>
    </section>
    <section class="recordings">
      <h2>Recordings</h2>
      <div id="recordings" class="recordings-list">Loading…</div>
    </section>
  </main>
  <script>
    const cam = document.getElementById('cam');
    const fsBtn = document.getElementById('fs-btn');
    fsBtn.addEventListener('click', async () => {
      if (!document.fullscreenElement) {
        await cam.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    });
    document.addEventListener('fullscreenchange', () => {
      fsBtn.textContent = document.fullscreenElement ? 'Exit full screen' : 'Full screen';
    });

    async function refreshStatus() {
      const res = await fetch('/status');
      const data = await res.json();
      const motion = data.motion ? 'yes' : 'no';
      document.getElementById('status').textContent = `motion: ${motion} | recording: ${data.recording}`;
    }
    setInterval(refreshStatus, 2000);
    refreshStatus();

    async function loadRecordings() {
      const res = await fetch('/recordings-list');
      const data = await res.json();
      const list = document.getElementById('recordings');
      const player = document.getElementById('player');
      const playerVideo = document.getElementById('player-video');
      const playerTitle = document.getElementById('player-title');
      const playerClose = document.getElementById('player-close');

      playerClose.addEventListener('click', () => {
        playerVideo.pause();
        playerVideo.removeAttribute('src');
        playerVideo.load();
        player.style.display = 'none';
      });

      if (!data.items.length) {
        list.textContent = 'No recordings yet.';
        return;
      }
      list.innerHTML = '';
      for (const item of data.items) {
        const btn = document.createElement('button');
        btn.className = 'btn rec-btn';
        btn.type = 'button';
        btn.textContent = item.filename;
        btn.addEventListener('click', () => {
          playerTitle.textContent = item.filename;
          playerVideo.src = `/recordings/${item.filename}`;
          player.style.display = 'block';
          playerVideo.play();
        });
        const meta = document.createElement('span');
        meta.className = 'rec-meta';
        meta.textContent = `(${item.size_kb} KB)`;
        btn.appendChild(meta);
        list.appendChild(btn);
      }
    }
    loadRecordings();
  </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(INDEX_HTML)


@app.route("/stream")
def stream():
    def generate():
        while True:
            frame, _, _ = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            data = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            )
            time.sleep(1.0 / max(1, STREAM_FPS))

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    _, _, motion = camera.get_frame()
    return {
        "motion": bool(motion),
        "recording": camera.recording,
        "last_motion_time": camera.last_motion_time,
    }


@app.route("/recordings/<path:filename>")
def recordings(filename):
    return send_from_directory(RECORDINGS_DIR, filename, as_attachment=False)


@app.route("/recordings-list")
def recordings_list():
    items = []
    try:
        for name in os.listdir(RECORDINGS_DIR):
            if not name.lower().endswith(RECORD_EXT.lower()):
                continue
            path = os.path.join(RECORDINGS_DIR, name)
            if not os.path.isfile(path):
                continue
            stat = os.stat(path)
            items.append({
                "filename": name,
                "mtime": stat.st_mtime,
                "size_kb": int(stat.st_size / 1024),
            })
    except FileNotFoundError:
        pass

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify({"items": items})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
