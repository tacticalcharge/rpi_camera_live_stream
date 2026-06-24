import os
import time
import threading
import logging
from datetime import datetime
from urllib.parse import quote
import re
import mimetypes
import shutil
import subprocess

import cv2
import requests
from flask import Flask, Response, render_template_string, send_file, jsonify, request
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
TRANSCODE_ENABLED = _get_env("TRANSCODE_ENABLED", lambda v: v.lower() == "true", True)
TRANSCODE_CODEC = _get_env("TRANSCODE_CODEC", str, "libx264")
TRANSCODE_PRESET = _get_env("TRANSCODE_PRESET", str, "veryfast")
TRANSCODE_CRF = _get_env("TRANSCODE_CRF", int, 23)
NTFY_ATTACH_METHOD = _get_env("NTFY_ATTACH_METHOD", str, "post").lower()
REC_OVERLAY_ENABLED = _get_env("REC_OVERLAY_ENABLED", lambda v: v.lower() == "true", True)
NOTIFY_COOLDOWN_SECONDS = _get_env("NOTIFY_COOLDOWN_SECONDS", int, 60)
NTFY_ENABLED = _get_env("NTFY_ENABLED", lambda v: v.lower() == "true", True)
NTFY_BASE_URL = _get_env("NTFY_BASE_URL", str, "https://ntfy.sh")
_raw_ntfy_topic = _get_env("NTFY_TOPIC", str, "").strip()
NTFY_TOPIC = _raw_ntfy_topic or "Cat Surveillance"
NTFY_TOKEN = _get_env("NTFY_TOKEN", str, "")
SITE_URL = _get_env("SITE_URL", str, "http://192.168.68.113:5000")
NTFY_ATTACHMENT_MODE = _get_env("NTFY_ATTACHMENT_MODE", str, "upload").lower()

os.makedirs(RECORDINGS_DIR, exist_ok=True)


class Camera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if FRAME_WIDTH and FRAME_WIDTH > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        if FRAME_HEIGHT and FRAME_HEIGHT > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        # Keep the capture buffer tiny to reduce latency on live stream.
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        self.lock = threading.Lock()
        self.last_frame = None
        self.last_frame_time = 0.0
        self.motion_detected = False
        self.last_motion_time = None
        self.recording = False
        self.writer = None
        self.recording_path = None
        self.recording_filename = None
        self.recording_start_time = None
        self.recording_start_dt = None
        self.last_notify_time = 0.0
        self.last_motion_state = False

        self.bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _notify(self, message, file_path=None, filename=None, ignore_cooldown=False):
        if not NTFY_ENABLED:
            return
        if not NTFY_TOPIC:
            logging.warning("NTFY_TOPIC is empty; skipping notification.")
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
                if not os.path.exists(file_path):
                    logging.warning("ntfy file missing: %s", file_path)
                    return
                if filename:
                    headers["Filename"] = filename
                headers["Content-Type"] = "application/octet-stream"
                with open(file_path, "rb") as handle:
                    if NTFY_ATTACHMENT_MODE == "url":
                        attach_url = f"{SITE_URL.rstrip('/')}/recordings/{quote(filename or os.path.basename(file_path))}"
                        headers["Attach"] = attach_url
                        resp = requests.post(url, data=(message or "").encode("utf-8"), headers=headers, timeout=10)
                    elif NTFY_ATTACHMENT_MODE == "none":
                        resp = requests.post(url, data=(message or "").encode("utf-8"), headers=headers, timeout=10)
                    else:
                        if NTFY_ATTACH_METHOD == "put" and filename:
                            attach_url = f"{url}/{quote(filename)}"
                            resp = requests.put(attach_url, data=handle, headers=headers, timeout=60)
                        else:
                            resp = requests.post(url, data=handle, headers=headers, timeout=60)
            else:
                resp = requests.post(url, data=(message or "").encode("utf-8"), headers=headers, timeout=5)
            if resp.status_code >= 300:
                logging.warning("ntfy notify failed (%s): %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logging.exception("ntfy notify error: %s", exc)

    def _start_recording(self, frame_shape):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"motion_{timestamp}{RECORD_EXT}"
        path = os.path.join(RECORDINGS_DIR, filename)
        height, width = frame_shape[:2]
        codecs_to_try = [RECORD_FOURCC, "avc1", "H264", "X264", "mp4v"]
        self.writer = None
        for codec in codecs_to_try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(path, fourcc, FPS, (width, height))
            if writer.isOpened():
                self.writer = writer
                if codec != RECORD_FOURCC:
                    logging.info("Recording codec fallback to %s", codec)
                break
            writer.release()

        if self.writer is None:
            logging.error("Failed to open VideoWriter for %s", path)
            return
        self.recording = True
        self.recording_path = path
        self.recording_filename = filename
        self.recording_start_time = time.time()
        self.recording_start_dt = datetime.now()
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] recording started: {path}")

    def _stop_recording(self, notify=False):
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.recording = False
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] recording stopped")
        if self.recording_path and TRANSCODE_ENABLED:
            self._transcode_recording(self.recording_path)
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
        self.recording_start_time = None
        self.recording_start_dt = None

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

    def _apply_recording_overlay(self, frame, now_dt):
        if not REC_OVERLAY_ENABLED:
            return frame
        overlay = frame.copy()
        timestamp = now_dt.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = ""
        if self.recording_start_time is not None:
            elapsed_seconds = int(time.time() - self.recording_start_time)
            minutes = elapsed_seconds // 60
            seconds = elapsed_seconds % 60
            elapsed = f"{minutes:02d}:{seconds:02d}"
        line1 = f"Recorded: {timestamp}"
        line2 = f"Elapsed: {elapsed}" if elapsed else ""
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        margin = 8
        x = 12
        y = 24

        def draw_label(text, y_pos):
            if not text:
                return
            (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
            cv2.rectangle(
                overlay,
                (x - margin, y_pos - h - margin),
                (x + w + margin, y_pos + margin),
                (0, 0, 0),
                -1,
            )
            cv2.putText(overlay, text, (x, y_pos), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

        draw_label(line1, y)
        if line2:
            draw_label(line2, y + 24)
        return overlay

    def _transcode_recording(self, path):
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            logging.warning("ffmpeg not found; skipping transcode for %s", path)
            return False
        base, ext = os.path.splitext(path)
        temp_path = f"{base}.tmp{ext}"
        cmd = [
            ffmpeg,
            "-y",
            "-i",
            path,
            "-c:v",
            TRANSCODE_CODEC,
            "-preset",
            TRANSCODE_PRESET,
            "-crf",
            str(TRANSCODE_CRF),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            temp_path,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                logging.warning("ffmpeg transcode failed for %s: %s", path, result.stderr.strip()[:300])
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                return False
            os.replace(temp_path, path)
            logging.info("Transcoded recording to H.264 for %s", path)
            return True
        except Exception as exc:
            logging.warning("ffmpeg transcode error for %s: %s", path, exc)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            motion, boxes = self._detect_motion(frame)
            now = time.time()
            now_dt = datetime.now()

            if motion:
                self.last_motion_time = now
                if not self.last_motion_state:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] motion detected")
                if not self.recording:
                    self._start_recording(frame.shape)
                if self.writer is not None:
                    record_frame = self._apply_recording_overlay(frame, now_dt)
                    self.writer.write(record_frame)
            else:
                if self.last_motion_state:
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] motion no longer detected")
                if self.recording and self.last_motion_time is not None:
                    if now - self.last_motion_time >= NO_MOTION_SECONDS:
                        self._stop_recording(notify=True)
                if self.recording and self.writer is not None:
                    record_frame = self._apply_recording_overlay(frame, now_dt)
                    self.writer.write(record_frame)

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
    .recordings-header { width: min(100%, 1100px); display: flex; align-items: center; justify-content: space-between; gap: 10px; }
    .recordings-actions { display: flex; align-items: center; gap: 8px; }
    .recordings-status { font-size: 12px; color: #aaa; }
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
      <div class="recordings-header">
        <h2>Recordings</h2>
        <div class="recordings-actions">
          <button class="btn" id="reencode-btn" type="button">Re-encode recordings</button>
          <span class="recordings-status" id="reencode-status"></span>
        </div>
      </div>
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
        btn.textContent = item.display_name || item.filename;
        btn.title = item.filename;
        btn.addEventListener('click', () => {
          playerTitle.textContent = item.display_name || item.filename;
          playerVideo.src = `/recordings/${item.filename}`;
          player.style.display = 'block';
          playerVideo.play();
        });
        const meta = document.createElement('span');
        meta.className = 'rec-meta';
        const recordedAt = item.recorded_at ? `Recorded: ${item.recorded_at}` : 'Recorded: unknown';
        meta.textContent = `(${recordedAt} | ${item.size_kb} KB)`;
        btn.appendChild(meta);
        list.appendChild(btn);
      }
    }
    loadRecordings();

    const reencodeBtn = document.getElementById('reencode-btn');
    const reencodeStatus = document.getElementById('reencode-status');
    reencodeBtn.addEventListener('click', async () => {
      reencodeBtn.disabled = true;
      reencodeStatus.textContent = 'Re-encoding…';
      try {
        const res = await fetch('/recordings-reencode', { method: 'POST' });
        const data = await res.json();
        if (!res.ok) {
          reencodeStatus.textContent = data.error || 'Re-encode failed';
        } else {
          reencodeStatus.textContent = `Done. ${data.success} ok, ${data.failed} failed.`;
          loadRecordings();
        }
      } catch (err) {
        reencodeStatus.textContent = 'Re-encode failed';
      } finally {
        reencodeBtn.disabled = false;
      }
    });
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
        last_sent_time = 0.0
        while True:
            frame, frame_time, _ = camera.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            if frame_time <= last_sent_time:
                time.sleep(0.005)
                continue
            ret, jpeg = cv2.imencode(".jpg", frame)
            if not ret:
                continue
            data = jpeg.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
            )
            last_sent_time = frame_time
            time.sleep(1.0 / max(1, STREAM_FPS))

    resp = Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


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
    path = os.path.join(RECORDINGS_DIR, filename)
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "application/octet-stream"
    return send_file(path, mimetype=mime, as_attachment=False)


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
            recorded_at = datetime.fromtimestamp(stat.st_mtime)
            match = re.match(r"^motion_(\d{8})_(\d{6})", name)
            if match:
                try:
                    recorded_at = datetime.strptime(
                        f"{match.group(1)}_{match.group(2)}",
                        "%Y%m%d_%H%M%S",
                    )
                except ValueError:
                    pass
            recorded_at_display = recorded_at.strftime("%b %d, %Y %H:%M:%S")
            display_name = f"Motion - {recorded_at_display}"
            items.append({
                "filename": name,
                "display_name": display_name,
                "recorded_at": recorded_at_display,
                "mtime": stat.st_mtime,
                "size_kb": int(stat.st_size / 1024),
            })
    except FileNotFoundError:
        pass

    items.sort(key=lambda x: x["mtime"], reverse=True)
    return jsonify({"items": items})


@app.route("/recordings-reencode", methods=["POST"])
def recordings_reencode():
    if not TRANSCODE_ENABLED:
        return jsonify({"error": "Transcoding disabled"}), 400
    if not shutil.which("ffmpeg"):
        return jsonify({"error": "ffmpeg not found"}), 400

    success = 0
    failed = 0
    try:
        for name in os.listdir(RECORDINGS_DIR):
            if not name.lower().endswith(RECORD_EXT.lower()):
                continue
            path = os.path.join(RECORDINGS_DIR, name)
            if not os.path.isfile(path):
                continue
            if camera._transcode_recording(path):
                success += 1
            else:
                failed += 1
    except FileNotFoundError:
        pass

    return jsonify({"success": success, "failed": failed})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
