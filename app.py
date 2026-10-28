"""
Deteksi pelanggaran K3 real-time berbasis web.

Jalankan:  python app.py
Buka:      http://localhost:5000   (dari HP/laptop lain: http://IP-LAPTOP:5000)

Butuh: pip install flask ultralytics opencv-python
"""
import cv2, os, csv, time, threading
from datetime import datetime
from flask import Flask, Response, jsonify, send_from_directory
from ultralytics import YOLO

# ====== PENGATURAN ======
MODEL_PATH = "models/best.pt"
SOURCE = 0                  # 0 = webcam, "video.mp4", atau "rtsp://..." untuk CCTV
CONF = 0.5
IMGSZ = 320                 # ukuran inferensi — lebih kecil = lebih cepat (320/416/640)
SKIP_FRAME = 2              # jalankan YOLO tiap N frame; frame antara pakai hasil sebelumnya
MIN_TINGGI_PERSON = 0.35    # orang valid minimal 35% tinggi frame
COOLDOWN = 5                # detik jeda antar catatan pelanggaran
VIOLATION_CLASSES = {"no_helmet", "no_coverall", "no_safety_shoes"}
LABEL = {
    "no_helmet": "Tidak pakai helm",
    "no_coverall": "Tidak pakai baju praktek",
    "no_safety_shoes": "Tidak pakai sepatu safety",
}
# ========================

BUKTI_DIR = "output/bukti"
LOG_CSV = "output/log_pelanggaran.csv"
os.makedirs(BUKTI_DIR, exist_ok=True)
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        csv.writer(f).writerow(["waktu", "pelanggaran", "file_bukti"])

app = Flask(__name__)
model = YOLO(MODEL_PATH)

state = {"jpeg": None, "violations": [], "persons": 0, "fps": 0.0}
log_rows = []          # 50 catatan terakhir, dipakai tabel di web
lock = threading.Lock()


def _pilih_webcam():
    """Pilih kamera fisik: OBS Virtual Camera punya fps_native == -1, webcam fisik > 0."""
    if isinstance(SOURCE, int):
        # Scan index 0..3, ambil yang fps_native > 0 (bukan OBS)
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            ok, _ = cap.read()
            print(f"[KAMERA] Index {idx}: fps_native={fps}, bisa_baca={ok}")
            if ok and fps > 0:
                print(f"[KAMERA] Pilih index {idx} sebagai webcam fisik (fps={fps})")
                return cap
            cap.release()
        # Fallback: pakai index SOURCE apa adanya
        print(f"[KAMERA] Fallback ke index {SOURCE}")
        return cv2.VideoCapture(SOURCE)
    else:
        return cv2.VideoCapture(SOURCE)


def detect_loop():
    cap = _pilih_webcam()
    last_saved, t_prev = 0, time.time()
    frame_n = 0
    last_res = None      # hasil deteksi terakhir (dipakai ulang saat skip)
    last_annotated = None
    last_viol = set()
    last_persons = []

    while True:
        ok, frame = cap.read()
        if not ok:
            if isinstance(SOURCE, str) and not SOURCE.startswith("rtsp"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            time.sleep(0.5)
            continue

        frame_n += 1
        H, W = frame.shape[:2]

        if frame_n % SKIP_FRAME == 0:
            # Biarkan YOLO yang handle resize + letterbox secara internal
            # sehingga koordinat bbox otomatis akurat di frame asli
            res = model(frame, conf=CONF, verbose=False, imgsz=IMGSZ)[0]
            H = frame.shape[0]

            persons = []
            for b in res.boxes:
                if model.names[int(b.cls)] == "person":
                    x1, y1, x2, y2 = map(float, b.xyxy[0])
                    if (y2 - y1) / H >= MIN_TINGGI_PERSON:
                        persons.append((x1, y1, x2, y2))

            viol = set()
            for b in res.boxes:
                nama = model.names[int(b.cls)]
                if nama not in VIOLATION_CLASSES:
                    continue
                x1, y1, x2, y2 = map(float, b.xyxy[0])
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if any(px1 <= cx <= px2 and py1 <= cy <= py2 for px1, py1, px2, py2 in persons):
                    viol.add(nama)

            annotated = res.plot()

            last_viol, last_persons, last_annotated = viol, persons, annotated

        else:
            # Frame skip: tampilkan frame asli dengan anotasi sebelumnya
            annotated = last_annotated if last_annotated is not None else frame
            viol = last_viol
            persons = last_persons

        if viol and time.time() - last_saved > COOLDOWN:
            waktu = datetime.now()
            nama_file = waktu.strftime("%Y%m%d_%H%M%S.jpg")
            cv2.imwrite(os.path.join(BUKTI_DIR, nama_file), annotated)
            teks = ", ".join(LABEL[v] for v in sorted(viol))
            with open(LOG_CSV, "a", newline="") as f:
                csv.writer(f).writerow([waktu.isoformat(timespec="seconds"), teks, nama_file])
            with lock:
                log_rows.insert(0, {"waktu": waktu.strftime("%H:%M:%S"),
                                    "pelanggaran": teks, "bukti": nama_file})
                del log_rows[50:]
            last_saved = time.time()

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 65])
        now = time.time()
        with lock:
            state["jpeg"] = buf.tobytes()
            state["violations"] = [LABEL[v] for v in sorted(viol)]
            state["persons"] = len(persons)
            state["fps"] = round(1 / max(now - t_prev, 1e-3), 1)
        t_prev = now


def mjpeg():
    while True:
        with lock:
            data = state["jpeg"]
        if data:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n"
        time.sleep(0.03)


@app.route("/video")
def video():
    return Response(mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    with lock:
        return jsonify({k: state[k] for k in ("violations", "persons", "fps")})


@app.route("/log")
def log():
    with lock:
        return jsonify(log_rows)


@app.route("/bukti/<path:nama>")
def bukti(nama):
    return send_from_directory(BUKTI_DIR, nama)


@app.route("/")
def index():
    return PAGE


PAGE = """<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pemantau K3</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#e8e9eb; --panel:#f7f7f5; --ink:#1b2430; --muted:#5d6775; --line:#c9ccd1;
    --ok:#1e7f4f; --bad:#c62828; --warn:#f2b705;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 "IBM Plex Sans",system-ui,sans-serif}
  header{display:flex;align-items:center;gap:12px;padding:14px 24px;background:var(--ink);color:#fff}
  header .mark{width:14px;height:14px;background:var(--warn);transform:rotate(45deg)}
  header h1{font-size:18px;font-weight:600;margin:0}
  main{display:grid;grid-template-columns:minmax(0,2fr) minmax(300px,1fr);gap:20px;padding:20px 24px;max-width:1400px;margin:0 auto}
  @media (max-width:900px){main{grid-template-columns:1fr;padding:14px}}
  .video{background:#000;border:2px solid var(--ink);position:relative}
  .video img{display:block;width:100%;height:auto}
  .side{display:flex;flex-direction:column;gap:20px;min-width:0}
  .panel{background:var(--panel);border:1px solid var(--line);padding:16px}
  .panel h2{font-size:15px;font-weight:600;margin:0 0 10px}
  #status{border-left:8px solid var(--ok)}
  #status.bad{border-left-color:var(--bad);background:#fbeaea}
  #status .big{font-size:22px;font-weight:600;margin:0}
  #status ul{margin:8px 0 0;padding-left:20px}
  .meta{display:flex;gap:18px;color:var(--muted);font-size:14px;margin-top:10px}
  table{width:100%;border-collapse:collapse;font-size:14px}
  th{font-weight:500;color:var(--muted);text-align:left;padding:4px 6px 6px;border-bottom:1px solid var(--line)}
  td{padding:7px 6px;border-bottom:1px solid var(--line);vertical-align:top}
  td a{color:var(--ink)}
  .empty{color:var(--muted);font-size:14px;margin:0}
  #logwrap{max-height:420px;overflow:auto}
  :focus-visible{outline:3px solid var(--warn);outline-offset:2px}
</style>
</head>
<body>
<header><span class="mark"></span><h1>Pemantau K3 langsung</h1></header>
<main>
  <section class="video"><img src="/video" alt="Tayangan kamera dengan hasil deteksi"></section>
  <aside class="side">
    <div id="status" class="panel" role="status" aria-live="polite">
      <p class="big" id="headline">Menghubungkan kamera...</p>
      <ul id="detail"></ul>
      <div class="meta"><span id="persons">Orang: -</span><span id="fps">FPS: -</span></div>
    </div>
    <div class="panel">
      <h2>Riwayat pelanggaran</h2>
      <div id="logwrap">
        <p class="empty" id="empty">Belum ada pelanggaran tercatat.</p>
        <table id="tbl" hidden>
          <thead><tr><th>Jam</th><th>Pelanggaran</th><th>Bukti</th></tr></thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </aside>
</main>
<script>
const $ = id => document.getElementById(id);
async function pollStatus(){
  try{
    const s = await (await fetch('/status')).json();
    const bad = s.violations.length > 0;
    $('status').classList.toggle('bad', bad);
    $('headline').textContent = bad ? 'Ada pelanggaran K3' : 'Semua sesuai standar';
    $('detail').innerHTML = s.violations.map(v => '<li>'+v+'</li>').join('');
    $('persons').textContent = 'Orang terdeteksi: ' + s.persons;
    $('fps').textContent = 'FPS: ' + s.fps;
  }catch(e){ $('headline').textContent = 'Koneksi ke server terputus'; }
}
async function pollLog(){
  try{
    const rows = await (await fetch('/log')).json();
    $('empty').hidden = rows.length > 0;
    $('tbl').hidden = rows.length === 0;
    $('tbl').querySelector('tbody').innerHTML = rows.map(r =>
      '<tr><td>'+r.waktu+'</td><td>'+r.pelanggaran+'</td><td><a href="/bukti/'+r.bukti+'" target="_blank">Lihat foto</a></td></tr>'
    ).join('');
  }catch(e){}
}
setInterval(pollStatus, 500);
setInterval(pollLog, 2000);
pollStatus(); pollLog();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    threading.Thread(target=detect_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, threaded=True)