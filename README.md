# Sistem Deteksi Pelanggaran K3 Berbasis Web

Sistem pemantauan APD (Alat Pelindung Diri) secara real-time menggunakan YOLOv11 dan Flask. Mendeteksi pelanggaran K3 seperti tidak memakai helm, baju praktek, atau sepatu safety, lalu menyimpan bukti foto dan log CSV secara otomatis.

---

## Struktur Proyek

```
K3_APD/
├── app.py                      # Aplikasi web deteksi real-time (file utama)
├── requirements.txt            # Daftar library yang dibutuhkan
├── README.md                   # Dokumentasi ini
│
├── models/
│   └── best.pt                 # Model YOLOv11 hasil training
│
├── gambar_mentah/              # Video bahan mentah untuk training
│   ├── video1_lengkap.mp4
│   ├── video2_tanpa_helm.mp4
│   ├── video3_tanpa_baju.mp4
│   ├── video4_tanpa_sepatu.mp4
│   └── video5_tanpa_apd.mp4
│
├── hasil_label/                # Dataset hasil export Roboflow (YOLOv11)
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── data.yaml
│
├── training/                   # Hasil training untuk laporan
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── val_batch0_pred.jpg
│   └── K3_APD.ipynb            # Notebook Google Colab training
│
├── demo/                       # Bahan tes dan hasil demo
│   ├── video_tes.mp4           # Video baru (bukan dari data training)
│   └── rekaman_demo.mp4        # Rekam layar browser saat demo
│
└── output/                     # Dibuat otomatis saat app.py berjalan
    ├── bukti/                  # Foto pelanggaran (*.jpg)
    └── log_pelanggaran.csv     # Catatan seluruh pelanggaran
```

---

## Cara Instalasi

### 1. Prasyarat
- Python 3.9 atau lebih baru
- Webcam atau kamera (bisa juga pakai file video)

### 2. Install Library

```bash
pip install -r requirements.txt
```

### 3. Letakkan Model

Pastikan file model berada di:
```
models/best.pt
```

---

## Cara Menjalankan

```bash
python app.py
```

Lalu buka browser dan akses:

| Perangkat | URL |
|-----------|-----|
| Laptop sendiri | http://localhost:5000 |
| HP / laptop lain (1 jaringan) | http://IP-LAPTOP:5000 |

---

## Pengaturan (`app.py`)

Edit bagian `PENGATURAN` di baris atas `app.py`:

| Parameter | Default | Keterangan |
|-----------|---------|------------|
| `MODEL_PATH` | `models/best.pt` | Lokasi file model |
| `SOURCE` | `0` | `0` = webcam, path video, atau `rtsp://...` untuk CCTV |
| `CONF` | `0.6` | Ambang batas confidence (0.0–1.0) |
| `MIN_TINGGI_PERSON` | `0.35` | Minimal tinggi orang (35% frame) agar dihitung valid |
| `COOLDOWN` | `5` | Jeda detik antar penyimpanan bukti pelanggaran |

---

## Kelas Pelanggaran yang Dideteksi

| Kelas Model | Label Tampilan |
|-------------|----------------|
| `no_helmet` | Tidak pakai helm |
| `no_coverall` | Tidak pakai baju praktek |
| `no_safety_shoes` | Tidak pakai sepatu safety |

---

## Output Otomatis

Setiap pelanggaran terdeteksi akan menghasilkan:

- **Foto bukti** → disimpan di `output/bukti/YYYYMMDD_HHMMSS.jpg`
- **Log CSV** → dicatat di `output/log_pelanggaran.csv` dengan kolom:
  - `waktu` — timestamp ISO
  - `pelanggaran` — jenis pelanggaran
  - `file_bukti` — nama file foto

---

## Endpoint API

| Endpoint | Keterangan |
|----------|------------|
| `GET /` | Halaman web pemantauan |
| `GET /video` | Stream MJPEG kamera |
| `GET /status` | Status deteksi saat ini (JSON) |
| `GET /log` | 50 riwayat pelanggaran terakhir (JSON) |
| `GET /bukti/<nama>` | Akses foto bukti pelanggaran |

---

## Lisensi

Proyek ini dibuat untuk keperluan tugas akhir / laporan. Tidak untuk distribusi komersial.
