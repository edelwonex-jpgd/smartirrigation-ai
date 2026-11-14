# 🌱 SmartIrrigation AI

**Prototipe aplikasi web AI untuk edukasi agroekoteknologi tanaman cabai.** Menggunakan model `RandomForestClassifier` yang dilatih pada **data sintetis**, bukan data sensor nyata. Termasuk formulir input, prediksi kelas kebutuhan irigasi, estimasi defisit kadar air, analisis massal CSV, evaluasi model, dan unduh hasil.



## Struktur

- `app.py`: antarmuka Streamlit
- `model.py`: generator dataset, validasi, pelatihan dan prediksi ML
- `generate_data.py`: skrip membuat ulang dataset deterministik
- `data/synthetic_irrigation.csv`: dataset latihan sintetis (2.400 baris)
- `requirements.txt`: paket Python
- `.streamlit/config.toml`: tema aplikasi
- `tests/test_model.py`: pengujian model

## Jalankan secara lokal (Windows / macOS / Linux)

Instal Python 3.11 atau 3.12, buka terminal di folder proyek, lalu:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Buka URL localhost yang ditampilkan terminal (biasanya `http://localhost:8501`). Model dilatih ulang dari CSV saat aplikasi pertama kali dijalankan, lalu di-cache selama sesi server. Tidak ada file pickle/joblib yang harus dipercaya atau diunggah.



```

## Dataset & pendekatan

Dataset dibuat dengan NumPy RNG seed 42, 2.400 baris. Variabel: kadar air tanah volumetrik (%), kapasitas lapang (%), titik layu permanen (%), suhu (°C), prakiraan hujan (mm), kedalaman akar (cm), fase pertumbuhan. Label kelas dibuat dengan aturan simulasi yang menggunakan fraksi air tersedia dan faktor penyesuaian sederhana. Model mempelajari label tersebut menggunakan Random Forest, dilatih pada 80% baris dan diuji pada 20% baris dengan stratifikasi. Hasil evaluasi hanya mengukur peniruan aturan generator data.

Perkiraan defisit air memakai `max(0, FC - theta) * depth_cm / 10` dalam satuan mm; kedalaman bruto dibagi 0,85. Persentase harus bersatuan **volumetric water content**, bukan persentase relatif keluaran sensor belum dikalibrasi. Rumus ini tidak mensimulasikan neraca air harian dan bukan penentuan jadwal irigasi.

## Pengembangan untuk penelitian nyata

1. Kalibrasi ESP32 dan sensor kelembapan ke kadar air volumetrik pada tanah yang diteliti.
2. Ukur kapasitas lapang, titik layu, kedalaman akar efektif, suhu, hujan dan hasil produksi secara nyata.
3. Susun label keputusan irigasi berdasarkan protokol eksperimen dan pengukuran; jangan memakai label sintetis untuk klaim efektivitas agronomis.
4. Lakukan validasi pada musim/lokasi dan data uji terpisah. Catat metrik F1, precision-recall serta dampak perlakuan terhadap pertumbuhan/hasil.
5. Jika kelak mengotomatisasi pompa, tambahkan kontrol keselamatan, batas waktu, sensor fail-safe, dan override manual.

## Referensi deployment

- https://docs.streamlit.io/deploy/streamlit-community-cloud
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
