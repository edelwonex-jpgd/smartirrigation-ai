# SmartIrrigation AI — Future Farming Edition

Website **Streamlit** untuk demonstrasi klasifikasi kebutuhan irigasi tanaman cabai dengan **Random Forest**, menggunakan *dataset sintetis*. Desain dark-glassmorphism, grafik Plotly, evaluasi model, serta analisis CSV massal.

## Deployment gratis di Streamlit Cloud

1. Unduh ZIP lalu ekstrak. Upload **seluruh isi folder** (bukan ZIP) ke repository GitHub, melalui **Add file → Upload files**.
2. Pastikan `app.py`, `model.py`, `requirements.txt`, `generate_data.py` dan `data/synthetic_irrigation.csv` ada. Folder `.streamlit` berisi `config.toml` untuk tema.
3. Buka https://share.streamlit.io → **Create app** → pilih repository, branch `main`, main file `app.py` → **Deploy**.
4. Bila dataset Anda sebelumnya diunggah ke **root** (`synthetic_irrigation.csv`), `model.py` versi ini mendukung lokasi root maupun folder `data/`.
5. Sebaiknya upload **dengan mempertahankan struktur folder** untuk `.streamlit/config.toml` dan `tests/test_model.py`. Perhatikan GitHub drag-and-drop harus memasukkan folder, bukan hanya file satu per satu.

**Penting saat update dari versi sebelumnya:** Timpa minimal `app.py`, `model.py`, `requirements.txt`. `data/synthetic_irrigation.csv` lama tetap kompatibel. Konfigurasi tema `.streamlit/config.toml` sangat dianjurkan.

## Jalankan lokal

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Tes

```bash
python -m unittest discover -s tests -v
```

## Catatan ilmiah

Dataset sintetis berisi 2.400 baris yang dibuat dengan aturan hipotetis. Akurasi pada data uji sintetis **tidak** membuktikan ketepatan lapangan. Persentase kadar air input harus berupa **volumetric water content (VWC)**; nilai sensor perlu kalibrasi. Estimasi air net/bruto bersifat ilustratif, tidak memperhitungkan ET, infiltrasi atau hujan efektif. Tidak ada kendali pompa.
