# Aplikasi Peningkatan Mutu Citra

Proyek ini adalah aplikasi pengolahan citra digital untuk meningkatkan kualitas gambar menggunakan metode peningkatan mutu citra. Aplikasi menyediakan dua antarmuka:

- **Desktop GUI** berbasis PyQt5 (`main.py`)
- **Web GUI** berbasis Streamlit (`web_app.py`)

## Pembuat

1. Roy Adiyta - 2301020093  
2. Fadillah Nanda Maulana - 2301020088

## Latar Belakang

Pada pengolahan citra digital, kualitas citra sering menurun karena noise, pencahayaan yang tidak merata, atau ketajaman yang rendah. Aplikasi ini dibuat untuk membantu proses eksperimen dan pembelajaran peningkatan mutu citra secara interaktif.

## Tujuan

- Menerapkan konsep peningkatan mutu citra dalam bentuk aplikasi yang mudah digunakan.
- Menyediakan visualisasi perbandingan citra sebelum dan sesudah proses.
- Mempermudah analisis histogram sebagai dasar evaluasi distribusi intensitas citra.

## Fungsi Utama

### 1) Histogram Specification

Menyesuaikan distribusi intensitas citra input agar mendekati citra referensi.

### 2) Convolution (Mask Processing)

Melakukan filtering menggunakan kernel/mask:

- `None` (tanpa filter)
- `Smoothing`
- `Sharpening`
- `Sobel Edge`
- `Laplacian Edge`

## Struktur Berkas Penting

- `main.py` -> aplikasi Desktop GUI (PyQt5)
- `web_app.py` -> aplikasi Web GUI (Streamlit)
- `image_processing.py` -> fungsi inti pemrosesan citra
- `requirements.txt` -> daftar dependency Python

## Kebutuhan Sistem

- Python 3.9 atau lebih baru
- Pip (Python package manager)
- Sistem operasi Windows/Linux/macOS

## Instalasi

1. Buka terminal pada folder proyek.
2. (Opsional tapi disarankan) aktifkan virtual environment.
3. Install dependency:

```bash
pip install -r requirements.txt
```

## Cara Menjalankan

### Opsi A - Web GUI (Direkomendasikan)

```bash
python -m streamlit run web_app.py
```

Lalu buka URL yang muncul di terminal (umumnya `http://localhost:8501`).

### Opsi B - Desktop GUI

```bash
python main.py
```

## Cara Penggunaan

1. Upload gambar input.
2. (Opsional) Upload gambar referensi untuk Histogram Specification.
3. Pilih filter pada menu Convolution (Mask Processing).
4. Atur parameter:
   - `Kernel Size`
   - `Sharpen Strength`
5. Aktifkan/Nonaktifkan Histogram Specification sesuai kebutuhan.
6. Pilih mode histogram:
   - `RGB`
   - `Grayscale`
7. Lihat hasil sebelum/sesudah dan histogram.
8. Simpan hasil akhir ke file PNG.

## Troubleshooting

- **Perintah `streamlit` tidak dikenali**
  - Jalankan dengan:
  - `python -m streamlit run web_app.py`

- **Gagal membaca gambar**
  - Pastikan file gambar valid (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`).

- **Histogram specification aktif tapi hasil tidak berubah**
  - Pastikan gambar referensi sudah di-upload dan berbeda dengan gambar input.


