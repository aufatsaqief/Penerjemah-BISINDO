# Penerjemah BISINDO

Proyek web yang mengenali huruf bahasa isyarat BISINDO. MediaPipe HandLandmarker mengambil 21 titik per tangan (dua tangan didukung), lalu titik-titik itu diubah menjadi 86 angka fitur. Anda merekam contoh sendiri per huruf, melatih model MLP kecil dengan TensorFlow.js di browser, lalu berisyarat. Huruf yang stabil dan cukup yakin masuk ke kotak teks hasil.

Fitur: kelas "Netral" agar model tidak menebak saat tangan diam, ekspor dan impor data JSON, hapus data, dan cara menambah huruf baru..
## Fitur
- Antarmuka web sederhana
- Kamus data BISINDO
- Penerjemahan ke huruf otomatis

## Cara menjalankan
1. Buka file `index.html` di browser.
2. Atau jalankan server statis lokal jika diperlukan.

## Teknologi
- HTML
- CSS
- JavaScript
- TensorFlow.js
-  MediaPipe HandLandmarker
