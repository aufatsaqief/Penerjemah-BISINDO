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

## Deploy ke GitHub Pages
Repository GitHub Anda saat ini adalah:
- `https://github.com/aufatsaqief/Penerjemah-BISINDO.git`

Maka URL GitHub Pages yang benar adalah:
- `https://aufatsaqief.github.io/Penerjemah-BISINDO/`

Langkah deploy:
1. Pastikan repo sudah tersimpan di GitHub dan branch default adalah `main`.
2. Buka repository GitHub di browser.
3. Masuk ke `Settings` > `Pages`.
4. Pilih `Source: GitHub Actions`.
5. Push ke branch `main`, lalu workflow akan otomatis deploy.

## Teknologi
- HTML
- CSS
- JavaScript
- TensorFlow.js
-  MediaPipe HandLandmarker
