# Penerjemah BISINDO

Proyek demo pengenal isyarat BISINDO untuk huruf terbatas A–E. Aplikasi ini menggunakan MediaPipe HandLandmarker untuk menangkap landmark tangan dan model MLP TensorFlow.js untuk klasifikasi huruf di browser. Semua pemrosesan video berlangsung secara lokal di perangkat pengguna; tidak ada unggahan video atau data ke server.

Catatan penting: demo ini  bukan pengganti penerjemah bahasa isyarat yang profesional. Verifikasi isyarat BISINDO tetap harus dilakukan dengan sumber resmi atau penutur asli.

## Cara menjalankan lokal
1. Buka folder proyek di editor.
2. Jalankan server lokal seperti Live Server.
3. Pastikan aplikasi dibuka lewat http://localhost, bukan file:// karena kamera tidak akan bekerja pada file static mentah.
4. Klik "Mulai kamera" lalu rekam data untuk kelas yang tersedia.

## Rekam, latih, dan unduh model
1. Pilih kelas seperti A, B, C, D, atau E dan klik untuk merekam sampel.
2. Setelah data cukup, klik "Latih model".
3. Setelah model siap, klik "Unduh model" untuk menyimpan file model di perangkat.
4. Letakkan file hasil unduhan ke folder model/ di repositori:
   - model/bisindo-model.json
   - model/bisindo-model.weights.bin
5. Jika Anda ingin memakai dataset bawaan, letakkan file dataset di folder data/ dengan nama:
   - data/bisindo-dataset.json


## Catatan keamanan & etika
- Video diproses di browser lokal dan tidak diunggah.
- Ini adalah demo terbatas untuk pendekatan edukasi dan eksperimen.
- Untuk kebutuhan komunikasi nyata, gunakan sumber resmi atau penutur BISINDO asli untuk validasi isyarat.

  
