# Penerjemah BISINDO

Proyek demo pengenal isyarat BISINDO untuk huruf alfabet. Aplikasi ini menggunakan MediaPipe HandLandmarker untuk menangkap landmark tangan dan model MLP TensorFlow.js untuk klasifikasi huruf di browser. Semua pemrosesan video berlangsung secara lokal di perangkat pengguna; tidak ada unggahan video atau data ke server.

Catatan penting: demo ini hanya untuk eksperimen pengenalan huruf dan bukan pengganti penerjemah bahasa isyarat profesional. Verifikasi isyarat BISINDO tetap harus dilakukan dengan sumber resmi atau penutur asli.

## Panduan BISINDO
Gambar berikut adalah panduan visual untuk menggunakan BISINDO.

![Panduan BISINDO](Panduan%20BISINDO.png)

Bentuk R dan Z pada proyek ini memiliki penyesuaian berikut:
- R: gunakan tangan kiri, luruskan kelingking secara mendatar, dan genggam jari-jari lainnya.
- Z: rapatkan atau kerucutkan jari-jari, lalu gerakkan tangan mengikuti lintasan Z. Deteksi menggunakan landmark tangan, bukan lengan.

Penyesuaian ini merupakan panduan khusus proyek. Pastikan bentuk huruf BISINDO diverifikasi dengan sumber resmi atau penutur BISINDO.

## Cara menjalankan lokal
1. Buka folder proyek di editor.
2. Jalankan server lokal seperti Live Server.
3. Pastikan aplikasi dibuka lewat http://localhost, bukan file:// karena kamera tidak akan bekerja pada file static mentah.
4. Klik "Mulai kamera" lalu rekam data untuk kelas yang tersedia.

## Rekam, latih, dan unduh model
1. Pilih kelas huruf yang tersedia dan klik untuk merekam sampel.
2. Setelah data cukup, klik "Latih model".
3. Setelah model siap, klik "Unduh model" untuk menyimpan file model di perangkat.
4. Model yang digunakan aplikasi tersedia di repositori:
   - model/bisindo-model(3).json
   - model/bisindo-model.weights(3).bin
5. Dataset bawaan tersedia di:
   - data/bisindo-dataset1.json


## Catatan keamanan & etika
- Video diproses di browser lokal dan tidak diunggah.
- Ini adalah demo terbatas untuk pendekatan edukasi dan eksperimen.
- Untuk kebutuhan komunikasi nyata, gunakan sumber resmi atau penutur BISINDO asli untuk validasi isyarat.

  
