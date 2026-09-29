import { HandLandmarker, FilesetResolver, DrawingUtils } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

/* ===== Konfigurasi ===== */
const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
// R: kelingking terangkat lalu diayun turun
const R_POSE_CLASS = "R_pose";
const R_DROP_THRESHOLD = 0.15;
const R_WINDOW_MS = 600;

// Z: lintasan pergelangan membentuk pola kanan-diagonal-kanan
const Z_WINDOW_MS = 1500;
const Z_MIN_SEGMENT = 0.05;
const CLASSES = ["Netral", ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ"]; // "Netral" = tangan diam/santai, bukan huruf
const TARGET = 200;        // sampel per kelas per sesi rekam
const SAMPLE_GAP_MS = 60;  // jeda antar sampel agar datanya bervariasi
const COUNTDOWN_S = 3;
const HOLD_MS = 500;       // huruf harus stabil selama ini sebelum masuk ke teks
const MIN_CONF = 0.98;
const N_FEATURES = 86;     // 2 tangan x 42 + selisih posisi pergelangan (x, y)
const DATA_KEY = "bisindo-data";
const MODEL_KEY = "localstorage://bisindo-model-v2";

/* ===== DOM & state ===== */
const $ = (id) => document.getElementById(id);
const vid = $("vid"), ov = $("ov"), ctx = ov.getContext("2d");
let landmarker, drawer, model = null, running = false, lastVideoTime = -1;
let data = JSON.parse(localStorage.getItem(DATA_KEY) || '{"X":[],"y":[]}');
let rec = null;            // { ci, n, go, last }
let text = "", cand = null, candSince = 0, locked = false;
let rTrail = [], zTrail = [];
let frames = 0, fpsTimer = performance.now();

const log = (t) => ($("log").textContent = t);
const say = (t) => { $("msg").style.display = "grid"; $("msg").textContent = t; };

/* ===== Fitur: landmark -> vektor 86 angka ===== */
// Tiap tangan dinormalisasi terhadap pergelangan dan skala, jadi tidak
// bergantung pada posisi/jarak. Slot dipisah berdasarkan Left/Right.
function toFeatures(res) {
  const slots = [null, null], wrist = [null, null];
  res.landmarks.forEach((lm, i) => {
    const s = res.handedness[i][0].categoryName === "Left" ? 0 : 1;
    if (slots[s]) return;
    const b = lm[0];
    const p = lm.flatMap((q) => [q.x - b.x, q.y - b.y]);
    const sc = Math.max(...p.map(Math.abs)) || 1;
    slots[s] = p.map((v) => v / sc);
    wrist[s] = b;
  });
  const zero = Array(42).fill(0);
  const both = wrist[0] && wrist[1];
  return [...(slots[0] || zero), ...(slots[1] || zero),
          both ? wrist[1].x - wrist[0].x : 0, both ? wrist[1].y - wrist[0].y : 0];
}

/* ===== Data ===== */
const saveData = () => localStorage.setItem(DATA_KEY, JSON.stringify(data));
const count = (ci) => data.y.filter((v) => v === ci).length;

function renderClasses() {
  const box = $("classes");
  box.innerHTML = "";
  CLASSES.forEach((name, ci) => {
    const b = document.createElement("button");
    b.className = "ghost";
    b.textContent = `${name} (${count(ci)})`;
    b.onclick = () => startRecording(ci);
    box.appendChild(b);
  });

  const select = $("delClass");
  const prev = select?.value ?? "0";
  select.innerHTML = "";
  CLASSES.forEach((name, ci) => {
    const option = document.createElement("option");
    option.value = String(ci);
    option.textContent = name;
    select.appendChild(option);
  });
  select.value = CLASSES.some((_, ci) => String(ci) === prev) ? prev : "0";
}

async function loadBundledDataset() {
  if (data.y.length) return; // never overwrite data the visitor already recorded
  try {
    const d = await (await fetch("data/bisindo-dataset1.json")).json();
    if (!d.classes.every((c, i) => c === CLASSES[i])) return;
    data = { X: d.X, y: d.y };
    saveData(); renderClasses();
    log(`Dataset bawaan dimuat: ${data.y.length} sampel.`);
  } catch {}
}
loadBundledDataset();

function startRecording(ci) {
  if (!running) return say("Mulai kamera dulu.");
  if (rec) return;
  rec = { ci, n: 0, go: false, last: 0 };
  let c = COUNTDOWN_S;
  say(`Siapkan isyarat "${CLASSES[ci]}"... ${c}`);
  const t = setInterval(() => {
    if (--c > 0) return say(`Siapkan isyarat "${CLASSES[ci]}"... ${c}`);
    clearInterval(t);
    $("msg").style.display = "none";
    rec.go = true;
  }, 1000);
}

function recordSample(f, now) {
  if (!rec?.go || now - rec.last < SAMPLE_GAP_MS) return;
  rec.last = now;
  data.X.push(f.map((v) => +v.toFixed(4)));
  data.y.push(rec.ci);
  $("letter").textContent = `Merekam "${CLASSES[rec.ci]}": ${++rec.n}/${TARGET}`;
  if (rec.n >= TARGET) {
    rec = null;
    saveData();
    renderClasses();
    $("letter").textContent = "Rekaman selesai";
  }
}

/* ===== Latih model (MLP) ===== */
async function train() {
  const present = new Set(data.y);
  if (data.y.length < 60 || present.size < 2) return log("Data belum cukup. Rekam minimal 2 kelas.");
  $("train").disabled = true;
  const idx = [...data.y.keys()].sort(() => Math.random() - 0.5); // acak sebelum validationSplit
  const xs = tf.tensor2d(idx.map((i) => data.X[i]));
  const ys = tf.oneHot(tf.tensor1d(idx.map((i) => data.y[i]), "int32"), CLASSES.length);
  model?.dispose();
  model = tf.sequential({ layers: [
    tf.layers.dense({ inputShape: [N_FEATURES], units: 64, activation: "relu" }),
    tf.layers.dropout({ rate: 0.2 }),
    tf.layers.dense({ units: 32, activation: "relu" }),
    tf.layers.dense({ units: CLASSES.length, activation: "softmax" }),
  ] });
  model.compile({ optimizer: tf.train.adam(0.005), loss: "categoricalCrossentropy", metrics: ["accuracy"] });
  await model.fit(xs, ys, {
    epochs: 60, batchSize: 32, validationSplit: 0.15,
    callbacks: { onEpochEnd: (e, l) =>
      log(`Epoch ${e + 1}/60 · loss ${l.loss.toFixed(3)} · akurasi latih ${(l.acc * 100).toFixed(0)}% · akurasi validasi ${((l.val_acc ?? 0) * 100).toFixed(0)}%`) },
  });
  await model.save(MODEL_KEY);
  xs.dispose(); ys.dispose();
  $("train").disabled = false;
  $("state").textContent = "Model siap";
}

/* ===== Prediksi & penyusun teks ===== */
function predict(f) {
  const t = tf.tidy(() => model.predict(tf.tensor2d([f])));
  const p = t.dataSync();
  t.dispose();
  let best = 0;
  p.forEach((v, i) => { if (v > p[best]) best = i; });
  return { label: CLASSES[best], conf: p[best] };
}

function showPrediction({ label, conf }) {
  $("pct").textContent = Math.round(conf * 100) + "%";
  $("arc").style.strokeDashoffset = 314 * (1 - conf);
  $("letter").textContent = conf >= MIN_CONF ? `Terdeteksi: ${label}` : "Belum yakin";
  $("badge").classList.toggle("idle", conf < MIN_CONF || label === "Netral");
}

function buildText(label, conf, now) {
  if (!label || conf < MIN_CONF || label === "Netral") { cand = null; locked = false; return; }
  if (label !== cand) { cand = label; candSince = now; locked = false; return; }
  if (!locked && now - candSince >= HOLD_MS) {
    text += label; locked = true;
    $("out").textContent = text;
  }
}

// Bandingkan posisi pinky relatif wrist agar perubahan ukuran tangan tidak dominan.
function detectR(landmarks, now) {
  const wrist = landmarks[0];
  const middleTip = landmarks[12];
  const pinkyTip = landmarks[20];
  const handHeight = Math.hypot(middleTip.x - wrist.x, middleTip.y - wrist.y) || 1;
  const position = (pinkyTip.y - wrist.y) / handHeight;
  rTrail = rTrail.filter((point) => now - point.time <= R_WINDOW_MS);
  if (rTrail.length && position - rTrail[0].position >= R_DROP_THRESHOLD) {
    rTrail = [];
    return true;
  }
  rTrail.push({ position, time: now });
  return false;
}

// Koordinat gambar: Z dibaca sebagai kanan, diagonal kiri-bawah, lalu kanan.
function detectZ(wrist, now) {
  zTrail = zTrail.filter((point) => now - point.time <= Z_WINDOW_MS);
  zTrail.push({ x: wrist.x, y: wrist.y, time: now });

  for (let i = 0; i < zTrail.length - 3; i++) {
    const start = zTrail[i];
    for (let j = i + 1; j < zTrail.length - 2; j++) {
      const top = zTrail[j];
      const dx1 = top.x - start.x, dy1 = top.y - start.y;
      if (dx1 < Z_MIN_SEGMENT || Math.abs(dy1) > Z_MIN_SEGMENT) continue;
      for (let k = j + 1; k < zTrail.length - 1; k++) {
        const diagonal = zTrail[k];
        const dx2 = diagonal.x - top.x, dy2 = diagonal.y - top.y;
        if (dx2 > -Z_MIN_SEGMENT || dy2 < Z_MIN_SEGMENT) continue;
        for (let l = k + 1; l < zTrail.length; l++) {
          const end = zTrail[l];
          const dx3 = end.x - diagonal.x, dy3 = end.y - diagonal.y;
          if (dx3 >= Z_MIN_SEGMENT && Math.abs(dy3) <= Z_MIN_SEGMENT) {
            zTrail = [];
            return true;
          }
        }
      }
    }
  }
  return false;
}

/* ===== Loop ===== */
function loop() {
  if (!running) return;
  requestAnimationFrame(loop);
  if (vid.readyState < 2 || vid.currentTime === lastVideoTime) return;
  lastVideoTime = vid.currentTime;
  const now = performance.now();
  if (ov.width !== vid.videoWidth) { ov.width = vid.videoWidth; ov.height = vid.videoHeight; }

  const res = landmarker.detectForVideo(vid, now);
  ctx.clearRect(0, 0, ov.width, ov.height);
  res.landmarks.forEach((lm) => {
    drawer.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#39ff9c", lineWidth: 3 });
    drawer.drawLandmarks(lm, { color: "#3d8bff", fillColor: "#0a1a3a", radius: 4 });
  });

  if (res.landmarks.length) {
    const f = toFeatures(res);
    recordSample(f, now);
    if (model && !rec) {
      const r = predict(f);
      showPrediction(r);
      if (r.label === R_POSE_CLASS && r.conf >= MIN_CONF) {
        const detected = detectR(res.landmarks[0], now);
        buildText(null, 0, now);
        if (detected) {
          cand = "R"; candSince = now - HOLD_MS; locked = false;
          buildText("R", 1, now);
        }
      } else {
        rTrail = [];
        buildText(r.label === R_POSE_CLASS ? null : r.label, r.conf, now);
      }
    } else {
      rTrail = [];
    }
    if (detectZ(res.landmarks[0][0], now)) {
      cand = "Z"; candSince = now - HOLD_MS; locked = false;
      buildText("Z", 1, now);
    }
  } else {
    rTrail = [];
    zTrail = [];
    buildText(null, 0, now);
    if (!rec) $("letter").textContent = "Tangan belum terdeteksi";
  }

  frames++;
  if (now - fpsTimer >= 1000) { $("fps").textContent = `${Math.round((frames * 1000) / (now - fpsTimer))} FPS`; frames = 0; fpsTimer = now; }
}

/* ===== Kamera ===== */
async function startCamera() {
  $("start").disabled = true;
  say("Memuat model tangan...");
  try {
    const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO", numHands: 2,
      minHandDetectionConfidence: 0.3, minHandPresenceConfidence: 0.3, minTrackingConfidence: 0.3,
    });
    drawer = new DrawingUtils(ctx);
    vid.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: { ideal: 30 } } });
    await vid.play();
    $("msg").style.display = "none";
    running = true;
    loop();
  } catch (e) {
    say(`Gagal: ${e.message}. Buka lewat localhost/HTTPS dan izinkan kamera.`);
    $("start").disabled = false;
  }
}

/* ===== Event ===== */
$("start").onclick = startCamera;
$("train").onclick = train;
$("dl").onclick = () =>
  model ? model.save("downloads://bisindo-model") : log("Belum ada model. Latih model dulu.");
$("space").onclick = () => { text += " "; $("out").textContent = text; };
$("del").onclick = () => { text = text.slice(0, -1); $("out").textContent = text; };
$("clr").onclick = () => { text = ""; $("out").textContent = ""; };
$("reset").onclick = () => {
  if (!confirm("Hapus semua data latih?")) return;
  data = { X: [], y: [] }; saveData(); renderClasses();
};
$("delOne").onclick = () => {
  const ci = Number($("delClass").value);
  const name = CLASSES[ci];
  const total = count(ci);
  if (total === 0) {
    log(`Kelas "${name}" belum punya data.`);
    return;
  }
  if (!confirm(`Hapus ${total} sampel kelas "${name}"?`)) return;
  data.X = data.X.filter((_, i) => data.y[i] !== ci);
  data.y = data.y.filter((v) => v !== ci);
  saveData();
  renderClasses();
  log(`Data kelas "${name}" dihapus. Latih ulang model.`);
};
$("export").onclick = () => {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify({ classes: CLASSES, ...data })], { type: "application/json" }));
  a.download = "bisindo-dataset.json";
  a.click();
};
$("import").onclick = () => $("file").click();
$("file").onchange = async (e) => {
  try {
    const d = JSON.parse(await e.target.files[0].text());
    if (!d.classes.every((c, i) => c === CLASSES[i])) throw new Error("urutan kelas tidak cocok");
    data = { X: d.X, y: d.y }; saveData(); renderClasses();
    log(`Data dimuat: ${data.y.length} sampel.`);
  } catch (err) { log("Gagal impor: " + err.message); }
};

renderClasses();
async function initModel() {
  let loaded = false;
  for (const src of [MODEL_KEY, "model/bisindo-model(3).json"]) {
    try {
      const m = await tf.loadLayersModel(src);
      if (m.outputs[0].shape[1] !== CLASSES.length) continue; // class count mismatch: skip this model
      model = m;
      $("state").textContent = "Model siap";
      loaded = true;
      return;
    } catch {}
  }
  if (!loaded) log("Daftar kelas berubah. Latih ulang model.");
}
initModel();
