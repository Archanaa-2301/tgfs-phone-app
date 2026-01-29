console.log("✅ app.js running");

let model;
let video = document.getElementById("camera");
let canvas = document.getElementById("overlay");
let ctx = canvas.getContext("2d");
let lastFrame = null;
let lastBox = null;
let running = false;

/* ---------- CAMERA ---------- */
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false
  });
  video.srcObject = stream;

  return new Promise(resolve => {
    video.onloadedmetadata = () => resolve();
  });
}

/* ---------- MODEL ---------- */
async function loadModel() {
  model = await tf.loadGraphModel("./tfjs_model/model.json");
  console.log("✅ Model loaded");
}

/* ---------- LIGHT MOTION CHECK ---------- */
function hasSceneChanged(current, threshold = 25) {
  if (!lastFrame) {
    lastFrame = current;
    return true;
  }

  let diff = 0;
  for (let i = 0; i < current.length; i += 40) {
    diff += Math.abs(current[i] - lastFrame[i]);
  }

  lastFrame = current;
  return diff > threshold * 100;
}

/* ---------- INFERENCE ---------- */
async function runInference() {
  if (!model || video.readyState < 2) return;

  /* 🔹 Cheap frame check (VERY FAST) */
  const tiny = tf.tidy(() =>
    tf.browser.fromPixels(video)
      .resizeNearestNeighbor([64, 64])
      .mean(2)
      .toFloat()
  );

  const frameData = await tiny.data();
  tf.dispose(tiny);

  if (!hasSceneChanged(frameData)) {
    drawLastBox();
    return;
  }

  /* 🔹 Heavy YOLO only when needed */
  const input = tf.tidy(() =>
    tf.browser.fromPixels(video)
      .resizeNearestNeighbor([416, 416])
      .toFloat()
      .div(255)
      .expandDims(0)
  );

  await model.executeAsync(input);
  tf.dispose(input);

  /* 🔹 TEMP: simulated detection result */
  lastBox = { x: 70, y: 50, w: 160, h: 120 };
  drawLastBox();
}

/* ---------- DRAW ---------- */
function drawLastBox() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!lastBox) return;

  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;
  ctx.strokeRect(lastBox.x, lastBox.y, lastBox.w, lastBox.h);
}

/* ---------- REALTIME LOOP ---------- */
function startRealtime() {
  if (running) return;
  running = true;

  async function loop() {
    await runInference();
    requestAnimationFrame(loop);
  }

  loop();
}

/* ---------- INIT ---------- */
async function init() {
  await setupCamera();
  await loadModel();
  document.getElementById("status").innerText =
    "Status: real-time optimized inference";
  startRealtime();
}

init();
