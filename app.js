console.log("✅ app.js running on phone");

let model;
let video = document.getElementById("camera");
let canvas = document.getElementById("overlay");
let ctx = canvas.getContext("2d");
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

/* ---------- INFERENCE ---------- */
async function runInference() {
  if (!model || video.readyState < 2) return;

  const imgTensor = tf.tidy(() => {
    return tf.browser.fromPixels(video)
      .resizeNearestNeighbor([640, 640])
      .toFloat()
      .div(255.0)
      .expandDims(0);
  });

  await model.executeAsync(imgTensor);

  // 🔥 CLEAR PREVIOUS DRAWINGS
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 🟢 TEMP PROOF: GREEN BOX (REAL-TIME)
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;
  ctx.strokeRect(60, 40, 180, 140);

  tf.dispose(imgTensor);
}

/* ---------- REAL-TIME LOOP ---------- */
function startRealtime() {
  if (running) return;
  running = true;

  async function loop() {
    if (!running) return;
    await runInference();
    requestAnimationFrame(loop);
  }

  loop();
}

/* ---------- INIT ---------- */
async function init() {
  await setupCamera();
  await loadModel();
  document.getElementById("status").innerText = "Status: running real-time inference";
  startRealtime();
}

init();
