console.log("✅ app.js running");

let model;
const video = document.getElementById("camera");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
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
  console.log("Inputs:", model.inputs);
  console.log("Outputs:", model.outputs);
}

/* ---------- YOLO DECODE (ROBUST) ---------- */
function decodeYOLO(data, shape) {
  const detections = [];

  // CASE 1: [1, N, 6] → x,y,w,h,conf,class
  if (shape.length === 3 && shape[2] === 6) {
    for (let i = 0; i < shape[1]; i++) {
      const base = i * 6;
      const x = data[base];
      const y = data[base + 1];
      const w = data[base + 2];
      const h = data[base + 3];
      const conf = data[base + 4];
      const cls = data[base + 5];

      if (conf > 0.4) {
        detections.push({ x, y, w, h, conf, cls });
      }
    }
  }

  // CASE 2: [1, N, 85] → YOLOv5/8 COCO-style
  if (shape.length === 3 && shape[2] > 6) {
    const numClasses = shape[2] - 5;

    for (let i = 0; i < shape[1]; i++) {
      const base = i * shape[2];
      const x = data[base];
      const y = data[base + 1];
      const w = data[base + 2];
      const h = data[base + 3];
      const objConf = data[base + 4];

      let maxClass = -1;
      let maxScore = 0;

      for (let c = 0; c < numClasses; c++) {
        const score = data[base + 5 + c];
        if (score > maxScore) {
          maxScore = score;
          maxClass = c;
        }
      }

      const conf = objConf * maxScore;
      if (conf > 0.4) {
        detections.push({ x, y, w, h, conf, cls: maxClass });
      }
    }
  }

  return detections;
}

/* ---------- INFERENCE ---------- */
async function runInference() {
  if (!model || video.readyState < 2) return;

  const input = tf.tidy(() =>
    tf.browser.fromPixels(video)
      .resizeNearestNeighbor([640, 640])
      .toFloat()
      .div(255)
      .expandDims(0)
  );

  const output = await model.executeAsync(input);
  const tensor = Array.isArray(output) ? output[0] : output;

  const data = await tensor.data();
  const shape = tensor.shape;

  console.log("🧠 Output shape:", shape);
  console.log("🧠 Sample values:", data.slice(0, 10));

  const detections = decodeYOLO(data, shape);

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  detections.forEach(det => {
    const bx = det.x * canvas.width;
    const by = det.y * canvas.height;
    const bw = det.w * canvas.width;
    const bh = det.h * canvas.height;

    ctx.strokeStyle = "red";
    ctx.lineWidth = 2;
    ctx.strokeRect(
      bx - bw / 2,
      by - bh / 2,
      bw,
      bh
    );

    ctx.fillStyle = "red";
    ctx.fillText(
      `conf:${det.conf.toFixed(2)}`,
      bx,
      by - 4
    );
  });

  tf.dispose([input, ...(Array.isArray(output) ? output : [output])]);
}

/* ---------- LOOP ---------- */
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
    "Status: running YOLO detection";
  startRealtime();
}

init();
