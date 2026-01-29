console.log("✅ app.js running on phone");

let model;
let video = document.getElementById("camera");

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

/* ---------- INFERENCE ---------- */
async function runInference() {
  console.log("▶ Running inference");

  const imgTensor = tf.tidy(() => {
    return tf.browser.fromPixels(video)
      .resizeNearestNeighbor([640, 640])
      .toFloat()
      .div(255.0)
      .expandDims(0);
  });

  const output = await model.executeAsync(imgTensor);

  console.log("✅ Output:", output);

  tf.dispose(imgTensor);
  if (Array.isArray(output)) output.forEach(t => t.dispose());
  else output.dispose();
}

/* ---------- INIT ---------- */
async function init() {
  await setupCamera();
  await loadModel();
  console.log("🚀 Ready on phone");
}

init();
