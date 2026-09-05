import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ── DOM refs ──────────────────────────────────────────────────────────
const container = document.getElementById("canvas-container");
const queryInput = document.getElementById("query-input");
const queryBtn = document.getElementById("btn-query");
const statusText = document.getElementById("query-status");
const uploadInput = document.getElementById("upload-input");
const uploadBtn = document.getElementById("btn-upload");
const legendMin = document.getElementById("legend-label-min");
const legendMax = document.getElementById("legend-label-max");
const infoValue = document.getElementById("info-value");
const infoLabel = document.getElementById("info-label");
const loadingSpinner = document.getElementById("loading-overlay");

// ── State ─────────────────────────────────────────────────────────────
let taskId = null;
let points = null; // THREE.Points — semantic point cloud
let currentMin = 0;
let currentMax = 1;
let instanceBoxes = []; // THREE.Box3Helper[]
let pollingTimer = null;

// ── Scene, camera, renderer, controls ──────────────────────────────────
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f2f5);

const camera = new THREE.PerspectiveCamera(
  60,
  container.clientWidth / container.clientHeight,
  0.01,
  1000,
);
camera.position.set(0, 0, 5);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

window.addEventListener("resize", () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ── Helpers ───────────────────────────────────────────────────────────

function showLoading() { loadingSpinner.classList.add("active"); }
function hideLoading() { loadingSpinner.classList.remove("active"); }
function setStatus(msg) { statusText.textContent = msg; }
function setInfo(msg) { infoValue.textContent = msg; }

function parseFloat32Array(response) {
  return response.arrayBuffer().then((buf) => new Float32Array(buf));
}

function enableQueryUI(enabled) {
  queryInput.disabled = !enabled;
  queryBtn.disabled = !enabled;
  instanceBtn.disabled = !enabled;
}

// ── Render point cloud ────────────────────────────────────────────────

function renderPointCloud(xyz, meta, colorsIn = null) {
  const n = meta.num_points;

  if (points) {
    points.geometry.dispose();
    points.material.dispose();
    scene.remove(points);
  }

  instanceBoxes.forEach(box => scene.remove(box));
  instanceBoxes = [];

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(xyz, 3));

  let colorArr;
  if (colorsIn && colorsIn.length === n * 3) {
    colorArr = colorsIn;
  } else {
    colorArr = new Float32Array(n * 3);
    colorArr.fill(0.5);
  }
  geom.setAttribute("color", new THREE.BufferAttribute(colorArr, 3));

  const mat = new THREE.PointsMaterial({
    size: 0.02,
    vertexColors: true,
    sizeAttenuation: true,
  });

  points = new THREE.Points(geom, mat);
  scene.add(points);

  const cx = (meta.bbox_min[0] + meta.bbox_max[0]) / 2;
  const cy = (meta.bbox_min[1] + meta.bbox_max[1]) / 2;
  const cz = (meta.bbox_min[2] + meta.bbox_max[2]) / 2;
  const dx = meta.bbox_max[0] - meta.bbox_min[0];
  const dy = meta.bbox_max[1] - meta.bbox_min[1];
  const dz = meta.bbox_max[2] - meta.bbox_min[2];
  const diag = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1;

  controls.target.set(cx, cy, cz);
  camera.position.set(cx + diag * 1.5, cy + diag * 0.5, cz + diag * 1.5);
  camera.lookAt(cx, cy, cz);
  controls.update();

  setInfo(`Points: ${n.toLocaleString()}`);
}

// ── Load point cloud ──────────────────────────────────────────────────

async function loadPointCloud(tid) {
  setStatus("Loading point cloud...");
  try {
    const [metaResp, xyzResp, colorsResp] = await Promise.all([
      fetch(`/api/task/${tid}/meta`),
      fetch(`/api/task/${tid}/xyz`),
      fetch(`/api/task/${tid}/colors`),
    ]);
    if (!metaResp.ok || !xyzResp.ok) {
      setStatus("Point cloud not ready yet — waiting...");
      return false;
    }
    const meta = await metaResp.json();
    const xyz = await parseFloat32Array(xyzResp);
    let colors = null;
    if (colorsResp.ok) {
      colors = await parseFloat32Array(colorsResp);
    }
    renderPointCloud(xyz, meta, colors);
    enableQueryUI(true);
    setStatus(`Loaded ${meta.num_points.toLocaleString()} points. Ready for queries.`);
    return true;
  } catch (err) {
    setStatus(`Point cloud load error: ${err.message}`);
    return false;
  }
}

// ── Poll for task completion ──────────────────────────────────────────

async function pollUntilReady(tid) {
  setStatus("Waiting for reconstruction to complete...");
  const maxPolls = 300; // ~5 minutes at 1s intervals
  for (let i = 0; i < maxPolls; i++) {
    try {
      const resp = await fetch(`/api/task/${tid}/status`);
      if (!resp.ok) break;
      const data = await resp.json();

      setInfo(`${data.status} — ${data.stage || ""}`);

      if (data.status === "ERROR") {
        setStatus(`Reconstruction failed: ${data.error_message || "unknown error"}`);
        return false;
      }
      if (data.status === "SUCCESS" && data.has_semantic) {
        setStatus("Reconstruction complete. Loading data...");
        return true;
      }
    } catch (e) {
      // continue polling
    }
    await new Promise(r => setTimeout(r, 1000));
  }
  setStatus("Timed out waiting for reconstruction.");
  return false;
}

// ── Upload zip ────────────────────────────────────────────────────────

async function uploadZip() {
  const file = uploadInput.files[0];
  if (!file) { setStatus("Select a .zip file to upload."); return; }
  if (!file.name.toLowerCase().endsWith(".zip")) {
    setStatus("File must be a .zip archive.");
    return;
  }

  showLoading();
  setStatus("Uploading and submitting reconstruction...");
  setInfo("Uploading...");
  enableQueryUI(false);

  try {
    const formData = new FormData();
    formData.append("zip_file", file);

    const resp = await fetch("/api/upload", { method: "POST", body: formData });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      setStatus(`Upload failed: ${err.error || resp.statusText}`);
      setInfo("Upload failed");
      hideLoading();
      return;
    }

    const result = await resp.json();
    taskId = result.task_id;
    infoLabel.textContent = "TASK";
    setInfo(`ID: ${taskId.slice(0, 8)}… — Submitted`);

    // Poll backend until reconstruction completes
    const ready = await pollUntilReady(taskId);
    if (!ready) { hideLoading(); return; }

    // Load semantic point cloud
    await loadPointCloud(taskId);

    setStatus("Done. Ready for queries.");
  } catch (err) {
    setStatus(`Upload error: ${err.message}`);
    setInfo("Upload error");
  } finally {
    hideLoading();
  }
}

// ── Render bounding boxes ─────────────────────────────────────────────

function renderInstanceBoxes(objects) {
  instanceBoxes.forEach(box => scene.remove(box));
  instanceBoxes = [];

  const palette = [
    0xff6b6b, 0x4ecdc4, 0x45b7d1, 0xf9ca24, 0x6ab04c,
    0xeb4d4b, 0x7ed6df, 0xe056a0, 0x686de0, 0x30336b,
  ];

  objects.forEach((object, index) => {
    const aabb = object.aabb;
    if (!aabb?.min || !aabb?.max) return;
    const color = new THREE.Color(palette[index % palette.length]);
    const box = new THREE.Box3(
      new THREE.Vector3(aabb.min.x, aabb.min.y, aabb.min.z),
      new THREE.Vector3(aabb.max.x, aabb.max.y, aabb.max.z),
    );
    const helper = new THREE.Box3Helper(box, color);
    scene.add(helper);
    instanceBoxes.push(helper);
  });
}

// ── Query ─────────────────────────────────────────────────────────────

async function runQuery() {
  const text = queryInput.value.trim();
  if (!text) { setStatus("Enter query text."); return; }
  if (!taskId) { setStatus("Upload a zip and reconstruct first."); return; }

  setStatus("Querying...");
  try {
    const resp = await fetch("/api/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId, text }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      setStatus(`Query failed: ${err.error || resp.statusText}`);
      return;
    }

    const result = await resp.json();
    const objects = result.candidates || [];
    renderInstanceBoxes(objects);

    if (result.matched && objects.length > 0) {
      const object = objects[0];
      setStatus(
        `"${text}": ${objects.length} catalog match(es), ${object.point_count} points, confidence ${object.confidence.toFixed(3)}`
      );
    } else {
      setStatus(`"${text}": no matches (max score ${result.top_match_score.toFixed(3)})`);
    }
  } catch (err) {
    setStatus(`Query error: ${err.message}`);
  }
}

// ── Event wiring ──────────────────────────────────────────────────────
uploadBtn.addEventListener("click", uploadZip);
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") runQuery();
});
