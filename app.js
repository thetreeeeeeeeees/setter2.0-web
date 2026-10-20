// Setter: generates Kilter Board climbs in the browser with a diffusion model (DDIM sampling),
// and shows how each climb forms over the denoising steps.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.logLevel = "error";   // hide ONNX Runtime's informational warnings
// multi-threaded WASM needs cross-origin isolation, which GitHub Pages doesn't provide
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const RING_RADIUS = 4.0;   // board units, as in visualize.py
const RING_WIDTH = 0.7;
// Step 1 isn't shown: at the noisiest step, recovering the clean climb divides by about 0.00005,
// so its "prediction" is just amplified noise. The timeline starts at step 2 (index 1).
const FIRST_SHOWN = 1;
const VIEW_HELP = {
  prediction: "The climb the model would produce if it stopped at this step.",
  confidence: "Every hole shaded by how strongly the model currently leans toward a hold there.",
  relative: "Holes shaded by how likely each is to be a hold compared with the most likely hold on the entire board right now.",
  noisy: "The current noise the model is cleaning up. It starts as pure noise and sharpens into the climb.",
};

const $ = (id) => document.getElementById(id);
const els = {
  grade: $("grade"), angle: $("angle"), angleOut: $("angle-out"), seed: $("seed"), dice: $("dice"),
  steps: $("steps"), stepsOut: $("steps-out"), eta: $("eta"), etaOut: $("eta-out"),
  guidance: $("guidance"), guidanceOut: $("guidance-out"),
  generate: $("generate"), bar: $("bar"), status: $("status"),
  canvas: $("board"), play: $("play"), step: $("step"), stepLabel: $("step-label"), viewHelp: $("view-help"),
  summary: $("summary"), copyLink: $("copy-link"), copyFrames: $("copy-frames"), download: $("download"),
};

let board = null;      // board.json
let image = null;      // board.jpg
let session = null;    // ONNX Runtime session
let backend = "";
let backendKind = "";  // "webgpu" or "wasm"
let simulateFailureAt = 0;   // testing only: make the n-th model call fail as if the GPU had been lost
let run = null;        // the latest generation: { settings, snapshots: [{ t, x0, xt }], holds }
let view = "noisy";           // the tab shown when the page opens (must match the "active" tab in index.html)
let shown = 0;         // which step the board shows
let busy = false;
let cancel = false;
let playTimer = null;

// ---------- seeded random numbers ----------

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(rand) {
  // Box-Muller; 1 - rand() is in (0, 1], so the log never sees 0
  let spare = null;
  return () => {
    if (spare !== null) { const s = spare; spare = null; return s; }
    const r = Math.sqrt(-2 * Math.log(1 - rand()));
    const theta = 2 * Math.PI * rand();
    spare = r * Math.sin(theta);
    return r * Math.cos(theta);
  };
}

// ---------- sampler (mirrors Diffusion.ddim_sample in the Python code) ----------

function roundHalfEven(v) {          // torch.round rounds halves to even
  const f = Math.floor(v), d = v - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

function timesteps(T, steps) {       // evenly spaced from T-1 down to 0, duplicates removed
  const ts = [];
  for (let k = 0; k < steps; k++) {
    const v = roundHalfEven((T - 1) - (T - 1) * k / (steps - 1));
    if (ts.length === 0 || ts[ts.length - 1] !== v) ts.push(v);
  }
  return ts;
}

async function predictNoise(x, t, cond, guidance) {
  const { channels: C, height: H, width: W } = board.grid;
  const N = x.length, B = guidance > 0 ? 2 : 1;      // with guidance: conditional and unconditional together
  const xs = new Float32Array(B * N);
  xs.set(x);
  if (B === 2) xs.set(x, N);
  if (simulateFailureAt > 0 && --simulateFailureAt === 0) throw new Error("simulated GPU failure (testing)");
  const out = await session.run({
    x: new ort.Tensor("float32", xs, [B, C, H, W]),
    t: new ort.Tensor("float32", new Float32Array(B).fill(t), [B]),
    cond: new ort.Tensor("float32", Float32Array.from(B === 2 ? [...cond, ...cond] : cond), [B, 2]),
    drop: new ort.Tensor("float32", Float32Array.from(B === 2 ? [0, 1] : [0]), [B]),
  });
  const d = out.eps.data;
  const eps = new Float32Array(N);
  if (B === 1) eps.set(d.subarray(0, N));
  else for (let j = 0; j < N; j++) eps[j] = (1 + guidance) * d[j] - guidance * d[N + j];
  return eps;
}

async function sample(settings, onStep) {
  const { channels: C, height: H, width: W } = board.grid;
  const N = C * H * W;
  const randn = gaussian(mulberry32(settings.seed));
  const cond = [(settings.angle - board.cond_mean[0]) / board.cond_std[0],
                (settings.difficulty - board.cond_mean[1]) / board.cond_std[1]];
  const ts = timesteps(board.T, settings.steps);
  let x = new Float32Array(N);
  for (let j = 0; j < N; j++) x[j] = randn();
  const snapshots = [];
  for (let k = 0; k < ts.length; k++) {
    const i = ts[k], prev = k + 1 < ts.length ? ts[k + 1] : null;
    const ab = board.alpha_bar[i], abPrev = prev === null ? 1 : board.alpha_bar[prev];
    const eps = await predictNoise(x, i, cond, settings.guidance);
    const sa = Math.sqrt(ab), s1 = Math.sqrt(1 - ab);
    const x0 = new Float32Array(N);
    for (let j = 0; j < N; j++) {
      const v = Math.min(1, Math.max(-1, (x[j] - s1 * eps[j]) / sa));   // the clean climb, clamped to [-1, 1]
      x0[j] = v;
      eps[j] = (x[j] - sa * v) / s1;                                      // the noise consistent with it
    }
    snapshots.push({ t: i, x0, xt: x });
    if (onStep && (await onStep(k, ts.length, snapshots)) === false) return null;   // cancelled
    if (prev === null) break;
    const sigma = settings.eta * Math.sqrt((1 - abPrev) / (1 - ab) * (1 - ab / abPrev));
    const cx0 = Math.sqrt(abPrev), ceps = Math.sqrt(Math.max(0, 1 - abPrev - sigma * sigma));
    const next = new Float32Array(N);
    for (let j = 0; j < N; j++) next[j] = cx0 * x0[j] + ceps * eps[j];
    if (sigma > 0) for (let j = 0; j < N; j++) next[j] += sigma * randn();
    x = next;
  }
  return snapshots;
}

// ---------- decoding ----------

function cell(arr, hole) {
  const plane = board.grid.height * board.grid.width, base = hole.row * board.grid.width + hole.col;
  const v = new Array(board.grid.channels);
  for (let ch = 0; ch < v.length; ch++) v[ch] = arr[ch * plane + base];
  return v;
}

function decode(arr) {                // each hole takes its strongest channel; anything but "empty" is a hold
  const holds = [];
  for (const hole of board.holes) {
    const v = cell(arr, hole);
    let best = 0;
    for (let ch = 1; ch < v.length; ch++) if (v[ch] > v[best]) best = ch;
    if (best > 0) holds.push({ hole, role: best });
  }
  return holds;
}

function relativeLeanings(arr) {      // per hole: chance of a hold (from "empty"), relative to the most likely hole
  const cells = board.holes.map((hole) => {
    const v = cell(arr, hole);
    let role = 1;
    for (let ch = 2; ch < v.length; ch++) if (v[ch] > v[role]) role = ch;
    return { hole, role, q: (1 - v[0]) / 2 };    // empty value +1 -> 0, -1 -> 1
  });
  const max = Math.max(...cells.map((c) => c.q));
  return cells.map((c) => ({ ...c, p: max > 0 ? c.q / max : 0 }));
}

function leanings(arr) {              // per hole: most likely role and how strongly it beats "empty" (0..1)
  return board.holes.map((hole) => {
    const v = cell(arr, hole);
    let role = 1;
    for (let ch = 2; ch < v.length; ch++) if (v[ch] > v[role]) role = ch;
    const score = (v[role] - v[0]) / 2;
    return { hole, role, p: Math.min(1, Math.max(0, (score + 1) / 2)) };
  });
}

function framesString(holds) {
  return holds.slice().sort((a, b) => a.hole.id - b.hole.id)
    .map((h) => `p${h.hole.id}r${board.roles[h.role].id}`).join("");
}

// ---------- drawing ----------

function draw() {
  const ctx = els.canvas.getContext("2d");
  ctx.drawImage(image, 0, 0);
  const snap = run?.snapshots[shown];
  if (!snap) return;                                 // nothing to show yet (e.g. during step 1)
  const { left, right, top } = board.image.edges;
  const scale = els.canvas.width / (right - left);
  const ring = (hole, color, alpha) => {
    ctx.globalAlpha = alpha;
    ctx.strokeStyle = color;
    ctx.lineWidth = Math.max(1, RING_WIDTH * scale);
    ctx.beginPath();
    ctx.arc((hole.x - left) * scale, (top - hole.y) * scale, RING_RADIUS * scale, 0, 2 * Math.PI);
    ctx.stroke();
  };
  if (view === "prediction") {
    for (const h of decode(snap.x0)) ring(h.hole, board.roles[h.role].color, 1);
  } else {
    const cells = view === "relative" ? relativeLeanings(snap.x0) : leanings(view === "confidence" ? snap.x0 : snap.xt);
    for (const c of cells) {
      const alpha = c.p ** 2;                       // fade out the in-between values
      if (alpha > 0.04) ring(c.hole, board.roles[c.role].color, alpha);
    }
  }
  ctx.globalAlpha = 1;
}

function showStep(k) {
  shown = k;
  els.step.value = String(k);
  const snap = run.snapshots[k];
  els.stepLabel.textContent = `step ${k + 1}/${run.total} · noise ${snap.t}`;
  draw();
}

// ---------- UI ----------

function yieldToBrowser() {           // lets the page repaint and handle clicks; unlike requestAnimationFrame,
  return new Promise((resolve) => {   // this keeps running when the tab is in the background
    const channel = new MessageChannel();
    channel.port1.onmessage = () => resolve();
    channel.port2.postMessage(null);
  });
}

function settingsFromUI() {
  return {
    difficulty: Number(els.grade.value), angle: Number(els.angle.value),
    seed: Math.max(0, Math.floor(Number(els.seed.value) || 0)),
    steps: Number(els.steps.value), eta: Number(els.eta.value), guidance: Number(els.guidance.value),
  };
}

function syncOutputs() {
  els.angleOut.textContent = `${els.angle.value}°`;
  els.stepsOut.textContent = els.steps.value;
  els.etaOut.textContent = Number(els.eta.value).toFixed(1);
  els.guidanceOut.textContent = Number(els.guidance.value).toFixed(1);
}

function setStatus(text, fraction) {
  els.status.textContent = text;
  if (fraction !== undefined) els.bar.style.width = `${Math.round(fraction * 100)}%`;
}

function stopPlaying() {
  clearInterval(playTimer);
  playTimer = null;
  els.play.textContent = "▶";
  els.play.setAttribute("aria-label", "Play");
}

function setPlayHighlight(on) {        // red, pulsing play button after a climb finishes, until it's used
  els.play.classList.toggle("ready", on);
}

function togglePlay() {
  setPlayHighlight(false);
  if (playTimer) return stopPlaying();
  if (shown >= run.snapshots.length - 1) showStep(FIRST_SHOWN);
  els.play.textContent = "❚❚";
  els.play.setAttribute("aria-label", "Pause");
  playTimer = setInterval(() => {
    if (shown >= run.snapshots.length - 1) return stopPlaying();
    showStep(shown + 1);
  }, Math.max(40, 4000 / run.snapshots.length));       // whole timeline in about 4 seconds
}

function summarize(holds) {
  const count = (r) => holds.filter((h) => h.role === r).length;
  const starts = count(1), finishes = count(3);
  const warn = [];
  if (starts < 1 || starts > 2) warn.push(`${starts} start holds`);
  if (finishes < 1 || finishes > 2) warn.push(`${finishes} finish holds`);
  const grade = board.grades.find((g) => g.difficulty === run.settings.difficulty)?.name ?? run.settings.difficulty;
  let html = `<strong>${grade} at ${run.settings.angle}°</strong>, seed ${run.settings.seed}: ${holds.length} holds ` +
    `(${starts} start, ${count(2)} hand, ${finishes} finish, ${count(4)} foot).`;
  if (warn.length) html += ` <span class="bad">Unusual: ${warn.join(", ")}. Try another seed.</span>`;
  html += ` <span class="hint">Made in ${run.seconds.toFixed(1)} s on ${backend}.</span>`;
  els.summary.innerHTML = html;
}

async function generate(retrySettings = null) {
  if (busy) { cancel = true; return; }
  stopPlaying();
  setPlayHighlight(false);
  busy = true;
  cancel = false;
  const settings = retrySettings ?? settingsFromUI();
  let switchToCPU = false;
  const started = performance.now();
  run = { settings, snapshots: [], total: settings.steps, holds: [], seconds: 0 };
  shown = FIRST_SHOWN;
  draw();                                            // clear the previous climb from the board
  els.generate.textContent = "Stop";
  els.step.disabled = els.play.disabled = true;
  [els.copyLink, els.copyFrames, els.download].forEach((b) => (b.disabled = true));
  try {
    const snapshots = await sample(settings, async (k, total, snaps) => {
      run.snapshots = snaps;
      run.total = total;
      els.step.max = String(total - 1);
      if (k >= FIRST_SHOWN) showStep(k);
      const perStep = (performance.now() - started) / 1000 / (k + 1);
      setStatus(`Step ${k + 1} of ${total} · about ${Math.ceil(perStep * (total - k - 1))} s left`, (k + 1) / total);
      await yieldToBrowser();                               // let the board repaint between steps
      return !cancel;
    });
    if (snapshots === null) {
      setStatus("Stopped.", 0);
    } else {
      run.seconds = (performance.now() - started) / 1000;
      run.holds = decode(snapshots[snapshots.length - 1].x0);
      summarize(run.holds);
      setStatus("Done! Drag the slider or press play to watch the climb form.", 1);
      setPlayHighlight(true);
      [els.copyLink, els.copyFrames, els.download].forEach((b) => (b.disabled = false));
    }
  } catch (err) {
    console.error(err);
    // Phones (iOS Safari especially) can take the GPU away mid-run, e.g. while zooming; after that the GPU
    // session can't be used again. Switch to the CPU once and redo the same climb (same settings and seed).
    if (backendKind === "webgpu") switchToCPU = true;
    else setStatus(`Something went wrong: ${err.message}`, 0);
  } finally {
    busy = false;
    els.generate.textContent = "Generate climb";
    els.step.disabled = els.play.disabled = run.snapshots.length <= FIRST_SHOWN;
  }
  if (switchToCPU) {
    els.generate.disabled = true;
    try {
      setStatus("The GPU stopped responding. Switching to the CPU and starting this climb again…", 0);
      await session.release().catch(() => {});
      await startModel(["wasm"]);
      els.generate.disabled = false;
      generate(settings);
    } catch (err) {
      console.error(err);
      setStatus(`Something went wrong: ${err.message}`, 0);
    }
  }
}

function shareURL() {
  const s = run.settings;
  const params = new URLSearchParams({ grade: s.difficulty, angle: s.angle, seed: s.seed, steps: s.steps, eta: s.eta, guidance: s.guidance });
  return `${location.origin}${location.pathname}?${params}`;
}

async function copy(text, button) {
  try {
    await navigator.clipboard.writeText(text);
    const old = button.textContent;
    button.textContent = "Copied";
    setTimeout(() => (button.textContent = old), 1200);
  } catch {
    prompt("Copy this:", text);
  }
}

function applyURLParams() {
  const p = new URLSearchParams(location.search);
  const set = (el, key) => { if (p.has(key)) el.value = p.get(key); };
  set(els.grade, "grade"); set(els.angle, "angle"); set(els.seed, "seed");
  set(els.steps, "steps"); set(els.eta, "eta"); set(els.guidance, "guidance");
  syncOutputs();
  return p.has("seed");
}

async function fetchWithProgress(url, onProgress, expectedBytes = 0) {
  // Memory matters here: iOS Safari kills tabs that use too much (pinch-zooming adds more on top), so when the
  // file size is known the download goes straight into one buffer instead of pieces that are copied together.
  const res = await fetch(url);
  if (!res.ok) throw new Error(`couldn't download ${url} (${res.status})`);
  // the real file size: from board.json if given (GitHub Pages compresses the model, so its Content-Length is
  // the compressed size), otherwise from the server when it isn't compressing
  const announced = res.headers.get("Content-Encoding") ? 0 : Number(res.headers.get("Content-Length")) || 0;
  const total = expectedBytes || announced;
  const exact = total > 0;
  const reader = res.body.getReader();
  let buffer = exact ? new Uint8Array(total) : null;
  let chunks = exact ? null : [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (buffer && received + value.length > buffer.length) {   // bigger than announced: fall back to pieces
      chunks = [buffer.subarray(0, received)];
      buffer = null;
    }
    if (buffer) buffer.set(value, received);
    else chunks.push(value);
    received += value.length;
    onProgress(received, total);
  }
  if (buffer) return received === buffer.length ? buffer : buffer.subarray(0, received);
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const c of chunks) { bytes.set(c, offset); offset += c.length; }
  return bytes;
}

async function createSession(bytes, order) {
  const { channels: C, height: H, width: W } = board.grid;
  for (const ep of order) {
    try {
      const s = await ort.InferenceSession.create(bytes, { executionProviders: [ep] });   // no extra copy
      // one test run: catches backends that load but can't run the model, and warms up the GPU
      await s.run({
        x: new ort.Tensor("float32", new Float32Array(C * H * W), [1, C, H, W]),
        t: new ort.Tensor("float32", Float32Array.from([500]), [1]),
        cond: new ort.Tensor("float32", new Float32Array(2), [1, 2]),
        drop: new ort.Tensor("float32", new Float32Array(1), [1]),
      });
      return { session: s, kind: ep, backend: ep === "webgpu" ? "your GPU (WebGPU)" : "your CPU (WebAssembly)" };
    } catch (err) {
      console.warn(`ONNX Runtime backend ${ep} unavailable:`, err);
    }
  }
  throw new Error("this browser couldn't run the model");
}

async function startModel(order) {     // downloads the model (usually from the browser's cache after the first time)
  let bytes = await fetchWithProgress("assets/model.onnx", (got, total) =>
    setStatus(`Downloading model… ${(got / 1e6).toFixed(0)}${total ? ` of ${(total / 1e6).toFixed(0)}` : ""} MB`,
              total ? got / total : 0), board.model_bytes);
  setStatus("Starting the model…", 1);
  ({ session, backend, kind: backendKind } = await createSession(bytes, order));
  bytes = null;                                      // ONNX Runtime has its own copy now; let the browser free this one
}

async function init() {
  syncOutputs();
  setStatus("Loading board…", 0);
  board = await (await fetch("assets/board.json")).json();
  image = new Image();
  await new Promise((resolve, reject) => {          // the load event is more dependable than image.decode()
    image.onload = resolve;
    image.onerror = () => reject(new Error("couldn't load the board image"));
    image.src = `assets/${board.image.file}`;
  });
  els.canvas.width = image.naturalWidth;
  els.canvas.height = image.naturalHeight;
  draw();

  for (const g of board.grades) els.grade.add(new Option(g.name, g.difficulty, false, g.difficulty === 20));
  document.querySelectorAll(".legend i").forEach((el, k) => el.style.setProperty("--c", board.roles[k + 1].color));
  const autoRun = applyURLParams();

  // ?backend=wasm forces the CPU path (the fallback for browsers without WebGPU)
  const forced = new URLSearchParams(location.search).get("backend");
  await startModel(forced ? [forced] : "gpu" in navigator ? ["webgpu", "wasm"] : ["wasm"]);
  setStatus(`Ready. Running on ${backend}.`, 0);
  els.generate.disabled = false;
  els.generate.textContent = "Generate climb";
  if (autoRun) generate();
}

// events
[els.angle, els.steps, els.eta, els.guidance].forEach((el) => el.addEventListener("input", syncOutputs));
els.dice.addEventListener("click", () => { els.seed.value = String(Math.floor(Math.random() * 1e6)); });
els.generate.addEventListener("click", () => generate());
els.step.addEventListener("input", () => { setPlayHighlight(false); stopPlaying(); showStep(Number(els.step.value)); });
els.play.addEventListener("click", togglePlay);
els.copyLink.addEventListener("click", () => copy(shareURL(), els.copyLink));
els.copyFrames.addEventListener("click", () => copy(framesString(run.holds), els.copyFrames));
els.download.addEventListener("click", () => {
  const a = document.createElement("a");
  a.download = `climb-seed${run.settings.seed}.png`;
  a.href = els.canvas.toDataURL("image/png");
  a.click();
});
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  view = tab.dataset.view;
  document.querySelectorAll(".tab").forEach((t) => {
    t.classList.toggle("active", t === tab);
    t.setAttribute("aria-selected", String(t === tab));
  });
  els.viewHelp.textContent = VIEW_HELP[view];
  draw();
}));
els.viewHelp.textContent = VIEW_HELP[view];

window.setter = { sample, decode, timesteps, get board() { return board; }, get backend() { return backendKind; },
                  simulateFailure(n) { simulateFailureAt = n; } };   // for testing from the console

init().catch((err) => {
  console.error(err);
  setStatus(`Couldn't start: ${err.message}`, 0);
});
