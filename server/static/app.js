import * as THREE from 'three';
import { OrbitControls } from '/static/vendor/OrbitControls.js';

/* ------------------------------------------------------------------ *
 * DOM
 * ------------------------------------------------------------------ */

const $ = (id) => document.getElementById(id);
const el = {
  canvas: $('canvas'),
  browse: $('btn-browse'), fileInput: $('file-input'), dropzone: $('dropzone'),
  uploadStatus: $('upload-status'), meshSelect: $('mesh-select'), meshStats: $('mesh-stats'),
  panelDir: $('panel-direction'), panelMold: $('panel-mold'), panelResult: $('panel-result'),
  az: $('az'), el: $('el'), azVal: $('az-val'), elVal: $('el-val'),
  suggest: $('btn-suggest'), fromView: $('btn-from-view'), suggestStatus: $('suggest-status'),
  undercutStats: $('undercut-stats'),
  margin: $('margin'), marginVal: $('margin-val'),
  blockShape: $('block-shape'), grid: $('grid'), gridVal: $('grid-val'),
  optKeys: $('opt-keys'), optSpout: $('opt-spout'), optVents: $('opt-vents'),
  build: $('btn-build'), buildStatus: $('build-status'), buildProgress: $('build-progress'),
  resultStats: $('result-stats'), downloads: $('downloads'),
  layers: $('layers'), legend: $('legend'), hud: $('hud'),
  jobsBtn: $('btn-jobs'), jobsDrawer: $('jobs-drawer'), jobsList: $('jobs-list'),
  storeUsage: $('store-usage'),
};

const state = {
  meshId: null,
  jobId: null,
  poller: null,
  heatAbort: null,
  heatTimer: null,
  layer: 'part',
  lastReport: null,
};

/* ------------------------------------------------------------------ *
 * three.js scene
 * ------------------------------------------------------------------ */

const renderer = new THREE.WebGLRenderer({ canvas: el.canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14171c);

// Scans of people are conventionally Z-up; matching that keeps orbiting sane.
const camera = new THREE.PerspectiveCamera(42, 1, 0.5, 20000);
camera.up.set(0, 0, 1);
camera.position.set(300, -320, 220);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

scene.add(new THREE.HemisphereLight(0xbcd0e6, 0x2a2f38, 0.85));
const key = new THREE.DirectionalLight(0xffffff, 1.5);
key.position.set(1, -1.4, 1.2);
scene.add(key);
const fill = new THREE.DirectionalLight(0xa8c4e0, 0.6);
fill.position.set(-1.1, 0.9, -0.6);
scene.add(fill);

const layers = { part: null, half_a: null, half_b: null, parting: null };
const gizmo = new THREE.Group();
scene.add(gizmo);

const NEUTRAL = new THREE.Color(0x9aa7b6);

function resize() {
  const w = el.canvas.clientWidth, h = el.canvas.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(el.canvas);
resize();

(function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
})();

/* ------------------------------------------------------------------ *
 * GGM2 binary mesh format (see glovegen/viewer_format.py)
 * ------------------------------------------------------------------ */

function decodeGGM2(buffer) {
  const view = new DataView(buffer);
  const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== 'GGM2') throw new Error(`bad mesh payload (magic ${magic})`);
  const nFaces = view.getUint32(4, true);
  return { nFaces, positions: new Float32Array(buffer, 8, nFaces * 9) };
}

function buildGeometry({ nFaces, positions }) {
  const geom = new THREE.BufferGeometry();
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  const colors = new Float32Array(nFaces * 9);
  for (let i = 0; i < nFaces * 3; i++) {
    colors[i * 3] = NEUTRAL.r; colors[i * 3 + 1] = NEUTRAL.g; colors[i * 3 + 2] = NEUTRAL.b;
  }
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geom.computeVertexNormals();
  geom.computeBoundingSphere();
  return geom;
}

function makeMesh(geom, opts = {}) {
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: 0.62, metalness: 0.02,
    side: THREE.DoubleSide, ...opts,
  });
  return new THREE.Mesh(geom, mat);
}

/* ------------------------------------------------------------------ *
 * API
 * ------------------------------------------------------------------ */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail ?? detail; } catch { /* non-JSON */ }
    throw new Error(detail);
  }
  return res.json();
}

function say(node, text, cls = '') {
  node.hidden = false;
  node.textContent = text;
  node.className = `status ${cls}`;
}

function statsTable(node, rows) {
  node.hidden = false;
  node.innerHTML = rows
    .map(([k, v, cls = '']) => `<div><dt>${k}</dt><dd class="${cls}">${v}</dd></div>`)
    .join('');
}

/* ------------------------------------------------------------------ *
 * upload
 * ------------------------------------------------------------------ */

function upload(file) {
  const body = new FormData();
  body.append('file', file);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/meshes');
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) {
      say(el.uploadStatus, `uploading ${file.name} — ${Math.round((e.loaded / e.total) * 100)}%`);
    }
  };
  xhr.onload = () => {
    if (xhr.status >= 400) {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail ?? msg; } catch { /* ignore */ }
      say(el.uploadStatus, `upload failed: ${msg}`, 'err');
      return;
    }
    const { mesh } = JSON.parse(xhr.responseText);
    say(el.uploadStatus, 'preparing mesh…');
    waitForMesh(mesh.id);
    refreshMeshList(mesh.id);
  };
  xhr.onerror = () => say(el.uploadStatus, 'upload failed (network)', 'err');
  xhr.send(body);
}

async function waitForMesh(meshId) {
  for (;;) {
    const { mesh } = await api(`/api/meshes/${meshId}`);
    if (mesh.state === 'ready') {
      say(el.uploadStatus, `ready — ${mesh.stats.faces.toLocaleString()} triangles`, 'ok');
      await selectMesh(meshId);
      return;
    }
    if (mesh.state === 'failed') {
      say(el.uploadStatus, `preparation failed: ${mesh.error}`, 'err');
      return;
    }
    await new Promise((r) => setTimeout(r, 700));
  }
}

async function refreshMeshList(selectId) {
  const { meshes } = await api('/api/meshes');
  el.meshSelect.hidden = meshes.length < 2;
  el.meshSelect.innerHTML = meshes
    .map((m) => `<option value="${m.id}">${m.name} (${m.state})</option>`)
    .join('');
  if (selectId) el.meshSelect.value = selectId;
}

/* ------------------------------------------------------------------ *
 * mesh display
 * ------------------------------------------------------------------ */

async function selectMesh(meshId) {
  state.meshId = meshId;
  const { mesh } = await api(`/api/meshes/${meshId}`);
  const s = mesh.stats;
  statsTable(el.meshStats, [
    ['triangles', s.faces.toLocaleString()],
    ['size', s.extents_mm.map((v) => v.toFixed(0)).join(' × ') + ' mm'],
    ['volume', `${s.volume_cm3.toFixed(1)} cm³`],
    ['closed', s.closed ? 'yes' : `${s.boundary_edges} open edges`,
      s.closed ? 'ok' : 'bad'],
  ]);

  const buf = await (await fetch(`/api/meshes/${meshId}/viewer.bin`)).arrayBuffer();
  setLayer('part', buildGeometry(decodeGGM2(buf)));
  frameModel(layers.part.geometry.boundingSphere);

  ['half_a', 'half_b', 'parting'].forEach((k) => disposeLayer(k));
  el.layers.querySelectorAll('button').forEach((b) => {
    b.disabled = b.dataset.layer !== 'part';
  });
  showLayer('part');

  el.panelDir.classList.remove('disabled');
  el.panelMold.classList.remove('disabled');
  el.panelResult.hidden = true;
  requestHeatmap();
}

function disposeLayer(name) {
  const m = layers[name];
  if (!m) return;
  scene.remove(m);
  m.geometry.dispose();
  m.material.dispose();
  layers[name] = null;
}

function setLayer(name, geom, opts) {
  disposeLayer(name);
  const mesh = makeMesh(geom, opts);
  mesh.visible = state.layer === name;
  layers[name] = mesh;
  scene.add(mesh);
  return mesh;
}

function showLayer(name) {
  state.layer = name;
  for (const [k, m] of Object.entries(layers)) if (m) m.visible = k === name;
  el.layers.querySelectorAll('button').forEach((b) => {
    b.classList.toggle('active', b.dataset.layer === name);
  });
  el.legend.hidden = name !== 'part';
  gizmo.visible = name === 'part';
}

function frameModel(sphere) {
  if (!sphere) return;
  const r = Math.max(sphere.radius, 1);
  controls.target.copy(sphere.center);
  camera.near = r / 200;
  camera.far = r * 60;
  camera.position.copy(sphere.center).add(new THREE.Vector3(r * 1.5, -r * 1.7, r * 1.1));
  camera.updateProjectionMatrix();
  controls.update();
  buildGizmo(sphere);
}

/* ------------------------------------------------------------------ *
 * pull direction + gizmo
 * ------------------------------------------------------------------ */

function currentDirection() {
  const az = (Number(el.az.value) * Math.PI) / 180;
  const ev = (Number(el.el.value) * Math.PI) / 180;
  return new THREE.Vector3(
    Math.cos(ev) * Math.cos(az),
    Math.cos(ev) * Math.sin(az),
    Math.sin(ev),
  ).normalize();
}

function setDirection(v) {
  const d = new THREE.Vector3(v[0], v[1], v[2]).normalize();
  el.el.value = Math.round((Math.asin(THREE.MathUtils.clamp(d.z, -1, 1)) * 180) / Math.PI);
  el.az.value = Math.round((Math.atan2(d.y, d.x) * 180) / Math.PI);
  syncDirLabels();
}

function syncDirLabels() {
  el.azVal.textContent = `${el.az.value}°`;
  el.elVal.textContent = `${el.el.value}°`;
  orientGizmo();
  const d = currentDirection();
  el.hud.textContent = `d = (${d.x.toFixed(3)}, ${d.y.toFixed(3)}, ${d.z.toFixed(3)})`;
}

function buildGizmo(sphere) {
  gizmo.clear();
  const r = Math.max(sphere?.radius ?? 50, 1);
  const len = r * 1.35;
  for (const [sign, colour] of [[1, 0x52a8ff], [-1, 0xff9a52]]) {
    const shaft = new THREE.Mesh(
      new THREE.CylinderGeometry(r * 0.012, r * 0.012, len, 12),
      new THREE.MeshBasicMaterial({ color: colour }),
    );
    shaft.position.set(0, (sign * len) / 2, 0);
    const head = new THREE.Mesh(
      new THREE.ConeGeometry(r * 0.045, r * 0.11, 16),
      new THREE.MeshBasicMaterial({ color: colour }),
    );
    head.position.set(0, sign * len, 0);
    if (sign < 0) head.rotation.z = Math.PI;
    gizmo.add(shaft, head);
  }
  if (sphere) gizmo.position.copy(sphere.center);
  orientGizmo();
}

function orientGizmo() {
  // Gizmo is modelled along +Y; rotate that onto the pull direction.
  gizmo.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), currentDirection());
}

/* ------------------------------------------------------------------ *
 * heatmap
 * ------------------------------------------------------------------ */

function requestHeatmap() {
  clearTimeout(state.heatTimer);
  state.heatTimer = setTimeout(fetchHeatmap, 140);
}

async function fetchHeatmap() {
  if (!state.meshId || !layers.part) return;
  state.heatAbort?.abort();
  const ctrl = new AbortController();
  state.heatAbort = ctrl;
  const d = currentDirection();
  try {
    const res = await fetch(`/api/meshes/${state.meshId}/heatmap`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ direction: [d.x, d.y, d.z] }),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(await res.text());
    const stats = JSON.parse(res.headers.get('X-Glovegen-Stats') || '{}');
    paintSeverity(new Uint8Array(await res.arrayBuffer()));
    showUndercutStats(stats);
  } catch (err) {
    if (err.name !== 'AbortError') say(el.suggestStatus, `heatmap failed: ${err.message}`, 'err');
  }
}

function paintSeverity(sev) {
  const geom = layers.part.geometry;
  const colours = geom.getAttribute('color');
  const n = Math.min(sev.length, colours.count / 3);
  const c = new THREE.Color();
  for (let f = 0; f < n; f++) {
    const s = sev[f] / 255;
    if (s <= 0) c.setRGB(NEUTRAL.r, NEUTRAL.g, NEUTRAL.b);
    // amber for shallow (easy to flex past) through to red for deeply trapped
    else c.setHSL(THREE.MathUtils.lerp(0.11, 0.0, Math.min(1, s)), 0.78, 0.52);
    for (let v = 0; v < 3; v++) colours.setXYZ(f * 3 + v, c.r, c.g, c.b);
  }
  colours.needsUpdate = true;
}

function showUndercutStats(s) {
  if (!s.undercut_area_fraction && s.undercut_area_fraction !== 0) return;
  const pct = s.undercut_area_fraction * 100;
  const cls = pct < 0.5 ? 'ok' : pct < 5 ? 'warn' : 'bad';
  statsTable(el.undercutStats, [
    ['undercut area', `${pct.toFixed(3)} %`, cls],
    ['trapped mold', `${(s.trapped_volume_cm3 ?? 0).toFixed(3)} cm³`, cls],
    ['of cast volume', `${(s.undercut_percent ?? 0).toFixed(4)} %`, cls],
    ['block volume', `${(s.block_volume_cm3 ?? 0).toFixed(0)} cm³`],
  ]);
  el.hud.textContent =
    `d = (${currentDirection().toArray().map((v) => v.toFixed(3)).join(', ')})   ` +
    `undercut ${pct.toFixed(3)}%`;
}

/* ------------------------------------------------------------------ *
 * jobs
 * ------------------------------------------------------------------ */

function stopPolling() {
  clearInterval(state.poller);
  state.poller = null;
}

function pollJob(jobId, { onProgress, onDone, onFail }) {
  stopPolling();
  state.jobId = jobId;
  state.poller = setInterval(async () => {
    let job;
    try { ({ job } = await api(`/api/jobs/${jobId}`)); }
    catch (err) { stopPolling(); onFail?.(err.message); return; }
    onProgress?.(job);
    if (job.state === 'done') { stopPolling(); onDone?.(job); }
    else if (job.state === 'failed' || job.state === 'interrupted') {
      stopPolling(); onFail?.(job.message || job.state);
    }
  }, 700);
}

function moldConfig() {
  return {
    block_margin: Number(el.margin.value),
    block_shape: el.blockShape.value,
    parting: { grid: Number(el.grid.value) },
    keys: { enabled: el.optKeys.checked },
    spout: { enabled: el.optSpout.checked },
    vents: { enabled: el.optVents.checked },
  };
}

el.suggest.onclick = async () => {
  el.suggest.disabled = true;
  say(el.suggestStatus, 'searching pull directions…');
  try {
    const { job } = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mesh_id: state.meshId, kind: 'analyze', config: moldConfig() }),
    });
    pollJob(job.id, {
      onProgress: (j) => say(el.suggestStatus, `${j.message} (${Math.round(j.progress * 100)}%)`),
      onDone: (j) => {
        el.suggest.disabled = false;
        setDirection(j.result.direction);
        const uc = j.result.best.undercut_percent;
        say(el.suggestStatus, `best direction: ${uc.toFixed(4)}% of cast trapped`, 'ok');
        fetchHeatmap();
      },
      onFail: (msg) => { el.suggest.disabled = false; say(el.suggestStatus, msg, 'err'); },
    });
  } catch (err) {
    el.suggest.disabled = false;
    say(el.suggestStatus, err.message, 'err');
  }
};

el.fromView.onclick = () => {
  const d = new THREE.Vector3().subVectors(controls.target, camera.position).normalize();
  setDirection([d.x, d.y, d.z]);
  requestHeatmap();
};

el.build.onclick = async () => {
  el.build.disabled = true;
  el.panelResult.hidden = true;
  el.buildProgress.hidden = false;
  el.buildProgress.querySelector('.bar').style.width = '0%';
  const d = currentDirection();
  say(el.buildStatus, 'starting…');
  try {
    const { job } = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        mesh_id: state.meshId,
        kind: 'mold',
        config: { ...moldConfig(), direction: [d.x, d.y, d.z] },
      }),
    });
    pollJob(job.id, {
      onProgress: (j) => {
        say(el.buildStatus, j.message);
        el.buildProgress.querySelector('.bar').style.width = `${Math.round(j.progress * 100)}%`;
      },
      onDone: async (j) => {
        el.build.disabled = false;
        el.buildProgress.querySelector('.bar').style.width = '100%';
        say(el.buildStatus, 'mold built', 'ok');
        await showResult(j);
      },
      onFail: (msg) => {
        el.build.disabled = false;
        el.buildProgress.hidden = true;
        say(el.buildStatus, msg, 'err');
      },
    });
  } catch (err) {
    el.build.disabled = false;
    el.buildProgress.hidden = true;
    say(el.buildStatus, err.message, 'err');
  }
};

async function showResult(job) {
  const r = job.result.report;
  state.lastReport = r;
  const sep = r.separation || {};
  const feat = r.features || {};
  statsTable(el.resultStats, [
    ['half A', `${r.mold.half_a_volume_cm3.toFixed(0)} cm³`],
    ['half B', `${r.mold.half_b_volume_cm3.toFixed(0)} cm³`],
    ['split error', `${r.mold.split_volume_error_cm3.toFixed(5)} cm³`],
    ['halves open', sep.opens ? 'yes' : 'NO', sep.opens ? 'ok' : 'bad'],
    ['cast interference', `${(sep.cast_interference_mm3 ?? 0).toFixed(2)} mm³`,
      sep.rigid_cast_demolds ? 'ok' : 'warn'],
    ['rigid cast demolds', sep.rigid_cast_demolds ? 'yes' : 'flex needed',
      sep.rigid_cast_demolds ? 'ok' : 'warn'],
    ['keys', feat.keys?.count ?? 0],
    ['spout', feat.spout?.length_mm ? `${feat.spout.length_mm.toFixed(0)} mm` : 'none'],
    ['vents', feat.vents?.count ?? 0],
  ]);

  el.downloads.innerHTML = Object.entries(job.parts || {})
    .map(([name, meta]) =>
      `<a href="/api/jobs/${job.id}/files/${name}" download>` +
      `${meta.label}<span>${(meta.bytes / 1e6).toFixed(1)} MB</span></a>`)
    .join('');
  el.panelResult.hidden = false;

  for (const which of ['half_a', 'half_b', 'parting']) {
    try {
      const buf = await (await fetch(`/api/jobs/${job.id}/preview/${which}.bin`)).arrayBuffer();
      setLayer(which, buildGeometry(decodeGGM2(buf)),
        which === 'parting' ? { transparent: true, opacity: 0.85 } : {});
      el.layers.querySelector(`[data-layer="${which}"]`).disabled = false;
    } catch { /* preview is optional */ }
  }
  showLayer('half_a');
}

/* ------------------------------------------------------------------ *
 * history drawer
 * ------------------------------------------------------------------ */

el.jobsBtn.onclick = async () => {
  el.jobsDrawer.hidden = !el.jobsDrawer.hidden;
  if (el.jobsDrawer.hidden) return;
  const { jobs } = await api('/api/jobs');
  el.jobsList.innerHTML = jobs.length
    ? jobs.map((j) => `
        <div class="job-row">
          <span class="k">${j.kind}</span>
          <span class="s ${j.state}">${j.state}</span>
          ${j.kind === 'mold' && j.state === 'done'
            ? `<button data-load="${j.id}">load</button>` : '<span></span>'}
        </div>`).join('')
    : '<p class="hint">nothing yet</p>';
  el.jobsList.querySelectorAll('[data-load]').forEach((b) => {
    b.onclick = async () => {
      const { job } = await api(`/api/jobs/${b.dataset.load}`);
      el.jobsDrawer.hidden = true;
      if (job.mesh_id !== state.meshId) await selectMesh(job.mesh_id);
      await showResult(job);
    };
  });
};

/* ------------------------------------------------------------------ *
 * wiring
 * ------------------------------------------------------------------ */

el.browse.onclick = () => el.fileInput.click();
el.fileInput.onchange = () => el.fileInput.files[0] && upload(el.fileInput.files[0]);

['dragenter', 'dragover'].forEach((ev) =>
  el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.add('over'); }));
['dragleave', 'drop'].forEach((ev) =>
  el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.remove('over'); }));
el.dropzone.addEventListener('drop', (e) => {
  const f = e.dataTransfer.files[0];
  if (f) upload(f);
});

el.meshSelect.onchange = () => selectMesh(el.meshSelect.value);

for (const slider of [el.az, el.el]) {
  slider.addEventListener('input', () => { syncDirLabels(); requestHeatmap(); });
}
el.margin.addEventListener('input', () => { el.marginVal.textContent = `${el.margin.value} mm`; });
el.grid.addEventListener('input', () => { el.gridVal.textContent = el.grid.value; });

el.layers.querySelectorAll('button').forEach((b) => {
  b.onclick = () => !b.disabled && showLayer(b.dataset.layer);
});

/* ------------------------------------------------------------------ *
 * boot
 * ------------------------------------------------------------------ */

(async function boot() {
  syncDirLabels();
  buildGizmo(null);
  try {
    const s = await api('/api/status');
    el.storeUsage.textContent =
      `${s.storage.meshes} mesh · ${(s.storage.bytes / 1e6).toFixed(0)} MB · ttl ${s.storage.ttl_hours}h`;
  } catch { /* status is cosmetic */ }
  await refreshMeshList();
  const { meshes } = await api('/api/meshes');
  const ready = meshes.find((m) => m.state === 'ready');
  if (ready) { el.meshSelect.value = ready.id; await selectMesh(ready.id); }
})();
