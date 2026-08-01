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
  panelFeatures: $('panel-features'), featureList: $('feature-list'),
  addKey: $('btn-add-key'), addSpout: $('btn-add-spout'), addVent: $('btn-add-vent'),
  resetPlan: $('btn-plan-reset'),
  applyFeatures: $('btn-apply-features'), featuresStatus: $('features-status'),
  featuresProgress: $('features-progress'), placeHint: $('place-hint'),
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
  // feature editing
  plan: null,           // the plan being edited
  autoPlan: null,       // what the mold job proposed, for "Reset"
  sourceJobId: null,    // job currently on screen
  featureStatus: {},    // item id -> { status, reason } from the last apply
  placing: null,        // 'key' | 'vent' while picking a spot in the viewport
  pullDirection: [0, 0, 1],
  defaults: null,       // MoldConfig defaults, from /api/status
  paramBounds: null,    // per-kind clamp ranges, from /api/status
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
// Where the knobs and holes are, drawn over whichever layer is showing.
const markers = new THREE.Group();
scene.add(markers);
const raycaster = new THREE.Raycaster();

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

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

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
  clearPlan();
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
 * feature plan — interactive, once the mold exists
 *
 * The mold job proposes every knob and hole automatically and returns that
 * proposal as data. Editing it here and re-applying only re-cuts the
 * features: the direction search, the parting surface and the split are
 * already done and none of them depend on where a knob sits.
 * ------------------------------------------------------------------ */

// Labels and steps for the sizes a feature exposes. The ranges are replaced
// by the server's on boot (and enforced there regardless) — see
// glovegen.features.PARAM_BOUNDS.
const PARAM_UI = {
  key: [
    ['radius', 'r', 0.5, 50, 0.5],
    ['height', 'h', 0.5, 50, 0.5],
    ['draft_deg', 'draft°', 0, 45, 1],
    ['clearance', 'fit', 0, 3, 0.05],
  ],
  spout: [
    ['inner_radius', 'inner r', 0.5, 60, 0.5],
    ['outer_radius', 'outer r', 0.5, 80, 0.5],
  ],
  vent: [['radius', 'r', 0.2, 20, 0.1]],
};
const KIND_LABEL = { key: 'Knob', spout: 'Spout', vent: 'Vent' };
const KIND_COLOUR = { key: 0x52a8ff, spout: 0x46b07a, vent: 0xe0a23c };

function adoptParamBounds(bounds) {
  for (const [kind, params] of Object.entries(bounds || {})) {
    for (const spec of PARAM_UI[kind] || []) {
      const range = params[spec[0]];
      if (Array.isArray(range)) { [spec[2], spec[3]] = range; }
    }
  }
}

function defaultParams(kind) {
  const d = state.defaults || {};
  const k = d.keys || {}, s = d.spout || {}, v = d.vents || {};
  if (kind === 'key') {
    return {
      radius: k.radius ?? 5, height: k.height ?? 4,
      draft_deg: k.draft_deg ?? 20, clearance: k.clearance ?? 0.25,
    };
  }
  if (kind === 'spout') {
    return { inner_radius: s.inner_radius ?? 4, outer_radius: s.outer_radius ?? 9 };
  }
  return { radius: v.radius ?? 0.9 };
}

function clonePlan(plan) {
  return plan ? JSON.parse(JSON.stringify(plan)) : null;
}

function setPlan(plan, { statuses = null } = {}) {
  state.plan = clonePlan(plan);
  state.featureStatus = statuses || {};
  renderPlan();
}

function clearPlan() {
  state.plan = null;
  state.autoPlan = null;
  state.sourceJobId = null;
  state.featureStatus = {};
  setPlacing(null);
  el.panelFeatures.hidden = true;
  el.featuresStatus.hidden = true;
  drawMarkers();
}

function planDirty(message) {
  say(el.featuresStatus, message || 'edited — re-apply to cut this into the mold');
}

function renderPlan() {
  const plan = state.plan;
  el.panelFeatures.hidden = !plan;
  if (!plan) { drawMarkers(); return; }

  const seen = {};
  el.featureList.innerHTML = plan.items.length
    ? plan.items.map((item) => {
      seen[item.kind] = (seen[item.kind] || 0) + 1;
      const st = state.featureStatus[item.id] || {};
      const skipped = st.status === 'skipped';
      const params = (PARAM_UI[item.kind] || []).map(([name, label, min, max, step]) =>
        `<label>${label}<input type="number" data-id="${esc(item.id)}" data-param="${name}"
           value="${item.params[name]}" min="${min}" max="${max}" step="${step}"></label>`).join('');
      return `
        <div class="feat${item.enabled ? '' : ' off'}${skipped ? ' skipped' : ''}">
          <div class="feat-head">
            <input type="checkbox" data-toggle="${esc(item.id)}" ${item.enabled ? 'checked' : ''}>
            <span class="dot ${item.kind}"></span>
            <span class="name">${KIND_LABEL[item.kind]} ${seen[item.kind]}</span>
            <span class="note">${esc(item.source === 'user' ? 'placed by hand' : item.note)}</span>
            <button class="del" data-del="${esc(item.id)}" title="remove">✕</button>
          </div>
          <div class="feat-params">${params}</div>
          ${skipped ? `<div class="feat-why">skipped — ${esc(st.reason)}</div>` : ''}
        </div>`;
    }).join('')
    : '<p class="empty">Nothing planned. Add a knob or a vent to place one by hand.</p>';

  drawMarkers();
}

function itemById(id) {
  return (state.plan?.items || []).find((i) => i.id === id);
}

el.featureList.addEventListener('input', (e) => {
  const t = e.target;
  if (!t.dataset.param) return;
  const item = itemById(t.dataset.id);
  const value = Number(t.value);
  if (!item || !Number.isFinite(value)) return;
  // Deliberately no re-render: that would yank focus out of the field being
  // typed into. The markers are the live feedback.
  item.params[t.dataset.param] = value;
  drawMarkers();
  planDirty();
});

el.featureList.addEventListener('change', (e) => {
  const id = e.target.dataset.toggle;
  if (!id) return;
  const item = itemById(id);
  if (!item) return;
  item.enabled = e.target.checked;
  renderPlan();
  planDirty();
});

el.featureList.addEventListener('click', (e) => {
  const id = e.target.dataset.del;
  if (!id) return;
  state.plan.items = state.plan.items.filter((i) => i.id !== id);
  renderPlan();
  planDirty();
});

/* ---- placing a feature by clicking in the viewport ---- */

// Knobs and spouts are both centred on the parting surface, so they are placed
// against it; a vent starts at the cast, so it is placed against the scan.
const PLACE_ON = { key: 'parting', spout: 'parting', vent: 'part' };
const PLACE_HINT = {
  key: 'Click the parting surface to drop a knob · Esc to cancel',
  spout: 'Click the parting surface where the mold should be filled · Esc to cancel',
  vent: 'Click the scan where air would be trapped · Esc to cancel',
};

function setPlacing(kind) {
  state.placing = kind;
  el.placeHint.hidden = !kind;
  for (const [k, button] of [['key', el.addKey], ['spout', el.addSpout], ['vent', el.addVent]]) {
    button.classList.toggle('active', kind === k);
  }
  if (!kind) return;
  el.placeHint.textContent = PLACE_HINT[kind];
  const layer = PLACE_ON[kind];
  if (layers[layer]) showLayer(layer);
}

function addItem(kind, point) {
  state.uid = (state.uid || 0) + 1;
  state.plan.items.push({
    id: `${kind}-u${state.uid}`,
    kind,
    enabled: true,
    source: 'user',
    position: [point.x, point.y, point.z],
    params: defaultParams(kind),
    note: 'placed by hand',
  });
  renderPlan();
  planDirty(`${KIND_LABEL[kind].toLowerCase()} placed — re-apply to cut it`);
}

let pressedAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => {
  pressedAt = [e.clientX, e.clientY];
});
renderer.domElement.addEventListener('pointerup', (e) => {
  const from = pressedAt;
  pressedAt = null;
  if (!state.placing || !from) return;
  // An orbit ends in a pointerup too; only a press that barely moved is a click.
  if (Math.hypot(e.clientX - from[0], e.clientY - from[1]) > 4) return;

  const target = layers[PLACE_ON[state.placing]];
  if (!target) return;
  const rect = renderer.domElement.getBoundingClientRect();
  raycaster.setFromCamera(new THREE.Vector2(
    ((e.clientX - rect.left) / rect.width) * 2 - 1,
    -((e.clientY - rect.top) / rect.height) * 2 + 1,
  ), camera);
  const hit = raycaster.intersectObject(target, false)[0];
  if (!hit) {
    say(el.featuresStatus, 'nothing there — click on the surface itself', 'err');
    return;
  }
  addItem(state.placing, hit.point);
  setPlacing(null);
});

addEventListener('keydown', (e) => { if (e.key === 'Escape') setPlacing(null); });

/* ---- markers ---- */

function drawMarkers() {
  for (const m of markers.children) { m.geometry.dispose(); m.material.dispose(); }
  markers.clear();
  if (!state.plan) return;

  const pull = new THREE.Vector3(...state.pullDirection).normalize();
  const pour = new THREE.Vector3(...(state.plan.pour_axis || [0, 0, 1])).normalize();
  const up = new THREE.Vector3(0, 1, 0);

  for (const item of state.plan.items) {
    const st = state.featureStatus[item.id] || {};
    const p = item.params;
    let geom, axis = pull;
    if (item.kind === 'key') {
      geom = new THREE.CylinderGeometry(p.radius, p.radius, Math.max(p.height, 1), 20);
    } else if (item.kind === 'spout') {
      geom = new THREE.ConeGeometry(p.outer_radius, p.outer_radius * 2.2, 24);
      axis = pour;
    } else {
      geom = new THREE.SphereGeometry(Math.max(p.radius, 1.5), 14, 10);
    }
    const mesh = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({
      color: st.status === 'skipped' ? 0xe05c4b : KIND_COLOUR[item.kind],
      transparent: true,
      opacity: item.enabled ? 0.55 : 0.15,
      depthWrite: false,
    }));
    mesh.quaternion.setFromUnitVectors(up, axis);
    mesh.position.set(...item.position);
    markers.add(mesh);
  }
}

/* ---- buttons ---- */

el.addKey.onclick = () => setPlacing(state.placing === 'key' ? null : 'key');
el.addSpout.onclick = () => setPlacing(state.placing === 'spout' ? null : 'spout');
el.addVent.onclick = () => setPlacing(state.placing === 'vent' ? null : 'vent');
el.resetPlan.onclick = () => {
  if (!state.autoPlan) return;
  setPlan(state.autoPlan);
  planDirty('back to the automatic proposal — re-apply to cut it');
};

el.applyFeatures.onclick = async () => {
  if (!state.plan || !state.sourceJobId) return;
  setPlacing(null);
  el.applyFeatures.disabled = true;
  el.featuresProgress.hidden = false;
  el.featuresProgress.querySelector('.bar').style.width = '0%';
  say(el.featuresStatus, 'starting…');
  try {
    const { job } = await api('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        kind: 'features',
        config: { source_job: state.sourceJobId, plan: state.plan },
      }),
    });
    pollJob(job.id, {
      onProgress: (j) => {
        say(el.featuresStatus, j.message);
        el.featuresProgress.querySelector('.bar').style.width = `${Math.round(j.progress * 100)}%`;
      },
      onDone: async (j) => {
        el.applyFeatures.disabled = false;
        el.featuresProgress.querySelector('.bar').style.width = '100%';
        await showResult(j);
        const skipped = (j.result.report.features.items || [])
          .filter((i) => i.status === 'skipped');
        if (skipped.length) {
          say(el.featuresStatus, `${skipped.length} could not be placed — see below`, 'err');
        } else {
          say(el.featuresStatus, 'features re-cut', 'ok');
        }
      },
      onFail: (msg) => {
        el.applyFeatures.disabled = false;
        el.featuresProgress.hidden = true;
        say(el.featuresStatus, msg, 'err');
      },
    });
  } catch (err) {
    el.applyFeatures.disabled = false;
    el.featuresProgress.hidden = true;
    say(el.featuresStatus, err.message, 'err');
  }
};

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
  // The plan on screen belongs to the previous mold; this build proposes a new one.
  clearPlan();
  const d = currentDirection();
  state.pullDirection = [d.x, d.y, d.z];
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

  // The knobs and holes become editable from here on: this job cached the
  // halves as they came out of the split, so re-cutting them is cheap.
  state.sourceJobId = job.id;
  if (r.pull_direction) state.pullDirection = r.pull_direction;
  const statuses = {};
  for (const item of feat.items || []) {
    statuses[item.id] = { status: item.status, reason: item.reason || '' };
  }
  if (job.result.plan) {
    if (job.kind === 'mold' || !state.autoPlan) state.autoPlan = job.result.plan;
    setPlan(job.result.plan, { statuses });
  }

  const rows = [
    ['half A', `${r.mold.half_a_volume_cm3.toFixed(0)} cm³`],
    ['half B', `${r.mold.half_b_volume_cm3.toFixed(0)} cm³`],
    ['halves open', sep.opens ? 'yes' : 'NO', sep.opens ? 'ok' : 'bad'],
    ['cast interference', `${(sep.cast_interference_mm3 ?? 0).toFixed(2)} mm³`,
      sep.rigid_cast_demolds ? 'ok' : 'warn'],
    ['rigid cast demolds', sep.rigid_cast_demolds ? 'yes' : 'flex needed',
      sep.rigid_cast_demolds ? 'ok' : 'warn'],
    ['keys', feat.keys?.count ?? 0],
    ['spout', feat.spout?.length_mm ? `${feat.spout.length_mm.toFixed(0)} mm` : 'none'],
    ['vents', feat.vents?.count ?? 0],
  ];
  if (job.kind === 'mold') {
    rows.splice(2, 0, ['split error', `${r.mold.split_volume_error_cm3.toFixed(5)} cm³`]);
  }
  const skipped = (feat.items || []).filter((i) => i.status === 'skipped').length;
  if (skipped) rows.push(['not placed', skipped, 'bad']);
  statsTable(el.resultStats, rows);

  el.downloads.innerHTML = Object.entries(job.parts || {})
    .map(([name, meta]) =>
      `<a href="/api/jobs/${job.id}/files/${name}" download>` +
      `${meta.label}<span>${(meta.bytes / 1e6).toFixed(1)} MB</span></a>`)
    .join('');
  el.panelResult.hidden = false;

  for (const which of ['half_a', 'half_b', 'parting']) {
    try {
      const res = await fetch(`/api/jobs/${job.id}/preview/${which}.bin`);
      if (!res.ok) throw new Error(res.statusText);
      setLayer(which, buildGeometry(decodeGGM2(await res.arrayBuffer())),
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
          ${(j.kind === 'mold' || j.kind === 'features') && j.state === 'done'
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
    state.defaults = s.defaults;
    adoptParamBounds(s.feature_params);
    el.storeUsage.textContent =
      `${s.storage.meshes} mesh · ${(s.storage.bytes / 1e6).toFixed(0)} MB · ttl ${s.storage.ttl_hours}h`;
  } catch { /* status is cosmetic */ }
  await refreshMeshList();
  const { meshes } = await api('/api/meshes');
  const ready = meshes.find((m) => m.state === 'ready');
  if (ready) { el.meshSelect.value = ready.id; await selectMesh(ready.id); }
})();
