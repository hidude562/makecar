// makecar viewer — three.js frontend.  World frame: +X forward, +Y left, +Z up (Z-up camera).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

// ---------------------------------------------------------------- API
// The static web build sets window.makecarTransport to run the same API in a Pyodide worker.
const WEB = !!window.makecarTransport;
const transport = window.makecarTransport || (async (url, body) => {
  const r = await fetch(url, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
  return { ok: r.ok, statusText: r.statusText, buf: await r.arrayBuffer() };
});
const decode = (buf) => new TextDecoder().decode(buf);
const errorOf = (r) => { try { return JSON.parse(decode(r.buf)).error || r.statusText; } catch (e) { return r.statusText || 'request failed'; } };
const api = {
  async json(url, body) {
    const r = await transport(url, body);
    if (!r.ok) throw new Error(errorOf(r));
    return JSON.parse(decode(r.buf));
  },
  async bin(url, body) {
    const r = await transport(url, body);
    if (!r.ok) throw new Error(errorOf(r));
    return parseMCB(r.buf);
  },
  async text(url) {
    const r = await transport(url);
    if (!r.ok) throw new Error(errorOf(r));
    return decode(r.buf);
  },
  async download(url, body, filename) {
    const r = await transport(url, body);
    if (!r.ok) throw new Error(errorOf(r));
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([r.buf]));
    a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  },
};
function parseMCB(buf) {
  const dv = new DataView(buf);
  if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'MCB1') throw new Error('bad mesh payload');
  const hlen = dv.getUint32(4, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 8, hlen)));
  const base = 8 + hlen;
  for (const p of header.parts) {
    p.pos = new Float32Array(buf, base + p.pos[0], p.pos[1]);
    p.idx = new Uint32Array(buf, base + p.idx[0], p.idx[1]);
    p.map = p.map && p.map[1] ? new Uint32Array(buf, base + p.map[0], p.map[1]) : null;
  }
  return header;
}
const b64f32 = (s) => { const b = atob(s); const a = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) a[i] = b.charCodeAt(i); return new Float32Array(a.buffer); };
const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => { const e = document.createElement(tag); for (const [k, v] of Object.entries(attrs)) { if (k === 'class') e.className = v; else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else if (v !== null && v !== undefined) e.setAttribute(k, v); } for (const k of kids) e.append(k); return e; };
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const msg = (s, ok = true) => { const m = $('#msg'); m.textContent = s; m.style.color = ok ? '' : '#ff8080'; };

// ---------------------------------------------------------------- scene
const canvas = $('#c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.localClippingEnabled = true;
renderer.outputColorSpace = THREE.SRGBColorSpace;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1b1e24);
const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 200);
camera.up.set(0, 0, 1);
const orbit = new OrbitControls(camera, canvas);
orbit.enableDamping = true; orbit.dampingFactor = 0.12; orbit.target.set(0, 0, 0.7);
orbit.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
scene.add(new THREE.HemisphereLight(0xdfe7f5, 0x3a3630, 0.9));
const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(-4, 5, 6); scene.add(key);
const fill = new THREE.DirectionalLight(0xbfd0ff, 0.5); fill.position.set(5, -3, 2); scene.add(fill);
const grid = new THREE.GridHelper(12, 24, 0x39404b, 0x272c34); grid.rotation.x = Math.PI / 2; scene.add(grid);
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 0, -1), 2.2); // keeps z < constant
const groups = { body: new THREE.Group(), parts: new THREE.Group(), seams: new THREE.Group(), conns: new THREE.Group() };
Object.values(groups).forEach(g => scene.add(g));
const gizmo = new TransformControls(camera, canvas);
gizmo.setSize(0.8); scene.add(gizmo);
gizmo.addEventListener('dragging-changed', (e) => { orbit.enabled = !e.value; if (!e.value) onGizmoEnd(); });
const proxy = new THREE.Object3D(); scene.add(proxy);

function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); }
new ResizeObserver(resize).observe(canvas); resize();
function animate() { requestAnimationFrame(animate); orbit.update(); renderer.render(scene, camera); }
animate();

const opts = { flat: false, wire: false, xray: false, cut: 2.2, autoAssemble: true, showConns: true, showLabels: false, showSeams: true };
function makeMaterial(p) {
  const color = new THREE.Color(p.color[0], p.color[1], p.color[2]);
  const isBody = p.component === 'body';
  const m = new THREE.MeshStandardMaterial({
    color, roughness: Math.max(0.05, 1 - p.shininess), metalness: p.material === 'paint' ? 0.35 : (p.shininess > 0.7 ? 0.7 : 0.05),
    transparent: p.alpha < 0.999 || (isBody && opts.xray), opacity: (isBody && opts.xray) ? 0.28 : p.alpha,
    flatShading: opts.flat, wireframe: opts.wire, side: THREE.DoubleSide, clippingPlanes: [clipPlane],
    emissive: color.clone().multiplyScalar(p.emissive || 0),
  });
  if (p.material === 'glass' || p.alpha < 0.999) m.depthWrite = false;
  return m;
}
function buildParts(header, group, tag) {
  while (group.children.length) { const c = group.children.pop(); c.geometry.dispose(); c.material.dispose(); }
  for (const p of header.parts) {
    if (tag === 'body' && p.component !== 'body') continue;
    if (tag === 'parts' && p.component === 'body') continue;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p.pos.slice(), 3));
    g.setIndex(new THREE.BufferAttribute(p.idx.slice(), 1));
    g.computeVertexNormals();
    const mesh = new THREE.Mesh(g, makeMaterial(p));
    mesh.userData = { part: p.name, component: p.component, material: p.material, map: p.map, color: p.color, alpha: p.alpha, shininess: p.shininess, emissive: p.emissive };
    group.add(mesh);
  }
}
function refreshMaterials() {
  for (const g of [groups.body, groups.parts]) for (const m of g.children) {
    const p = m.userData; m.material.dispose(); m.material = makeMaterial({ ...p, component: p.component });
  }
  clipPlane.constant = opts.cut;
}

// ---------------------------------------------------------------- state
let state = null;            // /api/state
let connectors = [];         // latest connector list (from state or a body/assembly payload)
let selected = null;         // selected connector name
let hovered = null;          // connector under the pointer (label shown on hover)
let body = null;             // {full: Float32Array positions, normals, mirror, nVerts}
const sculpt = { on: false, delta: null, strokes: [], radius: 0.18, strength: 0.006, sign: 1, mirror: true, dragging: false };

async function loadState() { state = await api.json('/api/state'); connectors = state.connectors; renderSidebar(); updateHeader(); }
async function loadBody(header) {
  buildParts(header, groups.body, 'body');
  const x = header.extra;
  body = { full: b64f32(x.full_positions_b64), normals: b64f32(x.full_normals_b64), mirror: x.mirror, nVerts: x.n_vertices };
  if (!sculpt.delta || sculpt.delta.length !== body.full.length) sculpt.delta = new Float32Array(body.full.length);
  connectors = x.connectors; buildSeams(x.seams); buildConnectors(); updateStatus(x.measurements);
}
async function fetchBody() { loadBody(await api.bin('/api/body.bin')); }
async function fetchAssembly() {
  const t = performance.now();
  const h = await api.bin('/api/assembly.bin');
  buildParts(h, groups.parts, 'parts');
  connectors = h.extra.connectors; buildConnectors();
  $('#faces').textContent = `${(h.extra.stats.body_faces + h.extra.stats.component_faces).toLocaleString()} faces · ${h.extra.stats.components} components · ${h.extra.stats.connectors} connectors`;
  $('#timing').textContent = `assemble ${((performance.now() - t) / 1000).toFixed(1)}s`;
  if (state) { state.assembled = true; }
  renderConnList();
}
const autoAssemble = debounce(() => { if (opts.autoAssemble) fetchAssembly().catch(e => msg(e.message, false)); }, 500);
function updateStatus(m) { if (!m) return; $('#meas').textContent = `L ${m.length.toFixed(3)}  W ${m.width.toFixed(3)}  H ${m.height.toFixed(3)}  WB ${m.wheelbase.toFixed(3)} m`; }
function updateHeader() { $('#cfg-name').textContent = state.config_path ? state.config_path.split('/').pop() : (state.config.name || 'untitled'); $('#dirty').hidden = !state.dirty; }
function markDirty() { if (state) { state.dirty = true; $('#dirty').hidden = false; } }

// ---------------------------------------------------------------- seams & connectors overlay
function buildSeams(seams) {
  while (groups.seams.children.length) { const c = groups.seams.children.pop(); c.geometry.dispose(); }
  const mat = new THREE.LineBasicMaterial({ color: 0x0d0f12, transparent: true, opacity: 0.8 });
  for (const pts of seams || []) {
    const g = new THREE.BufferGeometry().setFromPoints(pts.map(p => new THREE.Vector3(p[0], p[1], p[2] + 0.002)));
    groups.seams.add(new THREE.Line(g, mat));
  }
  groups.seams.visible = opts.showSeams;
}
const kindColor = { point: 0xf2c318, circle: 0x5fd1ff, rectangle: 0xff6fd8, polygon: 0x8ff26f };
const labelCache = new Map();
function labelSprite(text, color) {
  const k = text + color; if (labelCache.has(k)) return labelCache.get(k).clone();
  const c = document.createElement('canvas'); const ctx = c.getContext('2d'); ctx.font = '600 22px system-ui'; const w = Math.ceil(ctx.measureText(text).width) + 16; c.width = w; c.height = 30;
  ctx.font = '600 22px system-ui'; ctx.fillStyle = 'rgba(20,22,26,0.75)'; ctx.fillRect(0, 0, w, 30); ctx.fillStyle = '#' + color.toString(16).padStart(6, '0'); ctx.fillText(text, 8, 22);
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false, transparent: true }));
  sp.scale.set(w / 640, 30 / 640, 1); labelCache.set(k, sp); return sp.clone();
}
function frameFrom(c) { const f = c.frame; const m = new THREE.Matrix4().makeBasis(new THREE.Vector3(...f.x_axis), new THREE.Vector3(...f.y_axis), new THREE.Vector3(...f.z_axis)); m.setPosition(new THREE.Vector3(...f.origin)); return m; }
function buildConnectors() {
  while (groups.conns.children.length) groups.conns.remove(groups.conns.children[0]);
  for (const c of connectors) {
    const color = kindColor[c.kind] || 0xffffff; const sel = c.name === selected;
    const g = new THREE.Group(); g.userData.conn = c.name; g.applyMatrix4(frameFrom(c));
    const lm = new THREE.LineBasicMaterial({ color, transparent: true, opacity: sel ? 1 : 0.85, depthTest: !sel });
    if (c.kind === 'point') {
      const s = new THREE.Mesh(new THREE.SphereGeometry(sel ? 0.03 : 0.018, 12, 8), new THREE.MeshBasicMaterial({ color, depthTest: !sel })); g.add(s);
    } else {
      const local = c.points.map(p => new THREE.Vector3(...p)); const inv = frameFrom(c).invert();
      const pts = local.map(p => p.applyMatrix4(inv)); pts.push(pts[0].clone());
      g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), lm));
      const pick = new THREE.Mesh(new THREE.PlaneGeometry(Math.max(c.width || 0.1, 0.05), Math.max(c.height || 0.1, 0.05)), new THREE.MeshBasicMaterial({ visible: false, side: THREE.DoubleSide }));
      pick.userData.pick = c.name; g.add(pick);
    }
    // normal tick + local axes
    const ax = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0, sel ? 0.18 : 0.09)]);
    g.add(new THREE.Line(ax, new THREE.LineBasicMaterial({ color, depthTest: false })));
    if (sel) { g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0.12, 0, 0)]), new THREE.LineBasicMaterial({ color: 0xff5555, depthTest: false }))); g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(0, 0.12, 0)]), new THREE.LineBasicMaterial({ color: 0x55ff77, depthTest: false }))); }
    if (opts.showLabels || sel || c.name === hovered) { const l = labelSprite(c.name, color); l.position.set(0, 0, (c.kind === 'point' ? 0.06 : 0.04) + (sel ? 0.16 : 0.08)); if (sel) l.scale.multiplyScalar(1.5); g.add(l); }
    if (c.kind === 'point') { const pk = new THREE.Mesh(new THREE.SphereGeometry(0.05, 6, 4), new THREE.MeshBasicMaterial({ visible: false })); pk.userData.pick = c.name; g.add(pk); }
    groups.conns.add(g);
  }
  groups.conns.visible = opts.showConns;
  attachGizmo();
}
function attachGizmo() {
  const c = connectors.find(x => x.name === selected);
  if (!c) { gizmo.detach(); return; }
  proxy.matrix.copy(frameFrom(c)); proxy.matrix.decompose(proxy.position, proxy.quaternion, proxy.scale); proxy.scale.set(1, 1, 1);
  proxy.userData.base = { pos: proxy.position.clone(), quat: proxy.quaternion.clone(), conn: c };
  gizmo.attach(proxy);
}
async function onGizmoEnd() {
  const b = proxy.userData.base; if (!b) return;
  const c = b.conn; const prev = c.override || {};
  const dPos = proxy.position.clone().sub(b.pos);
  const dq = b.quat.clone().invert().multiply(proxy.quaternion); // rotation in the connector's local frame
  const e = new THREE.Euler().setFromQuaternion(dq, 'XYZ');
  const s = (proxy.scale.x + proxy.scale.y + proxy.scale.z) / 3;
  const add = (a, b2) => [0, 1, 2].map(i => +((a?.[i] || 0) + b2[i]).toFixed(4));
  const ov = { ...prev };
  if (dPos.length() > 1e-5) ov.translate = add(prev.translate, [dPos.x, dPos.y, dPos.z]);
  if (Math.abs(e.x) + Math.abs(e.y) + Math.abs(e.z) > 1e-4) ov.rotate_deg = add(prev.rotate_deg, [e.x, e.y, e.z].map(THREE.MathUtils.radToDeg));
  if (Math.abs(s - 1) > 1e-4) ov.scale = +(((prev.scale || 1) * s).toFixed(4));
  await setOverride(c.name, ov);
}
async function setOverride(name, ov) {
  try {
    const r = await api.json('/api/connector', { name, override: ov });
    connectors = r.connectors; markDirty(); buildConnectors(); renderConnDetail(); autoAssemble();
    msg(`override ${name}`);
  } catch (e) { msg(e.message, false); }
}

// picking
const ray = new THREE.Raycaster(); const ndc = new THREE.Vector2();
canvas.addEventListener('pointerdown', (ev) => {
  if (ev.button !== 0 || gizmo.dragging) return;
  const r = canvas.getBoundingClientRect(); ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(ndc, camera);
  if (sculpt.on) { sculpt.dragging = true; orbit.enabled = false; sculpt.strokes.push(sculpt.delta.slice()); brush(); return; }
  if (!opts.showConns) return;
  const hits = ray.intersectObjects(groups.conns.children, true).filter(h => h.object.userData.pick);
  if (hits.length) { select(hits[0].object.userData.pick); }
});
canvas.addEventListener('pointermove', (ev) => {
  if (!sculpt.dragging && opts.showConns && !gizmo.dragging) {
    const r0 = canvas.getBoundingClientRect(); ndc.set(((ev.clientX - r0.left) / r0.width) * 2 - 1, -((ev.clientY - r0.top) / r0.height) * 2 + 1); ray.setFromCamera(ndc, camera);
    const h = ray.intersectObjects(groups.conns.children, true).filter(x => x.object.userData.pick)[0];
    const name = h ? h.object.userData.pick : null; if (name !== hovered) { hovered = name; buildConnectors(); }
  }
  if (!sculpt.dragging) return; const r = canvas.getBoundingClientRect(); ndc.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1); ray.setFromCamera(ndc, camera); brush(); });
canvas.addEventListener('pointerup', () => { if (sculpt.dragging) { sculpt.dragging = false; orbit.enabled = true; } });
canvas.addEventListener('contextmenu', e => e.preventDefault());
function select(name) { selected = name; buildConnectors(); renderConnList(); renderConnDetail(); switchTab('connectors'); }

// ---------------------------------------------------------------- sculpt brush
function brush() {
  const hits = ray.intersectObjects(groups.body.children, false);
  if (!hits.length || !body) return;
  const hp = hits[0].point; const r2 = sculpt.radius * sculpt.radius; const P = body.full, N = body.normals, D = sculpt.delta;
  const touched = new Set();
  for (let i = 0; i < body.nVerts; i++) {
    const dx = P[3 * i] + D[3 * i] - hp.x, dy = P[3 * i + 1] + D[3 * i + 1] - hp.y, dz = P[3 * i + 2] + D[3 * i + 2] - hp.z;
    const d2 = dx * dx + dy * dy + dz * dz; if (d2 > r2) continue;
    const t = 1 - Math.sqrt(d2) / sculpt.radius; const w = t * t * (3 - 2 * t) * sculpt.strength * sculpt.sign;
    D[3 * i] += N[3 * i] * w; D[3 * i + 1] += N[3 * i + 1] * w; D[3 * i + 2] += N[3 * i + 2] * w; touched.add(i);
    if (sculpt.mirror) { const m = body.mirror[i]; if (m !== i && !touched.has(m)) { D[3 * m] += N[3 * i] * w; D[3 * m + 1] -= N[3 * i + 1] * w; D[3 * m + 2] += N[3 * i + 2] * w; touched.add(m); } }
  }
  applySculpt();
}
function applySculpt() {
  for (const m of groups.body.children) {
    const map = m.userData.map; if (!map) continue; const a = m.geometry.attributes.position; const arr = a.array;
    for (let k = 0; k < map.length; k++) { const i = map[k]; arr[3 * k] = body.full[3 * i] + sculpt.delta[3 * i]; arr[3 * k + 1] = body.full[3 * i + 1] + sculpt.delta[3 * i + 1]; arr[3 * k + 2] = body.full[3 * i + 2] + sculpt.delta[3 * i + 2]; }
    a.needsUpdate = true; m.geometry.computeVertexNormals();
  }
  let n = 0, mx = 0; for (let i = 0; i < body.nVerts; i++) { const d = Math.hypot(sculpt.delta[3 * i], sculpt.delta[3 * i + 1], sculpt.delta[3 * i + 2]); if (d > 1e-6) { n++; mx = Math.max(mx, d); } }
  $('#sc-status').textContent = n ? `${n} vertices displaced, max ${(mx * 1000).toFixed(1)} mm` : 'no sculpt yet';
}

// ---------------------------------------------------------------- sidebar: shape
function renderSidebar() { renderModifiers(); renderConnList(); renderComponents(); loadConfigText(); }
const setMod = debounce(async (vals) => {
  try { const t = performance.now(); loadBody(await api.bin('/api/modifiers', { values: vals })); $('#timing').textContent = `body ${((performance.now() - t) / 1000).toFixed(2)}s`; markDirty(); autoAssemble(); } catch (e) { msg(e.message, false); }
}, 40);
let pendingVals = {};
function queueMod(name, v) { pendingVals[name] = v; const vals = pendingVals; setMod(vals); pendingVals = {}; pendingVals = { ...vals }; }
function renderModifiers() {
  const box = $('#modifiers'); box.innerHTML = ''; const filter = ($('#mod-filter').value || '').toLowerCase();
  const groupsM = {}; for (const m of state.modifiers) { if (filter && !m.name.toLowerCase().includes(filter) && !(m.description || '').toLowerCase().includes(filter)) continue; (groupsM[m.group] ||= []).push(m); }
  const order = ['style', 'face', 'plan', 'section', 'proportions', 'stance', 'greenhouse', 'front', 'rear', 'lower_body', 'sculpt', 'custom'];
  const keys = Object.keys(groupsM).sort((a, b) => (order.indexOf(a) + 99 * (order.indexOf(a) < 0)) - (order.indexOf(b) + 99 * (order.indexOf(b) < 0)));
  for (const g of keys) {
    const det = el('details', { class: 'group', open: ['style', 'face', 'proportions', 'custom'].includes(g) || filter ? '' : null });
    det.append(el('summary', {}, g, el('small', {}, `${groupsM[g].length}`)));
    for (const m of groupsM[g]) {
      const row = el('div', { class: 'mod' + (Math.abs(m.value - m.default) > 1e-9 ? ' changed' : ''), title: m.description || '' });
      const range = el('input', { type: 'range', min: m.min, max: m.max, step: 0.01, value: m.value });
      const num = el('input', { type: 'number', min: m.min, max: m.max, step: 0.01, value: (+m.value).toFixed(2) });
      const label = el('label', {}, el('span', {}, m.name.includes('/') ? m.name.split('/')[1] : m.name), range);
      const apply = (v) => { v = Math.max(m.min, Math.min(m.max, +v)); m.value = v; range.value = v; num.value = v.toFixed(2); row.classList.toggle('changed', Math.abs(v - m.default) > 1e-9); queueMod(m.name, v); };
      range.addEventListener('input', () => apply(range.value)); num.addEventListener('change', () => apply(num.value));
      row.append(label, num, el('button', { class: 'rst', title: 'reset', onclick: () => apply(m.default) }, '↺'));
      det.append(row);
    }
    box.append(det);
  }
}
$('#mod-filter').addEventListener('input', renderModifiers);
$('#btn-reset-mods').addEventListener('click', () => { const vals = {}; for (const m of state.modifiers) if (Math.abs(m.value - m.default) > 1e-9) { vals[m.name] = m.default; m.value = m.default; } setMod(vals); renderModifiers(); });
$('#btn-random').addEventListener('click', () => { const vals = {}; for (const m of state.modifiers) if (['proportions', 'greenhouse', 'front', 'rear', 'stance', 'lower_body'].includes(m.group) && m.min < 0) { const v = +(Math.max(-0.6, Math.min(0.6, (Math.random() - 0.5) * 0.9))).toFixed(2); vals[m.name] = v; m.value = v; } setMod(vals); renderModifiers(); });

// ---------------------------------------------------------------- sidebar: connectors
function renderConnList() {
  const box = $('#conn-list'); if (!box) return; box.innerHTML = ''; const f = ($('#conn-filter').value || '').toLowerCase();
  for (const c of connectors) {
    if (f && !c.name.toLowerCase().includes(f) && !c.tags.join(' ').includes(f)) continue;
    const row = el('div', { class: 'ci' + (c.name === selected ? ' sel' : '') + (c.disabled ? ' off' : ''), onclick: () => select(c.name) },
      el('span', { class: 'kind ' + c.kind, title: c.kind }), el('span', { class: 'name' }, c.name), el('span', { class: 'comp' }, c.component || (c.disabled ? 'disabled' : '—')));
    box.append(row);
  }
}
$('#conn-filter').addEventListener('input', renderConnList);
const kindClass = { point: 'PointConnector', circle: 'CircleConnector', rectangle: 'RectangleConnector', polygon: 'PolygonConnector' };
const accepts = (comp, kind) => comp.accepts.some(a => a === 'Connector' || a === kindClass[kind] || (a === 'PolygonConnector' && ['rectangle', 'circle', 'polygon'].includes(kind)));
function renderConnDetail() {
  const box = $('#conn-detail'); const c = connectors.find(x => x.name === selected);
  if (!c) { box.hidden = true; return; } box.hidden = false; box.innerHTML = '';
  const ov = c.override || {}; const tr = ov.translate || [0, 0, 0], ro = ov.rotate_deg || [0, 0, 0];
  box.append(el('h3', {}, el('span', { class: 'kind ' + c.kind }), ' ', c.name), el('p', { class: 'hint' }, `${c.kind} · tags: ${c.tags.join(', ')} · owner: ${c.owner}`));
  const grid = el('div', { class: 'grid' });
  const numIn = (v, cb) => { const i = el('input', { type: 'number', step: 0.005, value: (+v).toFixed(3) }); i.addEventListener('change', () => cb(+i.value)); return i; };
  const update = (patch) => setOverride(c.name, { ...ov, ...patch });
  grid.append(el('span', {}, 'move'), ...[0, 1, 2].map(i => numIn(tr[i], v => { const t = [...tr]; t[i] = v; update({ translate: t }); })));
  grid.append(el('span', {}, 'rotate°'), ...[0, 1, 2].map(i => numIn(ro[i], v => { const t = [...ro]; t[i] = v; update({ rotate_deg: t }); })));
  if (c.kind === 'circle') grid.append(el('span', {}, 'radius'), numIn(c.radius, v => update({ radius: v })), el('span'), el('span'));
  if (c.kind === 'rectangle') grid.append(el('span', {}, 'size'), numIn(c.width, v => update({ width: v })), numIn(c.height, v => update({ height: v })), el('span'));
  if (c.kind === 'polygon') grid.append(el('span', {}, 'scale'), numIn(ov.scale || 1, v => update({ scale: v })), el('span'), el('span'));
  box.append(grid);
  const rowG = el('div', { class: 'row' });
  for (const mode of ['translate', 'rotate', 'scale']) rowG.append(el('button', { class: gizmo.mode === mode ? 'active' : '', onclick: () => { gizmo.setMode(mode); renderConnDetail(); } }, mode));
  rowG.append(el('button', { onclick: () => setOverride(c.name, null), title: 'remove this connector\'s override' }, 'Reset'));
  box.append(rowG);
  // component assignment
  const sel = el('select'); sel.append(el('option', { value: '' }, '(default)'));
  for (const comp of state.components) if (accepts(comp, c.kind)) sel.append(el('option', { value: comp.name, selected: comp.name === c.component ? '' : null }, comp.name));
  const optsTa = el('textarea', { placeholder: '{"spokes": 7}' }); optsTa.value = Object.keys(c.options || {}).length ? JSON.stringify(c.options, null, 1) : '';
  const dis = el('input', { type: 'checkbox' }); dis.checked = !!c.disabled;
  const applyAssign = async () => { let o = null; try { o = optsTa.value.trim() ? JSON.parse(optsTa.value) : null; } catch (e) { return msg('options must be JSON', false); } try { await api.json('/api/assign', { selector: c.name, component: sel.value || null, options: o }); markDirty(); await fetchAssembly(); msg(`assigned ${c.name}`); } catch (e) { msg(e.message, false); } };
  box.append(el('div', { class: 'row' }, el('span', { class: 'hint' }, 'component'), sel, el('button', { onclick: applyAssign }, 'Apply')), optsTa,
    el('label', { class: 'chk' }, dis, ' disabled (no component attached)'));
  dis.addEventListener('change', async () => { try { await api.json('/api/disable', { selector: c.name, disabled: dis.checked }); markDirty(); await fetchAssembly(); } catch (e) { msg(e.message, false); } });
  const o = c.frame.origin; box.append(el('p', { class: 'hint' }, `origin ${o.map(v => v.toFixed(3)).join(', ')} · normal ${c.frame.z_axis.map(v => v.toFixed(2)).join(', ')}` + (c.kind !== 'point' ? ` · ${(c.width || 0).toFixed(3)} × ${(c.height || 0).toFixed(3)} m` : '')));
}
function renderComponents() { const box = $('#comp-list'); box.innerHTML = ''; for (const c of state.components) box.append(el('div', { class: 'comp' }, el('b', {}, c.name), c.morphable ? ' · morphable' : '', el('div', { class: 'a' }, `accepts ${c.accepts.join(', ')}` + (c.default_for.length ? ` · default for ${c.default_for.join(', ')}` : '')), el('div', {}, c.description || ''), Object.keys(c.options).length ? el('div', { class: 'a' }, 'options ' + JSON.stringify(c.options)) : '')); }

// ---------------------------------------------------------------- sidebar: sculpt
$('#sc-enable').addEventListener('change', e => { sculpt.on = e.target.checked; if (sculpt.on) { gizmo.detach(); msg('sculpt mode: left-drag on the body'); } else attachGizmo(); });
$('#sc-radius').addEventListener('input', e => { sculpt.radius = +e.target.value; $('#sc-radius-v').value = sculpt.radius.toFixed(2); });
$('#sc-strength').addEventListener('input', e => { sculpt.strength = +e.target.value; $('#sc-strength-v').value = sculpt.strength.toFixed(3); });
$('#sc-mode').addEventListener('change', e => { sculpt.sign = +e.target.value; });
$('#sc-mirror').addEventListener('change', e => { sculpt.mirror = e.target.checked; });
$('#sc-undo').addEventListener('click', () => { const s = sculpt.strokes.pop(); if (s) { sculpt.delta = s; applySculpt(); } });
$('#sc-clear').addEventListener('click', () => { sculpt.delta.fill(0); sculpt.strokes = []; applySculpt(); });
$('#sc-save').addEventListener('click', async () => {
  const name = $('#sc-name').value.trim() || 'sculpt'; const bytes = new Uint8Array(sculpt.delta.buffer); let bin = ''; for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  try { const r = await api.json('/api/target', { name, offsets_b64: btoa(bin) }); sculpt.delta.fill(0); sculpt.strokes = []; state.modifiers = r.modifiers; await fetchBody(); renderModifiers(); markDirty(); msg(`saved ${r.path} as ${r.modifier}`); $('#sc-status').textContent = `saved as ${r.modifier} (slider under "custom")`; } catch (e) { msg(e.message, false); }
});

// ---------------------------------------------------------------- sidebar: config
async function loadConfigText() { try { $('#cfg-text').value = await api.text('/api/config.yaml'); } catch (e) {} }
$('#cfg-reload').addEventListener('click', loadConfigText);
$('#cfg-apply').addEventListener('click', async () => { try { state = await api.json('/api/config', { yaml: $('#cfg-text').value }); connectors = state.connectors; await fetchBody(); renderSidebar(); updateHeader(); autoAssemble(); $('#cfg-status').textContent = 'applied'; } catch (e) { $('#cfg-status').textContent = e.message; msg(e.message, false); } });
const carName = () => state.config.name || 'untitled';
$('#cfg-export').addEventListener('click', async () => {
  try {
    $('#cfg-status').textContent = 'exporting…';
    if (WEB) { await api.download('/api/export.zip', { formats: ['obj', 'json'] }, `${carName()}.zip`); $('#cfg-status').textContent = `downloaded ${carName()}.zip`; return; }
    const r = await api.json('/api/export', { formats: ['obj', 'json'] }); $('#cfg-status').textContent = 'wrote ' + r.files.map(f => f.split('/').slice(-2).join('/')).join(', ');
  } catch (e) { $('#cfg-status').textContent = e.message; }
});
$('#btn-save').addEventListener('click', async () => {
  try {
    if (WEB) { await api.download('/api/config.yaml', null, `${carName()}.yaml`); state.dirty = false; updateHeader(); msg(`downloaded ${carName()}.yaml`); return; }
    let path = state.config_path; if (!path) { path = prompt('Save config as (path):', 'configs/untitled.yaml'); if (!path) return; }
    const r = await api.json('/api/save', { path }); state.config_path = r.saved; state.dirty = false; updateHeader(); loadConfigText(); msg('saved ' + r.saved);
  } catch (e) { msg(e.message, false); }
});
$('#btn-assemble').addEventListener('click', () => fetchAssembly().catch(e => msg(e.message, false)));

// ---------------------------------------------------------------- toolbar
function switchTab(name) { document.querySelectorAll('.tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === name)); document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.id === 'tab-' + name)); if (name === 'config') loadConfigText(); }
document.querySelectorAll('.tabs button').forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab)));
function setView(v) {
  const L = 5.2; const t = new THREE.Vector3(0, 0, 0.7);
  const views = { '3q': [L * 0.75, L * 0.55, L * 0.42], front: [L * 1.6, 0, 0.9], side: [0, L * 1.6, 0.8], top: [0.001, 0, L * 1.8], rear: [-L * 1.6, 0, 0.9] };
  camera.position.set(...views[v]); orbit.target.copy(t); orbit.update();
}
document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => setView(b.dataset.view)));
$('#opt-connectors').addEventListener('change', e => { opts.showConns = e.target.checked; groups.conns.visible = opts.showConns; if (!opts.showConns) gizmo.detach(); else attachGizmo(); });
$('#opt-labels').addEventListener('change', e => { opts.showLabels = e.target.checked; buildConnectors(); });
$('#opt-seams').addEventListener('change', e => { opts.showSeams = e.target.checked; groups.seams.visible = opts.showSeams; });
$('#opt-flat').addEventListener('change', e => { opts.flat = e.target.checked; refreshMaterials(); });
$('#opt-wire').addEventListener('change', e => { opts.wire = e.target.checked; refreshMaterials(); });
$('#opt-xray').addEventListener('change', e => { opts.xray = e.target.checked; refreshMaterials(); });
$('#opt-cut').addEventListener('input', e => { opts.cut = +e.target.value; clipPlane.constant = opts.cut; });
$('#opt-auto').addEventListener('change', e => { opts.autoAssemble = e.target.checked; });
window.addEventListener('keydown', e => { if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return; if (e.key === 'g') gizmo.setMode('translate'); if (e.key === 'r') gizmo.setMode('rotate'); if (e.key === 's') gizmo.setMode('scale'); if (e.key === 'Escape') { selected = null; buildConnectors(); renderConnList(); renderConnDetail(); } });

// ---------------------------------------------------------------- boot
(async function boot() {
  try {
    setView('3q'); await loadState(); await fetchBody(); await fetchAssembly(); msg('ready');
  } catch (e) { msg('failed to load: ' + e.message, false); console.error(e); }
})();
