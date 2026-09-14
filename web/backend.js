// Runs the makecar viewer API inside a Pyodide worker, so the editor works as a static site.
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
let nextId = 1;
let dead = null;

const fail = (text) => {
  dead = text;
  const m = document.querySelector('#msg');
  if (m) { m.textContent = text; m.style.color = '#ff8080'; }
  const buf = new TextEncoder().encode(JSON.stringify({ error: text })).buffer;
  for (const resolve of pending.values()) resolve({ ok: false, statusText: text, buf: buf.slice(0) });
  pending.clear();
};
worker.onerror = (e) => { e.preventDefault(); fail('the python worker crashed: ' + (e.message || 'could not load')); };

worker.onmessage = ({ data }) => {
  if (data.type === 'status') {
    const m = document.querySelector('#msg');
    if (m) { m.textContent = data.text; m.style.color = data.error ? '#ff8080' : ''; }
    return;
  }
  const p = pending.get(data.id);
  if (!p) return;
  pending.delete(data.id);
  p({ ok: data.status < 400, statusText: data.statusText || '', buf: data.buf });
};

const config = new URLSearchParams(location.search).get('config');
worker.postMessage({ type: 'init', config });

window.makecarTransport = (url, body) => new Promise((resolve) => {
  if (dead) { resolve({ ok: false, statusText: dead, buf: new TextEncoder().encode(JSON.stringify({ error: dead })).buffer }); return; }
  const id = nextId++;
  pending.set(id, resolve);
  worker.postMessage({ type: 'call', id, method: body ? 'POST' : 'GET', path: url, body: body ? JSON.stringify(body) : null });
});
