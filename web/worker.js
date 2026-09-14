// Pyodide worker: loads the makecar package and answers viewer API calls with makecar.viewer.session.dispatch.
import { loadPyodide } from 'https://cdn.jsdelivr.net/pyodide/v314.0.6/full/pyodide.mjs';

const PYODIDE = 'https://cdn.jsdelivr.net/pyodide/v314.0.6/full/';

const status = (text, error = false) => postMessage({ type: 'status', text, error });
const encoder = new TextEncoder();
let ready = null;
let handle = null;

async function fetchOk(path, as) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${r.statusText}`);
  return as === 'json' ? r.json() : as === 'text' ? r.text() : r.arrayBuffer();
}

async function init(requested) {
  const t0 = performance.now();
  status('loading python…');
  const [pyodide, manifest] = await Promise.all([loadPyodide({ indexURL: PYODIDE }), fetchOk('manifest.json', 'json')]);
  status('loading numpy…');
  await pyodide.loadPackage(['numpy', 'pyyaml']);
  const name = manifest.configs.includes(requested) ? requested : manifest.default;
  status(`loading makecar (${name})…`);
  const [pkg, yaml, cache] = await Promise.all([
    fetchOk('makecar.zip'),
    fetchOk(`configs/${name}.yaml`, 'text'),
    manifest.cache[name] ? fetchOk(manifest.cache[name]) : null,
  ]);
  const FS = pyodide.FS;
  pyodide.unpackArchive(pkg, 'zip', { extractDir: '/app' });
  for (const dir of ['/cache', '/work', '/work/configs']) FS.mkdirTree(dir);
  if (cache) FS.writeFile('/cache/' + manifest.cache[name].split('/').pop(), new Uint8Array(cache));
  FS.writeFile(`/work/configs/${name}.yaml`, yaml);
  status('building the body…');
  pyodide.runPython(`
import json, os, sys
sys.path.insert(0, "/app")
os.environ["MAKECAR_CACHE"] = "/cache"
os.chdir("/work")
from makecar.config import CarConfig
from makecar.viewer.session import ViewerSession, dispatch

session = ViewerSession(CarConfig.load("configs/${name}.yaml"), output_dir="/work/output")

def handle(method, path, body):
    return dispatch(session, method, path, json.loads(body) if body else None)
`);
  handle = pyodide.globals.get('handle');
  status(`python ready in ${((performance.now() - t0) / 1000).toFixed(1)}s`);
}

function call(method, path, body) {
  const out = handle(method, path, body);
  try {
    const bytes = out.get(2);
    const view = bytes.getBuffer('u8');
    const buf = view.data.slice().buffer;
    view.release();
    bytes.destroy();
    return { status: out.get(0), buf };
  } finally {
    out.destroy();
  }
}

onmessage = async ({ data }) => {
  if (data.type === 'init') {
    ready = init(data.config).catch((e) => { status('failed to start python: ' + e.message, true); throw e; });
    return;
  }
  let reply;
  try {
    await ready;
    reply = call(data.method, data.path, data.body);
  } catch (e) {
    reply = { status: 500, buf: encoder.encode(JSON.stringify({ error: String(e.message || e) })).buffer };
  }
  postMessage({ id: data.id, ...reply }, [reply.buf]);
};
