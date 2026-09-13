"""The viewer's HTTP API, end to end against a real server thread."""
import base64
import json
import threading
import http.client

import numpy as np
import pytest

from makecar.config import CarConfig
from makecar.viewer.server import ViewerSession, make_server
from makecar.viewer.packing import pack_parts, mesh_parts
from makecar.geometry import primitives as P


def parse_mcb(buf: bytes):
    assert buf[:4] == b"MCB1"
    hlen = int.from_bytes(buf[4:8], "little")
    header = json.loads(buf[8 : 8 + hlen])
    base = 8 + hlen
    for p in header["parts"]:
        p["pos"] = np.frombuffer(buf, dtype=np.float32, count=p["pos"][1], offset=base + p["pos"][0]).reshape(-1, 3)
        p["idx"] = np.frombuffer(buf, dtype=np.uint32, count=p["idx"][1], offset=base + p["idx"][0]).reshape(-1, 3)
        p["map"] = np.frombuffer(buf, dtype=np.uint32, count=p["map"][1], offset=base + p["map"][0]) if p["map"][1] else None
    return header


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("viewer")
    cfgp = tmp / "car.yaml"
    cfgp.write_text("name: viewer_test\nbody:\n  style: sedan\n  custom_targets: targets\n")
    session = ViewerSession(CarConfig.load(cfgp), output_dir=tmp / "out")
    httpd = make_server(session, "127.0.0.1", 0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield {"port": port, "session": session, "tmp": tmp}
    httpd.shutdown()


class Client:
    def __init__(self, port):
        self.port = port

    def get(self, path):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        c.request("GET", path)
        r = c.getresponse()
        return r.status, r.getheader("Content-Type"), r.read()

    def post(self, path, obj):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=300)
        c.request("POST", path, body=json.dumps(obj), headers={"Content-Type": "application/json"})
        r = c.getresponse()
        return r.status, r.getheader("Content-Type"), r.read()


def test_packing_roundtrip():
    m = P.box(1, 2, 3, material="paint")
    m.materials["paint"] = m.materials.get("paint") or __import__("makecar.geometry.mesh", fromlist=["Material"]).Material("paint", (1, 0, 0))
    parts = mesh_parts(m, component="body")
    blob = pack_parts(parts, {"k": 1})
    hlen = int.from_bytes(blob[4:8], "little")
    assert hlen % 4 == 0, "buffer offsets must stay 4-byte aligned for browser typed arrays"
    h = parse_mcb(blob)
    assert h["extra"] == {"k": 1}
    for odd in ({"k": 1, "pad": "x"}, {"k": 1, "pad": "xy"}, {"k": 1, "pad": "xyz"}):
        assert int.from_bytes(pack_parts(parts, odd)[4:8], "little") % 4 == 0
    p = h["parts"][0]
    assert p["material"] == "paint" and p["pos"].shape == (8, 3) and p["idx"].shape == (12, 3)
    assert p["map"] is not None and len(p["map"]) == 8  # body parts carry the full-mesh vertex map
    assert p["idx"].max() < len(p["pos"])


def test_static_and_state(server):
    c = Client(server["port"])
    status, ctype, body = c.get("/")
    assert status == 200 and b"makecar viewer" in body and "text/html" in ctype
    status, ctype, body = c.get("/static/app.js")
    assert status == 200 and b"TransformControls" in body
    assert c.get("/static/../server.py")[0] == 404
    status, _, body = c.get("/api/state")
    st = json.loads(body)
    assert status == 200
    assert {m["name"] for m in st["modifiers"]} >= {"wheelbase", "style/suv", "face/wedge"}
    assert any(c_["name"] == "steering_wheel" for c_ in st["connectors"])
    from makecar.body.params import BodyParams
    assert st["measurements"]["wheelbase"] == pytest.approx(BodyParams().wheelbase, abs=0.02)
    assert st["custom_targets_dir"].endswith("targets")


def test_body_binary_and_modifiers(server):
    c = Client(server["port"])
    status, ctype, body = c.get("/api/body.bin")
    assert status == 200 and ctype == "application/octet-stream"
    h = parse_mcb(body)
    assert all(p["component"] == "body" for p in h["parts"])
    n = h["extra"]["n_vertices"]
    full = np.frombuffer(base64.b64decode(h["extra"]["full_positions_b64"]), dtype=np.float32).reshape(-1, 3)
    assert len(full) == n and len(h["extra"]["mirror"]) == n
    # every part vertex maps back onto the full mesh position
    for p in h["parts"]:
        assert np.allclose(full[p["map"]], p["pos"], atol=1e-5)
    # mirror index really mirrors y
    mir = np.asarray(h["extra"]["mirror"])
    assert np.allclose(full[mir][:, 1], -full[:, 1], atol=1e-3)
    wb0 = h["extra"]["measurements"]["wheelbase"]
    status, _, body = c.post("/api/modifiers", {"values": {"wheelbase": 1.0}})
    h2 = parse_mcb(body)
    assert status == 200 and h2["extra"]["measurements"]["wheelbase"] > wb0 + 0.4
    assert server["session"].config.body["modifiers"] == {"wheelbase": 1.0}
    # style slider maps back into body.style
    c.post("/api/modifiers", {"values": {"style/suv": 0.5, "wheelbase": 0.0}})
    assert server["session"].config.body["style"] == {"suv": 0.5}
    assert server["session"].config.body["modifiers"] == {}
    c.post("/api/modifiers", {"values": {"style/suv": 0.0}})
    assert server["session"].config.body["style"] == "sedan"


def test_connector_override_assign_disable(server):
    c = Client(server["port"])
    status, _, body = c.post("/api/connector", {"name": "steering_wheel", "override": {"translate": [0, 0, 0.05], "radius": 0.2}})
    r = json.loads(body)
    assert status == 200 and r["dirty"] is True
    sw = next(x for x in r["connectors"] if x["name"] == "steering_wheel")
    assert sw["radius"] == pytest.approx(0.2) and sw["override"] == {"translate": [0, 0, 0.05], "radius": 0.2}
    assert server["session"].config.connector_overrides()["steering_wheel"]["radius"] == 0.2
    # assemble and confirm the component followed
    status, _, body = c.post("/api/assemble", {})
    h = parse_mcb(body)
    names = {p["component"] for p in h["parts"]}
    assert "steering_wheel" in names and "body" in names
    assert h["extra"]["stats"]["components"] > 50
    # reset override
    c.post("/api/connector", {"name": "steering_wheel", "override": None})
    assert "steering_wheel" not in server["session"].config.connector_overrides()
    # assignment + disable
    status, _, body = c.post("/api/assign", {"selector": "wheel", "component": "wheel.alloy", "options": {"spokes": 7}})
    assert status == 200 and server["session"].config.components["assign"]["wheel"] == {"component": "wheel.alloy", "options": {"spokes": 7}}
    c.post("/api/disable", {"selector": "antenna", "disabled": True})
    assert "antenna" in server["session"].config.components["disable"]
    c.post("/api/disable", {"selector": "antenna", "disabled": False})
    assert "antenna" not in server["session"].config.components["disable"]
    status, _, body = c.post("/api/assign", {"selector": "grille", "component": "wheel.alloy"})
    # wheel on a rectangle is rejected at assemble time
    status2, _, body2 = c.post("/api/assemble", {})
    assert status2 == 500 and b"cannot attach" in body2
    c.post("/api/assign", {"selector": "grille", "component": None, "options": None})
    assert "grille" not in server["session"].config.components["assign"]


def test_config_yaml_roundtrip_and_save(server):
    c = Client(server["port"])
    status, ctype, body = c.get("/api/config.yaml")
    assert status == 200 and b"style" in body or b"name" in body
    yaml_text = body.decode() + "\npalette:\n  paint: '#123456'\n"
    status, _, body = c.post("/api/config", {"yaml": yaml_text})
    st = json.loads(body)
    assert status == 200 and st["config"]["palette"]["paint"] == "#123456"
    status, _, body = c.post("/api/config", {"yaml": "body: {style: spaceship}"})
    assert status == 500 and b"unknown" in body
    out = server["tmp"] / "saved.yaml"
    status, _, body = c.post("/api/save", {"path": str(out)})
    assert status == 200 and out.exists()
    again = CarConfig.load(out)
    assert again.raw["palette"]["paint"] == "#123456"
    assert server["session"].dirty is False


def test_sculpt_target_save(server):
    c = Client(server["port"])
    n = server["session"].car_body.library.base.n_vertices
    off = np.zeros((n, 3), dtype=np.float32)
    off[:50, 2] = 0.02
    status, _, body = c.post("/api/target", {"name": "Roof Bump!", "offsets_b64": base64.b64encode(off.tobytes()).decode()})
    r = json.loads(body)
    assert status == 200 and r["modifier"] == "custom/Roof_Bump"
    assert (server["tmp"] / "targets" / "Roof_Bump.target").exists()
    assert any(m["name"] == "custom/Roof_Bump" and m["value"] == 1.0 for m in r["modifiers"])
    assert server["session"].config.body["modifiers"]["custom/Roof_Bump"] == 1.0
    # and it really displaces the body
    status, _, body = c.get("/api/body.bin")
    h = parse_mcb(body)
    full = np.frombuffer(base64.b64decode(h["extra"]["full_positions_b64"]), dtype=np.float32).reshape(-1, 3)
    c.post("/api/modifiers", {"values": {"custom/Roof_Bump": 0.0}})
    h0 = parse_mcb(c.get("/api/body.bin")[2])
    full0 = np.frombuffer(base64.b64decode(h0["extra"]["full_positions_b64"]), dtype=np.float32).reshape(-1, 3)
    assert np.allclose((full - full0)[:50, 2], 0.02, atol=1e-4)


def test_export(server):
    c = Client(server["port"])
    status, _, body = c.post("/api/export", {"formats": ["json"]})
    r = json.loads(body)
    assert status == 200 and any(f.endswith("assembly.json") for f in r["files"])
