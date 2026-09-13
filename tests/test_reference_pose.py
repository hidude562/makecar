"""Pinhole camera and silhouette tools used to fit 3/4 reference photos."""
import numpy as np
import pytest

from makecar.reference.pose import PinholeCamera, silhouette, iou, align_to_mask, refine_pose, resize_mask
from makecar.body.params import BodyParams
from makecar.body.generator import BodyGenerator


def test_front_view_orientation_and_pinhole_scaling():
    cam = PinholeCamera(yaw=0, pitch=0, roll=0, distance=10, focal=1000, cx=500, cy=300, tz=0.0)
    c, left, up = cam.project([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(c[:2], [500, 300])
    assert left[0] > 500 and abs(left[1] - 300) < 1e-9, "the car's left must appear on image right from the front"
    assert up[1] < 300, "world up must be image up"
    assert left[0] - 500 == pytest.approx(1000 * 1 / 10)


def test_side_view_nose_direction():
    cam = PinholeCamera(yaw=90, pitch=0, roll=0, distance=10, focal=1000, cx=500, cy=300, tz=0.0)
    nose, tail = cam.project([[2, 0, 0], [-2, 0, 0]])
    assert nose[0] < 500 < tail[0], "seen from the left side the nose points to image left"


def test_iou_and_resize():
    a = np.zeros((40, 40), bool); a[10:30, 10:30] = True
    b = np.zeros((40, 40), bool); b[20:30, 10:30] = True
    assert iou(a, a) == 1.0 and iou(a, b) == pytest.approx(0.5)
    small, s = resize_mask(a, 20)
    assert small.shape == (20, 20) and s == 0.5


@pytest.fixture(scope="module")
def body():
    m = BodyGenerator(BodyParams()).build()
    return m.vertices, m.triangulated()[0]


def test_refine_recovers_a_perturbed_pose(body):
    V, T = body
    W, H = 320, 180
    truth = PinholeCamera(yaw=38, pitch=9, roll=1.0, distance=8.0, focal=300, cx=W / 2 + 8, cy=H / 2 - 5, tz=0.7)
    target = silhouette(V, T, truth, (W, H))
    start = PinholeCamera(yaw=30, pitch=6, roll=0.0, distance=8.0, focal=260, cx=W / 2, cy=H / 2, tz=0.7)
    start = align_to_mask(V, T, start, target, iters=2)
    cam, score = refine_pose(V, T, start, target, free=("yaw", "pitch", "roll", "focal", "cx", "cy"), maxiter=250)
    assert score > 0.95
    assert abs(cam.yaw - truth.yaw) < 3.0 and abs(cam.pitch - truth.pitch) < 3.0
