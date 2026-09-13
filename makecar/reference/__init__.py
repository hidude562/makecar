"""Reference-driven modelling: measure real cars from orthographic-ish images.

    calib = Calibration.from_hubs(front_px, rear_px, wheelbase, r_front, r_rear, front_on_right=True)
    top = trace_edge(image, u0, u1, band, polarity="enter", signal="luma")
    xz  = calib.to_world(top)

A two-point similarity (scale, roll, translation) is solved from the two hub
centres, whose world positions are exactly known from the published wheelbase
and tyre sizes, so no ground line or manual scale bar is needed.  Perspective is
not modelled: use photos taken square-on from a distance.
"""
from .trace import Calibration, trace_edge, clean_trace, resample_xz, draw_overlay, load_rgb  # noqa: F401
