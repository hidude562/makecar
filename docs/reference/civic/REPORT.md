# 2022 Honda Civic Si Sedan — reference-config report

## Target

This is a **config approximation**, not a replica or a licensed model, of the
U.S.-market **2022 Honda Civic Si Sedan** — the four-door sedan of the
eleventh-generation Civic.  The target is specifically the Si because its
235/40 R18 tyres and small deck-lid spoiler make the side profile a useful
sport-sedan reference.

## Published data and sources

The primary dimensional source is Honda's [2022 Civic Si announcement and
specification release](https://hondanews.com/en-US/honda-automobiles/releases/all-new-2022-honda-civic-si-brings-the-passion-sets-new-benchmark-for-performance).
The dimensional values were cross-checked against the [2022 Civic Si spec
listing](https://www.caranddriver.com/honda/civic-si/specs/2022/honda_civic-si_honda-civic-si_2022).
The release's [eleventh-generation Civic Sedan overview](https://hondanews.com/en-US/honda-automobiles/releases/all-new-11th-generation-civic-sedan-fully-revealed-in-production-form-with-sporty-design-advanced-technology-cutting-edge-safety-features)
was also used to confirm the body generation.

| Published specification | Value | Config decision |
|---|---:|---|
| Overall length | 184.0 in / 4.6736 m | Wheelbase + 0.8941 m front + 1.0439 m rear overhang |
| Overall width | 70.9 in / 1.8009 m | `width: 1.8009` |
| Overall height | 55.5 in / 1.4097 m | `roof_height: 1.408` (the generated roof crown supplies the final 1.4092 m) |
| Wheelbase | 107.7 in / 2.7356 m | `wheelbase: 2.7356` |
| Front / rear overhang | 35.2 / 41.1 in; 0.8941 / 1.0439 m | Direct base parameters |
| Front / rear track | 60.5 / 61.6 in; 1.5367 / 1.5646 m | Recorded reference; the engine exposes an arch track rather than a separately configurable hub track |
| Tyres | 235/40 R18 | `wheel_diameter: 0.6452`, `tire_width: 0.235` |
| Ground clearance | 5.3 in / 0.1346 m | `ground_clearance: 0.134` |

## Reference image

[`reference_side.jpg`](reference_side.jpg) is a clean manufacturer-style side
profile of the target car.  It is 37,370 bytes, below the 400 kB limit, and is
kept strictly as a private modelling reference.  Its cached source, identified
original asset and retrieval details are recorded in [`SOURCE.md`](SOURCE.md).

## Render-driven tuning

I looked at the final scale-matched [`comparison.png`](comparison.png) and
[`three_quarter.png`](three_quarter.png) with the image viewer.  The comparison
uses the reference image's car-width scale (193.43 px/m); the generated car is
orthographic and placed immediately below the reference.

Starting from the `sedan` macro, the config sets the published wheelbase and
overhangs directly, then uses a low roof, shallow shoulder and modest
`tumblehome` for the Civic's low, long greenhouse.  The hood was initially too
steep and wedge-like when compared directly: it was raised, straightened and
the `face/wedge` weight was reduced to `0.03`.  The final pass shortens the
windscreen run, carries the roof/rear-window shape rearward, uses the Si tyre
size and ten-spoke wheels, and switches the generic grille to honeycomb.  This
produces the Civic's long hood, low roof, close wheel-to-arch relationship and
short three-box rear at the correct scale.

### Intentional limits

The body profile is the closest expression available through config, but it is
still visibly an approximation.  In particular, the config vocabulary cannot
make the Civic's thin, joined LED headlamp/DRL signature, blacked-out Si grille
surround, precise tail-lamp wraparound, deck-lid spoiler, or its specific
crease topology.  The current generic headlamp, taillamp, mirror and bumper
components consequently make the three-quarter image read less specifically
than the side silhouette.  The fixed sedan fascia also stays more vertical and
faceted than the Civic's stamped front and rear corners.  These are vocabulary
limits; no files under `makecar/` were changed to work around them.

## Verification

```text
python3 docs/reference/civic/verify.py
length     published=4.6736 measured=4.6736 error=-0.00%
width      published=1.8009 measured=1.8129 error=+0.67%
height     published=1.4097 measured=1.4092 error=-0.03%
wheelbase  published=2.7356 measured=2.7356 error=+0.00%
```

All four acceptance dimensions are within 3%.  `verify.py` builds
`configs/civic.yaml` through `makecar.pipeline.build_car`, prints the table,
and regenerates both committed PNGs.

- `python3 -m makecar build configs/civic.yaml -o /tmp/civic` succeeded
  (61,491 faces).
- `python3 -m pytest -q` was run before and after this config-only task.  Both
  runs had the same pre-existing result: **1010 passed, 2 failed**.  The two
  failures are the sports-exhaust clearance checks in
  `tests/test_components_connectors_extra.py` (minimum 0.0010255 m where
  0.004 m is required); this work neither changes nor touches that engine
  code.
- `git status` was checked: no files below `makecar/` are modified.
