# Task 08 — A dense library of body detail targets

Branch: `agent/detail-targets`.  You own `makecar/body/*` (`generator.py`,
`params.py`, `shapes.py`, `styles.py`, `targets.py`, plus a new
`makecar/body/details.py` if you want one) and `tests/test_detail_targets.py`.
Do NOT edit anything under `makecar/components/`, `assembly.py` or `pipeline.py`.
Read `docs/tasks/README.md`, then `docs/agent_reports/05_shape_archetypes/REPORT.md`.

## Why

Three real cars were approximated using only the config
(`configs/{lamborghini,civic,tesla}.yaml`, and the reports in
`docs/reference/*/REPORT.md`).  They are not good enough, and the reason is
that the target vocabulary is too coarse to describe a real body surface:

* **Window openings are straight lines.**  The glass apertures are axis-aligned
  rectangles in (station, ring-index) grid space, and the belt and roof-rail
  profiles that bound them have only a handful of keypoints.  No car has a
  straight DLO: real ones have a Hofmeister kink, a rising or falling belt, a
  tapering rear, and radiused glass corners.
* **The hood is a single crown scalar.**  No spine, no power bulge, no fender
  peaks, no leading-edge dip, no curvature control along its length.
* **There are no corner controls.**  You cannot inset, chamfer or vertically
  taper a corner, so every car has the same plan-view corner treatment at every
  height.
* Similar gaps exist for the roof, the door skin, the arches, the tail and the
  lower body.

## What to build

**Dozens of targets, not a few.**  Aim for roughly 40–60 new unipolar or
bipolar modifiers organised into families, in the same style as the existing
`face/*`, `plan/*`, `section/*` archetypes: ordinary targets that blend, are
generated from the parametric generator wherever possible, and compose with
everything already there.  A suggested taxonomy, which you should extend where
you find a gap:

| family | targets to cover |
|---|---|
| `dlo/*` (the glass opening) | Hofmeister kink at the C-pillar, belt rise toward the rear, belt drop toward the front, rear taper of the opening, front and rear glass corner radii, frame thickness, quarter-window split, windshield wrap into the side glass, rear screen curvature |
| `hood/*` | transverse crown curvature (not just a scalar height), central spine, power bulge, fender peaks running fore-aft, leading-edge dip, edge chamfer, cowl step |
| `roof/*` | transverse arc strength, longitudinal crown, rear taper in plan, rail step and drip rail |
| `side/*` | convex and concave door skin, high and low character crease, lower-door undercut, sharp versus soft shoulder, front and rear hip flare |
| `corner/*` | front and rear plan inset, chamfer, and vertical taper so a corner can be narrower at the top than at the bottom |
| `arch/*` | round, square and trapezoid arch shapes, arch lip, per-axle flare, eyebrow |
| `tail/*` | deck lip spoiler, Kamm cut-off, boat-tail taper, deck curvature |
| `lower/*` | valance drop, diffuser, air dam, rocker flare |

Also add, as body parameters (components consume them, another agent handles the
wheels themselves): **per-axle wheel diameter and per-axle track**, so a car can
have a larger rear wheel and a wider rear track, with the arches following.

## How

1. **Give the profiles enough keypoints.**  Most of the straightness comes from
   `Profile` curves built from four or five keypoints derived from scalars.
   Where a target needs local control (a kink, a dip, a local radius), the
   profile needs a keypoint there.  Adding keypoints is expected.
2. **Build a regional deformation helper.**  Most of these targets are a smooth
   local displacement over a region of the body defined by a station range, a
   ring-index range and a falloff — the car equivalent of MakeHuman's local face
   targets.  Write that helper once (in `details.py`) and author the families
   through it, rather than hand-rolling each field.  Prefer generator parameters
   plus `DifferentialTargetBuilder` where the feature is genuinely a parameter
   of the loft; use the regional helper for local features that are not.
3. **Topology.**  Fixed topology must still hold across every style and every
   target value.  You may change the ring segmentation or station counts **once**
   if the extra resolution is genuinely needed (say so, and bump
   `LIBRARY_VERSION`) — but the same vertex and face count and the same face
   index tuples must then hold everywhere afterwards.
4. **Do not break the connector contract.**  Connector names, tags, frames and
   meta keys stay as they are; components are fitted to them and another agent
   is working on the wheels in parallel.  `tests/test_fit_sanity.py` must stay
   green for every style.

## Acceptance

* `python3 -m pytest -q` fully green, no test deleted or weakened, and new
  tests: each target moves the measurement it claims to move, family
  normalisation where families are mutually exclusive, topology invariance
  across all targets and styles, and connectors still following the mesh.
* All 15 sample configs still build.
* **Look at it.**  Into `docs/agent_reports/08_detail_targets/`, render a sweep
  strip (0, 0.5, 1.0) for every target, grouped by family, from the view that
  best shows it — the DLO family in side elevation, corners in plan, hood in
  three-quarter and front. View them, and keep tuning until each target visibly
  does what its name says and nothing tears or self-intersects.
* A `docs/agent_reports/08_detail_targets/REPORT.md` with the full list of
  targets, what each does, how it is built, and what you still cannot express.
