# Task 01 — Body surfacing realism

Branch: `agent/body-surfacing`.  Files you own: `makecar/body/generator.py`,
`makecar/body/params.py`, `makecar/body/styles.py`, `makecar/body/targets.py`,
`makecar/body/connectors.py` (only where noted), tests under `tests/test_body*.py`.
Read `docs/tasks/README.md` first.

## Goal
The body loft is recognisable but generic.  Make it read like a production car
body when rendered.  Work through the items below in order; each must be
visible in a render and covered by a test where a number can be checked.

1. **Front and rear ends.**  Today the nose/tail are a raked ring closed by two
   shrunken rings and an apex.  Replace with a proper fascia: a bumper volume
   that stands proud of the body sides at the bottom (bumper corner radius in
   plan ~0.25 m, in elevation the bumper face is near-vertical from
   `front_bumper_bottom` up to a bumper crease at ~0.55–0.65 m, then steps back
   ~0.03 m to the hood leading edge which overhangs the grille opening).  Add
   parameters (with sensible defaults per style) for bumper crease height, hood
   overhang and a front splitter/lower lip; rear: bumper with a licence-plate
   recess, a diffuser step at the bottom for sports/coupe, tailgate lower panel
   for hatch/wagon/SUV/van.  Keep the apertures (headlight/taillight zones) and
   the `nose_ring`/`tail_ring`/`apex_*` vertex groups meaningful so
   `emit_connectors` keeps working (grille, plates, badges, exhausts).  Adding
   rings to the fascia is fine (bump `N_FASCIA`, `LIBRARY_VERSION`).
2. **Fender/hood crease and shoulder.**  Real bodies have a crisp crease where
   the fender top meets the hood (and a softer one along the shoulder above the
   belt).  Give the section a proper hood-to-fender step and make the shoulder
   line (E→F) a definite radius rather than a soft roll; add
   `shoulder_radius` and `fender_crease` parameters + modifiers.
3. **Rocker and lower body.**  Add a rocker panel with a flat vertical face and
   a lower flare-in to the underbody (a real sill is ~0.15 m tall, slightly
   inboard of the door skin); doors should tuck under to the sill with a small
   step (door bottom edge).  Wheel-arch lips: a flange ~0.02 m wide standing
   proud of the fender, following the arch, not just a bulge.
4. **Greenhouse.**  Glass must sit ~8 mm below the pillar surface (recess the
   F→G glass strip slightly and the roof-arc glass zones) with a visible frame
   band; add a drip rail / roof edge radius; A-pillar and C-pillar widths as
   parameters.  Side glass top edge should follow the roof rail curve, not a
   straight line.
5. **Underbody.**  A flat floor pan with a longitudinal tunnel bump (only
   visible in low renders), rear valance under the bumper, front air dam.
6. **Style presets.**  Tune the eight presets against real reference
   dimensions (state your references in the report — e.g. Camry/Golf/Outback/
   RAV4/F-150/Mustang/MX-5/Sienna).  Windshield rake, cowl height, belt line,
   roof height, overhangs and wheel size must be within a few cm of the class.

## Acceptance
* `python3 -m pytest -q` green, new tests for each new parameter's effect
  (measure the mesh: e.g. bumper face stands proud by ≥ the parameter value).
* `docs/agent_reports/01_body_surfacing/`: before/after renders for sedan,
  hatchback, suv, pickup, sports from front-3/4, rear-3/4, side, front, plus a
  close-up of the nose and of the A-pillar/glass recess.  Use
  `makecar.pipeline.build_car` + `Renderer` (see `render_views`).
* All 15 sample configs still build via `python -m makecar samples -o /tmp/x`
  (you may need to update presets/params; keep config keys backward compatible).
* REPORT.md as described in the README of this folder.
