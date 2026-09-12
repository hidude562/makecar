# Task 09 — Wheels and tyres worth looking at

Branch: `agent/wheels-tyres`.  You own the wheel and tyre components in
`makecar/components/exterior.py` and `tests/test_wheels_tyres.py`.  Do NOT edit
anything under `makecar/body/` — another agent is reworking the body targets in
parallel and will add per-axle wheel diameter and track parameters.
Read `docs/tasks/README.md` first.

## Why

The wheels are the most-looked-at part of a car and ours do not hold up in a
close render.  Current gaps:

* the tyre is a revolved profile with shallow tread and no real sidewall
  detail — no bead, no shoulder blocks, no lettering relief, no aspect-ratio
  control, so a 35-profile tyre looks like a 65-profile one;
* the rim is a dished disc with spokes cut by material rather than a real
  barrel with a concave face, spoke draft and a proper lip;
* there is one spoke pattern; real wheels differ enormously and that difference
  is most of a car's character;
* brake discs, calipers and the gap behind the spokes read as flat.

## What to build

1. **Tyre.**  Aspect ratio as a first-class option (a 245/35 R20 must look
   different from a 205/65 R15): sidewall height derived from section width and
   aspect, with a correct bead seat, sidewall curvature, shoulder radius and a
   tread band whose width follows the section width.  Tread pattern as real
   geometry with selectable families (directional V, asymmetric block,
   all-terrain lug, slick) and a groove depth option.  Sidewall lettering as
   shallow relief is welcome if it stays within budget.
2. **Rim.**  A real barrel with an outer and inner lip, a concave or convex
   face controlled by an offset/dish option, spokes as lofted solids with draft
   and a fillet where they meet the hub and the rim, and a bolt circle with
   recessed lug seats.  Spoke families as a selectable option: `mesh`,
   `five_spoke`, `twin_five`, `multi_spoke`, `turbine`, `dish`, `steel_cap`.
   Make it a `MorphableComponent` family where that is the natural fit, so
   sizes fit their connector the way the existing wheel does.
3. **Brakes.**  Vented disc with visible vanes between two faces, a drilled or
   slotted option, a caliper that actually straddles the disc with pad backing
   plates, and a dust shield behind.
4. **Per-axle sizing.**  When the body starts emitting different front and rear
   wheel connectors (bigger rear, wider rear track), your components must simply
   fit them; they already read radius and width from the connector, so verify
   that a staggered setup works and add a test using synthetic connectors so you
   are not blocked waiting for the body agent.

Keep the four-wheel budget under about 40k faces at default options, and make
the expensive detail (tread blocks, lettering, disc vanes) scale down through
options so a 15-car sample run stays quick.

## Acceptance

* `python3 -m pytest -q` green, plus new tests: aspect ratio changes sidewall
  height, each spoke family produces a distinct face count and topology, the
  caliper straddles the disc without intersecting it, staggered front/rear
  connectors fit, and the face budget holds.
* **Look at it.**  Into `docs/agent_reports/09_wheels_tyres/`: each spoke family
  rendered head-on and at three-quarter, a tread close-up per pattern, a
  low-profile versus high-profile tyre comparison, and a brake close-up with the
  wheel hidden. View them and iterate until they read as real wheels.
* `docs/agent_reports/09_wheels_tyres/REPORT.md`.
