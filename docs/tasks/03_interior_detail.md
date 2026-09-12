# Task 03 — Interior detail and accuracy

Branch: `agent/interior-detail`.  Files you own: `makecar/components/interior.py`,
`makecar/components/interior_helpers.py`, `makecar/body/connectors_extra.py`
(`extra_interior_connectors` only), `tests/test_interior*.py` (new).  Do NOT
edit the body generator or `connectors.py`.  Read `docs/tasks/README.md`, then
`makecar/components/base.py`, `interior.py`, `interior_helpers.py`, and the
interior part of `makecar/body/connectors.py` for the connector contract.

## Goal
Make the cabin read like a real production interior at real ergonomic scale,
and make it *enclosed*: from the driver's-eye view nothing should look
unfinished.  In order:

1. **Ergonomics check.**  Verify against SAE-style packaging numbers: H-point
   to heel ~0.8–0.9 m horizontal; steering wheel centre 0.55–0.65 m ahead of
   and 0.30–0.35 m above the H-point, rim diameter 0.37 m, column 20–25° to
   horizontal; seat cushion 0.50 m long, 0.48–0.52 m wide, backrest 0.60–0.65 m,
   headrest top ≥ 0.80 m above the H-point... Write these as assertions in a
   test that reads the connectors/meshes for all 8 styles and fix whatever
   fails (report the corrections).
2. **Seats**: sculpted cushion and backrest with side bolsters and a centre
   panel with stitch lines (thin inset grooves as geometry), headrest on two
   posts, seat rails on the floor, side airbag tag/recline lever, rear bench
   with a fold seam and three headrests; `seat.sport` variant with deeper
   bolsters and an integrated headrest.  Seat belts: buckle stalk beside the
   cushion, belt strap from the B-pillar (emit a `belt_anchor_*` connector from
   `extra_interior_connectors` at the B-pillar belt height) down to the buckle.
3. **Dashboard**: instrument binnacle hood, a proper centre stack (screen,
   HVAC, buttons with real 10–15 mm button sizes), air vents with louvres,
   glovebox seam, passenger-side airbag seam, dash-to-windshield cowl trim
   strip along the windshield base, A-pillar trims (new
   `pillar_trim_*` connectors from `extra_interior_connectors`, built from the
   windshield aperture edge), steering column stalks (two), instrument cluster
   with speedo/tacho faces, needles, and a hooded shade.
4. **Steering wheel**: rim with a round cross-section that flattens at the
   bottom for sports, three spokes with thumb rests, airbag pad with a badge,
   horn pad edge.
5. **Console / floor / doors**: gear lever with a boot, handbrake lever or
   e-brake switch, cup holders with a sliding cover, armrest lid seam; floor
   mats (four) as slightly raised plates; pedals with rubber pads and a foot
   rest; door sill scuff plates; door cards with an armrest, pull handle,
   window-switch cluster, speaker grille with a real mesh pattern, map pocket.
6. **Headliner and closure**: sun visors with vanity mirrors, dome light lens,
   grab handles on a base, B-pillar trims and rear parcel shelf with speaker
   grilles; the inner side of the wheel houses in the trunk/hatch area for
   hatch/wagon/SUV so cargo views don't show the outer shell.

## Acceptance
* Tests green + ergonomic assertion test for all 8 styles + tests for new
  connectors and components.
* `docs/agent_reports/03_interior_detail/`: driver's-eye, passenger-eye,
  roof-off cutaway, rear-seat view, close-ups of a seat, the cluster and the
  door card — before and after, sedan and suv.
* Interior face budget under ~50k faces per car.
* REPORT.md.
