# Task 06 — Refit the exhaust route to the archetype-shaped underbody

Branch: `agent/shape-integration`.  Files you own: `makecar/components/exterior.py`,
`makecar/body/connectors_extra.py`, and the test files that cover them.  Do NOT
change `makecar/body/generator.py`, `params.py`, `styles.py`, `shapes.py` or
`targets.py` — the archetype shapes are the reference now.  Read
`docs/tasks/README.md`, then `docs/agent_reports/05_shape_archetypes/REPORT.md`
and the exhaust section of `docs/agent_reports/04_integration/REPORT.md`.

## Situation
The new `face/*`, `plan/*` and `section/*` shape archetypes reshape the
underbody, and the sports style's defaults (`face/wedge`, `section/tumblehome`)
lower and narrow its floor pan.  The swept exhaust system no longer fits:

```
FAILED tests/test_components_connectors_extra.py::test_exhaust_cans_and_branch_turn_clear_current_pan[sports]
FAILED tests/test_components_connectors_extra.py::test_exhaust_pan_and_tip_fit_follow_morphs[sports-modifiers0]
```

The measured minimum clearance is **1.0 mm** against a required **4 mm**.  The
existing code in `Exhaust.build_local` already fits the *cans* between the road
and the tunnel roof using `floor_grid`, but the **branch turn pipes are not
clearance-checked at all** — they are swept along their path and can graze the
pan wherever the archetypes lower it.

## What to do
1. Apply the same pan-clearance treatment to the whole route, not just the two
   cans: the tunnel run, the branch turns and the terminal legs must all keep at
   least the required clearance below the measured floor grid, on every style
   and with the archetype sliders at their extremes.  Derive it from the actual
   measured pan, never from a per-style constant.
2. Keep everything the exhaust already guarantees: continuity between adjacent
   sweep rings, the branches meeting the exact `exhaust_L/R` tip connectors, and
   the catalyst/silencer ground clearance the integration report claims.
3. **Do not weaken the clearance tests** and do not change the archetype
   geometry to make room.  If a route genuinely cannot fit on some extreme
   combination, reroute it (a flatter oval section, a different branch split
   point, tucking beside the tunnel rather than under it) and say so.
4. Re-check the other styles and the archetype extremes for the same class of
   problem, and add tests for whichever you fix.

## Acceptance
* `python3 -m pytest -q` fully green.
* All 15 samples build: `python3 -m makecar samples -c configs -o /tmp/si_samples`.
* Renders into `docs/agent_reports/06_shape_integration/`: an underbody view of
  the sports, sedan and pickup before and after, plus the rear three-quarter of
  the sports.  Look at them and confirm the route is not visibly grazing or
  poking through the floor.
* `docs/agent_reports/06_shape_integration/REPORT.md`.
