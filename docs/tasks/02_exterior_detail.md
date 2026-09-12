# Task 02 — Exterior component detail

Branch: `agent/exterior-detail`.  Files you own: `makecar/components/exterior.py`
(you may split it into a package `makecar/components/exterior/`),
`makecar/body/connectors_extra.py` (`extra_exterior_connectors` only),
`tests/test_components*.py` additions.  Do NOT edit `makecar/body/generator.py`
or `makecar/body/connectors.py` (another agent owns them).  Read
`docs/tasks/README.md` first, then `makecar/components/base.py`, `fills.py`,
`exterior.py`, `makecar/body/connectors.py` (to learn the connector contract).

## Goal
Every exterior component should look like the real part at real scale.  In
order of visual impact:

1. **Wheels** (MorphableComponent `wheel.alloy`): tyre with a real tread
   pattern (circumferential grooves + angled blocks, as geometry, not just
   material), sidewall profile with a shoulder, rim with a visible lip, spokes
   as actual 3-D spokes (lofted, with depth) instead of a painted disc — keep
   the morphable part (radius/width/rim_ratio/dish) fixed-topology and make the
   spoke count a build option; brake disc + caliper behind the spokes (caliper
   colour option), lug nuts, centre cap, valve stem.  Add `wheel.steel` (steel
   wheel with a hubcap) as a second component.
2. **Lamps**: headlight with a real internal structure — two projector bowls or
   a reflector bowl + LED bar, a DRL strip along the top edge of the aperture,
   amber indicator segment at the outboard end; lens as a slightly bulged pane.
   Taillight with a light-guide ring graphic, reversing lamp, reflex reflector
   strip and a body-colour bezel.  Both must adapt to the aperture polygon
   (Coons fill of the exact grid is already available via `fill_connector`).
3. **Front/rear fascia furniture**: fog lamps (new PointConnectors emitted from
   `extra_exterior_connectors` at the bumper corners, plus a `light.fog`
   component), tow-hook cover, lower grille surround, rear reflectors, licence
   plate lamps, rear fog/reverse.  Grille: add `grille.honeycomb` and
   `grille.mesh` variants; the slat grille needs real slat depth and a badge
   cut-out.
4. **Mirrors, wipers, trim**: door mirror with a proper housing loft (not a
   box), turn-signal strip, base plinth; windshield wipers (two, parked, from a
   new `wipers` RectangleConnector at the cowl emitted from
   `extra_exterior_connectors`); black window trim/frames around the side glass
   apertures (component on the glass connector may add it); door handles with
   a finger recess; side markers/reflectors; roof antenna base.
5. **Underside**: exhaust system (pipe run from the front along the tunnel to
   the tips — a `sweep_profile` along a path derived from measurements), rear
   diffuser fins for sports/coupe, mud flaps option, tow eye.

Every component keeps working for every connector the body emits for all 8
styles (radii, widths and aperture shapes differ a lot — check the pickup,
sports and van cases).

## Acceptance
* Tests green + new tests (e.g. wheel tread face count grows with `tread_blocks`,
  fog lamp connectors emitted for every style, headlight fits every style's
  aperture with no inverted faces).
* `docs/agent_reports/02_exterior_detail/`: close-up renders of each part
  (wheel at 3 angles, headlight, taillight, mirror, grille, underside) before
  and after, for sedan and sports.
* Face budget: the four wheels together under 24k faces; whole car under ~120k.
* REPORT.md.
