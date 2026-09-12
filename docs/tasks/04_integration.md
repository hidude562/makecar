# Task 04 — Integrate the reshaped body with the detail components

Branch: `agent/integration` (already checked out in this worktree, based on
the three-way merge of `agent/body-surfacing`, `agent/exterior-detail` and
`agent/interior-detail`).  Read `docs/tasks/README.md` first, then the three
reports in `docs/agent_reports/0{1,2,3}_*/REPORT.md` so you know what each
work stream intended.

## Situation
Three agents worked in parallel from the same base:

* **01 body surfacing** reshaped the body: stepped bumpers with a crease and a
  rear plate pocket, folded fender creases, radiused shoulders, vertical
  rockers, recessed glazing, a floor pan with a tunnel and lower bumper skirts,
  and eight presets retuned to real reference dimensions (see its report).
* **02 exterior detail** and **03 interior detail** built components that were
  fitted to the *old* body surface.

The merge is committed and the engine builds, but **25 tests fail** because
component geometry no longer matches the reshaped body.  Your job is to make
the components fit the new body.  Run `python3 -m pytest -q` to see them.

## Hard rules
* **Do not change the body to suit the components.**  `makecar/body/generator.py`,
  `params.py` and `styles.py` are the reference now; the body's shape, its
  fixed topology and the retuned presets are correct and must not regress.
  You may only touch `makecar/body/connectors.py` where a *mount point* is
  demonstrably in the wrong place on the new surface (e.g. sitting inside the
  new bumper volume), and then only to move the mount, never to change the shell.
* **Do not weaken a test to make it pass.**  If a test encodes a real
  requirement (clearance, no inverted faces, symmetry, continuity), fix the
  geometry.  You may only adjust a test's expectation when the expectation
  itself was tied to the old body's shape, and then you must say so in the
  report with the reason and the new expected value.
* Keep every existing public contract: connector names, tags, frame
  conventions, meta keys, component names and their options.

## The failures, by cluster
1. **Exhaust** (`test_exhaust_routes_to_existing_tips`,
   `test_exhaust_geometry_is_continuous_and_meets_tips`, most styles).  The
   swept system no longer meets the `exhaust_L/R` tips: the new rear bumper and
   floor pan moved both the tips and the tunnel.  Re-derive the tunnel and
   branch paths from the current measurements and the actual tip connectors,
   and keep the catalyst/silencer ground clearance the exterior report claims.
2. **Intake / grille / plate** (`test_intake_backing_clears_uncut_fascia`,
   `test_grille_variants_and_badge_clearance`).  The stepped bumper changed the
   fascia plane, so the intake backing and plate plinth no longer clear it.
   Re-fit them against the new fascia, keeping the ray-based clearance test
   honest.
3. **Lamp panes** (`test_lamp_panes_fit_apertures_without_inverted_faces`,
   pickup and sports).  The new aperture grids fold the pane on those styles.
4. **Mirror mounts** (`test_extra_mirror_symmetry_and_mount_directions`, coupe
   and sports).  Mount direction/symmetry broke on the reshaped shoulder.
5. **Interior** (`test_dashboard_controls_and_closure_all_styles[sports]`).  The
   retuned sports cabin is tighter; re-pack the controls rather than shrinking
   the clearance requirement.

## Also required
* After the suite is green, run `python3 -m makecar samples -c configs -o /tmp/int_samples`
  and confirm all 15 sample cars build.
* **Look at the result.**  Render sedan, sports, pickup and SUV from front-3/4,
  rear-3/4 and side, plus close-ups of the nose, the rear bumper/exhaust area
  and a wheel, into `docs/agent_reports/04_integration/`, and view them.  Fix
  anything that is visibly wrong even if no test covers it: parts floating off
  the surface, geometry poking through the body, doubled or missing trim.
* `tests/test_fit_sanity.py` must stay green for every style — it is the
  whole-car fit guard (interior inside the shell, exterior only protruding
  where a real car does, nothing below the road, wheels touching it).

## Acceptance
* `python3 -m pytest -q` fully green, with no test deleted or weakened except
  as justified above.
* All 15 samples build.
* `docs/agent_reports/04_integration/REPORT.md`: what you changed per cluster,
  any test expectations you retuned and why, the renders you looked at, and
  anything still visibly imperfect.
