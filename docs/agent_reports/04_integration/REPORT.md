# 04 — Body/component integration

## Result

Integrated on `agent/integration`, starting at `c8a44076540c091a88970cf1022c56c894d97ffe`.
Final engine revision: **`78d260b`**; subsequent commits contain evidence/report files only.

- Baseline: **25 failed, 608 passed** (`66.01s`).
- Final: **812 passed** (`129.42s`), including the unchanged whole-car fit suite. See [test output](pytest_final.txt).
- All **15 sample cars** rebuilt successfully through the actual CLI after the final suite passed. See [sample output and export checks](samples_final.txt).
- Rendered and viewed all six required views of sedan, sports, pickup and SUV, with matched baseline comparisons.
- Reference body shape, topology, presets and target library are unchanged. No new runtime dependencies. Small local commits; **nothing pushed**.

## Changes by failure cluster

### Exhaust and lower valance

The floor centreline is now the raised tunnel crown, not the flat pan. Routing now samples the complete measured A→B underbody cross-sections, retaining them as additive, JSON-serializable `floor_grid` metadata. The trunk follows the tunnel; branches descend beneath the flat pan **before** turning sideways, follow pan slope changes, and use rounded bends with slope-aware tube clearance. Catalyst/silencer sections fit both the road and the measured ceiling over their complete footprints. The original 50mm ground-clearance requirement remains.

Visual inspection caught a second problem after endpoint tests were green: the old high tips occupied painted bumper volume, while clear underfloor feeds formed exposed hooks below them. A shared, mount-only `exhaust_mount` helper now samples the actual lower rear-valance edge. The 38mm-radius outlets sit 4mm beneath it. Examples, left mount XYZ in metres; right mirrors Y:

| Style | Before | After |
|---|---|---|
| Sedan | `(-2.484274, .383855, .375)` | `(-2.484000, .383855, .233)` |
| Sports | `(-1.938711, .336114, .345)` | `(-1.908500, .336114, .203)` |
| Pickup | `(-3.074122, .515194, .505)` | `(-3.075500, .515194, .363)` |
| SUV | `(-2.327160, .425853, .485)` | `(-2.327000, .425853, .343)` |

The shared derivation gives exact branch endpoints and rearward terminal tangents without changing connector names, radii or frames. `dual` outlets now sit horizontally rather than vertically; circular/oval inlet connections join the central feed to the sleeve(s). `dual` and `length` remain available, and the existing exact 2:1 dual/single face-count test remains unchanged.

Added pan/can/terminal clearance, actual-shell outlet, modifier and option regressions. Original sweep continuity and ground assertions were not weakened. The routes are deliberately visible where a low camera can see below the valance; see limitations below.

### Intake, grille, plate and rear furniture

- Grille backing now seats its **back**, not merely its face, ahead of the measured whole-footprint fascia mount. Removed the obsolete lower-intake rake compensation.
- Stock EU front plate plinth is 18mm deep instead of the old 65mm pedestal, with its back reaching the skin. Taller US plates and modified fascias that overlap either insert receive enough projection to clear the actual assembly. Ray regressions cover both upper and lower inserts, including raised pickup/SUV bumpers and lowered sports hood fronts.
- Removed the doubled default nose roundel: the separate body badge remains; the additional grille badge is opt-in with `badge: true`. Its slat clearance is retained. No phantom badge-shaped gap remains when disabled.
- Thin slats retain a real flat face instead of a knife-edge bevel. The 45mm sports opening gets two default slats, eliminating crowded crosshatching. Explicit `slats: 5` and `slats: 14` still produce those counts; other component option names remain available.
- Rear plate lamps now fit the real recessed pocket. Rear reflectors moved 50mm upward onto the upright face rather than the downward diffuser return.

The failing test named `extra_mirror_symmetry_and_mount_directions` was actually detecting those rear-reflector directions on coupe/sports. The side-mirror mounts did not need moving. Added all-style tests verify real mirrored side-mirror vertices and rearward-facing mirror panes as well as rear-furniture visibility.

### Lamp panes and optics

Pickup's tall aperture exposed an incorrect assumption that the longest grid axis was horizontal. Lamp UV orientation now follows world height; the DRL is above the projectors and the indicator is outboard.

Sports has a genuinely crossing tiny noncorner boundary notch in the average mount-plane projection. Exact preservation of that notch cannot produce an embedded outward pane. The component-only fit moves one source sample **1.78736mm** onto its neighbouring chord, under a physical black **2.5mm-radius seal**. Four corners, connector points/grid and the reference body remain unchanged. The octagonal seal's inscribed radius exceeds the allowance. All other stock lamp sample displacements are zero.

Housing and lens share the fitted grid. Concave cells use the valid alternate diagonal instead of flipping isolated triangle winding. This also fixes a previously hidden wagon taillight triangle. Every stock headlight/taillight now has strictly positive exported triangle normals against its original connector normal, consistent disk topology and a simple boundary.

A compatibility follow-up prevents newly introduced whole-build exceptions for supported scalar modifiers. Mild sports hood changes use a genuine original-normal fit with a bounded adaptive seal (about 2.06–2.13mm notch correction for `-.25`/`-.5`). Strongly wrapped apertures use boundary-constrained triangulation in a feasible projection within the connector-normal hemisphere, with explicit `ComponentResult.info["fit_warnings"]`. Tests cover eight reported modifier/style pairs at seven values, preserve corners and bound boundary allowance to 3mm. Mirrored fallbacks are deterministic.

A truly unprojectable boundary gets an explicitly warned, bounded convex-envelope pane rather than an intersecting fill or a new CLI abort. **That fallback is not an exact-aperture-fit guarantee.** Extreme optics remain approximate; see the specific visible failure below.

### Sports interior and visual integration fixes

Dashboard packing now shares the actual console-nose rise calculation with the console. The sports HVAC/screen/centre-vent stack moves upward **10.333mm**, retaining full control dimensions and a **10mm** HVAC-to-console gap. Added low-H-point and left/right-drive cases rather than reducing clearance.

Full-context wheel renders also revealed carpet/firewall slabs through pickup/SUV front rims. The carpet's outboard nose and lower firewall now wrap around measured wheel houses; the upper cowl width and central floor remain. Exact triangle clipping against wheel cylinders guards all styles, rather than deleting obstructing faces.

Pickup's last default bench previously protruded through the cab into the bed. Only its rear seat mounts were repacked: all three default row connectors remain, the front row stays fixed, and equal pitch becomes **721.5mm**. Rows two/three move forward **138.5/277mm**, leaving **29.02mm** behind the final default headrest at the trimmed wall. Sampled solid inter-row gaps remain above 267mm. Explicit two-row pickup retains its original 860mm pitch. Custom rear-bench recline is not silently overridden: actual `cab_rear_clearance` and warnings expose infeasible requested poses.

## Test expectations retuned — and why

Only these two old-shape numerical assumptions were replaced:

1. **Grille badge radius:** the old fixed `.046m` threshold assumed a 40mm badge. The reshaped sedan opening is 105mm high and produces a 30.45mm badge radius. The test now measures the actual radius and retains **6mm additional radial clearance**, giving `.03645m` on sedan. It explicitly requests `badge: true` after removal of the doubled default badge.
2. **Intake ray positions:** old rays at local Y `±.060m` missed the smaller new insert footprints. Rays now use `±.43 * connector.height` and zero, covering the real 70–140mm stock intake heights. The original **greater-than-2mm** skin clearance remains.

No tests were deleted. Stock lamp assertions were **strengthened** to actual exported triangles; the old taillight exception relative to an already folded reference was removed. Exhaust continuity, symmetry, console gaps, road clearance and `tests/test_fit_sanity.py` were not relaxed. The suite gained **179 cases** overall.

## Verification and reproducibility

From the integration worktree:

```sh
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q
OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 python3 -m makecar samples -c configs -o /tmp/int_samples
```

The CLI describes 11 configs; their variants expand to 15 cars. All 15 output directories contain the expected 11 nonempty files: body/car OBJ+MTL, connectors/assembly JSON, blueprint/connector-map SVG, preview/interior PNG and specification text. All JSON/SVG files parse, PNGs verify, and exported OBJ face counts match the CLI. Sample assemblies span **57,777–64,791 faces**, below the approximately 120k budget. The generated sports sample preview was also viewed.

Verified with `git diff c8a4407 --exit-code --` that these files are unchanged:

- `makecar/body/generator.py`
- `makecar/body/params.py`
- `makecar/body/styles.py`
- `makecar/body/targets.py`
- `tests/test_fit_sanity.py`

`git diff --check` is clean. Runtime requirements remain NumPy/PyYAML; PIL was used only to inspect/verify images and make contact sheets.

## Renders actually viewed

Each linked sheet contains front three-quarter, rear three-quarter, side, nose, rear bumper/exhaust and wheel views. All four final sheets and all twelve full-size closeups were opened and inspected after the final engine freeze; baseline comparisons were also viewed.

| Style | Baseline sheet | Final sheet | Final complete-assembly faces |
|---|---|---|---:|
| Sedan | [Before](baseline_sedan_contact_sheet.png) | [After](after_sedan_contact_sheet.png) | 61,169 |
| Sports | [Before](baseline_sports_contact_sheet.png) | [After](after_sports_contact_sheet.png) | 58,083 |
| Pickup | [Before](baseline_pickup_contact_sheet.png) | [After](after_pickup_contact_sheet.png) | 65,207 |
| SUV | [Before](baseline_suv_contact_sheet.png) | [After](after_suv_contact_sheet.png) | 61,227 |

Full-size files live in `baseline/<style>/` and `after/<style>/` as:
`front_three_quarter.png`, `rear_three_quarter.png`, `side.png`, `nose_closeup.png`, `rear_bumper_exhaust_closeup.png`, `wheel_closeup.png`.

The required set uses the real NumPy Renderer, **800×600 with 2× supersampling**, identical paint/light/camera settings locked in [cameras.json](cameras.json), and the **complete assembly for every view**. Nothing is omitted, clipped or recoloured to conceal an intersection. Per-style manifests record imported-source hashes, PNG hashes, settings and zero removed faces. Final hashes match committed engine `78d260b`; baseline is a fixed `git archive` of `c8a4407`, not current code relabelled as before.

Supplemental comparisons are explicitly diagnostic, not substitutes for full-context views:

- `exhaust_{before,after}_{sedan,sports}_{underbody,side,rear_closeup}.png`: 12 images, selected shell/wheels/tips/system with a grey, below-lit pan to expose routing.
- `sports_controls_{before,after}_{closeup,front,oblique}.png` and pickup/SUV `footwell`/`wheel_footwell` comparisons: 18 images showing isolated controls, shaped footwells and contextual wheels.
- `lamps/`: 18 stock lamp comparisons and 36 modifier comparisons, each front/oblique/seam-context before and after.

Total: **140 verified PNGs**, including eight contact sheets; nine render JSON files parse. Reproduce the source images with:

```sh
export OPENBLAS_NUM_THREADS=1
python3 docs/agent_reports/04_integration/render_integration.py baseline
python3 docs/agent_reports/04_integration/render_integration.py after
python3 docs/agent_reports/04_integration/render_exhaust.py --phase both
python3 docs/agent_reports/04_integration/render_interior.py --phase before
python3 docs/agent_reports/04_integration/render_interior.py --phase after
python3 docs/agent_reports/04_integration/render_lamps.py
python3 docs/agent_reports/04_integration/render_lamps.py --modifiers
```

## Remaining visible/model limitations

- These remain generic, visibly faceted low-poly cars, not exact replicas. The protected shell and simplified materials were not redesigned.
- **Exhaust feeds are conspicuous in low views**, especially sports: real curved branches and inlet hardware remain visible beneath the valance, including in the CLI preview. They are connected and clear the shell/road, but less visually concealed than a production car's underbody. No pan penetration, omitted geometry or camera change was used to hide them.
- Default pickup retains the existing three-row connector contract, although **`seat_rows: 2` is the sensible stock-like crew-cab choice**. Large custom three-row bench recline can still exceed the cab; it now reports the actual negative clearance instead of silently changing the requested seat.
- Extreme custom lamps are not universally realistic. In particular **sports `hood_front_height: -1` leaves a rearward-facing/folded protected aperture**: the [context render](lamps/sports_hood_front_height_-1_headlight_after_seam_context.png) shows fascia occlusion and exposed optical fragments. The warning-bearing embedded fallback restores build compatibility, **not feasible shell packaging**. Sports `fender_crease: 1` also shows fan-like optical faceting. Mild `hood_front_height: -.25` fits cleanly. Avoid those extreme configurations when an unobscured stock-like lamp is required; an exact fix would require body changes outside this brief.
