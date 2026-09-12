# Task 02 — Exterior component detail

## Outcome

Implemented on `agent/exterior-detail`, without changing `makecar/body/generator.py`, `makecar/body/connectors.py`, the interior extension function, or the body's fixed topology. Runtime dependencies remain NumPy and PyYAML. All changes are committed locally; nothing was pushed.

**Verification:** baseline **280 passed**; final **390 passed** with `python3 -m pytest -q`. Added 110 tests in `tests/test_components_detail.py` and `tests/test_components_connectors_extra.py`.

## What changed

### Wheels

- `wheel.alloy` retains its `MorphableComponent` library and radius/width/rim_ratio/dish modifiers. The fixed-topology carcass, barrel, rolled lips and hub are morphed first; count-dependent details are built afterward using the fitted dimensions.
- Actual circumferential tread channels and angled raised blocks, with 4mm block relief; shoulder/sidewall profile; open gaps between solid, dished radial spoke lofts.
- Brake disc and hat, edge ventilation detail, curved caliper, five hexagonal lugs, centre cap and valve stem. Brake surfaces are separated to prevent z-fighting.
- Added `wheel.steel`: stamped dish/webs with open ventilation slots and a domed hubcap. It shares the fitted tyre and hardware.
- Options: `spokes` (3–12), `spoke_width`, `tread_blocks` (0–64 per lane), `tread_grooves` (0–4), `rim_ratio`, `dish`, `hubcap`, `rim_color`, `caliper_color`. Counts are bounded to protect the face budget.

### Lamps

- Headlamps now contain two reflector/projector bowls, convex inner projector panes, an upper DRL and an outboard amber indicator. `projector: false` selects one bowl plus an LED bar.
- Taillamps have a continuous red light-guide ring, clear reversing segment, geometrically fluted reflex strip and palette-matched painted bezel.
- Both outer panes use `fill_connector` with the exact body aperture grid, a slight bulge and unchanged boundary corners (apart from a 1mm normal clearance).
- Internal UVs use physical arc length, not raw semantic grid indices. This was essential: the original nonuniform sampling otherwise crowded the projectors onto the hood-side edge. Long guide segments are subdivided before surface mapping so they do not disappear inside curved housings.

### Fascia furniture

- Added default components for corner fog lamps, front/rear tow-hook covers, rear reflex reflectors, rear fog/reversing lamps, rear plate lamps and side markers.
- Slat grille has real 30mm section depth and an open roundel cut-out. Added `grille.honeycomb` (open 22mm-deep cells) and `grille.mesh` (clipped diagonal wires).
- Lower intake uses honeycomb and an open surround. Backing and cells sit ahead of the **uncut** fascia, with a rake-following lower insert. The front plate uses a 65mm mounting plinth to clear the insert, including its bottom border on sports bodies.
- Rear plate lamps point downward rather than upward; rear auxiliary lamp colours follow connector metadata.

### Mirrors, glazing and trim

- Lofted mirror housing, rear-facing convex pane, rubber surround, mounting plinth/stalk and a signal strip following the housing surface. Left/right parts are world-symmetric despite their opposite local-Y directions.
- Two parked wipers: exact windshield-grid blade paths and surface normals, rubber blades, spines, articulated arms and pivots.
- Separate rubber mouldings around side-glass apertures; pull handles over shallow sculpted finger-recess bowls; roof antenna gasket/base.

### Underside

- `exhaust.system` sweeps a tunnel pipe from the front axle region, adds catalyst/silencer cans, then branches to the exact existing exhaust-tip mounts. The original `exhaust.tip` and its `dual` option remain separate and compatible.
- Low cars use vertically flattened cans to retain at least **50mm catalyst/silencer ground clearance**. This is a geometry envelope, not a roadworthiness or engineering certification.
- Sports/coupe bodies (including positive style blends) receive a rear diffuser with tapered fins. Added a small forged rear tow eye.
- Mud flaps are opt-in through `body.hints.mud_flaps: true`; each has mounting rivets. Conditional connector emission keeps default assemblies free of unattached optional mounts.

## Connector decisions

All new mounts are emitted by `extra_exterior_connectors` only, derived from morphed vertices/measurements rather than canonical body parameters.

- Point mounts: `fog_front_L/R`, `tow_hook_cover_front/rear`, `rear_reflector_L/R`, `rear_fog_L`, `rear_reverse_R`, `plate_lamp_L/R`, `side_marker_L/R`, `exhaust_system`, `tow_eye`.
- Rectangle mounts: `wipers`; sporting `diffuser`; opt-in `mud_flap_front/rear_L/R`.
- Wipers carry world-space blade paths, normals, pivots and source UV coordinates. Exhaust carries world-space tunnel/branch paths and exact tip endpoints. Metadata is JSON-serializable.
- Fascia furniture frames use local +Z outward and +Y up. Existing connector names, tags and frames are untouched.
- The long exhaust component is mounted near the wheelbase midpoint, not at a rear tip, so its geometry remains sensibly centred relative to its connector.

Example configuration:

```yaml
body:
  style: sports
  hints: {mud_flaps: true}
components:
  defaults: true
  assign:
    wheel:
      component: wheel.alloy   # or wheel.steel
      options: {spokes: 7, tread_blocks: 48, dish: 0.05, caliper_color: "#bb3025"}
    grille:
      component: grille.honeycomb
      options: {cell_size: 0.025}
```

The `grille` tag also matches the lower intake, as in the existing selector contract. Use more specific assignment selectors when only one mount should change. Components can otherwise be disabled using their connector names or tags.

## Verification and face budgets

Checked sedan, hatchback, wagon, SUV, pickup, coupe, sports and van, plus a morphed/style-blended body.

- Wheel target topology across all 16 combined modifier extrema; actual tread/spoke count growth and geometry depth; maximum-option face budgets; brake/spoke separation; hardware and colour options; steel end-cap winding.
- Every headlamp pane has outward-facing cells in all eight styles. Pane corners reproduce the exact aperture. Tail panes preserve the original grid's cell orientations rather than silently flipping isolated cells (see limitation below).
- Connector frame handedness, side symmetry, unique names, dimensions, JSON serialization and movement when the mesh is translated.
- New solid parts checked for nondegenerate triangles, closed welded edges, consistent winding and positive signed volume. Whole assemblies are intentionally not one watertight solid.
- Both wiper blades recover the exact windshield samples and have 0.4–0.6m lengths; arm attachments and pivot centres match the contract.
- Every adjacent exhaust sweep ring is joined by the expected quads; branch junctions and terminal planes meet the original tip connectors. Catalyst/silencer clearance is tested explicitly.
- Ray/triangle tests compare intake backing and the full front-plate face/border against the actual uncut fascia and insert, catching the occlusions found in renders.
- Existing body topology, assembly, config, export and component tests remain green. `git diff --check` passes.

| Style | Complete car, including opt-in mud flaps |
|---|---:|
| Sedan | 49,166 faces |
| Hatchback | 47,742 |
| Wagon | 48,842 |
| SUV | 48,506 |
| Pickup | 54,098 |
| Coupe | 47,636 |
| Sports | 47,204 |
| Van | 48,608 |

Four default alloy wheels: **14,728 faces**. With maximum supported spoke/tread/groove options: **19,592** alloy or **20,968** steel faces for four wheels, both below 24,000. All measured cars are comfortably below 120,000 faces.

Default rendered assemblies, without mud flaps: sedan **48,830** faces, sports **46,868**. Baselines were **22,542** and **20,700**, respectively.

## Visual inspection and screenshots

Rendered and viewed both styles before and after, with matched cameras, material palette, lighting, 960×720 resolution and 2× supersampling. The baseline was frozen before editing production code. Mirror cameras were reframed using the union of before/after bounds, then **both** stages rerendered. Lens-removed inspections are explicitly labelled; their geometry is not repositioned.

Start with:

- [Sedan before/after contact sheet](comparison_sedan.png)
- [Sports before/after contact sheet](comparison_sports.png)
- [Sedan options and furniture](extras_sedan.png)
- [Sports options and furniture](extras_sports.png)

Full matched views are in `before/{sedan,sports}/` and `after/{sedan,sports}/`:

- `wheel_01_outboard.png`, `wheel_02_oblique.png`, `wheel_03_tread.png`, `wheel_context.png`
- `headlight_isolated.png`, `headlight_lens_removed.png`, `headlight_context.png`
- `taillight_isolated.png`, `taillight_lens_removed.png`, `taillight_context.png`
- `mirror_housing_isolated.png`, `mirror_glass_isolated.png`, `mirror_context.png`
- `grille_isolated.png`, `grille_context.png`
- `underside_oblique.png`, `underside_bottom.png`, `underside_furniture_isolated.png`, `exhaust_tip_isolated.png`
- `whole_front_three_quarter.png`, `whole_rear_three_quarter.png`

`after/extras/{sedan,sports}/` includes steel wheels at three angles, both optional grille variants, three mirror-signal angles, fog lamps, wipers, handles, rear furniture/plate lamps/reflectors and normal-angle lamp views. Sports also has `underside_diffuser_isolated.png`.

**129 PNGs total:** 42 before, 42 after, 41 supplemental, four contact sheets. Manifests record the exact source hashes, cameras, selections, face counts and lens-removal settings.

Reproduce from the worktree root:

```bash
python3 -m pytest -q
python3 docs/agent_reports/02_exterior_detail/render_exterior.py before
python3 docs/agent_reports/02_exterior_detail/render_exterior.py after --extras
```

The renderer script needs only NumPy. PIL was used only to assemble/inspect contact sheets. The committed `before_meshes.pkl` is this task's trusted local baseline, allowing exact old images to be regenerated without reverting source; **never load untrusted pickle snapshots**. After snapshots and render caches are ignored and regenerable.

## Open issues / deliberately limited scope

1. **Inherited taillight grids:** wagon/SUV/van contain cells pointing against the aperture's average normal even before these changes (9/7/15 cells, respectively, at 3× subdivision on the left lamp). Exact body geometry is preserved; tests ensure no new orientation reversals relative to the unmodified fill. Resolving the underlying tail-grid folds/strong reversals belongs to the body-generator workstream. Headlight acceptance passes for every style.
2. **Existing sports fascia layout is tight:** upper grille and lower intake overlap in elevation. Backing and plate visibility are fixed, but moving the underlying mounts was excluded by the protected connector contract. There are also two roundels (the original body badge and the grille's inset badge); `badge: false` disables the grille roundel/cut-out.
3. These are lightweight geometric proxies, not a manufacturing model: no tyre lettering/sipes, brake hydraulics, optical refraction, true mirror reflections or cut-through handle recesses. The shallow handle bowl stays above the unchanged door skin. Existing plate characters remain placeholders.
4. The built-in rasterizer uses flat/two-sided shading and approximate alpha sorting. It is useful for inspecting silhouettes, occlusion and details, but not a substitute for the numerical winding checks or a physically based renderer.

## Commit sequence

- `d11f567` — alloy/steel wheels, geometric tread and brakes.
- `88074fb` — aperture-fitted lighting, grilles, mirrors and trim.
- `4c768ca` — detail connectors, fascia furniture, wipers and underside.
- `b302465` — render-driven corrections and all-style acceptance tests.
- `9cc6152` — raked lower-intake visibility and ray regression tests.
- `0fbcf45` — full plate-border clearance on the sports intake.
- Final documentation commit — this report, reproducible before/after renders and manifests.
