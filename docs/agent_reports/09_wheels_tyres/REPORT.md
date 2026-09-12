# Task 09 — Wheels and tyres

## Result

Implemented on `agent/wheels-tyres`. Production changes are confined to the wheel classes in `makecar/components/exterior.py`; regression tests are in `tests/test_wheels_tyres.py`. **Nothing under `makecar/body/` was edited.** NumPy/PyYAML remain the only runtime dependencies. No pushing.

- Baseline: **1,648 passed** (`python3 -m pytest -q`, 615.74 s).
- Added **58 tests**. Full suite after the brake implementation: **1,706 passed** (620.79 s).
- **Final full suite after the render-driven caliper refinement: 1,706 passed** (`python3 -m pytest -q`, 589.66 s).
- Latest focused wheel/legacy-contract run: **71 passed**, 117 unrelated tests deselected.
- Four default wheels: **18,064 polygon faces / 37,216 triangles**, below the brief's approximately 40k-face budget.
- The stricter, unchanged legacy maximum-option test also passes: **23,760 alloy / 23,920 steel polygon faces** for four wheels, both below 24k.

## What changed

### Tyres and sizing

- `aspect_ratio` is a percentage, e.g. **35**, not `.35`. Sidewall height is `section_width * aspect_ratio / 100`; bead-seat radius is `connector.radius - sidewall_height`.
- Connector radius remains **tyre outer radius**, and `meta["tire_width"]` remains full section width in metres. Geometry respects the connector frame independently on each axle/side; no global wheel radius or track is consulted.
- Bead shoulders, curved sidewalls, a width-dependent tread band and wrapping shoulder blocks replace the old coarse profile. The outer/inboard rolled rim lips seat against the bead region.
- Four geometric tread families: `directional_v`, `asymmetric_block`, `all_terrain`, `slick`. All-terrain uses larger voids, deeper lugs and additional shoulder blocks; slick has a continuous tread surface, not invisible blocks.
- Groove depth does not enlarge the tyre. Zero depth removes tread detail; zero block count switches to continuous ribs that still reach the connector's outer radius.
- Optional shallow **0.8 mm rubber relief** spells `RADIAL` and the rounded fitted size on the outboard sidewall. It uses a tiny built-in geometric glyph set, not textures or a font dependency.

Exact size checks and rendered comparison:

| Tyre | Connector radius | Section width | Sidewall height | Bead-seat radius |
|---|---:|---:|---:|---:|
| 245/35 R20 | 339.75 mm | 245 mm | 85.75 mm | 254 mm |
| 205/65 R15 | 323.75 mm | 205 mm | 133.25 mm | 190.5 mm |

`aspect_ratio` takes precedence over `rim_ratio`. If omitted, the legacy radius-dependent `rim_ratio` default remains. The connector determines the outside size; an aspect option alone does **not** promise a particular integer R-size on an arbitrary body.

### Rims and families

- A genuinely open, stepped barrel with a drop centre and rolled lips; a darker **physical inner barrel surface**, not a disc painted between spokes.
- Independent solid spoke lofts with 20 mm section depth, 12% front-face draft, edge breaks up to 2 mm, and flared root/tip transitions approximating forged fillets. Dish varies smoothly along their length.
- `dish` supports concave and convex faces, clamped to **−25…90 mm**. Aero-dish front/back skins share a radial profile with a constant 12 mm axial wall, so negative dish cannot fold them through one another.
- Five recessed conical lug seats, actual hex heads below their mouths, centre cap/badge and valve stem. The hub perimeter is refined independently of the five-hole count.
- Seven selectable families share the fitted carcass/barrel:

| Family | Face design | Faces / wheel, other options default |
|---|---|---:|
| `mesh` | 24 crossing narrow lofts, diamond openings | 5,428 |
| `five_spoke` | five broad forged-style spokes | 4,516 |
| `twin_five` | five branching pairs / ten solids | 4,756 |
| `multi_spoke` | 18 slender radial spokes | 5,140 |
| `turbine` | nine swept, widening blades | 4,708 |
| `dish` | concave/convex aero web and eight outer bridges | 5,172 |
| `steel_cap` | stamped dish, twelve open ventilation webs, domed cap | 4,892 |

`wheel.alloy` remains the registered default `MorphableComponent`; `wheel.steel` remains compatible and defaults to `steel_cap`. The legacy `spokes` control changes the count of the `five_spoke` family; other families have their own deliberate counts.

The existing four-modifier library and fixed core topology are preserved. Its bounded target coordinates are retained in `info["modifier_values"]`. Final fitted vertices are corrected analytically for coupled radius/ratio/width dimensions, rather than retaining the old missing cross-term or silently clipping large connectors to library limits. Exact fitted dimensions are recorded separately in result info. Synthetic tests include a rear connector beyond both the old radius and width slider ranges.

### Brakes

- Two **6 mm friction plates**, separated by an **8 mm ventilation gap**, with 16 actual vanes by default. These are not dark boxes pasted onto a solid rotor edge.
- `plain`, `drilled` and `slotted` patterns. Drilled rotors have 24 through-holes per plate; slotted rotors have twelve **1.5 mm blind recesses** per outward face. Rays through drilled centres pass through; rays into slots hit their recessed floors.
- Adjacent drilled bands share welded boundaries without coincident internal walls. The back friction plate is a reflected copy of the front, with winding corrected.
- Separate pad friction material and backing plates on both sides. Pad/disc clearance is **0.6 mm**; cheek/disc clearance is **5 mm**. The bridge crosses the rotor's axial interval only **outside its swept radius**.
- Tapered caliper cheeks, piston-housing bosses and a shorter, waisted outer bridge reveal the rotor straddle in the close-up. The complete caliper retains at least 6 mm spoke clearance in the tested size/dish combinations.
- A thin, turned-edge dust shield behind the disc has a genuine caliper cutout. It no longer passes through the inboard cheek.

## Options and economical builds

```yaml
components:
  assign:
    wheel_front_*:
      component: wheel.alloy
      options:
        spoke_family: twin_five
        aspect_ratio: 35
        dish: 0.045
        tread_pattern: directional_v
        groove_depth: 0.006
        disc_pattern: drilled
        lettering: true
    wheel_rear_*:
      component: wheel.alloy
      options:
        spoke_family: twin_five
        aspect_ratio: 30
        dish: 0.06
        tread_pattern: directional_v
        disc_pattern: drilled
```

Front/rear section widths, outer diameters and track still come from their body connectors. This example deliberately does not invent new body parameters while the body workstream is in progress.

| Option | Default / supported interpretation |
|---|---|
| `aspect_ratio` | unset; otherwise 20–90 percent |
| `rim_ratio` | legacy fallback when aspect is unset |
| `dish` | 0.04 m; −0.025…0.09 m |
| `spoke_family` | `five_spoke`; seven choices above |
| `spokes` | 5; legacy count 3–12 for `five_spoke` |
| `spoke_width` | 0.55; clamped 0.2–0.85 |
| `tread_pattern` | `asymmetric_block`; four choices above |
| `groove_depth` | automatic: 4 mm road / 10 mm all-terrain; clamped 0–18 mm and at most 25% of sidewall height |
| `tread_blocks` | 40 per lane; 0–64 |
| `tread_grooves` | 3; 0–4; all-terrain uses three lug lanes |
| `shoulder_blocks` | automatic: 16 per side road / 32 all-terrain; explicit 0–40 |
| `lettering` | false; optional outboard rubber relief |
| `disc_pattern` | `plain`, `drilled`, `slotted` |
| `disc_vanes` | 16; explicit 0–48 |
| `hubcap`, `rim_color`, `caliper_color` | preserved; cap visibility and material controls |

Cheap scene/sample settings:

```yaml
wheel:
  options: {tread_blocks: 0, shoulder_blocks: 0, disc_vanes: 0, lettering: false}
```

This reduces four wheels from **18,064 to 13,072 faces** without changing their fitted outer radius or section width. It deliberately omits the expensive ventilation/tread pieces rather than painting substitutes.

`measure_wheels.py` built **15 complete cars across all eight body styles**, with three width settings, then repeated them at low wheel detail. It checked all 60 wheel fits/ground contacts in each run:

- Default: **8.20 s**, cars **59,851–67,889 faces**.
- Low detail: **7.53 s**, cars **54,859–62,897 faces**.
- Warm body cache, no rendering or export; wall-clock figures are indicative and were measured with other verification running. Full measurements and triangle counts are in [`measurements.json`](measurements.json).

## Verification and render-driven iteration

New regression tests cover exact tyre sizing, independent aspect/OD behavior, staggered and mirrored frames, fitting beyond slider bounds, distinct family counts/connectivity, convex dish skins, solid draft, geometric groove depth, slick/zero-detail behavior, recessed lug heads, drilled/slot ray tests, real vane gaps, caliper/pad/spoke clearances, shield cutout, positive winding, nondegenerate triangles, closed welded solid parts, and face budgets.

Numerical inspection found and prompted fixes for:

1. Reflection reversing inboard shoulder winding — corrected with the mesh reflection transform.
2. A coarser rib crest periodically sinking into the carcass — crest sampling now matches the carcass; coplanar side-wall cells are combined without losing the curve.
3. Shield/inboard-caliper interference — added the cutout, not a material disguise.
4. Convex aero skins crossing — both surfaces now derive from one thickness-consistent graph.
5. Duplicate walls between drilled bands — removed and welded the shared seam.
6. A transient open barrel-profile endpoint during lip reshaping — closed it and added a solid-topology guard.

**Visual checks were not replaced by these tests.** I viewed head-on and three-quarter renders of every family, all four tread close-ups, the low/high-profile pair, exposed-brake face/edge views, the caliper close-up, sidewall relief, and the CLI coupe views. The final caliper was reshaped because its first close-up still read as a red block despite passing the separation tests. The all-terrain voids/depth were increased after its first render looked too similar to the road pattern. Darkening the real inner barrel made the hollow depth and polished outer lip clearer.

The actual CLI was driven before and after:

```bash
python3 -m makecar build configs/coupe.yaml -o /tmp/makecar-wheels-before
python3 -m makecar build configs/coupe.yaml -o /tmp/makecar-wheels-final
```

Both exited 0. The final coupe has **96 connectors, 94 components, 62,407 faces** and built/exported/rendered in **33.4 s** (baseline 58,431 faces, 35.3 s). The body's visible limitations in those context images are unchanged and outside this task.

## Screenshots and reproduction

**45 PNGs** are retained, with source hashes and exact standalone cameras in `before/manifest.json` and `after/manifest.json`.

Start with:

- [`before_after.png`](before_after.png) — matched default head-on, oblique and tread views.
- [`families_1.png`](families_1.png), [`families_2.png`](families_2.png), [`families_3.png`](families_3.png) — all seven families, two angles each.
- [`tread_patterns.png`](tread_patterns.png) — one actual tread close-up per family.
- [`profiles.png`](profiles.png) — 245/35 R20 versus 205/65 R15 at matched camera distance.
- [`brakes.png`](brakes.png) — plain/drilled/slotted rotors with the wheel hidden.
- [`after/caliper_drilled_close.png`](after/caliper_drilled_close.png) — visible rotor straddle, pads, bosses and bridge.
- [`after/sidewall_lettering_close.png`](after/sidewall_lettering_close.png) — shallow rubber relief.
- [`coupe_context_before.png`](coupe_context_before.png), [`coupe_context_after.png`](coupe_context_after.png) — full CLI-generated car sheets.

Full-resolution standalone images: `before/default_*.png`; `after/default_*.png`; `after/{family}_{head_on,three_quarter}.png`; `after/tread_{pattern}.png`; `after/profile_*.png`; `after/brake_{pattern}_{face,edge}.png`; `after/caliper_{pattern}_close.png`; and the lettering close-up.

Default before/after comparisons have identical cameras, lighting and option defaults. The family gallery deliberately uses **aspect 40, drilled discs and lettering enabled** to inspect optional details; its maximum is 7,236 faces per wheel. It is not presented as the default-options budget measurement. Tyre rubber colour was changed from near-black to charcoal to make physical relief more legible.

```bash
python3 -m pytest -q
python3 docs/agent_reports/09_wheels_tyres/render_wheels.py after
python3 docs/agent_reports/09_wheels_tyres/contact_sheets.py
python3 docs/agent_reports/09_wheels_tyres/measure_wheels.py
```

The renderer/measurement scripts use only the project runtime dependencies. `contact_sheets.py` uses Pillow **for verification only**. The committed before images were captured at baseline `4cb7fa4`; the `before` command is intended for that source and refuses to overwrite existing baseline images. No pickle snapshots or new renderer dependencies are needed.

## Deliberate limits / open issues

- These are lightweight render meshes, not wheel/brake manufacturing CAD or roadworthiness models. Spoke fillets are sampled flares/edge breaks; there are no tyre cords, a fully modelled inner liner, hydraulic plumbing, fastener threads, or load/contact-patch deformation.
- Individually tested hardware solids are closed after coordinate welding. The complete wheel is intentionally an assembly of adjoining/intersecting parts, not one boolean-unioned watertight volume; the tyre carcass terminates at its bead seats.
- The renderer uses flat/two-sided shading and no physically based reflections. Circumferential facets remain visible at very tight crops. Lettering is intentionally subtle, block-glyph moulding, not branded/typeset sidewall artwork.
- Spoke and tread families are discrete topology choices, not blendable macro targets. The fixed-topology core remains morphable, and every family uses the same physical fitting step.
- Current body connectors still choose their existing per-car sizing. Synthetic stagger tests establish readiness for future per-axle connectors; no unrelated body parameters or target libraries were changed here.

## Commits

1. `6f76b57` — physical tyre fitting, tread patterns, lofted face families, sizing regressions.
2. `76a4877` — vented brakes, recessed hardware, geometry corrections, lettering and solid tests.
3. `5f8b2ad` — caliper casting refinement from the close-up render.
4. `e388408` — reproducible verification scripts, measurements and viewed PNGs.
5. Final documentation commit — this report and final verification result.
