# Task 01 — Body surfacing

Date: 2026-09-12 · Branch: `agent/body-surfacing`

## Result

Implemented all six body work items: stepped fascias, folded hood/fender edges, radiused shoulders, rockers and arch flanges, framed recessed greenhouse, floor tunnel/skirts, and dimensioned presets. Changes stay in the task-owned body files, body tests, and this report directory. No dependencies were added; nothing was pushed.

**Tests: 415 passed** (`python3 -m pytest -q`, 36.63 s). Baseline was **280 passed**. The final post-metadata-fix sample CLI run built **15/15 cars**, exit code 0; the largest sample assembly has **27,024 faces**.

## What changed

### 1. Front and rear ends

* Replaced the two shrinking cap rings with six corresponding fascia rings per end: corner transition, bumper face, broad landing, pocket rim and pocket floor. The final fan is flat, not a pointed nose/tail.
* The front has a near-vertical bumper below its crease, a stepped upper grille landing, an overhanging hood lip, and a lower splitter/air dam. Nominal front/rear overhangs include the new volumes rather than accumulating arbitrary cap extensions.
* The rear has a 640 × 230 mm pocket mouth and a 580 × 180 mm recessed floor around the existing 520 × 110 mm plate. Coupe/sports have stronger diffuser setbacks; hatch/wagon/SUV/van have inset lower tailgate panels.
* Aligned crease-height samples across the cap rings and transitioned the last four body sections into the same perimeter parameterization. This removes the opposed triangle normals seen in early corner/step prototypes.
* Preset plan corner runs are 250–290 mm, with end widths chosen for roughly 200–250 mm lateral corner rounding.

New controls: `bumper_projection`, `bumper_crease_height`, `hood_overhang`, `front_splitter`, `rear_bumper_crease_height`, `plate_recess`, `diffuser_step`, `tailgate_panel`.

### 2. Fender/hood crease and shoulder

* Explicit 6 mm-wide hood/deck return folds replace a purely decorative seam. Default fold depth is 12 mm, tuned to 10–18 mm by class.
* E→F now contains a sampled circular quarter-radius, followed by the shoulder landing; it is not an unconstrained soft spline roll.
* `shoulder_radius` and `fender_crease` have dimensional modifiers and mesh tests. Roof-height and shoulder-inset behavior remain independent.

### 3. Rocker, door hem and arch flange

* C→D is a flat vertical rocker, normally 150 mm tall, inboard of the door skin. B→C flares inward to the floor.
* A 12 mm door-hem step separates the door from the sill.
* The arch starts at the unchanged circle-fit landmark D, extends radially by 20 mm, then returns inward by 12 mm. This is a flange with two distinct faces, not another fender bulge. Changing its width does not move the wheel-circle landmarks.

New controls: `rocker_height`, `door_step`, `arch_lip_width`.

### 4. Greenhouse

* A-pillar lean now approaches the windshield direction instead of leaving the previous very broad triangular pillar. `a_pillar_width` controls its cowl band; the existing `c_pillar_width` now has a modifier.
* Dedicated 25 mm longitudinal frame rows and 20 mm side-strip margins surround the glass. All aperture vertices, including their boundaries, sink 8 mm along their local surface normals; adjoining frame vertices remain on the outer shell.
* Side-glass upper edges use multiple stations along an arched roof rail, rather than a single straight door-width chord. Added rolled roof edges and a 4 mm drip bead.
* Quarter-glass rectangles and pickup bed zones now activate from the **morphed mesh**. The existing target library carries vertex offsets but retains base metadata; without this synchronization, SUV/wagon/van quarter glass was recessed but still painted over. Nested metadata is copied before activation, so style builds cannot contaminate subsequent sedan builds.

New controls: `a_pillar_width`, `glass_recess`, `roof_edge_radius`, `drip_rail`; added modifier for `c_pillar_width`.

### 5. Underbody

* Flat floor strips outside a 280 mm-wide, 65 mm-high longitudinal tunnel, fading out before both bumpers.
* Separate front air-dam and rear-valance depths below the painted bumper (35/45 mm by default).
* Interior floor measurements use B, the flat pan, not A, the newly raised tunnel crown. Raising the tunnel therefore does not raise seats or cabin-floor connectors.
* The normal dark-material underside views are retained. Additional **clay, below-lit diagnostic views** expose the tunnel and skirt geometry rather than pretending that black-underbody beauty renders show these details clearly.

New controls: `tunnel_height`, `tunnel_width`, `air_dam_height`, `rear_valance_height`.

### 6. Presets

The eight presets use stock reference wheelbases, widths and nominal tyre outside diameters. Overhang splits and greenhouse hard points are estimates unless identified as factory dimensions. These are class/proportion references, **not exact replicas or scanned surfaces**.

All dimensions below are metres. Measured values are from the actual `build_car` body, not just the parameter dataclass. The extra 20 mm of width comes from the existing door-skin bulge; the sedan/coupe/sports tail rake adds 20 mm to overall length.

| Style / stock reference | Factory L × W × H | Factory WB | Tyre | Measured body L × W × roof skin |
|---|---|---:|---|---|
| Sedan — 2024 Camry LE FWD [1] | 4.879 × 1.839 × 1.445 | 2.824 | 215/55R17; Ø0.6683 | 4.899 × 1.859 × 1.445 |
| Hatchback — 2020 Golf Life 1.0 eTSI [2] | 4.284 × 1.789 × 1.491 | 2.619 | 205/55R16; Ø0.6319 | 4.284 × 1.809 × 1.491 |
| Wagon — 2024 Outback Base [3] | 4.854 × 1.854 × 1.679 | 2.746 | 225/65R17; Ø0.7243 | 4.854 × 1.874 × 1.640 |
| SUV — 2024 RAV4 LE FWD [4] | 4.595 × 1.854 × 1.702 | 2.690 | 225/65R17; Ø0.7243 | 4.595 × 1.874 × 1.660 |
| Pickup — 2021 F-150 XL SuperCrew 4×2, 5.5-ft bed [5] | 5.885 × 2.029 × 1.920 | 3.693 | 245/70R17; Ø0.7748 | 5.885 × 2.049 × 1.920 |
| Coupe — 2025 Mustang EcoBoost Fastback [6] | 4.811 × 1.915 × 1.397 | 2.718 | 235/55R17; Ø0.6903 | 4.831 × 1.935 × 1.397 |
| Sports — 2025 MX-5 RF Grand Touring [7] | 3.914 × 1.735 × 1.245 | 2.309 | 205/45R17; Ø0.6163 | 3.934 × 1.755 × 1.245 |
| Van — 2024 Sienna LE FWD [8] | 5.174 × 1.994 × 1.740 | 3.061 | 235/65R17; Ø0.7373 | 5.174 × 2.014 × 1.740 |

Height datum caveats: Outback and RAV4 **bare roof-skin heights are estimates** excluding external equipment, not factory roof-skin measurements. Sienna's 1.740 m specification excludes roof rails. VW's cited August 2020 sheet specifically gives 2.619 m wheelbase and a 4.284–4.286 m equipment-dependent length. F-150 factory front/rear overhangs are 0.955/1.234 m; the preset adds 3 mm to the rear to reconcile independently rounded factory length and wheelbase.

Chosen greenhouse and overhang hard points:

| Style | Cowl Z | Belt Z parameter | Roof Z | Windshield horizontal run | Endpoint angle above horizontal | Front / rear overhang |
|---|---:|---:|---:|---:|---:|---|
| Sedan | 1.00 | 0.95 | 1.445 | 0.85 | 27.6° | 0.975 / 1.080 |
| Hatchback | 1.04 | 0.98 | 1.491 | 0.72 | 32.1° | 0.850 / 0.815 |
| Wagon | 1.15 | 1.09 | 1.640 | 0.84 | 30.3° | 0.995 / 1.113 |
| SUV | 1.16 | 1.10 | 1.660 | 0.65 | 37.6° | 0.915 / 0.990 |
| Pickup | 1.38 | 1.28 | 1.920 | 0.66 | 39.3° | 0.955 / 1.237 |
| Coupe | 0.99 | 0.93 | 1.397 | 0.86 | 25.3° | 0.965 / 1.128 |
| Sports | 0.86 | 0.81 | 1.245 | 0.59 | 33.1° | 0.780 / 0.825 |
| Van | 1.15 | 1.09 | 1.740 | 0.98 | 31.0° | 0.970 / 1.143 |

Cowl/belt heights and windshield endpoints are explicitly **modeling estimates**; manufacturer hard-point data was not available to independently certify their centimetre-level accuracy. The tests verify that the mesh actually realizes the chosen values. [measurements.json](measurements.json) includes every preset parameter, measured dimensions, topology, full-assembly bounds and face counts.

## Topology and connector compatibility

* `LIBRARY_VERSION = 11`; 61 differential modifiers, including 22 new/exposed detail controls.
* Fixed body topology: **59 main rings + 12 fascia rings**, **86 vertices/ring + 2 fan centres = 6,108 vertices**; **6,192 polygon faces** / **12,212 triangles** before aperture removal.
* All eight styles have identical vertex counts, face counts, face index tuples and group indices. Tests repeat the face-index comparison for both endpoints of **every differential modifier**.
* Shell faces: 5,620 normally; 5,588 with the two quarter apertures. Aperture removal is intentionally style-dependent; the full target mesh is not.
* The existing `nose_ring`, `tail_ring`, `apex_front`, `apex_rear`, arch, pillar and feature-line groups remain. Added explicit `fascia/*` face/pocket groups. Fan centres remain useful cap landmarks, but overall end measurements now use fascia extrema rather than a recessed plate-pocket centre.
* Existing connector names, types, tag sets and frame conventions are retained. Grille heights are fitted between crease and hood; front plates mount on the upright bumper. Whole-footprint triangle clipping prevents flat components being buried across fascia steps. Triangulation is shared across fascia mounts in each emission.
* Antenna placement uses physical clearance from the rear roof edge rather than the newly introduced 25 mm frame station.
* Full stock assemblies are **23,120–26,776 faces**, far below 120k.

## Verification

* Baseline: `python3 -m pytest -q` → **280 passed**.
* Final implementation: same command → **415 passed in 36.63 s**.
* New numerical checks cover every added control, exact glass normal offsets, unbroken frame bands, roof-rail curvature, circular shoulder radii, uniform arch flanges, rocker verticality, floor/tunnel separation, unchanged seat-floor mounts, rear plate depth, production dimensions, fixed topology across modifier endpoints, all-style filled-aperture normals, non-opposed fascia triangles, full grille/plate footprint clearance, and optional-zone/cache isolation.
* Sample CLI: `python3 -m makecar samples -o /tmp/makecar-body-surfacing-samples-final-20260912` → **15/15 built, exit 0**. The CLI calls these “11 sample configs”; variants expand them into the requested **15 cars**. Final assemblies range from 23,382 to 27,024 faces.
* Rendered through `makecar.pipeline.build_car` and `makecar.export.render.Renderer`, not hand-drawn proxies. **80 PNGs**, 800×600 with 2× supersampling: five styles × eight views × before/after.
* Validated all 80 PNG headers/dimensions and every local report link; all 15 final sample directories contain `car.obj`, `body.obj`, `assembly.json`, `connectors.json`, `blueprint.svg` and `preview.png`. Both JSON exports parse successfully in every directory.
* Visually inspected front/rear three-quarter, side/front, nose, A-pillar and underside renders throughout implementation. The closeups led to corrective iterations on fascia corner correspondence, lamp boundaries and the roof rail. Diagnostic underside lighting makes the floor tunnel visible.

Reproduce the final images:

```bash
python3 docs/agent_reports/01_body_surfacing/render_comparison.py after
```

The renderer embeds the original camera dimensions and also reuses the saved baseline camera JSON when present. Before images were captured from commit `f937856` before geometry editing; local snapshots used during this session were stored in `/tmp/makecar-body-surfacing-baseline`. To regenerate a new baseline, run the script against the baseline implementation, not current geometry labeled “before”. The required overview views contain every default component. Nose/A-pillar closeups intentionally omit obstructing accessories and lamps, showing the real shell plus assembled glass. The clay underside diagnostic changes **materials and lighting only**, identically in both phases.

## Screenshot inventory

Each cell links **before / after**. Ordinary underside images use the same default materials as the overviews; clay images are explicitly diagnostic.

| Style | Front three-quarter | Rear three-quarter | Side | Front |
|---|---|---|---|---|
| Sedan | [B](before_sedan_three_quarter_front.png) / [A](after_sedan_three_quarter_front.png) | [B](before_sedan_three_quarter_rear.png) / [A](after_sedan_three_quarter_rear.png) | [B](before_sedan_side.png) / [A](after_sedan_side.png) | [B](before_sedan_front.png) / [A](after_sedan_front.png) |
| Hatchback | [B](before_hatchback_three_quarter_front.png) / [A](after_hatchback_three_quarter_front.png) | [B](before_hatchback_three_quarter_rear.png) / [A](after_hatchback_three_quarter_rear.png) | [B](before_hatchback_side.png) / [A](after_hatchback_side.png) | [B](before_hatchback_front.png) / [A](after_hatchback_front.png) |
| SUV | [B](before_suv_three_quarter_front.png) / [A](after_suv_three_quarter_front.png) | [B](before_suv_three_quarter_rear.png) / [A](after_suv_three_quarter_rear.png) | [B](before_suv_side.png) / [A](after_suv_side.png) | [B](before_suv_front.png) / [A](after_suv_front.png) |
| Pickup | [B](before_pickup_three_quarter_front.png) / [A](after_pickup_three_quarter_front.png) | [B](before_pickup_three_quarter_rear.png) / [A](after_pickup_three_quarter_rear.png) | [B](before_pickup_side.png) / [A](after_pickup_side.png) | [B](before_pickup_front.png) / [A](after_pickup_front.png) |
| Sports | [B](before_sports_three_quarter_front.png) / [A](after_sports_three_quarter_front.png) | [B](before_sports_three_quarter_rear.png) / [A](after_sports_three_quarter_rear.png) | [B](before_sports_side.png) / [A](after_sports_side.png) | [B](before_sports_front.png) / [A](after_sports_front.png) |

| Style | Nose closeup | A-pillar/glass closeup | Underside | Clay underside diagnostic |
|---|---|---|---|---|
| Sedan | [B](before_sedan_nose_closeup.png) / [A](after_sedan_nose_closeup.png) | [B](before_sedan_a_pillar_closeup.png) / [A](after_sedan_a_pillar_closeup.png) | [B](before_sedan_underbody.png) / [A](after_sedan_underbody.png) | [B](before_sedan_underbody_diagnostic.png) / [A](after_sedan_underbody_diagnostic.png) |
| Hatchback | [B](before_hatchback_nose_closeup.png) / [A](after_hatchback_nose_closeup.png) | [B](before_hatchback_a_pillar_closeup.png) / [A](after_hatchback_a_pillar_closeup.png) | [B](before_hatchback_underbody.png) / [A](after_hatchback_underbody.png) | [B](before_hatchback_underbody_diagnostic.png) / [A](after_hatchback_underbody_diagnostic.png) |
| SUV | [B](before_suv_nose_closeup.png) / [A](after_suv_nose_closeup.png) | [B](before_suv_a_pillar_closeup.png) / [A](after_suv_a_pillar_closeup.png) | [B](before_suv_underbody.png) / [A](after_suv_underbody.png) | [B](before_suv_underbody_diagnostic.png) / [A](after_suv_underbody_diagnostic.png) |
| Pickup | [B](before_pickup_nose_closeup.png) / [A](after_pickup_nose_closeup.png) | [B](before_pickup_a_pillar_closeup.png) / [A](after_pickup_a_pillar_closeup.png) | [B](before_pickup_underbody.png) / [A](after_pickup_underbody.png) | [B](before_pickup_underbody_diagnostic.png) / [A](after_pickup_underbody_diagnostic.png) |
| Sports | [B](before_sports_nose_closeup.png) / [A](after_sports_nose_closeup.png) | [B](before_sports_a_pillar_closeup.png) / [A](after_sports_a_pillar_closeup.png) | [B](before_sports_underbody.png) / [A](after_sports_underbody.png) | [B](before_sports_underbody_diagnostic.png) / [A](after_sports_underbody_diagnostic.png) |

## Open issues and deliberate limits

1. **Pickup auto seating, outside this body brief:** the newly dimensioned SuperCrew cab reproduces the factory 3.957 m front-bumper-to-cab-back distance, but the existing interior auto-row algorithm can emit three rows. The third bench/headrests protrude into the bed in the default full-car pickup views. The shipped pickup config explicitly selects two rows and builds. Reported to the interior work stream; neither stream owns the row allocator in this task. An integrator should constrain pickup auto rows to two and check complete reclined-seat extents against the bulkhead. Body dimensions were not falsified to hide this.
2. **Glazing component inset:** the shell/aperture grid is 8 mm recessed. The unchanged `glass.tinted` component adds its own default 4 mm inset, so the assembled surface is approximately 12 mm below the frame. Selecting glass option `inset: 0` gives the shell's 8 mm seating; component defaults were not changed in this branch.
3. Existing generic lamp, grille, bumper-accessory and interior components still affect the full-car appearance. RF buttresses, detailed pickup bed stampings, a brand-specific grille or suspension hardware are not represented by this fixed-topology body loft.
4. Manufacturer greenhouse hard-point measurements were unavailable; cowl/belt/rake estimates are not independently certified to a few centimetres. Dimensions and tyre diameters with published references are checked numerically. Arbitrary combinations of every slider at its maximum are not claimed to be production-feasible; individual endpoints, existing blends and all shipped samples are tested.

## Sources

1. [Toyota 2024 Camry manufacturer brochure](https://www.toyota.com/content/dam/toyota/brochures/pdf/2024/camry_ebrochure.pdf); [LE tyre specification](https://www.cars.com/research/toyota-camry-2024/specs/).
2. [Volkswagen August 2020 Golf hybrid technical specifications](https://uploads.vw-mms.de/system/production/files/vwn/016/289/file/7679ef3a5204d765bb1980d8131c293751ebffdb/EN_Golf_Hybride_Specifications.pdf).
3. [Subaru 2024 Outback manufacturer specification sheet](https://s3.amazonaws.com/subarumedia.iconicweb.com/mediasite/attachments/2024_Outback_AAG_-_FINAL.pdf).
4. [Toyota 2024 RAV4 manufacturer brochure](https://www.toyota.com/content/dam/toyota/brochures/pdf/2024/rav4_ebrochure.pdf); [LE tyre specification](https://www.cars.com/research/toyota-rav4-2024/specs/).
5. [Ford 2021 F-150 technical specifications, dimensions and tyres](https://www.fromtheroad.ford.com/content/dam/fordmediasite/us/en/library/2021/specs/2021-F-150-Technical-Specs.pdf).
6. [Ford 2025 Mustang technical specifications](https://www.fromtheroad.ford.com/content/dam/fordmediasite/us/en/library/2025/specs/2025_Mustang_Technical_Specs.pdf).
7. [Mazda 2025 MX-5 manufacturer specification deck](https://news.mazdausa.com/download/25MY+MX-5+Spec+Deck.pdf).
8. [Toyota 2024 Sienna product information, manufacturer PDF hosted by MyAutoWorld](https://a4.myautoworld.com/wp-content/uploads/2024/02/2024_Toyota_Sienna_Product_Info_FINAL.pdf).
