# Task 05 — Face, footprint and section archetypes

2026-09-12 · `agent/shape-archetypes`

## Result

The sports car no longer wears the sedan's face. Its leading edge is **532 mm** above ground versus the sedan's **780 mm**, with a straight falling hood, narrow nose, thin swept lamp apertures and a raked upper fascia. The pickup's **1,360 mm** leading edge and vertical face sit at the opposite extreme.

The eight plan outlines also differ without glass, seams, wheels or accessories. See the [black silhouette comparison](compare_silhouette_plan.png): neutral blunt sedan; short round-corner hatch; long tapered estate forebody; shallow-waisted, square-ended SUV; straight-sided pickup; strongly waisted coupe; sports body tapered at both ends; and broad-backed MPV with a rounded, narrowing front. These are class archetypes, not exact replicas of the production cars used for the original dimensional references.

**Final code tests: 921 passed** (721 existing + 200 new), in 123.05 s. Body topology remains **6,108 vertices / 6,192 faces**, with identical face-index tuples. All eight default complete assemblies and all twelve sedan archetype endpoints build; default assemblies have **55,415–63,273 faces**, below the 120k budget.

**15/15 sample cars exported and validated.** Final matched renders, all twelve sweeps and both blends are committed: **237 PNGs and six JSON records**. The final sports face and eight pure plan outlines were visually inspected after the last geometry fixes.

## Generator controls

All existing parameter names remain. Twelve controls were added; lengths are metres:

| Parameter | Default | Meaning |
|---|---:|---|
| `hood_straightness` | 0 | Blend the original curved hood into a straight cowl-to-tip plane. Slope comes from the existing cowl/leading heights and actual finished horizontal run. |
| `nose_center_drop` | 0 | Lower the centre tip independently of the fenders. |
| `nose_center_extension` | 0 | Project the centre tip ahead of the outer corners. |
| `fender_crown_height` | 0 | Raise the front fenders independently of the centreline. |
| `fascia_roundness` | 0 | Additional elevation corner radius, carried through the concentric fascia rings. |
| `fascia_slope` | 0 | Continuous upper-fascia setback, separate from the existing abrupt grille step. |
| `nose_taper_length` | 0 | **Additional** longitudinal plan taper beyond `front_corner_length`; does not enlarge the fascia rake/roll itself. |
| `tail_taper_length` | 0 | Additional plan taper beyond `rear_corner_length`. |
| `nose_taper_exponent` | 2.2 | Independent front taper shape; lower is more progressive/pointed. |
| `tail_taper_exponent` | 2.2 | Independent rear taper shape. |
| `waist_depth` | 0 | Per-side door inset, fading smoothly to full-width axle shoulders. |
| `tumblehome` | 0 | Additional per-side roof-rail inset. |

The existing `front_corner_length` and `rear_corner_length` remain independent corner controls. Keeping taper runs separate proved essential: changing only the last few centimetres of the bumper produced numerically different but visually almost identical rectangles.

`LIBRARY_VERSION` is **14**. Shape deltas and style mixes are included in the cache fingerprint. No runtime dependencies were added; Pillow is used only by the verification renderer script.

## How the twelve archetypes are built

`shapes.py` holds dimensional deltas and descriptions. Every archetype is a normal unipolar differential target, generated from the caller's canonical `BodyParams`, not a replacement mesh.

| Archetype | Construction at full strength on the canonical sedan |
|---|---|
| `face/wedge` | Lower leading edge 160 mm and bumper bottom 65 mm; lower crease 100 mm; straighten the hood; add 150 mm of continuous upper-face setback; crown outer fenders 55 mm. The lower bumper clearance preserves room for the existing plate/intake components. |
| `face/upright` | Raise leading edge 190 mm; flatten hood; remove most upper rake and lip projection. |
| `face/shark` | Drop centre 120 mm, extend centre 160 mm, crown fenders 100 mm; add 130 mm upper-face rake. Extension exceeds the setback, so the tip really moves forward, not merely forward relative to the fenders. |
| `face/snub` | Move cowl forward 400 mm, lengthen windscreen 180 mm, raise leading edge 160 mm and enlarge plan/elevation corner radii. |
| `face/cabforward` | Move cowl forward 680 mm and up 100 mm; extend windscreen 500 mm and front-door run 400 mm; raise/round the nose. The default van has only about 238 mm from cowl to front lip. |
| `face/longhood` | Move cowl rearward 320 mm, raise leading edge 150 mm, straighten hood and reduce upper rake. Overall length is not inflated to fake a long hood. |
| `plan/pointed` | Reduce front/tail width ratios by .36/.26, extend their taper runs by .67/.58 m and reduce both exponents by 1.1. |
| `plan/square` | Increase front/tail width ratios by .20/.16, shorten corner runs 160 mm and increase taper exponents by 5.8. |
| `plan/cokebottle` | Inset doors 140 mm per side, preserving both axle shoulders; give the ends a gentler, longer taper. |
| `section/tumblehome` | Move roof rails inward 100 mm per side without narrowing the hips. |
| `section/slabside` | Broaden roof ratio .16, reduce shoulder inset 30 mm, tighten shoulder radius and reduce door bulge. |
| `section/domed` | Increase roof crown 95 mm, round shoulders and slightly narrow the rail. Roof-centre height stays unchanged. |

**Procedural field:** the shark's independent tip/fender relationship cannot come from a single centreline profile. `BodyGenerator._shape_nose` supplies a smooth transverse field across the hood and the actual fascia/lamp grids; the wedge uses its fender-crown part too. The field excludes wheel-circle landmarks and the cabin. These remain generator-derived differential targets, not additional morph-only sculpts. The other ten archetypes use loft/profile controls. Existing `sculpt/*` targets are unchanged.

## Tuned defaults and preset decisions

Every style macro inherits this mix. Explicit config entries replace individual weights, including zero.

| Style | Face | Plan | Section |
|---|---|---|---|
| sedan | neutral | neutral | neutral |
| hatchback | snub .65 | pointed .15 | domed .50 |
| wagon | upright .30 | square .30 + pointed .35 | neutral |
| SUV | upright .85 | square .55 + cokebottle .45 | slabside .60 |
| pickup | upright 1.00 | square 1.00 | slabside .80 |
| coupe | longhood .65 | cokebottle .80 | tumblehome .50 |
| sports | wedge .80 | pointed .85 | tumblehome .70 |
| van | cabforward .85 | square .40 + pointed .30 | slabside .50 |

Additional preset tuning, beyond the mix table:

- **Wagon:** neutral nose width ratio .64, extra front taper .70 m and front exponent .85. After its mix, the long narrowing forebody separates it from the sedan's parallel sides and blunt nose.
- **Van:** neutral nose/tail ratios .77/.94, front corner run .40 m, extra front taper .65 m and exponent 1.2. Its broad rear and rounded front are deliberately unlike the pickup's near-rectangle.
- **Sports:** diffuser setback reduced from 75 to 45 mm so the pointed tail does not fold around the fixed plate pocket. Tunnel width increased from 280 to 300 mm to restore the existing exhaust's full 4 mm pan-clearance allowance after the footprint change. This is underbody accommodation, not a connector/component change.

Wheelbase, overall reference length/width, roof-centre height and nominal wheel diameters are retained. Existing dimensional tests verify them on generated meshes. The task intentionally changes the estimated face/section/plan surfaces; the original reference models should not be read as claims of manufacturer-exact styling.

## Additive semantics and the ownership boundary

Each modifier is `[0, 1]`, default `0`, in `face`, `plan` or `section`. Within each family, weights are clamped and divided by their sum only when that sum exceeds one. Thus `.6 wedge` leaves 40% neutral; `1 wedge + 1 upright` becomes `.5 + .5`. Other families, dimensional sliders, sculpts and style weights are not normalized together. As in the existing morph engine, `None` means unspecified: it does not disable an inherited shape. Explicit zero does.

`style_params` includes its default shape mix. Each stored style macro subtracts the corresponding canonical archetype displacement; normal target application adds the inherited mix back. This preserves exact generated-style reproduction, linear partial styles and style blends while allowing individual defaults to be removed.

Example config:

```yaml
body:
  style: sports
  modifiers:
    face/wedge: 0       # explicitly disable the inherited wedge
    face/shark: 0.6
    plan/pointed: 0.35  # replace, not add to, the inherited .85
```

**Deliberate scope limitation:** `CarBody` and `pipeline.py` are outside this task's ownership. Resolution therefore happens in `BodyMorphableMesh.displacement`, for both direct library calls and `CarBody` builds. `resolve_shape_values(requested)` exposes the effective mix without mutating its input. `CarBody.resolve_values`, `BodyResult.modifier_values` and exported assembly modifier metadata still show **requested**, not expanded/normalized, values. Likewise the existing optional `.target` export enumerates requested targets and does not automatically include inherited shapes; exporting a residual style target alone is insufficient to reconstruct its default style. An integrator should have those metadata/export paths consume the same pure resolver. No monkey-patching or out-of-scope edits were used to conceal this limitation.

## Render-driven iterations

1. Captured the original eight styles from immutable baseline **`d7122f7`** (the brief merge; body geometry starts at `8a595e8`). The task brief was initially missing from this worktree and was read from the main checkout without changing working directories.
2. Added first-pass controls and inspected front, side, plan, three-quarter and nose views. The sports nose changed substantially, but several utility/estate outlines remained nearly rectangular.
3. Found an actual backward fold at the wedge's last hood stations. A steep setback cannot be squeezed into the original short corner run: it now gets a longer, independently bounded upper-rake transition, and the hood is straightened in its **finished x/z frame**, not merely before rake.
4. Propagated elevation radii into inner fascia rings and separated lower/upper corner radii to avoid crossed rounded bumper cells. Matched lower-lip samples between rings, eased the rear shoulder into the lamp roll and faded the terminal hood-return crease. This fixed the wagon taillight and sports headlight geometry without relaxing the lamp fitting allowance.
5. With **911 tests green**, rejected the first black silhouette sheet: sedan/wagon and SUV/van were still too similar. Extended the wagon/van front taper and strengthened the SUV's shallow waist, then viewed the pure silhouettes again. The final outlines differ even with both overall length and width normalized away: all 28 pairs exceed **2% RMS** and **9% peak** half-width-profile difference at 41 longitudinal cuts. The nearest pair is sedan/hatchback at about **2.25% RMS**. This numerical guard supplements, not substitutes for, viewing the sheets.
6. Viewed complete front, side and three-quarter assemblies and the isolated shell/glass diagnostics. Generic components were not hidden in complete-car views; black silhouettes contain the actual full body mesh, not hand-drawn profiles.
7. Independent review exposed failures in simple overrides despite green default-style tests: disabling the wagon's pointed target could make a taillight cell concave, and full pointed on the rounded hatch could reverse upper fascia samples. Blended the rear shoulder width toward the belt chain at the fascia transition and bounded rounded corner radii to preserve sample order. Added complete-assembly regressions for four reviewed cases; all **23 inherited defaults individually set to zero** now build successfully. Also restored `None`-weight compatibility. The final version-14 suite has **921 passing tests**.
8. The dome sweep exposed two small headliner tabs at the front roof corners at full strength. A diagnostic render removing only `headliner.fabric` confirmed their source; the shell/glass render is clean. This out-of-scope interior fitting limitation is retained in the full renders and documented below.

## Verification

- Baseline `python3 -m pytest -q`: **721 passed**, 97.39 s.
- Final code `python3 -m pytest -q`: **921 passed**, 123.05 s.
- New tests cover all twelve measurement directions; true wedge hood monotonicity/straightness; independent family normalization; zero/half/full interpolation; inherited defaults and explicit-zero overrides; custom bases; fresh/disk-cache consistency and invalidation; exact style reconstruction; symmetry; wheel-circle preservation; and **288 archetype/style/sweep-weight topology combinations**.
- Existing body, component, connector, interior, fascia-fit, pipeline and export suites all remain green. No existing tests were weakened.
- Public CLI `list-modifiers` lists exactly the twelve new names, each `[0, 1]`.
- All eight complete default styles and all twelve complete sedan endpoints build, independently of image rendering.
- Final version-14 `python3 -m makecar samples -c configs -o /tmp/shape_samples`: **15/15 samples exported**, exit 0. The CLI's header counts 11 config files; blends/random variants produce 15 cars. All 15 output folders contain the expected 11 nonempty files; JSON, PNG and SVG files parse successfully, and every exported OBJ face index is valid. Complete sample meshes contain **55,745–62,803 faces**, below 120k. Exported `body.obj` omits component apertures; the fixed-topology full morph mesh remains 6,192 faces.
- All **23 inherited defaults individually disabled** build complete assemblies. Regression tests also cover wagon square/domed overrides, wagon pointed disabled, hatchback full pointed, and unspecified (`None`) weights.
- Final image audit: all **237 PNGs** decode; **44 sweep/blend cases** are present; before/after and silhouette cameras exactly match `baseline.json`; all **16 normalized blend image pairs** (`.5 + .5` versus `1 + 1`, two blends × two variants × four views) are pixel-identical. Baseline metadata remains byte-identical to the archived baseline. The renderer's self-test passes.

## Images and reproduction

All cameras are orthographic and matched before/after, recorded in `baseline.json`. `after_manifest.json`, the silhouette manifests and `sweeps_manifest.json` record source hashes, cameras, configurations and measurements. Full views contain the actual default `build_car` assembly. Clay views intentionally show only shell plus opaque assembled glass so the generic grille/lamp/accessory parts cannot hide body shape; their captions identify this diagnostic mode. Silhouettes rasterize **all faces of the real full body mesh** flat black with no seams or components.

| View | Complete assembly comparison | Shell/glass comparison |
|---|---|---|
| Side | [before/after](compare_full_side.png) | [before/after](compare_clay_side.png) |
| Front | [before/after](compare_full_front.png) | [before/after](compare_clay_front.png) |
| Plan | [before/after](compare_full_plan.png) | [before/after](compare_clay_plan.png) |
| Front three-quarter | [before/after](compare_full_three_quarter_front.png) | [before/after](compare_clay_three_quarter_front.png) |
| Nose close-up | [before/after](compare_full_nose.png) | [before/after](compare_clay_nose.png) |

[Pure silhouette before/after](compare_silhouette_plan.png) · [final silhouette sheet](after_silhouette_plan.png)

Each single-archetype sweep shows the sedan at **0 / .5 / 1** in side, front, plan and front-three-quarter views. Blend sheets include `.5 + .5` and `1 + 1` to demonstrate normalization.

| Sweep | Complete assembly | Shell/glass |
|---|---|---|
| Wedge | [full](sweep_face_wedge_full.png) | [clay](sweep_face_wedge_clay.png) |
| Upright | [full](sweep_face_upright_full.png) | [clay](sweep_face_upright_clay.png) |
| Shark | [full](sweep_face_shark_full.png) | [clay](sweep_face_shark_clay.png) |
| Snub | [full](sweep_face_snub_full.png) | [clay](sweep_face_snub_clay.png) |
| Cab-forward | [full](sweep_face_cabforward_full.png) | [clay](sweep_face_cabforward_clay.png) |
| Long hood | [full](sweep_face_longhood_full.png) | [clay](sweep_face_longhood_clay.png) |
| Pointed | [full](sweep_plan_pointed_full.png) | [clay](sweep_plan_pointed_clay.png) |
| Square | [full](sweep_plan_square_full.png) | [clay](sweep_plan_square_clay.png) |
| Coke-bottle | [full](sweep_plan_cokebottle_full.png) | [clay](sweep_plan_cokebottle_clay.png) |
| Tumblehome | [full](sweep_section_tumblehome_full.png) | [clay](sweep_section_tumblehome_clay.png) |
| Slabside | [full](sweep_section_slabside_full.png) | [clay](sweep_section_slabside_clay.png) |
| Domed | [full](sweep_section_domed_full.png) | [clay](sweep_section_domed_clay.png) |
| Wedge + upright | [full](sweep_blend_wedge_upright_full.png) | [clay](sweep_blend_wedge_upright_clay.png) |
| Long hood + shark | [full](sweep_blend_longhood_shark_full.png) | [clay](sweep_blend_longhood_shark_clay.png) |

Final inventory: **176 standalone cells + 61 sheets = 237 PNGs**, 19,008,464 bytes (18.13 MiB), plus **six JSON records**. After/sweep/silhouette captures use the same frozen version-14 implementation (`474b3c5`), with source SHA256 **`1cf8ae59011bd8dc94c4f34e8107e8d66d4ff9aea34df377674116646c3ec69d`**, verified against the final worktree. Final artifacts are committed as `c225966`; render manifests distinguish requested and effective modifier weights.

```bash
python3 docs/agent_reports/05_shape_archetypes/render_shapes.py before --nose
python3 docs/agent_reports/05_shape_archetypes/render_shapes.py after --nose
python3 docs/agent_reports/05_shape_archetypes/render_shapes.py sweeps
python3 docs/agent_reports/05_shape_archetypes/render_shapes.py before --variants silhouette --views plan
python3 docs/agent_reports/05_shape_archetypes/render_shapes.py after --variants silhouette --views plan
python3 -m pytest -q
python3 -m makecar samples -c configs -o /tmp/shape_samples
```

For quick iteration use `after --variants clay --styles sedan sports --views front plan nose --quick --output /tmp/shape-check`. `before` explicitly uses the archived baseline implementation; it does not label current geometry as baseline.

## Remaining limits

- Complete-car views still use the same generic grille, fog-light, plate and mirror component families. The shell and lamp **apertures** vary, but this task does not invent brand-specific lighting/grille designs. The clay views make that distinction explicit.
- Cab-forward remains a fixed-topology MPV approximation with a cowl/frame band, not a bespoke uninterrupted windscreen-to-bumper stamping. Strongly domed sections and pointed noses are intentional stylized archetype endpoints. At [`face/longhood: 1` on the four-door sedan](sweep_face_longhood_full.png), the rear side window and door proportions become visibly squeezed: the long dash-to-axle is achieved without extending the car, not by redesigning its door layout.
- Archetype targets are additive displacements, not a nonlinear parameter solver. Explicit overrides on a style subtract/add canonical offsets; they are not equivalent to regenerating that style with a different parameter mix. Very aggressive combinations of every dimensional slider, sculpt and archetype are not claimed production-feasible.
- At **`section/domed: 1` on the sedan**, two small pale tabs from **`headliner.fabric`** protrude through the front roof corners. Compare the [complete dome sweep](sweep_section_domed_full.png) with the [clean shell/glass sweep](sweep_section_domed_clay.png). Removing only that component in a diagnostic render removed the tabs. This is an interior surface-fit limitation, not a body topology failure; the full verification images retain it. Interior/component/connector code is outside ownership and was not changed.
- Effective-value metadata and inherited `.target` export need the integration follow-up described above. No component, connector, assembler or pipeline files were modified. Final tested default assemblies have no outstanding fit-test failures.

No pushes were made.
