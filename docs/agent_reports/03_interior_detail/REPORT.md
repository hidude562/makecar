# Task 03 — Interior detail and accuracy

- Date: 2026-09-12
- Branch: `agent/interior-detail`
- Baseline geometry: `f937856`

Implemented within the owned component/helper files and `extra_interior_connectors` only. The body generator, `body/connectors.py`, assembler and pipeline were not edited. Shared-file integration follow-ups are listed below rather than hidden by changes outside this task's scope.

## What changed

### Packaging and seats

- Front cushions remain 500 × 500 mm, with a 620 mm reclined backrest. Added inset upholstery grooves and contrasting centre panels, named inspectable mesh groups, buckle stalks/release buttons, outboard airbag tags and recline levers. Existing risers and two floor rails remain.
- Explicitly defined the fitted H-point 95 mm ahead of the cushion rear edge and 50 mm above its nominal top. The floor-mount rectangle's origin is **not** the H-point. This gives 560 mm wheel reach without moving the public steering connector or falsifying its frame.
- Corrected the sports-style cushion-height slider clamp: requested 128 mm is no longer clamped to 140 mm.
- Corrected the undersized front headrest assembly. Baseline sedan/SUV headrest tops were only **636.5 mm above the H-point**; the upright 230 mm headrest on two raised posts now reaches **819.9 mm**. The backrest still meets the cushion rather than floating above it.
- Added `seat.sport`: deeper 70 mm bolsters and a continuous integrated headrest shell, without exposed posts. Select with `components.assign.seat_front: seat.sport`.
- Rear benches retain their geometric 60/40 fold seam and now default to three larger headrests. Explicit zero/one/two/three-headrest options work; a single headrest is centred.
- B-pillar `belt_anchor_L/R` mounts carry height adjusters, D-rings and solid 47 mm-wide shoulder webbing. Webbing destinations use the same fitted buckle helper as bucket/sport seats, including seat-height slider clamping.

### Dashboard and steering

- Added passenger airbag seam, fascia strips, gauge graduations/numerals, separate speedometer/tachometer faces and needles, and a closed rear binnacle backing.
- Fixed the existing concave hood-cap bug: fan triangulation bridged the C-shaped opening and obscured the gauges. End caps are now thin quads around the opening, not triangles across it.
- Raised the cluster relative to the measured wheel centre so the SUV gauges are visible through the upper wheel opening. Its complete face clears the curved dashboard surface.
- HVAC buttons are actual 12 × 10 mm solids. Repacked HVAC/screen/centre vents and lowered the console nose so tall-H-point interiors do not bury controls behind the console. Tests check actual mesh bounds for separation.
- Retained the infotainment screen, glovebox seam/latch and louvred vents; added fitted windshield-base cowl/demister trim.
- Steering wheels retain a 370 mm outside width and round 32 mm grip section, with three spokes, thumb rests, airbag pad/badge and horn-pad edge. Sports default to a flat bottom; the bottom spoke ends at the rim. The closed rim has matching seam rings and constant-radius grip sections. Both column stalks remain.

### Console, floor, doors and closure

- Added an e-brake switch, armrest lid seam, guide rails and a partly retracted sliding cupholder cover (`cup_cover` in `[0,1]`). Retained the gear lever, boot, knob and cup wells.
- Four raised, bordered mats follow the carpet surface. Pedals have pad ribs and a heel reference patch; manuals now also get a dead pedal.
- Door cards have open-topped map pockets, sill returns and metal scuff plates, in addition to their armrests, pull/release handles and window switches.
- Replaced solid speaker discs with crossed 1 mm ribs at 6 mm pitch, actual open cells and recessed backing. Parcel-shelf speakers use the same geometry.
- Added fitted A/B/C/rear/quarter pillar skins, belt-rail returns and lower side/footwell liners. A-pillar paths come directly from the windshield aperture edge; adjacent skins follow the actual morphed loft rather than leaving broad painted wedges exposed. The front liner continues around the footwell wheel-house intrusion. The upper firewall and end returns close behind the dashboard.
- Sun visors now have the correct lateral orientation, vanity mirrors and clips; grab handles have mounting bases. Existing fabric headliner and dome lens/switches remain.
- Added closed inner rear wheel houses at the measured axle, including hatch/wagon/SUV. They are independent of the short inherited trunk-floor connector, so longer cabins still receive them. Cargo floors have side and rear lining returns.
- Fixed legacy rearview-mirror geometry being placed near the rear window. Automatic header fitting is restricted to the stale body root; custom connectors still move their mirrors normally, and `fit_header: false` preserves the legacy mount explicitly.

## Verification

- Baseline: `python3 -m pytest -q` — **280 passed**.
- Final: `python3 -m pytest -q` — **356 passed in 16.22 s**. Added 76 cases in `tests/test_interior_detail.py`.
- All eight default styles pass mesh/connector ergonomic assertions, including both front seats and headroom clearance.
- Added exact seat-target face/index topology checks; constant-radius and closed-seam wheel checks; outward ribbon signed-volume checks; groove/open-grille geometry checks; all-style control clearance, closure, mats, headrests and wheel-house checks; right-hand-drive/manual/sport/option tests; old tag-assignment compatibility; clamped belt fitting; custom mirror placement; and explicit infeasible-headroom warnings.
- During iteration, new clearance tests caught console/HVAC overlap in four styles; reducing the nose rise and repacking controls resolved those failures.
- Body fixed-topology tests remain green. No body topology/library version changes were made.
- Ran the actual CLI, not just library calls:

  ```bash
  python3 -m makecar build configs/sedan.yaml -o <fresh-temp-directory>
  python3 -m makecar build configs/suv.yaml -o <fresh-temp-directory>
  ```

  Both exited 0. Verified nonempty OBJ/MTL, body OBJ, connector/assembly JSON and PNG exports; new connectors and `interior_detail` metadata serialize correctly. Neither sample reported seat-fit warnings. Final temporary outputs: `/tmp/makecar-interior-final.PFFmEI/` (`family_sedan`: 29,227 faces; configured `family_suv`: 30,515 faces).
- Rendered and **viewed** the sedan/SUV before/after contact sheets and full-size close-ups, plus sport-seat, sport-wheel, ceiling and cargo-liner inspections. Also viewed the CLI's interior composite. Closed eye views retain the complete car; their apparent closure is not achieved by hiding the body or removing glass.

### Default front-seat packaging, all eight styles

| Measurement | Result | Brief target |
|---|---:|---:|
| H-point to accelerator heel, horizontal | 875 mm | 800–900 mm |
| Wheel centre ahead of H-point | 560 mm | 550–650 mm |
| Wheel centre above H-point | 320 mm | 300–350 mm |
| Wheel outside width / normal rim diameter | 370 mm | 370 mm |
| Column angle to horizontal | 24° | 20–25° |
| Cushion length × width | 500 × 500 mm | 500 × 480–520 mm |
| Backrest length along reclined axis | 620 mm | 600–650 mm |
| Headrest top above H-point | 819.9 mm | ≥800 mm |

These are packaging checks against the brief's SAE-style numbers, **not a claim of SAE certification or an occupant/crash simulation**. Headrests are shown in a raised operating position.

### Face budget

Counts include **all** new detail parts, not merely the legacy assembler's interior-only subset. Even after triangulation, every default interior remains below 50k faces.

| Style | Interior polygons | Interior triangles | Complete car polygons | Front headroom clearance (m) |
|---|---:|---:|---:|---:|
| sedan | 17,227 | 30,786 | 29,227 | 0.130 |
| hatchback | 17,279 | 30,890 | 29,279 | 0.152 |
| wagon | 17,863 | 31,854 | 29,863 | 0.130 |
| suv | 17,911 | 31,950 | 29,923 | 0.250 |
| pickup | 17,049 | 30,594 | 29,073 | 0.330 |
| coupe | 15,579 | 27,826 | 27,441 | 0.096 |
| sports | 15,015 | 26,902 | 26,877 | 0.032 |
| van | 17,395 | 31,014 | 29,413 | 0.350 |

## Integration notes and remaining limits

1. **Interior-only subset classification:** add `"interior_detail"` to `makecar/assembly.py`'s `INTERIOR_TAGS` when integrating. That shared file is outside ownership. Until then, `CarAssembly.interior_mesh()` omits the new extension-root parts; full-car meshes, OBJ output and the rendered comparisons include them. Budget tests explicitly include them without double-counting if the shared tag list is later updated. Reusing old tags such as `dashboard` or `seat` was rejected because valid existing assignments would select incompatible new PointConnectors and crash.
2. **Stock cutaway:** the shared pipeline does not clip new pillar trims at its belt cut. The report renderer does, only for the labelled roof-off views. The CLI's stock cutaway consequently retains upper trim fragments; its closed driver view is unaffected. A shared-pipeline integration can adopt the report script's `pillar_trim` clipping rule.
3. **Infeasible custom cabins:** minimum adult backrest/headrest dimensions cannot coexist with arbitrary low headroom/high H-points. Default styles clear the supplied envelope; infeasible custom mounts expose `headroom_clearance` and `fit_warnings` rather than silently miniaturizing the seat. This does not guarantee collision-free arbitrary slider combinations. Webbing supports the bucket/sport fitting family, not arbitrary third-party seat geometry.
4. **Inherited body packaging:** broad pillar/aperture shapes, automatic row selection and sedan-like cargo/parcel-shelf classification remain body decisions. The body-surfacing session reported that its separate tuned F-150 SuperCrew preset emits three rows with the last bench entering the bed. That preset is not present/reproduced here; the current canonical pickup has two rows. Integration should clamp pickup automatic rows appropriately and check the full reclined seat extent against the cab rear in `interior_connectors`, not shrink or silently remove an emitted bench.
5. **Mirror metadata:** the old root connector frame is deliberately preserved for compatibility, while its corrected geometry reports the actual header mount in component info. A shared connector correction can eventually remove this legacy fitting path.
6. Detail is static, low-poly geometry. Mirrors are material faces rather than ray-traced reflections; screen content is schematic; seams and grilles are modeled rather than textured. The fixed two-light renderer is flat-shaded. No runtime dependencies were added; PIL is used only by the report-image script.

## Screenshots and reproduction

All matched individual images are 960 × 640 with 2× supersampling. Cameras and belt cutoffs are saved in [cameras.json](cameras.json) and shared exactly between phases. Close-ups isolate the indicated component tree; the cluster close-up includes its dashboard and siblings, so fascia occlusion is still visible. The rear-seat view looks rearward from between the front seatbacks.

Overview sheets:

- Sedan: [before](sedan_before_contact-sheet.png) / [after](sedan_after_contact-sheet.png)
- SUV: [before](suv_before_contact-sheet.png) / [after](suv_after_contact-sheet.png)

| View | Sedan before | Sedan after | SUV before | SUV after |
|---|---|---|---|---|
| Driver eye | [PNG](sedan_before_driver-eye.png) | [PNG](sedan_after_driver-eye.png) | [PNG](suv_before_driver-eye.png) | [PNG](suv_after_driver-eye.png) |
| Passenger eye | [PNG](sedan_before_passenger-eye.png) | [PNG](sedan_after_passenger-eye.png) | [PNG](suv_before_passenger-eye.png) | [PNG](suv_after_passenger-eye.png) |
| Roof-off cutaway | [PNG](sedan_before_roof-off-cutaway.png) | [PNG](sedan_after_roof-off-cutaway.png) | [PNG](suv_before_roof-off-cutaway.png) | [PNG](suv_after_roof-off-cutaway.png) |
| Rear seat | [PNG](sedan_before_rear-seat.png) | [PNG](sedan_after_rear-seat.png) | [PNG](suv_before_rear-seat.png) | [PNG](suv_after_rear-seat.png) |
| Seat close-up | [PNG](sedan_before_seat-closeup.png) | [PNG](sedan_after_seat-closeup.png) | [PNG](suv_before_seat-closeup.png) | [PNG](suv_after_seat-closeup.png) |
| Cluster close-up | [PNG](sedan_before_cluster-closeup.png) | [PNG](sedan_after_cluster-closeup.png) | [PNG](suv_before_cluster-closeup.png) | [PNG](suv_after_cluster-closeup.png) |
| Door-card close-up | [PNG](sedan_before_door-card-closeup.png) | [PNG](sedan_after_door-card-closeup.png) | [PNG](suv_before_door-card-closeup.png) | [PNG](suv_after_door-card-closeup.png) |

Supplemental after-only isolated inspections: [sport seat](sports_after_integrated-seat.png), [flat-bottom wheel](sports_after_flat-wheel.png), [ceiling](suv_after_ceiling.png), [cargo liners](suv_after_cargo-liners.png).

```bash
python3 docs/agent_reports/03_interior_detail/render_comparisons.py --phase after
python3 docs/agent_reports/03_interior_detail/render_comparisons.py --extras
python3 -m pytest -q
```

After rendering requires no temporary snapshots. Before images were captured from the unchanged original geometry saved before implementation. Regenerating them requires trusted baseline snapshots from `f937856`; the script documents that process and accepts `--snapshot-dir`. Never use current geometry as a substitute for the before phase.

Implementation commits: `030a74b` (packaging/seats/belts), `cefa59a` (detail/closure), `6f21d56` (compatibility/clearance corrections). No push was performed.
