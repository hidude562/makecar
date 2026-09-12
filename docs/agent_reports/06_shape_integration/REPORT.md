# 06 — Exhaust refit to the archetype-shaped underbody

2026-09-12 · `agent/shape-integration` · production revision **`4255c21`**

## Result

- Starting `2a4a5e7`: **2 failed, 1,010 passed**. Both failures named in the brief reproduced unchanged.
- Final suite: **1,648 passed**, 808.07 s. **636 tests added; no existing assertions removed or relaxed.**
- **15/15 sample cars exported and validated.**
- Independent **627-configuration audit**: no pan-clearance failures, can-ground violations, collapsed cans, below-road vertices, backward sweep edges or inward-facing side triangles.
- **20 matched before/after PNGs** captured, opened and inspected, including the complete sports rear three-quarter.
- No body shape files, component implementations or runtime dependencies changed. Small local commits; **nothing pushed**.

## Geometry change

Only `makecar/body/connectors_extra.py` changes production geometry. The route is fitted before `ExhaustSystem` consumes its existing world-space path metadata; the component's circular sweeps and can fitting remain intact.

The old branch logic checked centre heights and longitudinal pan slope, but missed lateral tunnel shoulders and floor changes between rings. The first failing assertion reported **1.025498 mm** on the sports left branch. Checking both sides independently exposed an actual **−1.836489 mm** penetration on the right: the loft's two sides use different exported triangle diagonals. Sedan `plan/pointed: 1` also failed, reaching **−6.302481 mm**.

The new `_ExhaustFloor` treatment:

1. Builds a conservative ceiling from the actual measured `floor_grid`, covering both quad diagonals and both sides. No style-specific height constants or body alterations.
2. Clips floor triangles over every radius-expanded segment footprint, including station and shoulder breaks between sweep rings. Sloping and horizontal supporting planes enclose the complete circular sweep, not just its sampled centres. Their crossover points are included in the clearance calculation.
3. Lowers and eases only the required route portions to retain **at least 4 mm** pan clearance, targeting 8 mm where a correction is needed. The tunnel, turns and terminal legs all participate. The shared junction remains coincident; the final 40 mm collar retains the exact `exhaust_L/R` origins and rearward tangent.
4. Coalesces branch samples closer than 40 mm before fitting. The old 1–3 mm ring spacing could fold the inside faces of a 50 mm diameter pipe, especially after a clearance correction. The continuous footprint check still protects floor breaks between the remaining rings.
5. Caches invariant XY clipping and unchanged clearance queries. Equal-X segments use a conservative horizontal bound rather than dividing by zero.

All default-style main paths retain the existing crown-minus-33-mm placement. No audited case required smaller pipe sections or moved tip mounts. The unchanged can fitter retains its **50 mm ground clearance** and noncollapsed oval sections.

## Numerical verification

[Before audit](audit_before.json) · [After audit](audit_after.json) · [Reproduction script](audit_exhaust.py)

Minimum vertical clearance over the audit matrix, in millimetres:

| Part | Before | After |
|---|---:|---:|
| Tunnel pipe | 8.470 | 8.470 |
| Catalyst/silencer | 8.000 | 8.000 |
| Branches | **−30.452** | **5.212** |
| Terminal legs | 8.534 | 8.152 |
| Tip hardware | 4.000 | 4.000 |

Before: **66 configurations** failed pan clearance and **256** contained locally folded sweep faces. After: **zero** of either. Minimum corresponding-ring edge progress is **12.668 mm**; minimum outward side-triangle cosine is **0.900233**. Can height remains at least **24 mm** and can ground clearance at least **50 mm**.

The independent audit intersects the actual exported pan/shell triangles, not the fitter's metadata or clearance helper. It samples vertices and edge midpoints, checks both branches and all three existing tip-option combinations, and checks actual exported side-triangle winding. Its accelerated ray oracle is compared with the unaccelerated test equations at `1e-12 m`. These local fold checks are not a general nonadjacent self-intersection proof.

Coverage includes eight default styles, all inherited overrides at 0/1, isolated endpoints, normalized mixes, and every ±1 combination of ground clearance/tunnel height/tunnel width with several extreme archetype mixes. The pytest suite additionally checks **all 432 full-strength face/plan/section corners** across the eight styles, plus **192 inherited endpoint cases**, eight default bend regressions and four envelope tests.

## Acceptance runs and exports

[Starting test output](pytest_before.txt) · [Final test output](pytest_after.txt) · [Sample CLI output](samples_final.txt) · [Export validation](samples_verified.json)

```bash
python3 -m pytest -q --tb=short
OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 python3 -m makecar samples -c configs -o /tmp/si_samples
```

The CLI's 11 config files expand to **15 cars**. All have the 11 expected nonempty files: **165 files total**, including 30 valid PNGs, 30 SVGs, 30 JSON files and 30 OBJs. OBJ face/line indices are valid; car mesh counts match assembly metadata. Sample assemblies contain **56,905–64,083 faces**, below the 120k budget.

The first export-verifier draft incorrectly compared all OBJ vertices with mesh-only metadata. The exporter appends separate seam-line vertices; validation now checks those separately, without changing the exporter or any test.

`git diff --check` passes. `generator.py`, `params.py`, `styles.py`, `shapes.py`, `targets.py`, `components/exterior.py` and `tests/test_fit_sanity.py` are unchanged from `2a4a5e7`. Audit and render source hashes match production revision `4255c21`; only `connectors_extra.py` differs from the baseline package. Superseded development runs were stopped after the bend revision; the complete final suite and sample export were rerun above.

## Images actually inspected

All cameras, lighting and scene selections are locked in [cameras.json](cameras.json). The before renderer imports an immutable git archive of `2a4a5e7`; the after renderer freezes the working package. Its recorded source hashes match the subsequently committed production revision. All 20 PNG hashes and dimensions were verified, and before/after settings are identical.

These three diagnostic views show the real shell, wheels, system and tips, with a grey below-lit pan; other furniture is explicitly omitted. No selected geometry is displaced, clipped or removed. Full manifests disclose every selection and material override.

| Style | Underbody | Low side | Rear close-up |
|---|---|---|---|
| Sports | [Before](before/sports/underbody.png) / [After](after/sports/underbody.png) | [Before](before/sports/side.png) / [After](after/sports/side.png) | [Before](before/sports/rear_closeup.png) / [After](after/sports/rear_closeup.png) |
| Sedan | [Before](before/sedan/underbody.png) / [After](after/sedan/underbody.png) | [Before](before/sedan/side.png) / [After](after/sedan/side.png) | [Before](before/sedan/rear_closeup.png) / [After](after/sedan/rear_closeup.png) |
| Pickup | [Before](before/pickup/underbody.png) / [After](after/pickup/underbody.png) | [Before](before/pickup/side.png) / [After](after/pickup/side.png) | [Before](before/pickup/rear_closeup.png) / [After](after/pickup/rear_closeup.png) |

**Complete sports rear three-quarter:** [Before](before/sports/rear_three_quarter.png) / [After](after/sports/rear_three_quarter.png). This view retains the complete assembly and original materials.

The final underbody, side and close-up views show continuous exposed routes without visible floor grazing or poke-through. Sports' formerly tight shoulder passage is lower and clearer; the outlets remain aligned below the valance. The millimetre clearance claim comes from geometry checks, not image resolution.

```bash
python3 docs/agent_reports/06_shape_integration/render_shape_integration.py before --output /tmp/si-before-repro
python3 docs/agent_reports/06_shape_integration/render_shape_integration.py after
python3 docs/agent_reports/06_shape_integration/audit_exhaust.py --baseline 2a4a5e7 --output /tmp/si-audit-before.json
python3 docs/agent_reports/06_shape_integration/audit_exhaust.py --output /tmp/si-audit-after.json
```

## Remaining limits

- **Extreme pipe-road packaging is not production-realistic.** Audit case **523**—sports with all other archetypes zero, `face/wedge = plan/pointed = section/tumblehome = 1`, `ground_clearance = -1`, `tunnel_height = 1`, `tunnel_width = -1`—leaves only **1.221 mm** beneath the pipe. It clears the pan and road geometrically, and the cans still clear the road by 50 mm, but this is not a usable road-car setup. The 50 mm guarantee is for the cans, not every pipe. Default styles have at least **56.794 mm** pipe-road clearance. Avoid that simultaneous extreme for realistic vehicles.
- The low-poly feeds remain conspicuous below the sports valance, as in the previous integration report. They were not hidden or replaced with disconnected stubs.
- The conservative fitter is not a general packaging solver for arbitrary manually supplied pan/path geometry. The validated scope is the style/archetype matrix above; arbitrary further combinations can still require a different route design.
