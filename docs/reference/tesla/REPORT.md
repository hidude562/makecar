# Tesla reference-car configuration report

**Target:** 2024 Tesla Model 3 RWD / Long Range (the 2023-refresh body sold as the 2024 Model 3 in the US).  This is a **config approximation in the style of the car**, not a licensed model or a replica.

## Sources

- [Tesla Model 3 owner's-manual dimensions](https://www.tesla.com/ownersmanual/model3/en_us/GUID-56562137-FC31-4110-A13C-9A9FC6657BF0.html): length, width excluding mirrors, height, wheelbase and ground clearance.
- [Tesla 2024+ Model 3 service manual — replacement tyre information](https://service.tesla.com/docs/Model3/ServiceManual/2024/en-us/GUID-92AB1F42-7358-45AD-BACC-90A41BB3E715.html): 235/45R18 tyre specification for the 18-inch wheel.
- [CC0 left-side 2023-refresh Model 3 photo](https://commons.wikimedia.org/wiki/File:Tesla_Model_3_(2023),_long_range,_Japan,_left-side.jpg), downloaded here as `model3_2024_side_reference.jpg` (201,689 bytes).  It is used only as private modelling reference.  Its refresh body is the relevant 2024 US model; it is a close side photograph rather than a technical orthographic drawing.

## Dimensional verification

All dimensions are metres.  The generated measurement is from `build_car`, rather than simply restating the values put into the config.

| Metric | Tesla published | Config measurement | Error |
|---|---:|---:|---:|
| Overall length | 4.720 | 4.720 | 0.00% |
| Width, mirrors excluded | 1.849 | 1.849 | 0.00% |
| Overall height | 1.440 | 1.441 | +0.04% |
| Wheelbase | 2.875 | 2.875 | 0.00% |
| Ground clearance | 0.138 | 0.138 | 0.00% |

The 0.668 m configured tyre diameter is the nominal outside diameter of the cited 235/45R18 fitment.  The config uses 0.855 m front and 0.955 m rear canonical overhangs; the resulting mesh, including its terminal bumper treatment, measures the published 4.720 m overall.

## What was tuned

`configs/tesla.yaml` uses the sedan topology with direct base parameters for the Tesla wheelbase, exterior envelope, ride height and wheel package.  I placed the cowl and roof hard points for the compact, long-glasshouse silhouette; lengthened the rear glass run and reduced the deck; reduced body bulge/arch flare; and used a mildly pointed plan and tumblehome.  After viewing the matched-scale comparison, I removed the straight-horizon wedge treatment: the curved lower hood reads closer to the reference's low rounded nose.

The car is battery-electric.  `components.disable` explicitly removes `grille`, `intake`, all `exhaust*`, `fuel_cap`, fog lamps, rear reflectors and tow eye.  It also removes the generic plate component: its black simulated characters read as a grille in this renderer.  Thus the finished front is deliberately blank and closed rather than pretending to have a radiator opening.

## Render audit

Run:

```bash
python3 docs/reference/tesla/verify.py
```

The script calls `makecar.pipeline.build_car` and `makecar.export.render`, prints the table above, and writes these inspected outputs:

- [matched-scale side comparison](comparison.png) — reference immediately above the config render; both use 270 px/m from their respective 4.720 m outlines
- [rendered side](rendered_side.png)
- [three-quarter exterior](rendered_three_quarter.png)
- [front](rendered_front.png)
- [nose close-up](rendered_nose.png)

I inspected the final comparison, three-quarter/front presentation, and nose close-up with the image viewer.  The exact wheelbase, low ride height, short rounded nose, four-door sedan proportions, large wheels, continuous falling rear glass and closed EV front distinguish the intended Model 3 approximation when set beside the reference.

## Known divergence / config-vocabulary finding

The matching envelope and wheel locations are strong, but the fixed sedan topology remains visibly more angular and three-boxed than the Model 3: its roof and side glazing form planar trapezoids, its C-pillar/deck break is heavier, and the black panoramic glass roof cannot be represented by the current body-plus-glass component vocabulary.  The generic twin-projector headlamps and rectangular rear lamps do not resemble the refresh's thin Tesla-specific lighting, and the generic alloy wheel does not reproduce Tesla's aero cover.  There is no config-only control for a continuous full-width black glass canopy, Tesla lamp signatures, a brand-specific fascia, or aero wheel-cover geometry.  Those omissions are documented rather than compensated by engine changes.

## Validation status

- `python3 -m makecar build configs/tesla.yaml -o /tmp/tesla` succeeds.
- No file under `makecar/` was modified.
- The baseline and final test suite have the same two pre-existing `sports` exhaust/pan-clearance failures in `tests/test_components_connectors_extra.py`; all other 1,010 tests pass.  This task changed no engine or test code.
