# Lamborghini Aventador LP 700-4 reference-config report

## Target

This is a **config approximation**, not a licensed replica, of the **2012
Lamborghini Aventador LP 700-4 coupé** (mid-engine V12).  It uses the
`lamborghini` config/reference slug.

## Published reference data

| Item | Published value | Source |
|---|---:|---|
| Overall length | 4,780 mm | [Period LP 700-4 technical-specification brochure scan](https://xr793.com/) |
| Width, excluding mirrors | 2,030 mm | [Period LP 700-4 technical-specification brochure scan](https://xr793.com/) |
| Overall height | 1,136 mm | [Period LP 700-4 technical-specification brochure scan](https://xr793.com/) |
| Wheelbase | 2,700 mm | [Period LP 700-4 technical-specification brochure scan](https://xr793.com/) |
| Tyres, front / rear | 255/35 ZR19 / 335/30 ZR20 | [Lamborghini dealer listing of the factory LP 700-4 specification](https://www.lamborghinidallas.com/) |
| Ground clearance | 125 mm | [Published Aventador specification listing](https://www.cardekho.com/lamborghini/aventador/specifications) |

The original Lamborghini technical-specification page for the first LP 700-4
is retired.  The period brochure scan is retained as the dimensional source;
the current registry entry is a useful cross-check for the model identity:
[LP 700-4 (2011–)](https://www.lamborghiniregistry.com/Aventador/LP700-4).
Published literature gives overall length and wheelbase, rather than a
manufacturer front/rear overhang split, so the config's nearly symmetric
base split is a side-photo-derived modelling choice.

## Reference image

`reference_side_source.jpg` is a downloaded 960 px source image; the tight
`reference_side.jpg` crop is the working side reference.  Both remain under
400 kB and are retained only as private modelling reference material.

- Source: [Wikimedia Commons — *Aventador LP resized.jpg*](https://commons.wikimedia.org/wiki/File:Aventador_LP_resized.jpg)
- Photographer: Basvandergronde, own work, 2013-08-01
- Licence: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)

## Measured result

`verify.py` builds `configs/lamborghini.yaml` with `makecar.pipeline.build_car`.
It renders an orthographic side image at the same physical pixel scale as the
reference (4.780 m = 750 px), then writes `comparison.png` and a front
three-quarter `three_quarter.png`.

| Dimension | Published (m) | Config measurement (m) | Difference |
|---|---:|---:|---:|
| Length | 4.780 | 4.780 | +0.01% |
| Width | 2.030 | 2.009 | -1.03% |
| Height | 1.136 | 1.132 | -0.33% |
| Wheelbase | 2.700 | 2.700 | -0.00% |
| Ground clearance | 0.125 | 0.125 | +0.00% |

All four acceptance dimensions are within 3%.

```bash
python3 docs/reference/lamborghini/verify.py
python3 -m makecar build configs/lamborghini.yaml -o /tmp/lamborghini
```

The final CLI build succeeded, producing a 54,495-face assembly.  I also ran
`python3 -m pytest -q` before and after the config work.  The suite is not
green on the supplied parent revision: both runs end at **2 failed, 1010
passed** in the existing sports-car exhaust/pan-clearance assertions in
`tests/test_components_connectors_extra.py` (minimum observed clearance
0.001025 m versus 0.004 m).  This config-only task neither changes nor masks
that engine/component failure; the same two failures were present before this
work began.

## Render-led tuning and inspection

I inspected the final `comparison.png` and `three_quarter.png` with the
image viewer.  The first pass measured 4.865 m long, 2.074 m wide and 1.148 m
high.  The final base dimensions reduce the physical envelope to the published
values after the wedge/pointed/tumblehome morph displacements.  I also moved
and lengthened the rear glass transition, lowered the rear deck, set the
published clearance, retained the short low nose, and used large rear-tyre
scale, wide haunches, a low belt, a pointed plan, and a tumblehome to make the
profile read as the Aventador's low-cab, wedge-shaped supercar silhouette.

The final side comparison does capture the low triangular nose, near-equal
front/rear overhangs, very short greenhouse, broad rear haunch and sharply
raked upper body.  It is deliberately compared at a shared scale rather than
only by matching the tabulated figures.

## Honest limits of the config vocabulary

The config cannot independently fit the LP 700-4's 19-inch front and 20-inch
rear wheels: `wheel_diameter` is one shared axle value, so it uses the 709 mm
rear tyre diameter and makes the front tyre visibly too large.  It also has no
separate front/rear track control.  The rear deck remains a generic lofted
trunk rather than the Aventador's distinct mid-engine cover, hexagonal vents
and rear buttresses.  The fixed component library supplies generic lamp,
intake, grille, badge, door and exhaust forms; it cannot express the
Aventador's Y-lamps, large side inlet, scissor doors, centre exhaust or active
aero without engine/component work, which this task intentionally does not
perform.  Those are the main reasons this should be called an approximation,
not a replica.
