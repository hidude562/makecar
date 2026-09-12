# Task 07 — Reference cars, using the config only

One agent per car.  You will be told which car in your prompt.

## The hard constraint

**You may not change the engine.**  Your entire deliverable is:

* `configs/<slug>.yaml` — one config file,
* `docs/reference/<slug>/` — your reference material, comparison images and a
  small `verify.py`,
* `docs/reference/<slug>/REPORT.md`.

Nothing under `makecar/` may be modified.  Not one line.  If the config
vocabulary cannot express something about the car, that is a finding to write
up, not a licence to edit the engine.  This task exists partly to measure how
expressive the config actually is, so an honest "cannot express X" is a
valuable result and a silent engine edit ruins the measurement.

## What the config can do

Read `README.md` for the schema, then get the live vocabulary:

```bash
python3 -m makecar list-styles        # 8 macro targets
python3 -m makecar list-modifiers     # 39 shape sliders, 4 sculpts, and the
                                      # face/*, plan/*, section/* archetypes
python3 -m makecar list-components    # components and their options
```

Also read `docs/agent_reports/05_shape_archetypes/REPORT.md` — the `face/*`,
`plan/*` and `section/*` archetype families are the main tool for getting a
specific car's character, and they blend.  `body.base_params` can set canonical
parameters directly when a slider cannot reach far enough; that is still config.

## Method

1. **Specs.**  Find published dimensions from the manufacturer where possible:
   overall length, width, height, wheelbase, front and rear overhang or track,
   tyre size and ground clearance.  Record each number with its source URL in
   the report.  Say which trim/model year you are targeting.
2. **Side-view reference.**  Fetch a clean side-elevation image of the car
   (a press side profile or a blueprint-style drawing is ideal) and save it
   under `docs/reference/<slug>/`.  Record the source URL and keep the file
   under about 400 kB.  Treat it strictly as a private modelling reference.
3. **Build and compare.**  Write `docs/reference/<slug>/verify.py` that builds
   your config, prints the measured dimensions against the published ones, and
   writes a comparison PNG: your side elevation rendered at the same scale as
   the reference, placed directly above or below it, plus a three-quarter view.
   Use `makecar.pipeline.build_car` and `makecar.export.render`; see how
   `makecar/pipeline.py` sets up its orthographic side camera.
4. **Iterate by eye.**  Look at the comparison image with the Read tool and
   keep tuning the config: greenhouse position and rake, hood length, roof
   apex, belt line, overhangs, wheel size and arch position, tail shape.  The
   numbers matching is not enough — the profile has to read as the car.

## Acceptance

* `python3 -m makecar build configs/<slug>.yaml -o /tmp/<slug>` succeeds.
* Measured length, width, height and wheelbase within **3%** of published, and
  say so in a table in the report with both numbers.
* `git status` shows **no modified files under `makecar/`**.
* `python3 -m pytest -q` still green (you changed no engine code, so it must be).
* A comparison image you have actually looked at, and a short honest paragraph
  on where the model diverges from the real car and which of those divergences
  the config vocabulary cannot currently express.
* Name it as an approximation, not a replica: the description field should say
  it is a config approximation in the style of the car, not a licensed model.
