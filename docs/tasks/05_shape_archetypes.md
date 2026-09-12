# Task 05 — Front-end and silhouette archetypes (the "face shape" family)

Branch: `agent/shape-archetypes`.  Files you own: `makecar/body/shapes.py` (new),
`makecar/body/params.py`, `makecar/body/generator.py`, `makecar/body/styles.py`,
`makecar/body/targets.py`, `tests/test_shape_archetypes.py` (new).  You may read
everything.  Do NOT edit component files, `makecar/body/connectors.py`,
`connectors_extra.py`, `assembly.py` or `pipeline.py` — if a connector ends up
in the wrong place because of a shape you added, say so in the report.
Read `docs/tasks/README.md` first, then `docs/agent_reports/01_body_surfacing/REPORT.md`.

## The problem

Every style currently shares one front-end and one plan-view outline.  Render an
orthographic side, front and plan of `sedan`, `sports`, `suv` and `coupe` and you
will see it: only the greenhouse differs.  The nose droops toward a low leading
edge while the bumper juts forward underneath it, which reads as a duck's bill,
and from above all four have the same blunt rounded-rectangle footprint.  A
sports car must not have a family sedan's face.

## The model to follow

MakeHuman does not give the head one "shape" scalar.  It ships a *family* of
head-shape targets — oval, round, square, rectangular, triangular, inverted
triangular, lozenge, diamond — each a unipolar 0..1 target that is added on top
of the base mesh, so a user can dial 0.6 square + 0.4 oval and get a coherent
blend.  Build the same thing for car exteriors, in three orthogonal families.

### Family `face/*` — front-end archetype (the priority)

| target | character to hit |
|---|---|
| `face/wedge` | hood falls in a straight, steep plane to a very low leading edge; nose tip low; upper fascia raked well back over a forward splitter (mid-engine sports) |
| `face/upright` | tall, near-vertical fascia; flat hood; high blunt leading edge; the face reads as a wall (truck, SUV, van) |
| `face/shark` | centreline nose tip pushed forward and dropped *below* the fender tops, which stand crowned and proud; grille leans forward at its base (classic GT) |
| `face/snub` | very short, high, softly rounded nose; large plan and elevation corner radii; hood almost merging into the fascia (city car) |
| `face/cabforward` | face and windshield base pushed forward, hood nearly eliminated, one continuous curve from bumper to roof (MPV) |
| `face/longhood` | hood plane extended forward and flattened, cowl pushed back, leading edge staying high; long dash-to-axle (muscle, classic GT) |

### Family `plan/*` — plan-view footprint

| target | character |
|---|---|
| `plan/pointed` | nose narrows to a rounded point; strong plan taper at both ends |
| `plan/square` | near-full-width nose and tail with tight corner radii |
| `plan/cokebottle` | waisted door section with shoulders standing wide over both arches |

### Family `section/*` — front/rear cross-section

| target | character |
|---|---|
| `section/tumblehome` | glasshouse leans in strongly over wide hips |
| `section/slabside` | vertical sides, minimal tumblehome, square shoulders |
| `section/domed` | strongly domed roof and rounded shoulders |

## Required semantics

1. **Unipolar and additive.**  Each is a `Modifier` with range [0, 1], default 0,
   in group `face`, `plan` or `section`, named exactly as above.  They are
   ordinary targets, so they compose with the existing `style/*` macros and the
   39 shape sliders.
2. **Blend normalisation within a family.**  `face/wedge: 1` plus
   `face/upright: 1` must not double-displace into nonsense.  When the weights
   within one family sum to more than 1, scale that family's weights so they sum
   to 1, and apply that in `CarBody.resolve_values` (or wherever you can do it
   without touching files you do not own — if the only clean place is outside
   your files, implement it in `targets.py`/`shapes.py` and report the
   limitation).  Weights summing to ≤ 1 pass through unchanged, so a single
   archetype at 0.6 is a 60% blend with the neutral base.
3. **Per-style defaults.**  Add a `STYLE_SHAPE_DEFAULTS` table in `styles.py`
   giving each of the eight styles its default mix, applied automatically when
   that style macro is used and overridable by the config.  Suggested starting
   point, which you should tune by eye:
   * sedan `{}` (the neutral base), hatchback `{face/snub: .5, section/domed: .3}`,
     wagon `{face/upright: .25, plan/square: .3}`,
     suv `{face/upright: .7, plan/square: .5, section/slabside: .6}`,
     pickup `{face/upright: .9, plan/square: .8, section/slabside: .8}`,
     coupe `{face/longhood: .5, plan/cokebottle: .4, section/tumblehome: .5}`,
     sports `{face/wedge: .8, plan/pointed: .5, section/tumblehome: .7}`,
     van `{face/cabforward: .8, plan/square: .5, section/slabside: .5}`.
4. **Fixed topology, as always.**  Same vertex and face count and the same face
   index tuples for every style and every archetype weight.  Bump
   `LIBRARY_VERSION` in `targets.py`.

## How to build them

Prefer new *generator parameters* plus `DifferentialTargetBuilder`, the way the
`style/*` macros are made, over hand-written displacement fields: the loft then
does the smoothing and the result will not look crude.  You will need real
shape knobs the generator does not have yet, at least:

* hood plane slope and straightness between the cowl and the leading edge (a
  wedge needs a straight steep plane, an upright a flat one with a sudden drop);
* nose tip height at the centreline, independent of the fender tops, plus a
  fender crown height so the shark nose can sit below them;
* plan-view nose and tail sharpness (the taper exponent is currently the
  hardcoded `2.2` in `plan_half_width`) and separate nose/tail corner radii;
* tumblehome: lateral lean of the section between the belt and the roof rail.

Keep every existing parameter name and default working.  Where a parameter set
cannot express an archetype, a procedural sculpt in `shapes.py` in the manner of
`sculpt_haunches` is acceptable — say which ones needed it.

## Acceptance

* `python3 -m pytest -q` green, including the existing body, fit-sanity,
  component and pipeline suites, plus new tests: each archetype at 1.0 moves the
  measurements in its defined direction (e.g. `face/wedge` lowers the leading
  edge and steepens the hood plane; `plan/square` raises nose plan width;
  `section/tumblehome` reduces roof width relative to hip width); family
  normalisation behaves; topology is invariant across all archetypes and styles.
* All 15 samples build: `python3 -m makecar samples -c configs -o /tmp/shape_samples`.
* **Look at it.**  Into `docs/agent_reports/05_shape_archetypes/`, render:
  a contact sheet of the eight styles before and after (side, front, plan,
  front-3/4, matched cameras); a sweep strip per archetype (0, 0.5, 1.0) on the
  sedan; and two blend strips proving mixes work
  (`face/wedge` + `face/upright`, `face/longhood` + `face/shark`).  View them and
  iterate until the sports car's face is obviously not the sedan's, and the
  eight styles are distinguishable in plan view alone.
* `docs/agent_reports/05_shape_archetypes/REPORT.md`: the parameters you added,
  how each archetype is built, which needed sculpts, the tuned per-style
  defaults, and anything still weak.
