# Agent task briefs

Each file here is a self-contained brief for an autonomous coding agent working
on one realism work stream, on its own branch/worktree.  Shared ground rules
for every brief:

* Read `README.md`, then the files the brief names.  Run `python3 -m pytest -q`
  before you start (it must be green) and before you finish (it must be green,
  and you must ADD tests for what you build).
* World frame: +X forward, +Y left, +Z up, ground z=0, metres.  Origin at the
  wheelbase midpoint.  Real-world scale matters: check every dimension you
  introduce against a real car.
* LOOK at your work.  `makecar.export.render.Renderer` renders PNGs
  (see `makecar/pipeline.py: render_views` and `cutaway_mesh`).  Write
  before/after comparison images into `docs/agent_reports/<task>/` and view
  them (the Read tool shows PNGs).  Do not declare something realistic without
  having looked at it from at least three angles, including a close-up.
* The body's fixed-topology invariant is sacred: `BodyGenerator` must return the
  same vertex/face count and face index tuples for ALL styles and parameter
  values (tests enforce it).  If you change `SEGMENTS`, station counts or the
  cap construction, bump `LIBRARY_VERSION` in `makecar/body/targets.py` and make
  sure connectors, zones, groups and seams still line up.
* Do not break the connector contract that components rely on (names, tags,
  frame conventions, meta keys documented in `makecar/body/connectors.py`).
  New connectors go in `makecar/body/connectors_extra.py`.
* Keep meshes light: a full car should stay under ~120k faces.  Prefer quads
  and lofts; use `makecar/geometry/primitives.py`.
* Commit on your branch in small, described commits (git user is preconfigured).
  No pushing.  When done, write `docs/agent_reports/<task>/REPORT.md`: what
  changed, what you verified and how, open issues, screenshots list.
