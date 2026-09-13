"""Interactive viewer: a local web app for editing targets and connectors in 3-D.

    python -m makecar viewer configs/sedan.yaml

The Python side owns the engine and the config; the browser owns rendering
and interaction.  Edits are ordinary config state (modifier values, connector
overrides, component assignments) or `.target` files, so anything done in the
viewer can be saved and reproduced from the command line.
"""
from .server import serve, ViewerSession  # noqa: F401
