"""Component library.  Importing this package registers all built-in components."""
from .base import (  # noqa: F401
    CarComponent, MorphableComponent, ComponentResult, BuildContext, Palette,
    REGISTRY, register, get_component, default_component_for, describe_all,
)
from . import exterior  # noqa: F401,E402
from . import interior  # noqa: F401,E402
