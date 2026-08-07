"""Built-in converters; import submodules to populate the registry."""

from . import action, classification, detection

__all__ = [
    "action",
    "classification",
    "detection",
    "MonolithUpActionConverter",
    "MonolithUpClassificationConverter",
    "MonolithUpDetectionConverter",
]

from .action import MonolithUpActionConverter
from .classification import MonolithUpClassificationConverter
from .detection import MonolithUpDetectionConverter
