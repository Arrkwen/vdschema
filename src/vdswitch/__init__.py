"""vdswitch — convert legacy dataset formats to vdschema."""

from . import converters  # noqa: F401 — register built-in converters
from .switch import switch

__all__ = ["switch"]
