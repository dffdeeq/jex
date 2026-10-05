"""jex - a Jev-style System One decision model built from an existing LLM."""

from .schema import SchemaError, parse_request

__all__ = ["SchemaError", "parse_request"]
__version__ = "0.1.0"
