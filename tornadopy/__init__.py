"""
TornadoPy - A Python library for tornado chart generation and analysis.

This library provides tools for processing Excel-based tornado data and
generating professional tornado charts for uncertainty analysis.
"""

from .processor import TornadoProcessor
from .plot import tornado_plot
from .distribution import distribution_plot

# Dynamic version from package metadata
try:
    from importlib.metadata import version
    __version__ = version("tornadopy")
except Exception:
    # Fallback for development installs
    __version__ = "0.0.0.dev"

__all__ = ["TornadoProcessor", "tornado_plot", "distribution_plot"]
