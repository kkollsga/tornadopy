"""
TornadoPy - A Python library for tornado chart generation and analysis.

This library provides tools for processing Excel-based tornado data and
generating professional tornado charts for uncertainty analysis.
"""

from .processor import TornadoProcessor
from .plot import tornado_plot

__version__ = "0.1.0"
__all__ = ["TornadoProcessor", "tornado_plot"]
