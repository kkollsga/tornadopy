"""
TornadoPy - A Python library for tornado chart generation and analysis.

This library provides tools for processing Excel-based tornado data and
generating professional tornado charts for uncertainty analysis.
"""

from typing import List

from .processor import Case, Dataset, FilteredDataset, read_clipboard
from .plot import tornado_plot
from .distribution import distribution_plot
from .correlation import correlation_plot


def _silence_pandas_read_clipboard_dtype_warning() -> None:
    """Wrap ``pandas.read_clipboard`` so it doesn't emit the wide-Petrel
    DtypeWarning.

    A Petrel volumetrics paste is ~1700 columns with mixed types per column
    (text in metadata rows, numbers in data rows). ``pd.read_clipboard``
    therefore emits a noisy ``DtypeWarning`` that flashes in the Jupyter cell
    above the plot. The warning is informational only — every consumer in
    tornadopy converts column values to strings before parsing — so we
    suppress it at import time, narrowly scoped to ``read_clipboard``.

    Opt out: ``pandas.read_clipboard = pandas.read_clipboard.__wrapped__``.
    """
    try:
        import pandas as _pd
        from pandas.errors import DtypeWarning as _DtypeWarning
    except ImportError:
        return
    if getattr(_pd.read_clipboard, "_tornadopy_wrapped", False):
        return
    import functools as _functools
    import warnings as _warnings

    _orig = _pd.read_clipboard

    @_functools.wraps(_orig)
    def _wrapper(*args, **kwargs):
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", category=_DtypeWarning)
            return _orig(*args, **kwargs)

    _wrapper._tornadopy_wrapped = True  # type: ignore[attr-defined]
    _pd.read_clipboard = _wrapper


_silence_pandas_read_clipboard_dtype_warning()

# Dynamic version from package metadata
try:
    from importlib.metadata import version
    __version__: str = version("tornadopy")
except Exception:
    # Fallback for development installs
    __version__: str = "0.0.0.dev"

__all__: List[str] = [
    "Dataset",
    "FilteredDataset",
    "Case",
    "tornado_plot",
    "distribution_plot",
    "correlation_plot",
    "read_clipboard",
]
