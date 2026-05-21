import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from .processor import Dataset, FilteredDataset

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


# --- Color schemes ---
_COLOR_MAP = {
    "red": {"light": "#FB877A", "dark": "#BA2A19"},
    "blue": {"light": "#66C3EB", "dark": "#0075A6"},
    "green": {"light": "#AED879", "dark": "#5A8E18"},
    "orange": {"light": "#F7CB66", "dark": "#B57E00"},
    "purple": {"light": "#A49CDD", "dark": "#4E4495"},
    "fuchsia": {"light": "#DD7BD0", "dark": "#951B84"},
    "yellow": {"light": "#FEEA66", "dark": "#BEA500"},
}


def _build_settings(color: str, settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the settings dict, resolving the colour scheme."""
    if color not in _COLOR_MAP:
        color = "blue"
    colors = _COLOR_MAP[color]

    s = {
        "figsize": (10, 6),
        "dpi": 160,
        "plot_bg_color": "#FAF0E6",
        "figure_bg_color": "white",
        # Colors
        "bar_color": colors["light"],
        "bar_outline_color": colors["dark"],
        "cumulative_color": "#BA2A19",  # dark red
        "text_color": "#000000",  # black
        "outline_color": "#000000",  # black
        "reference_color": "#000000",  # black
        # Lines & fonts
        "bar_linewidth": 1.2,
        "cumulative_linewidth": 2.5,
        "reference_width": 2.0,
        "reference_linestyle": "--",
        # Font sizes
        "title_fontsize": 15,
        "subtitle_fontsize": 11,
        "header_fontsize": 12,           # grid row/column headers
        "label_fontsize": 10,
        "tick_fontsize": 9,
        "reference_fontsize": 9,
        # Grid
        "grid_color": "#D5D8DC",
        "grid_alpha": 0.5,
        "grid_linewidth": 0.8,
        "minor_grid_alpha": 0.2,
        "minor_grid_linewidth": 0.5,
        "show_minor_grid": True,
        # Manual gridline intervals (None = automatic)
        "x_major_interval": None,
        "y_major_interval": None,
        "x_minor_interval": None,
        "y_minor_interval": None,
        # Layout
        "left_margin": 0.10,
        "right_margin": 0.90,
        "bottom_margin": 0.12,
        "top_margin": 0.88,
        # Grid-mode layout
        "grid_cell_width": 6.0,
        "grid_cell_height": 4.2,
        "grid_wspace": 0.30,
        "grid_hspace": 0.62,
        "cell_subtitle_pad": 5.0,
        # Percentile markers
        "show_percentile_markers": True,
        "marker_size": 8,
        "marker_color": "#FB877A",  # light red
        "marker_edge_color": "#BA2A19",  # dark red
        "marker_edge_width": 1.5,
        "marker_label_fontsize": 8,
        "marker_label_offset": 5,  # offset in percentage points
    }
    if settings:
        s.update(settings)
    return s


def _beautiful_bins(data, target_bins, manual_start=None, manual_end=None):
    """Create bins with nice round numbers."""
    data_min, data_max = data.min(), data.max()
    if manual_start is not None:
        data_min = manual_start
    if manual_end is not None:
        data_max = manual_end

    data_range = data_max - data_min
    raw_width = data_range / target_bins if target_bins else data_range

    if raw_width <= 0:
        # Degenerate (all values equal) — fall back to a unit-ish span.
        raw_width = abs(data_min) or 1.0

    magnitude = 10 ** np.floor(np.log10(raw_width))
    normalized = raw_width / magnitude

    if normalized < 1.5:
        beautiful_width = 1 * magnitude
    elif normalized < 3.5:
        beautiful_width = 2 * magnitude
    elif normalized < 7.5:
        beautiful_width = 5 * magnitude
    else:
        beautiful_width = 10 * magnitude

    bin_start = manual_start if manual_start is not None else (
        np.floor(data_min / beautiful_width) * beautiful_width
    )
    bin_end = manual_end if manual_end is not None else (
        np.ceil(data_max / beautiful_width) * beautiful_width
    )
    return np.arange(bin_start, bin_end + beautiful_width, beautiful_width)


def _draw_distribution(
    ax: "Axes",
    dist_meta: Dict[str, Any],
    s: Dict[str, Any],
    *,
    clip_min: Optional[float],
    clip_max: Optional[float],
    reference_case: Optional[float],
    target_bins: int,
    bin_number: Optional[int],
    bin_start: Optional[float],
    bin_end: Optional[float],
    unit_override: Optional[str],
    subtitle_prefix: Optional[str],
    show_xlabel: bool,
    show_ylabel: bool,
    show_cumulative_label: bool,
) -> Dict[str, float]:
    """Render one distribution histogram + cumulative curve onto ``ax``.

    Creates its own secondary axis. Returns the {p10, p50, p90} percentiles.
    """
    distribution_data = np.asarray(dist_meta["data"], dtype=np.float64)
    distribution_data = distribution_data[np.isfinite(distribution_data)]
    if len(distribution_data) == 0:
        raise ValueError("No valid data points")

    # --- Optional clipping: drop cases outside [clip_min, clip_max] ---
    if clip_min is not None and clip_max is not None and clip_min > clip_max:
        raise ValueError(
            f"clip_min ({clip_min}) must not exceed clip_max ({clip_max})."
        )
    if clip_min is not None or clip_max is not None:
        n_before = len(distribution_data)
        if clip_min is not None:
            distribution_data = distribution_data[distribution_data >= clip_min]
        if clip_max is not None:
            distribution_data = distribution_data[distribution_data <= clip_max]
        if len(distribution_data) == 0:
            raise ValueError(
                f"No data points remain after clipping to "
                f"[{clip_min}, {clip_max}]."
            )
        n_clipped = n_before - len(distribution_data)
        if n_clipped:
            warnings.warn(
                f"distribution_plot: dropped {n_clipped} of {n_before} "
                f"case(s) outside [{clip_min}, {clip_max}].",
                stacklevel=3,
            )

    # --- Percentiles (petroleum convention: P90 low, P10 high) ---
    p90 = np.percentile(distribution_data, 10)
    p50 = np.percentile(distribution_data, 50)
    p10 = np.percentile(distribution_data, 90)

    unit = unit_override if unit_override is not None else dist_meta.get("unit")
    property_name = dist_meta.get("property", "Value")
    unit_str = f" {unit}" if unit else ""

    # --- Per-cell subtitle ---
    stats_str = f"P90: {p90:.2f}   P50: {p50:.2f}   P10: {p10:.2f}{unit_str}"
    subtitle = f"{subtitle_prefix}  |  {stats_str}" if subtitle_prefix else stats_str
    ax.set_title(
        subtitle, fontsize=s["subtitle_fontsize"], color=s["text_color"],
        alpha=0.85, pad=s["cell_subtitle_pad"],
    )

    # --- Bins ---
    if bin_number is not None:
        data_min = bin_start if bin_start is not None else distribution_data.min()
        data_max = bin_end if bin_end is not None else distribution_data.max()
        bins = np.linspace(data_min, data_max, bin_number + 1)
    else:
        bins = _beautiful_bins(distribution_data, target_bins, bin_start, bin_end)

    ax.set_facecolor(s["plot_bg_color"])

    # --- Histogram (no gaps) ---
    ax.hist(
        distribution_data, bins=bins,
        color=s["bar_color"], edgecolor=s["bar_outline_color"],
        linewidth=s["bar_linewidth"], alpha=0.9, zorder=2,
    )

    # --- Gridline intervals ---
    if s["x_major_interval"] is not None:
        ax.xaxis.set_major_locator(MultipleLocator(s["x_major_interval"]))
    if s["x_minor_interval"] is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(s["x_minor_interval"]))
    elif s["show_minor_grid"]:
        if s["x_major_interval"] is not None:
            ax.xaxis.set_minor_locator(MultipleLocator(s["x_major_interval"] / 10))
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(10))

    if s["y_major_interval"] is not None:
        ax.yaxis.set_major_locator(MultipleLocator(s["y_major_interval"]))
    if s["y_minor_interval"] is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(s["y_minor_interval"]))
    elif s["show_minor_grid"]:
        if s["y_major_interval"] is not None:
            ax.yaxis.set_minor_locator(MultipleLocator(s["y_major_interval"] / 10))
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    # --- Grid ---
    ax.grid(axis='y', which='major', alpha=s["grid_alpha"], color=s["grid_color"],
            linewidth=s["grid_linewidth"], zorder=1)
    ax.grid(axis='x', which='major', alpha=s["grid_alpha"] * 0.7, color=s["grid_color"],
            linewidth=s["grid_linewidth"], zorder=1)
    if s["show_minor_grid"]:
        ax.grid(axis='y', which='minor', alpha=s["minor_grid_alpha"],
                color=s["grid_color"], linewidth=s["minor_grid_linewidth"], zorder=1)
        ax.grid(axis='x', which='minor', alpha=s["minor_grid_alpha"],
                color=s["grid_color"], linewidth=s["minor_grid_linewidth"], zorder=1)

    # --- Reference case line ---
    if reference_case is not None:
        ax.axvline(reference_case, color=s["reference_color"], lw=s["reference_width"],
                   linestyle=s["reference_linestyle"], zorder=3.5, alpha=0.7)
        ymax = ax.get_ylim()[1]
        ax.text(reference_case, ymax * 1.03, 'Ref case',
                ha='center', va='top', fontsize=s["reference_fontsize"],
                color=s["reference_color"], zorder=4)

    # --- Cumulative curve (% with higher value) ---
    sorted_data = np.sort(distribution_data)
    n = len(sorted_data)
    percentile_higher = 100 * (1 - np.arange(1, n + 1) / n)

    ax2 = ax.twinx()
    ax2.plot(sorted_data, percentile_higher, color=s["cumulative_color"],
             linewidth=s["cumulative_linewidth"], alpha=0.9, zorder=3,
             label="Cumulative Distribution")

    # --- Percentile markers on cumulative line ---
    if s["show_percentile_markers"]:
        for value, cumulative_pct, label in (
            (p90, 90.0, "P90"), (p50, 50.0, "P50"), (p10, 10.0, "P10"),
        ):
            ax2.plot(value, cumulative_pct, marker='o', markersize=s["marker_size"],
                     markerfacecolor=s["marker_color"], markeredgecolor=s["marker_edge_color"],
                     markeredgewidth=s["marker_edge_width"], zorder=4)
            ax2.text(value, cumulative_pct + s["marker_label_offset"], label,
                     ha='center', va='bottom', fontsize=s["marker_label_fontsize"],
                     color=s["marker_edge_color"], fontweight='bold', zorder=5)

    # --- Secondary axis (100% top, 0% bottom) ---
    ax2.set_ylim(0, 100)
    ax2.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax2.tick_params(axis='y', labelcolor=s["text_color"], colors=s["text_color"],
                    labelsize=s["tick_fontsize"])
    ax2.spines['right'].set_color(s["outline_color"])
    ax2.spines['right'].set_linewidth(1.1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
    if show_cumulative_label:
        ax2.set_ylabel("Cumulative Distribution", fontsize=s["label_fontsize"],
                       color=s["text_color"])

    # --- Axis labels ---
    if show_xlabel:
        if unit:
            disp = (property_name.upper()
                    if property_name.lower() in ['npv', 'stoiip', 'giip', 'hcpv']
                    else property_name.title())
            x_label = f"{disp} ({unit})"
        else:
            x_label = property_name.title() if property_name else "Value"
        ax.set_xlabel(x_label, fontsize=s["label_fontsize"], color=s["text_color"])
    if show_ylabel:
        ax.set_ylabel("Frequency", fontsize=s["label_fontsize"], color=s["text_color"])

    # --- Axis styling ---
    for spine in ['left', 'bottom', 'top']:
        ax.spines[spine].set_color(s["outline_color"])
        ax.spines[spine].set_linewidth(1.1)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', colors=s["text_color"], which='both',
                   labelsize=s["tick_fontsize"])

    return {"p10": p10, "p50": p50, "p90": p90}


def distribution_plot(
    ds: Union[Dataset, FilteredDataset],
    *,
    property: Union[str, List[str]],
    parameter: Optional[str] = None,
    filters: Union[Dict[str, Any], str, List[Union[Dict[str, Any], str]], None] = None,
    multiplier: Optional[float] = None,
    title: Optional[str] = None,
    unit: Optional[str] = None,
    outfile: Optional[Union[str, Path]] = None,
    target_bins: int = 20,
    color: str = "blue",
    reference_case: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
    settings: Optional[Dict[str, Any]] = None,
    bin_number: Optional[int] = None,
    bin_start: Optional[float] = None,
    bin_end: Optional[float] = None,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> Tuple["Figure", Union["Axes", Any], Optional[str]]:
    """
    Distribution histogram (with cumulative curve) for one property of one parameter.

    Grid mode: pass a list for ``property`` and/or ``filters`` to render a grid
    of subplots — properties stack across columns, filters stack down rows. A
    scalar ``property`` and scalar ``filters`` produce a single plot as before.

    Args:
        ds: Dataset.
        property: Property to plot (e.g. 'stoiip'), or a list of properties —
                  one column per property.
        parameter: Sheet name. Defaults to the first sheet — a warning is printed
                   when defaulted.
        filters: Spatial filter dict ({field: value(s)}) or stored-filter name.
                 Must not contain a 'property' key. A list of filters renders one
                 row per filter. (A FilteredDataset cannot be combined with a
                 filter list — pass a plain Dataset for filter grids.)
        multiplier: Optional display multiplier override.
        clip_min: Optional lower bound. Cases below it are dropped before
                  percentiles, bins and the cumulative curve are computed.
                  In display units (same as the x-axis).
        clip_max: Optional upper bound, applied the same way as clip_min.
        title, unit, outfile, target_bins, color, reference_case, figsize,
        settings, bin_number, bin_start, bin_end: Plot styling — same as before.

    Returns:
        ``(fig, ax, saved)`` for a single plot, or ``(fig, axes_2d, saved)`` in
        grid mode where ``axes_2d`` is a 2-D numpy array of Axes indexed
        ``[row][column]``.
    """
    # --- Resolve a FilteredDataset to (Dataset, filters) ---
    if isinstance(ds, FilteredDataset):
        if filters is not None:
            raise ValueError(
                "distribution_plot received both a FilteredDataset (which "
                "already carries a filter) and a filters= argument. Pick one."
            )
        view_filters = dict(ds.filters) if ds.filters else {}
        if ds.title:
            view_filters['title'] = ds.title
        filters = view_filters or None
        ds = ds.dataset

    if not isinstance(ds, Dataset):
        raise TypeError(
            "distribution_plot expects a Dataset or FilteredDataset as first "
            f"argument. Got {type(ds).__name__}."
        )

    # --- Normalise property / filters to lists; detect grid mode ---
    prop_is_list = isinstance(property, list)
    filt_is_list = isinstance(filters, list)
    properties = list(property) if prop_is_list else [property]
    filter_list = list(filters) if filt_is_list else [filters]
    is_grid = prop_is_list or filt_is_list

    if not properties or not filter_list:
        raise ValueError("distribution_plot: 'property' and 'filters' lists must be non-empty.")

    # --- Resolve parameter ---
    if parameter is None:
        params = ds.parameters()
        if not params:
            raise ValueError("Dataset has no parameters.")
        parameter = params[0]
        warnings.warn(
            f"distribution_plot: 'parameter' not specified — defaulting to "
            f"'{parameter}'. Available parameters: {params}",
            stacklevel=2,
        )

    s = _build_settings(color, settings)

    nrows, ncols = len(filter_list), len(properties)

    # --- Figure sizing ---
    if figsize is not None:
        fig_w, fig_h = figsize
    elif is_grid:
        fig_w = s["grid_cell_width"] * ncols
        fig_h = s["grid_cell_height"] * nrows
    else:
        fig_w, fig_h = s["figsize"]

    plt.close("all")
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_w, fig_h), dpi=s["dpi"], squeeze=False,
    )
    fig.patch.set_facecolor(s["figure_bg_color"])

    # --- Margins: tighter for grids, with room reserved for headers ---
    if is_grid:
        left = 0.085 + (0.055 if nrows > 1 else 0.0)
        right = 0.93
        bottom = 0.085
        top = 0.875 - (0.03 if ncols > 1 else 0.0)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                            wspace=s["grid_wspace"], hspace=s["grid_hspace"])
    else:
        fig.subplots_adjust(left=s["left_margin"], right=s["right_margin"],
                            bottom=s["bottom_margin"], top=s["top_margin"])

    # --- Draw each cell ---
    col_labels: List[str] = []
    row_labels: List[str] = []
    for c, prop in enumerate(properties):
        for r, flt in enumerate(filter_list):
            ax = axes[r][c]
            dist_meta = ds._distribution_data(
                parameter=parameter, property=prop, filters=flt, multiplier=multiplier,
            )
            if isinstance(dist_meta, dict) and 'data' not in dist_meta:
                raise ValueError(
                    "distribution_plot received multi-property data but expects a "
                    "single property per cell — pass plain strings in the list."
                )
            # In grid mode the filter name is the row header, so it is not
            # repeated in the per-cell subtitle.
            filter_name = dist_meta.get('filter_name')
            _draw_distribution(
                ax, dist_meta, s,
                clip_min=clip_min, clip_max=clip_max, reference_case=reference_case,
                target_bins=target_bins, bin_number=bin_number,
                bin_start=bin_start, bin_end=bin_end, unit_override=unit,
                subtitle_prefix=None if is_grid else filter_name,
                show_xlabel=(r == nrows - 1),
                show_ylabel=(c == 0),
                show_cumulative_label=(c == ncols - 1),
            )
            if r == 0:
                pn = dist_meta.get('property', prop) or prop
                col_labels.append(
                    pn.upper() if str(pn).lower() in ['npv', 'stoiip', 'giip', 'hcpv']
                    else str(pn).title()
                )
            if c == 0:
                row_labels.append(_filter_label(ds, flt, dist_meta, r))

    # --- Figure title ---
    if title is None:
        first = ds._distribution_data(parameter=parameter, property=properties[0],
                                      filters=filter_list[0], multiplier=multiplier)
        title = first.get('title', 'Distribution') if isinstance(first, dict) else 'Distribution'
    fig.text(0.5, 0.975, title, ha="center", va="top",
             fontsize=s["title_fontsize"], fontweight="bold", color=s["text_color"])

    # --- Grid row / column headers ---
    if is_grid and ncols > 1:
        for c in range(ncols):
            box = axes[0][c].get_position()
            fig.text((box.x0 + box.x1) / 2, top + 0.052, col_labels[c],
                     ha="center", va="center", fontsize=s["header_fontsize"],
                     fontweight="bold", color=s["text_color"])
    if is_grid and nrows > 1:
        for r in range(nrows):
            box = axes[r][0].get_position()
            fig.text(0.018, (box.y0 + box.y1) / 2, row_labels[r],
                     ha="center", va="center", rotation=90,
                     fontsize=s["header_fontsize"], fontweight="bold",
                     color=s["text_color"])

    # --- Save ---
    saved = None
    if outfile:
        outfile = Path(outfile)
        fig.savefig(outfile, bbox_inches="tight", facecolor=s["figure_bg_color"])
        saved = str(outfile)

    return fig, (axes if is_grid else axes[0][0]), saved


def _filter_label(ds: Dataset, flt: Any, dist_meta: Dict[str, Any], row_idx: int) -> str:
    """Best-effort human label for a filter, used as a grid row header."""
    name = dist_meta.get('filter_name')
    if name:
        return str(name)
    if isinstance(flt, str):
        return flt
    if isinstance(flt, dict):
        if flt.get('title'):
            return str(flt['title'])
        # Summarise the field selections compactly.
        parts = []
        for k, v in flt.items():
            if k in ('title', 'name'):
                continue
            parts.append(f"{k}={v}" if not isinstance(v, list) else f"{k}=[{len(v)}]")
        if parts:
            return ", ".join(parts)
    return f"Filter {row_idx + 1}"
