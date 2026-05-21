from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

from .processor import Dataset, FilteredDataset
from ._colors import _cell_color, _gap_fraction, _tint_pair

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes


def _build_settings(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the tornado settings dict from defaults + overrides."""
    s = {
        "figsize": (10, 7),
        "dpi": 160,
        "plot_bg_color": "#FAF0E6",
        "figure_bg_color": "white",
        # Default colours (used when color=None)
        "pos_light": "#A9CFF7",
        "neg_light": "#F5B7B1",
        "pos_dark": "#2E5BFF",
        "neg_dark": "#E74C3C",
        # Single-colour mode (color=<family>): tints + opacities
        "neg_shade": 700,        # palette shade for negative (left) bars
        "pos_shade": 400,        # palette shade for positive (right) bars
        "inner_opacity": 100,    # % opacity for inner p90-p10 bars
        "outer_opacity": 40,     # % opacity for outer min-max bars
        "shadow_offset": 2.0,    # inner-bar shadow offset (points, outward)
        # Lines & fonts
        "outline_color": "#2C3E50",
        "label_color": "#1C2833",
        "baseline_color": "#2C3E50",
        "reference_color": "#000000",
        "bar_linewidth": 0.8,
        "baseline_width": 2.0,
        "reference_width": 2.0,
        "bar_height": 0.6,
        "axis_buffer": 0.3,
        # Font sizes
        "title_fontsize": 15,
        "subtitle_fontsize": 11,
        "header_fontsize": 12,           # grid row/column headers
        "label_fontsize": 9,
        "value_fontsize": 6,
        "header_fontsize_value": 7.5,
        "reference_fontsize": 9,
        # Grid
        "grid_color": "#D5D8DC",
        "grid_major_alpha": 0.35,
        "grid_minor_alpha": 0.15,
        "hgrid_alpha": 0.55,
        "vgrid_alpha": 0.3,
        # Values & layout
        "show_values": ["min", "p10", "p90", "max"],
        "show_value_headers": True,
        "value_format": "{:.1f}",
        "value_offset": 0.01,
        "label_gap": 0.01,
        "left_margin": 0.18,
        "header_value_spacing": 0.09,
        # Grid-mode layout
        "grid_cell_width": 8.0,
        "grid_cell_height": 6.0,       # floor; height also scales with bar count
        "grid_bar_inches": 0.55,       # inches of cell height per tornado bar
        "grid_col_gap": 1.40,          # inches between columns (fits per-cell y labels)
        "grid_row_gap": 0.55,          # inches of blank space between rows
        "cell_subtitle_pad": 16.0,
        # Feature toggles
        "show_relative_values": False,
        "show_percentage_diff": True,
        "show_bar_shadows": True,
    }
    if settings:
        s.update(settings)
    return s


def _tornado_colors(color: Any, s: Dict[str, Any]) -> Dict[str, Tuple[Any, float]]:
    """Return the four ``(colour, alpha)`` styles for the tornado bar layers.

    With ``color=None`` the established default scheme is used (distinct
    positive/negative hues). Given a single colour, both signs share that
    family — negative bars take ``neg_shade``, positive bars ``pos_shade``,
    and inner/outer bars are separated by ``inner_opacity``/``outer_opacity``.
    """
    if color is None:
        return {
            "neg_outer": (s["neg_light"], 1.0),
            "neg_inner": (s["neg_dark"], 0.9),
            "pos_outer": (s["pos_light"], 1.0),
            "pos_inner": (s["pos_dark"], 0.9),
        }
    pos_fill, neg_fill = _tint_pair(color, s["pos_shade"], s["neg_shade"])
    inner_a = max(0.0, min(1.0, s["inner_opacity"] / 100.0))
    outer_a = max(0.0, min(1.0, s["outer_opacity"] / 100.0))
    return {
        "neg_outer": (neg_fill, outer_a),
        "neg_inner": (neg_fill, inner_a),
        "pos_outer": (pos_fill, outer_a),
        "pos_inner": (pos_fill, inner_a),
    }


def _filter_label(flt: Any, idx: int) -> str:
    """Best-effort human label for a filter, used as a grid row header."""
    if isinstance(flt, str):
        return flt
    if isinstance(flt, dict):
        if flt.get('title'):
            return str(flt['title'])
        if flt.get('name'):
            return str(flt['name'])
        parts = []
        for k, v in flt.items():
            if k in ('title', 'name'):
                continue
            parts.append(f"{k}=[{len(v)}]" if isinstance(v, list) else f"{k}={v}")
        if parts:
            return ", ".join(parts)
    return f"Filter {idx + 1}"


def _resolve_entry(entry: Any) -> Tuple[Dataset, Optional[Dict[str, Any]]]:
    """Resolve one ds entry to (Dataset, filter-or-None)."""
    if isinstance(entry, FilteredDataset):
        view_filters = dict(entry.filters) if entry.filters else {}
        if entry.title:
            view_filters['title'] = entry.title
        return entry.dataset, (view_filters or None)
    if isinstance(entry, Dataset):
        return entry, None
    raise TypeError(
        "tornado_plot expects a Dataset, a FilteredDataset, or a list of them "
        f"as the first argument. Got {type(entry).__name__}."
    )


def _resolve_rows(
    ds: Any,
    filters: Any,
) -> Tuple[List[Dataset], List[Optional[Dict[str, Any]]]]:
    """Resolve the ds + filters arguments into per-row (Dataset, filter) pairs."""
    if isinstance(ds, list):
        if not ds:
            raise ValueError("tornado_plot: 'ds' list is empty.")
        if filters is not None:
            raise ValueError(
                "tornado_plot: when 'ds' is a list, 'filters' must be None "
                "— each entry carries its own filter."
            )
        datasets, row_filters = [], []
        for entry in ds:
            d, f = _resolve_entry(entry)
            datasets.append(d)
            row_filters.append(f)
        return datasets, row_filters

    if isinstance(ds, FilteredDataset):
        if filters is not None:
            raise ValueError(
                "tornado_plot received both a FilteredDataset (which already "
                "carries a filter) and a filters= argument. Pick one."
            )
        d, f = _resolve_entry(ds)
        return [d], [f]

    if isinstance(ds, Dataset):
        filter_list = list(filters) if isinstance(filters, list) else [filters]
        if not filter_list:
            raise ValueError("tornado_plot: 'filters' list must be non-empty.")
        return [ds] * len(filter_list), filter_list

    raise TypeError(
        "tornado_plot expects a Dataset, a FilteredDataset, or a list of them "
        f"as the first argument. Got {type(ds).__name__}."
    )


def _draw_tornado(
    ax: "Axes",
    sections: List[Dict[str, Any]],
    s: Dict[str, Any],
    *,
    base: Optional[float],
    reference_case: Optional[float],
    unit_override: Optional[str],
    subtitle_prefix: Optional[str],
    subtitle_override: Optional[str],
    preferred_order: Optional[List[str]],
    show_xlabel: bool,
    show_param_labels: bool,
    show_ref_label: bool,
    bar_styles: Dict[str, Tuple[Any, float]],
    positive_text_black: bool,
) -> Dict[str, Any]:
    """Render one tornado chart onto ``ax``. Returns detected property/unit/filter."""
    # --- Auto-detect base, reference, filter_name, property_name, unit ---
    auto_base = auto_reference = auto_filter_name = None
    auto_property_name = auto_unit = None
    for sec in sections:
        if "base_case" in sec:
            auto_base = sec.get("base_case")
        if "reference_case" in sec:
            auto_reference = sec.get("reference_case")
        if "filter_name" in sec:
            auto_filter_name = sec.get("filter_name")
        if "property_name" in sec:
            auto_property_name = sec.get("property_name")
        if "unit" in sec:
            auto_unit = sec.get("unit")
        if any(v is not None for v in (auto_base, auto_reference, auto_filter_name,
                                       auto_property_name, auto_unit)):
            break

    if base is None and auto_base is not None:
        base = auto_base
    elif base is None:
        base = 0.0
    if reference_case is None and auto_reference is not None:
        reference_case = auto_reference

    property_name = auto_property_name
    detected_unit = unit_override if unit_override is not None else auto_unit

    # --- Per-cell subtitle ---
    if subtitle_override is not None:
        subtitle = subtitle_override
    else:
        unit_str = f" {detected_unit}" if detected_unit else ""
        case_values = []
        if base is not None:
            case_values.append(f"Base case: {base:.2f}")
        if reference_case is not None:
            case_values.append(f"Ref case: {reference_case:.2f}")
        if case_values and unit_str:
            case_values[-1] += unit_str
        stats_str = "   ".join(case_values)
        if subtitle_prefix and stats_str:
            subtitle = f"{subtitle_prefix}  |  {stats_str}"
        else:
            subtitle = subtitle_prefix or stats_str
    if subtitle:
        ax.set_title(subtitle, fontsize=s["subtitle_fontsize"], color=s["label_color"],
                     alpha=0.85, pad=s["cell_subtitle_pad"])

    # --- Prepare data ---
    data = []
    for sec in sections:
        if "minmax" not in sec and "range" not in sec:
            continue
        if "minmax" in sec:
            low, high = sec["minmax"]
        elif "range" in sec:
            low, high = base + sec["range"][0], base + sec["range"][1]
        else:
            continue
        data.append({
            "parameter": sec.get("parameter", sec.get("title", "")),
            "low": low, "high": high,
            "p90p10": sec.get("p90p10"),
            "label_pos": sec.get("label_pos", {}),
        })

    if not data:
        error_msgs = []
        for sec in sections:
            for err in sec.get("errors", []) or []:
                error_msgs.append(f"  - {sec.get('parameter', '<unknown>')}: {err}")
        if error_msgs:
            raise ValueError(
                "tornado_plot produced no bars — every parameter failed:\n"
                + "\n".join(error_msgs)
                + "\n\nUse ds.describe() to inspect available sheets, properties, "
                "and filter fields."
            )
        raise ValueError(
            "tornado_plot produced no bars (no minmax/p90p10 data in any section). "
            "Use ds.describe() to inspect available sheets and properties."
        )

    # --- Sort data by preferred order ---
    if preferred_order:
        order_map = {param: idx for idx, param in enumerate(preferred_order)}
        preferred_data, remaining_data = [], []
        for d in data:
            if d["parameter"] in order_map:
                preferred_data.append((order_map[d["parameter"]], d))
            else:
                remaining_data.append(d)
        preferred_data.sort(key=lambda x: x[0])
        data = [d for _, d in preferred_data] + remaining_data

    # --- Axis limits ---
    lows, highs = [d["low"] for d in data], [d["high"] for d in data]
    max_extent = max(abs(min(lows) - base), abs(max(highs) - base))
    buffer = max_extent * s["axis_buffer"]
    xmin, xmax = base - max_extent - buffer, base + max_extent + buffer
    xspan = xmax - xmin

    n_bars = len(data)
    header_value_offset = s["header_value_spacing"]
    y_padding = 0.5
    ymin = -y_padding
    ymax = n_bars - 1 + y_padding
    ref_case_offset = y_padding + 0.1

    ax.set_facecolor(s["plot_bg_color"])

    # --- Gridlines ---
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='x', which='major', alpha=s["vgrid_alpha"], color=s["grid_color"], zorder=0)
    ax.grid(axis='x', which='minor', alpha=s["vgrid_alpha"] * 0.7, color=s["grid_color"], zorder=0)
    for i in range(len(data)):
        ax.axhline(i, color=s["grid_color"], alpha=s["hgrid_alpha"], lw=0.6, zorder=0)

    # --- Reference case line ---
    if reference_case is not None:
        ax.axvline(reference_case, color=s["reference_color"], lw=s["reference_width"],
                   linestyle='--', zorder=0.5, alpha=0.3)
        # In a grid the per-cell subtitle already states the ref value, so the
        # in-plot 'Ref case' annotation is suppressed to avoid the duplicate.
        if show_ref_label:
            ax.text(reference_case, -ref_case_offset, 'Ref case',
                    ha='center', va='bottom', fontsize=s["reference_fontsize"],
                    color=s["reference_color"], zorder=4)

    # Inner-bar shadows point straight outward: left on the negative side,
    # right on the positive side.
    _so = s["shadow_offset"]
    neg_bar_shadow = [patheffects.withSimplePatchShadow(offset=(-_so, 0.0), alpha=0.25)]
    pos_bar_shadow = [patheffects.withSimplePatchShadow(offset=(_so, 0.0), alpha=0.25)]
    white_text_outline = [patheffects.withStroke(linewidth=1.2, foreground='black', alpha=0.4)]

    def has_space_for_label(text, width_in_data_units, xspan):
        char_width = 0.0065 * xspan * (s["value_fontsize"] / 8.0)
        return width_in_data_units > len(text) * char_width * 1.3

    no_color, no_alpha = bar_styles["neg_outer"]
    ni_color, ni_alpha = bar_styles["neg_inner"]
    po_color, po_alpha = bar_styles["pos_outer"]
    pi_color, pi_alpha = bar_styles["pos_inner"]

    # --- Draw bars ---
    for i, d in enumerate(data):
        low, high, p90p10 = d["low"], d["high"], d.get("p90p10")

        neg_outer_width = 0
        if low < base:
            neg_outer_width = min(high, base) - low
            ax.barh(i, neg_outer_width, left=low, height=s["bar_height"],
                    color=no_color, alpha=no_alpha, zorder=1)

        pos_outer_width = 0
        if high > base:
            pos_outer_width = high - max(low, base)
            ax.barh(i, pos_outer_width, left=max(low, base), height=s["bar_height"],
                    color=po_color, alpha=po_alpha, zorder=1)

        neg_inner_width = pos_inner_width = 0
        inner_neg = inner_pos = None
        if p90p10:
            p90, p10 = p90p10
            if p90 < base:
                neg_inner_width = min(p10, base) - p90
                ib = ax.barh(i, neg_inner_width, left=p90, height=s["bar_height"],
                             color=ni_color, alpha=ni_alpha, zorder=2)
                inner_neg = ib[0]
            if p10 > base:
                pos_inner_width = p10 - max(p90, base)
                ib = ax.barh(i, pos_inner_width, left=max(p90, base), height=s["bar_height"],
                             color=pi_color, alpha=pi_alpha, zorder=2)
                inner_pos = ib[0]

        # Outline on top — its rectangle also clips the inner-bar shadows so
        # they stay contained within the bar and never spill past the outline.
        outline = ax.barh(i, high - low, left=low, height=s["bar_height"],
                          facecolor="none", edgecolor=s["outline_color"],
                          linewidth=s["bar_linewidth"], zorder=3)
        if s["show_bar_shadows"]:
            if inner_neg is not None:
                inner_neg.set_path_effects(neg_bar_shadow)
                inner_neg.set_clip_path(outline[0])
            if inner_pos is not None:
                inner_pos.set_path_effects(pos_bar_shadow)
                inner_pos.set_clip_path(outline[0])

        # --- Value labels with automatic space checking ---
        pad = s["value_offset"] * xspan
        label_data = []
        if "min" in s["show_values"]:
            label_data.append(("min", low, "right", "outside"))
        if "max" in s["show_values"]:
            label_data.append(("max", high, "left", "outside"))

        if p90p10:
            p90, p10 = p90p10
            if "p90" in s["show_values"] and p90 < base:
                light_width = p90 - low
                display_val = p90 - base if s["show_relative_values"] else p90
                value_str = s["value_format"].format(display_val)
                if s["show_percentage_diff"] and base != 0:
                    pct = (p90 - base) / base * 100
                    value_str += f" ({'+' if pct > 0 else ''}{pct:.0f}%)"
                light_fits = has_space_for_label(value_str, light_width, xspan)
                inner_fits = has_space_for_label(value_str, neg_inner_width, xspan)
                if light_fits and inner_fits:
                    label_data.append(("p90", p90, "right",
                                       "outside" if light_width >= neg_inner_width else "inside"))
                elif light_fits:
                    label_data.append(("p90", p90, "right", "outside"))
                elif inner_fits:
                    label_data.append(("p90", p90, "right", "inside"))

            if "p10" in s["show_values"] and p10 > base:
                light_width = high - p10
                display_val = p10 - base if s["show_relative_values"] else p10
                value_str = s["value_format"].format(display_val)
                if s["show_percentage_diff"] and base != 0:
                    pct = (p10 - base) / base * 100
                    value_str += f" ({'+' if pct > 0 else ''}{pct:.0f}%)"
                light_fits = has_space_for_label(value_str, light_width, xspan)
                inner_fits = has_space_for_label(value_str, pos_inner_width, xspan)
                if light_fits and inner_fits:
                    label_data.append(("p10", p10, "left",
                                       "outside" if light_width >= pos_inner_width else "inside"))
                elif light_fits:
                    label_data.append(("p10", p10, "left", "outside"))
                elif inner_fits:
                    label_data.append(("p10", p10, "left", "inside"))

        for lbl, val, align, inside_or_outside in label_data:
            inside = inside_or_outside == "inside"
            ha = "right" if align == "right" else "left"
            if inside:
                ha = "left" if align == "right" else "right"
                x_pos = val + pad if align == "right" else val - pad
            else:
                x_pos = val - pad if align == "right" else val + pad

            display_val = val - base if s["show_relative_values"] else val
            value_str = s["value_format"].format(display_val)
            if s["show_percentage_diff"] and base != 0:
                pct = (val - base) / base * 100
                value_str += f" ({'+' if pct > 0 else ''}{pct:.0f}%)"

            color = s["label_color"]
            effects = None
            if inside and lbl in ("p10", "p90"):
                color = "white"
                effects = white_text_outline
            # Single-colour mode tints the positive bars light, so force all
            # positive-side text to black for legibility.
            if positive_text_black and val >= base:
                color = "black"
                effects = None

            if s["show_value_headers"]:
                ax.text(x_pos, i - header_value_offset, lbl, ha=ha, va='center',
                        fontsize=s["header_fontsize_value"], fontweight='bold',
                        color=color, zorder=4, path_effects=effects)
                ax.text(x_pos, i + header_value_offset, value_str, ha=ha, va='center',
                        fontsize=s["value_fontsize"], color=color, zorder=4,
                        path_effects=effects)
            else:
                ax.text(x_pos, i, value_str, ha=ha, va='center',
                        fontsize=s["value_fontsize"], color=color, zorder=4,
                        path_effects=effects)

        # --- Parameter labels (left of bars; left column only in a grid) ---
        if show_param_labels:
            label_x = xmin - s["label_gap"] * xspan
            ax.text(label_x, i, d["parameter"], ha='right', va='center',
                    fontsize=s["label_fontsize"], color=s["label_color"],
                    weight='bold', zorder=4)

    # --- Axis styling ---
    ax.axvline(base, color=s["baseline_color"], lw=s["baseline_width"], zorder=3)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # inverted: top to bottom
    ax.set_yticks([])

    if show_xlabel:
        if property_name and detected_unit:
            prop_display = (property_name.upper()
                            if property_name.lower() in ['npv', 'stoiip', 'giip', 'hcpv']
                            else property_name.title())
            xlabel = f"{prop_display} ({detected_unit})"
        elif property_name:
            xlabel = (property_name.upper()
                      if property_name.lower() in ['npv', 'stoiip', 'giip', 'hcpv']
                      else property_name.title())
        elif detected_unit:
            xlabel = detected_unit
        else:
            xlabel = "Effect"
        ax.set_xlabel(xlabel, fontsize=10, color=s["label_color"])

    for spine in ax.spines.values():
        spine.set_color(s["outline_color"])
        spine.set_linewidth(1.1)
    ax.tick_params(colors=s["outline_color"], which='both')

    return {"property_name": property_name, "unit": detected_unit,
            "filter_name": auto_filter_name}


def tornado_plot(
    ds: Union[Dataset, FilteredDataset, List[Union[Dataset, FilteredDataset]]],
    *,
    property: Union[str, List[str]],
    filters: Union[Dict[str, Any], str, List[Union[Dict[str, Any], str]], None] = None,
    multiplier: Optional[float] = None,
    color: Union[str, List[Any], None] = None,
    skip: Union[str, List[str], None] = None,
    case_selection: bool = False,
    selection_criteria: Optional[Dict[str, Any]] = None,
    title: str = "Tornado Chart",
    subtitle: Optional[str] = None,
    outfile: Optional[Union[str, Path]] = None,
    base: Optional[float] = None,
    reference_case: Optional[float] = None,
    unit: Optional[str] = None,
    filter_name: Optional[str] = None,
    preferred_order: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    settings: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional["Figure"], Union[Optional["Axes"], Any], Optional[str]]:
    """
    Render a tornado chart from a Dataset.

    Grid mode: pass a list for ``property`` and/or ``filters`` (or a list of
    datasets/views as ``ds``) to render a grid of tornado charts — properties
    stack across columns, rows stack down. A scalar ``property`` with a single
    dataset/filter produces a single chart.

    Args:
        ds: A Dataset, a FilteredDataset, or a **list** of either — one grid
            row per list entry. Each FilteredDataset in the list contributes
            its own filter as that row's selection.
        property: Property to plot (e.g. 'stoiip'), or a list of properties —
                  one column per property.
        filters: Spatial filter dict ({field: value(s)}) or stored-filter name.
                 Must not contain a 'property' key. A list of filters renders
                 one row per filter. Must be None when ``ds`` is a list (each
                 entry carries its own filter).
        multiplier: Optional display multiplier override.
        color: Optional single colour per chart. ``None`` keeps the default
               red/blue scheme. A palette family name (or any literal colour)
               tints both signs from one hue — negative bars at shade
               ``neg_shade`` (700), positive at ``pos_shade`` (400); inner
               p90-p10 bars at ``inner_opacity`` (100%), outer min-max bars at
               ``outer_opacity`` (40%). All four are tunable via ``settings``.
               In single-colour mode the positive bars are a light tint, so
               all positive-side value text is rendered black for legibility.
               In a grid, pass a flat list (per row) or nested list
               (``color[row][col]``, per cell).
        case_selection / selection_criteria: Forwarded to data extraction.
        title, subtitle, outfile, base, reference_case, unit, filter_name,
        preferred_order, figsize, settings: Plot styling. ``subtitle`` and
        ``filter_name`` apply to single-chart mode only.

    Returns:
        ``(fig, ax, saved)`` for a single chart, or ``(fig, axes_2d, saved)`` in
        grid mode where ``axes_2d`` is a 2-D numpy array of Axes indexed
        ``[row][column]``.

    Notes:
    - Each parameter (sheet) becomes one bar; tornado is intrinsically multi-parameter.
    """
    # --- Resolve ds + filters into per-row (Dataset, filter) pairs ---
    datasets, row_filters = _resolve_rows(ds, filters)

    # --- Normalise property to a list; detect grid mode ---
    properties = list(property) if isinstance(property, list) else [property]
    if not properties:
        raise ValueError("tornado_plot: 'property' list must be non-empty.")

    s = _build_settings(settings)
    nrows, ncols = len(datasets), len(properties)
    is_grid = nrows > 1 or ncols > 1

    # --- Figure sizing ---
    if figsize is not None:
        fig_w, fig_h = figsize
    elif is_grid:
        # Tornado cells need height proportional to the number of bars
        # (one bar per sheet) so labels stay legible.
        n_bars_est = max(1, len(datasets[0].parameters()) - 1)
        cell_h = max(s["grid_cell_height"], s["grid_bar_inches"] * n_bars_est + 2.0)
        fig_w = s["grid_cell_width"] * ncols
        fig_h = cell_h * nrows
    else:
        fig_w, fig_h = s["figsize"]

    plt.close("all")
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=s["dpi"],
                             squeeze=False)
    fig.patch.set_facecolor(s["figure_bg_color"])

    has_col_headers = is_grid and ncols > 1
    has_row_headers = is_grid and nrows > 1

    # --- Margins ---
    # Computed in inches then converted to fractions so padding stays
    # constant regardless of grid size (a fixed fraction margin balloons
    # into a wide white gap on large figures).
    if is_grid:
        m_left = 1.25 + (0.34 if has_row_headers else 0.0)   # param labels(+row hdr)
        m_right = 0.35
        m_bottom = 0.62
        m_top = 0.97 + (0.30 if has_col_headers else 0.0)    # fig title + subtitle(+col hdr)

        left = m_left / fig_w
        right = 1.0 - m_right / fig_w
        bottom = m_bottom / fig_h
        top = 1.0 - m_top / fig_h
        fig.subplots_adjust(
            left=left, right=right, bottom=bottom, top=top,
            wspace=_gap_fraction(s["grid_col_gap"], ncols, (right - left) * fig_w),
            hspace=_gap_fraction(s["grid_row_gap"], nrows, (top - bottom) * fig_h),
        )
    else:
        top = 0.88
        fig.subplots_adjust(left=s["left_margin"], right=0.95, bottom=0.12, top=top)

    # --- Draw each cell ---
    col_labels: List[str] = []
    row_labels: List[str] = []
    for c, prop in enumerate(properties):
        for r in range(nrows):
            ax = axes[r][c]
            sections = datasets[r]._tornado_data(
                property=prop, filters=row_filters[r], multiplier=multiplier, skip=skip,
                case_selection=case_selection, selection_criteria=selection_criteria,
            )
            if isinstance(sections, tuple):
                sections, _ = sections
            if isinstance(sections, dict):
                sections = [sections]

            cell_color = _cell_color(color, r, c)
            detected = _draw_tornado(
                ax, sections, s,
                base=base, reference_case=reference_case, unit_override=unit,
                subtitle_prefix=(None if is_grid else filter_name),
                subtitle_override=(subtitle if not is_grid else None),
                preferred_order=preferred_order,
                show_xlabel=(r == nrows - 1),
                show_param_labels=True,
                show_ref_label=not is_grid,
                bar_styles=_tornado_colors(cell_color, s),
                positive_text_black=(cell_color is not None),
            )
            if r == 0:
                pn = detected.get("property_name") or prop
                col_labels.append(
                    pn.upper() if str(pn).lower() in ['npv', 'stoiip', 'giip', 'hcpv']
                    else str(pn).title()
                )
            if c == 0:
                row_labels.append(detected.get("filter_name")
                                   or _filter_label(row_filters[r], r))

    # --- Figure title ---
    fig.text(0.5, 1.0 - 0.30 / fig_h, title, ha="center", va="top",
             fontsize=s["title_fontsize"], fontweight="bold", color=s["label_color"])

    # --- Grid row / column headers ---
    if has_col_headers:
        col_y = 1.0 - 0.64 / fig_h
        for c in range(ncols):
            box = axes[0][c].get_position()
            fig.text((box.x0 + box.x1) / 2, col_y, col_labels[c],
                     ha="center", va="center", fontsize=s["header_fontsize"],
                     fontweight="bold", color=s["label_color"])
    if has_row_headers:
        row_x = 0.30 / fig_w
        for r in range(nrows):
            box = axes[r][0].get_position()
            fig.text(row_x, (box.y0 + box.y1) / 2, row_labels[r],
                     ha="center", va="center", rotation=90,
                     fontsize=s["header_fontsize"], fontweight="bold",
                     color=s["label_color"])

    # --- Save ---
    saved = None
    if outfile:
        outfile = Path(outfile)
        fig.savefig(outfile, bbox_inches="tight", facecolor=s["figure_bg_color"])
        saved = str(outfile)

    return fig, (axes if is_grid else axes[0][0]), saved
