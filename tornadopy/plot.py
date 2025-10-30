from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter


def tornado_plot(
    sections,
    title="Tornado Chart",
    subtitle=None,
    outfile=None,
    base=None,
    reference_case=None,
    unit=None,
    preferred_order=None,
    settings=None
):
    """
    Tornado chart with:
    - inside/outside label control
    - automatic white p10/p90 text (inside only)
    - soft outline for white text
    - shadows only for outer bars (not p90/p10 overlays)
    - relative and percentage label modes
    - uses 'parameter' instead of 'title' for bar labels
    - expects 'p90p10' as [p90, p10] (low, high)
    - automatic space checking for p10/p90 labels (prefers largest space)
    - optional reference_case vertical line
    - preferred_order: list of parameter names to show first (in that order)
    - auto-detects base and reference_case from sections if not provided
    """
    # Auto-detect base and reference_case from sections
    auto_base = None
    auto_reference = None

    for sec in sections:
        if "base_case" in sec:
            auto_base = sec.get("base_case")
        if "reference_case" in sec:
            auto_reference = sec.get("reference_case")
        # Stop after finding the first entry with these values
        if auto_base is not None or auto_reference is not None:
            break

    # Use auto-detected values if not explicitly provided
    # Allow False to force defaults (0 for base, None for reference)
    if base is None and auto_base is not None:
        base = auto_base
    elif base is None:
        base = 0.0

    if reference_case is None and auto_reference is not None:
        reference_case = auto_reference
    # --- Default settings ---
    s = {
        "figsize": (10, 7),
        "dpi": 160,
        "plot_bg_color": "#FAF0E6",
        "figure_bg_color": "white",
        # Colors
        "pos_light": "#A9CFF7",
        "neg_light": "#F5B7B1",
        "pos_dark": "#2E5BFF",
        "neg_dark": "#E74C3C",
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
        "label_fontsize": 9,
        "value_fontsize": 6,
        "header_fontsize": 7.5,
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
        "header_value_spacing": 0.09,  # Spacing between header and value text
        # Feature toggles
        "show_relative_values": False,
        "show_percentage_diff": True,
        "show_bar_shadows": True,
    }
    if settings:
        s.update(settings)

    # --- Subtitle ---
    if subtitle is None:
        unit_str = f" {unit}" if unit else ""
        subtitle = f"Base case = {base:.2f}{unit_str}"

    # --- Prepare data ---
    data = []
    for sec in sections:
        # Skip entries that only contain base_case/reference_case (no actual data)
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
            "low": low,
            "high": high,
            "p90p10": sec.get("p90p10"),  # expects [p90, p10]
            "label_pos": sec.get("label_pos", {})
        })

    if not data:
        return None, None, None

    # --- Sort data by preferred order ---
    if preferred_order:
        # Create a mapping of parameter names to their preferred position
        order_map = {param: idx for idx, param in enumerate(preferred_order)}
        
        # Separate data into preferred and remaining
        preferred_data = []
        remaining_data = []
        
        for d in data:
            param = d["parameter"]
            if param in order_map:
                preferred_data.append((order_map[param], d))
            else:
                remaining_data.append(d)
        
        # Sort preferred data by their order
        preferred_data.sort(key=lambda x: x[0])
        
        # Combine: preferred first (in order), then remaining (original order)
        data = [d for _, d in preferred_data] + remaining_data

    # --- Axis limits ---
    lows, highs = [d["low"] for d in data], [d["high"] for d in data]
    max_extent = max(abs(min(lows) - base), abs(max(highs) - base))
    buffer = max_extent * s["axis_buffer"]
    xmin, xmax = base - max_extent - buffer, base + max_extent + buffer
    xspan = xmax - xmin

    # --- Figure setup ---
    plt.close("all")
    fig, ax = plt.subplots(figsize=s["figsize"], dpi=s["dpi"])
    fig.patch.set_facecolor(s["figure_bg_color"])
    ax.set_facecolor(s["plot_bg_color"])
    fig.subplots_adjust(left=s["left_margin"], right=0.95, bottom=0.12, top=0.88)

    # --- Calculate dynamic offsets based on plot dimensions ---
    # Calculate bars per inch to determine spacing
    n_bars = len(data)
    fig_height_inches = s["figsize"][1]
    plot_area_fraction = 0.88 - 0.12  # top - bottom margin
    effective_plot_height = fig_height_inches * plot_area_fraction

    # Space per bar in y-axis units (each bar is centered at integer y position)
    bar_spacing = 1.0  # bars are at y=0, 1, 2, ... with spacing of 1

    # For header/value separation within each bar (use customizable setting)
    header_value_offset = s["header_value_spacing"]

    # For reference case label (should be just outside plot area)
    # Calculate based on bar spacing to place it consistently above first bar
    ref_case_offset = bar_spacing * 0.7  # 70% of bar spacing above plot

    # --- Titles ---
    plot_box = ax.get_position()
    plot_center = (plot_box.x0 + plot_box.x1) / 2
    fig.text(plot_center, 0.97, title, ha="center", fontsize=s["title_fontsize"],
             fontweight="bold", color=s["label_color"])
    fig.text(plot_center, 0.93, subtitle, ha="center", fontsize=s["subtitle_fontsize"],
             color=s["label_color"], alpha=0.85)

    # --- Gridlines ---
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='x', which='major', alpha=s["vgrid_alpha"], color=s["grid_color"], zorder=0)
    ax.grid(axis='x', which='minor', alpha=s["vgrid_alpha"] * 0.7, color=s["grid_color"], zorder=0)
    for i in range(len(data)):
        ax.axhline(i, color=s["grid_color"], alpha=s["hgrid_alpha"], lw=0.6, zorder=0)

    # --- Reference case line (plotted below bars) ---
    if reference_case is not None:
        ax.axvline(reference_case, color=s["reference_color"], lw=s["reference_width"],
                   linestyle='--', zorder=0.5, alpha=0.3)
        # Add label above the plot area (negative y since axis is inverted)
        # Use dynamic offset based on bar spacing
        ax.text(reference_case, -ref_case_offset, 'Ref case',
               ha='center', va='bottom', fontsize=s["reference_fontsize"],
               color=s["reference_color"], zorder=4)

    # --- Shadows and effects ---
    outer_bar_shadow = [patheffects.withSimplePatchShadow(offset=(1.5, -1.5), alpha=0.25)]
    white_text_outline = [patheffects.withStroke(linewidth=1.2, foreground='black', alpha=0.4)]

    # --- Helper function to check if text fits ---
    def has_space_for_label(text, width_in_data_units, xspan):
        """Estimate if text fits in given width (in data coordinates)"""
        # Rough estimate: ~0.0065 * xspan per character at fontsize 8
        char_width = 0.0065 * xspan * (s["value_fontsize"] / 8.0)
        text_width = len(text) * char_width
        return width_in_data_units > text_width * 1.3  # 1.3x for padding

    # --- Draw bars ---
    for i, d in enumerate(data):
        low, high, p90p10 = d["low"], d["high"], d.get("p90p10")

        # --- Outer light bars (with shadow) ---
        # Negative side outer bar: from low to min(high, base)
        neg_outer_width = 0
        if low < base:
            neg_outer_width = min(high, base) - low
            bar = ax.barh(i, neg_outer_width, left=low, height=s["bar_height"],
                         color=s["neg_light"], zorder=1)
            if s["show_bar_shadows"]:
                bar[0].set_path_effects(outer_bar_shadow)
        
        # Positive side outer bar: from max(low, base) to high
        pos_outer_width = 0
        if high > base:
            pos_outer_width = high - max(low, base)
            bar = ax.barh(i, pos_outer_width, left=max(low, base), height=s["bar_height"],
                         color=s["pos_light"], zorder=1)
            if s["show_bar_shadows"]:
                bar[0].set_path_effects(outer_bar_shadow)

        # --- Dark p90–p10 overlays (no shadow) ---
        neg_inner_width = 0
        pos_inner_width = 0
        if p90p10:  # ✅ interpret correctly: [p90, p10]
            p90, p10 = p90p10
            # Negative side inner bar
            if p90 < base:
                neg_inner_width = min(p10, base) - p90
                ax.barh(i, neg_inner_width, left=p90, height=s["bar_height"],
                       color=s["neg_dark"], alpha=0.9, zorder=2)
            # Positive side inner bar
            if p10 > base:
                pos_inner_width = p10 - max(p90, base)
                ax.barh(i, pos_inner_width, left=max(p90, base), height=s["bar_height"],
                       color=s["pos_dark"], alpha=0.9, zorder=2)

        # Outline on top
        ax.barh(i, high - low, left=low, height=s["bar_height"], facecolor="none",
               edgecolor=s["outline_color"], linewidth=s["bar_linewidth"], zorder=3)

        # --- Value labels with automatic space checking ---
        pad = s["value_offset"] * xspan
        label_data = []
        
        # Min/max always added (they have space by design)
        if "min" in s["show_values"]:
            label_data.append(("min", low, "right", "outside"))
        if "max" in s["show_values"]:
            label_data.append(("max", high, "left", "outside"))

        # p90/p10 with space checking (prefer largest space)
        if p90p10:
            p90, p10 = p90p10
            
            # Check p90 (left side, negative)
            if "p90" in s["show_values"] and p90 < base:
                # Light area: space between left edge of outer bar and left edge of inner bar
                light_width = p90 - low
                inner_width = neg_inner_width
                
                # Format value string to check its length
                display_val = p90 - base if s["show_relative_values"] else p90
                value_str = s["value_format"].format(display_val)
                if s["show_percentage_diff"] and base != 0:
                    pct = (p90 - base) / base * 100
                    sign = "+" if pct > 0 else ""
                    value_str += f" ({sign}{pct:.0f}%)"
                
                # Check which spaces work
                light_fits = has_space_for_label(value_str, light_width, xspan)
                inner_fits = has_space_for_label(value_str, inner_width, xspan)
                
                # Prefer the largest space
                if light_fits and inner_fits:
                    if light_width >= inner_width:
                        label_data.append(("p90", p90, "right", "outside"))
                    else:
                        label_data.append(("p90", p90, "right", "inside"))
                elif light_fits:
                    label_data.append(("p90", p90, "right", "outside"))
                elif inner_fits:
                    label_data.append(("p90", p90, "right", "inside"))
                # else: skip (not enough space)
            
            # Check p10 (right side, positive)
            if "p10" in s["show_values"] and p10 > base:
                # Light area: space between right edge of inner bar and right edge of outer bar
                light_width = high - p10
                inner_width = pos_inner_width
                
                # Format value string
                display_val = p10 - base if s["show_relative_values"] else p10
                value_str = s["value_format"].format(display_val)
                if s["show_percentage_diff"] and base != 0:
                    pct = (p10 - base) / base * 100
                    sign = "+" if pct > 0 else ""
                    value_str += f" ({sign}{pct:.0f}%)"
                
                # Check which spaces work
                light_fits = has_space_for_label(value_str, light_width, xspan)
                inner_fits = has_space_for_label(value_str, inner_width, xspan)
                
                # Prefer the largest space
                if light_fits and inner_fits:
                    if light_width >= inner_width:
                        label_data.append(("p10", p10, "left", "outside"))
                    else:
                        label_data.append(("p10", p10, "left", "inside"))
                elif light_fits:
                    label_data.append(("p10", p10, "left", "outside"))
                elif inner_fits:
                    label_data.append(("p10", p10, "left", "inside"))
                # else: skip (not enough space)

        # Render all labels
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
                sign = "+" if pct > 0 else ""
                value_str += f" ({sign}{pct:.0f}%)"

            # Color & optional white outline
            color = s["label_color"]
            effects = None
            if inside and lbl in ("p10", "p90"):
                color = "white"
                effects = white_text_outline

            # Render with dynamic offsets
            if s["show_value_headers"]:
                # Use dynamic offset calculated based on plot dimensions
                ax.text(x_pos, i - header_value_offset, lbl, ha=ha, va='center',
                       fontsize=s["header_fontsize"], fontweight='bold',
                       color=color, zorder=4, path_effects=effects)
                ax.text(x_pos, i + header_value_offset, value_str, ha=ha, va='center',
                       fontsize=s["value_fontsize"], color=color, zorder=4,
                       path_effects=effects)
            else:
                ax.text(x_pos, i, value_str, ha=ha, va='center',
                       fontsize=s["value_fontsize"], color=color, zorder=4,
                       path_effects=effects)

        # --- Parameter labels ---
        label_x = xmin - s["label_gap"] * xspan
        ax.text(label_x, i, d["parameter"], ha='right', va='center',
               fontsize=s["label_fontsize"], color=s["label_color"],
               weight='bold', zorder=4)

    # --- Axis styling ---
    # Base case line
    ax.axvline(base, color=s["baseline_color"], lw=s["baseline_width"], zorder=3)
    
    ax.set_xlim(xmin, xmax)
    ax.set_yticks([])
    ax.invert_yaxis()
    ax.set_xlabel(unit or "Effect", fontsize=10, color=s["label_color"])
    for spine in ax.spines.values():
        spine.set_color(s["outline_color"])
        spine.set_linewidth(1.1)
    ax.tick_params(colors=s["outline_color"], which='both')

    # --- Save ---
    saved = None
    if outfile:
        outfile = Path(outfile)
        fig.savefig(outfile, bbox_inches="tight", facecolor=s["figure_bg_color"])
        saved = str(outfile)

    return fig, ax, saved