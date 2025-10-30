from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def distribution_plot(
    data,
    title="Distribution",
    unit=None,
    outfile=None,
    target_bins=20,
    color="blue",
    reference_case=None,
    settings=None
):
    """
    Generate a distribution histogram with cumulative curve.

    Features:
    - P90/P50/P10 subtitle with percentile values
    - Beautiful bin sizing with round numbers
    - No gaps between bars
    - Dark outline matching color scheme
    - Cumulative curve (dark red) showing % of cases with higher value
    - Secondary axis (right) for reading percentile values
    - Optional percentile markers on cumulative line
    - Optional reference case line
    - Minor gridlines with transparency
    - Manual gridline interval control

    Args:
        data: Array-like data (numpy array, list, or pandas Series)
        title: Chart title
        unit: Unit label for x-axis and subtitle
        outfile: Output file path (if specified, saves the figure)
        target_bins: Target number of bins for histogram (default 20)
        color: Color scheme - "red", "blue", "green", "orange", "purple", "fuchsia", "yellow"
        reference_case: Optional reference case value to plot as vertical line
        settings: Dictionary of visual settings to override defaults

    Returns:
        Tuple of (fig, ax, saved):
        - fig: Matplotlib figure object
        - ax: Matplotlib axes object (primary)
        - saved: Path to saved file (if outfile specified, otherwise None)

    Example:
        >>> from tornadopy import TornadoProcessor, distribution_plot
        >>> processor = TornadoProcessor("data.xlsb")
        >>> dist = processor.get_distribution(
        ...     parameter="Parameter1",
        ...     filters={"property": "npv"},
        ...     multiplier=1e-6
        ... )
        >>> fig, ax, saved = distribution_plot(
        ...     dist,
        ...     title="NPV Distribution",
        ...     unit="MM USD",
        ...     outfile="npv_distribution.png"
        ... )
    """
    # --- Color schemes ---
    color_map = {
        "red": {"light": "#FB877A", "dark": "#BA2A19"},
        "blue": {"light": "#66C3EB", "dark": "#0075A6"},
        "green": {"light": "#AED879", "dark": "#5A8E18"},
        "orange": {"light": "#F7CB66", "dark": "#B57E00"},
        "purple": {"light": "#A49CDD", "dark": "#4E4495"},
        "fuchsia": {"light": "#DD7BD0", "dark": "#951B84"},
        "yellow": {"light": "#FEEA66", "dark": "#BEA500"},
    }

    if color not in color_map:
        color = "blue"

    colors = color_map[color]

    # --- Default settings ---
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
        # Percentile markers
        "show_percentile_markers": False,
        "marker_size": 8,
        "marker_color": "#FB877A",  # light red
        "marker_edge_color": "#BA2A19",  # dark red
        "marker_edge_width": 1.5,
        "marker_label_fontsize": 8,
        "marker_label_offset": 5,  # offset in percentage points
    }
    if settings:
        s.update(settings)

    # --- Convert data to numpy array ---
    data = np.array(data)
    data = data[np.isfinite(data)]  # Remove NaN/Inf

    if len(data) == 0:
        raise ValueError("No valid data points")

    # --- Calculate percentiles ---
    p90 = np.percentile(data, 10)  # P90 = 10th percentile (low value)
    p50 = np.percentile(data, 50)  # P50 = 50th percentile (median)
    p10 = np.percentile(data, 90)  # P10 = 90th percentile (high value)

    # --- Subtitle ---
    unit_str = f" {unit}" if unit else ""
    subtitle = f"P90 = {p90:.2f}{unit_str}  |  P50 = {p50:.2f}{unit_str}  |  P10 = {p10:.2f}{unit_str}"

    # --- Calculate beautiful bins ---
    def get_beautiful_bins(data, target_bins):
        """Create bins with nice round numbers"""
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min

        # Calculate initial bin width
        raw_width = data_range / target_bins

        # Round to beautiful numbers (1, 2, 5, 10, 20, 50, 100, etc.)
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

        # Create bin edges
        bin_start = np.floor(data_min / beautiful_width) * beautiful_width
        bin_end = np.ceil(data_max / beautiful_width) * beautiful_width
        bins = np.arange(bin_start, bin_end + beautiful_width, beautiful_width)

        return bins

    bins = get_beautiful_bins(data, target_bins)

    # --- Figure setup ---
    plt.close("all")
    fig, ax = plt.subplots(figsize=s["figsize"], dpi=s["dpi"])
    fig.patch.set_facecolor(s["figure_bg_color"])
    ax.set_facecolor(s["plot_bg_color"])
    fig.subplots_adjust(
        left=s["left_margin"],
        right=s["right_margin"],
        bottom=s["bottom_margin"],
        top=s["top_margin"]
    )

    # --- Titles ---
    plot_box = ax.get_position()
    plot_center = (plot_box.x0 + plot_box.x1) / 2
    fig.text(plot_center, 0.97, title, ha="center", fontsize=s["title_fontsize"],
             fontweight="bold", color=s["text_color"])
    fig.text(plot_center, 0.93, subtitle, ha="center", fontsize=s["subtitle_fontsize"],
             color=s["text_color"], alpha=0.85)

    # --- Histogram (no gaps) ---
    counts, bin_edges, patches = ax.hist(
        data,
        bins=bins,
        color=s["bar_color"],
        edgecolor=s["bar_outline_color"],
        linewidth=s["bar_linewidth"],
        alpha=0.9,
        zorder=2
    )

    # --- Set gridline intervals (must be done after histogram to get proper limits) ---
    # X-axis intervals
    if s["x_major_interval"] is not None:
        ax.xaxis.set_major_locator(MultipleLocator(s["x_major_interval"]))

    if s["x_minor_interval"] is not None:
        ax.xaxis.set_minor_locator(MultipleLocator(s["x_minor_interval"]))
    elif s["show_minor_grid"]:
        if s["x_major_interval"] is not None:
            # Auto-calculate minor interval as 1/10th of major
            ax.xaxis.set_minor_locator(MultipleLocator(s["x_major_interval"] / 10))
        else:
            ax.xaxis.set_minor_locator(AutoMinorLocator(10))

    # Y-axis intervals
    if s["y_major_interval"] is not None:
        ax.yaxis.set_major_locator(MultipleLocator(s["y_major_interval"]))

    if s["y_minor_interval"] is not None:
        ax.yaxis.set_minor_locator(MultipleLocator(s["y_minor_interval"]))
    elif s["show_minor_grid"]:
        if s["y_major_interval"] is not None:
            # Auto-calculate minor interval as 1/10th of major
            ax.yaxis.set_minor_locator(MultipleLocator(s["y_major_interval"] / 10))
        else:
            ax.yaxis.set_minor_locator(AutoMinorLocator(10))

    # --- Grid (major and minor) - must be called after setting locators ---
    ax.grid(axis='y', which='major', alpha=s["grid_alpha"], color=s["grid_color"],
            linewidth=s["grid_linewidth"], zorder=1)
    ax.grid(axis='x', which='major', alpha=s["grid_alpha"] * 0.7, color=s["grid_color"],
            linewidth=s["grid_linewidth"], zorder=1)

    if s["show_minor_grid"]:
        ax.grid(axis='y', which='minor', alpha=s["minor_grid_alpha"],
                color=s["grid_color"], linewidth=s["minor_grid_linewidth"], zorder=1)
        ax.grid(axis='x', which='minor', alpha=s["minor_grid_alpha"],
                color=s["grid_color"], linewidth=s["minor_grid_linewidth"], zorder=1)

    # --- Reference case line (plotted over bars) ---
    if reference_case is not None:
        ax.axvline(reference_case, color=s["reference_color"], lw=s["reference_width"],
                   linestyle=s["reference_linestyle"], zorder=3.5, alpha=0.7)
        # Add label at top of plot area with reduced spacing
        ymax = ax.get_ylim()[1]
        ax.text(reference_case, ymax*1.03, 'Ref case',
               ha='center', va='top', fontsize=s["reference_fontsize"],
               color=s["reference_color"], zorder=4)

    # --- Cumulative curve (% with higher value) ---
    # Sort data and calculate cumulative percentages
    sorted_data = np.sort(data)
    n = len(sorted_data)

    # For each value, calculate what % of cases have HIGHER value
    # This is the reverse cumulative (100% - CDF)
    percentile_higher = 100 * (1 - np.arange(1, n + 1) / n)

    # Create secondary y-axis for cumulative curve
    ax2 = ax.twinx()
    ax2.plot(
        sorted_data,
        percentile_higher,
        color=s["cumulative_color"],
        linewidth=s["cumulative_linewidth"],
        alpha=0.9,
        zorder=3,
        label="Cumulative Distribution"
    )

    # --- Percentile markers on cumulative line ---
    if s["show_percentile_markers"]:
        # Calculate cumulative percentages for p90, p50, p10
        percentiles = [
            (p90, 90.0, "P90"),  # 90% of cases have higher value than p90
            (p50, 50.0, "P50"),  # 50% of cases have higher value than p50
            (p10, 10.0, "P10"),  # 10% of cases have higher value than p10
        ]

        for value, cumulative_pct, label in percentiles:
            # Plot marker
            ax2.plot(
                value,
                cumulative_pct,
                marker='o',
                markersize=s["marker_size"],
                markerfacecolor=s["marker_color"],
                markeredgecolor=s["marker_edge_color"],
                markeredgewidth=s["marker_edge_width"],
                zorder=4
            )

            # Add label above marker
            ax2.text(
                value,
                cumulative_pct + s["marker_label_offset"],
                label,
                ha='center',
                va='bottom',
                fontsize=s["marker_label_fontsize"],
                color=s["marker_edge_color"],
                fontweight='bold',
                zorder=5
            )

    # Configure secondary axis (100% at top, 0% at bottom)
    ax2.set_ylabel("Cumulative Distribution",
                   fontsize=s["label_fontsize"],
                   color=s["text_color"])
    ax2.set_ylim(0, 100)  # 0% at bottom, 100% at top
    ax2.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax2.tick_params(axis='y', labelcolor=s["text_color"], colors=s["text_color"])
    ax2.spines['right'].set_color(s["outline_color"])
    ax2.spines['right'].set_linewidth(1.1)

    # Add percentage ticks
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))

    # --- Axis labels ---
    ax.set_xlabel(unit or "Value", fontsize=s["label_fontsize"], color=s["text_color"])
    ax.set_ylabel("Frequency", fontsize=s["label_fontsize"], color=s["text_color"])

    # --- Axis styling ---
    for spine in ['left', 'bottom', 'top']:
        ax.spines[spine].set_color(s["outline_color"])
        ax.spines[spine].set_linewidth(1.1)
    ax.spines['right'].set_visible(False)  # Hide right spine on primary axis

    ax.tick_params(axis='both', colors=s["text_color"], which='both', labelsize=s["tick_fontsize"])
    ax2.tick_params(axis='y', colors=s["text_color"], which='both', labelsize=s["tick_fontsize"])

    # --- Save ---
    saved = None
    if outfile:
        outfile = Path(outfile)
        fig.savefig(outfile, bbox_inches="tight", facecolor=s["figure_bg_color"])
        saved = str(outfile)

    return fig, ax, saved
