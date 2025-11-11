"""
Correlation Matrix Visualization Module

This module provides functionality to visualize correlation matrices from
TornadoPy's correlation_grid() function results.
"""

import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from pathlib import Path


def correlation_plot(
    correlation_data,
    outfile=None,
    figsize=None,
    settings=None
):
    """
    Create a beautiful correlation matrix heatmap.

    Parameters
    ----------
    correlation_data : dict
        Dictionary returned by TornadoProcessor.correlation_grid() containing:
        - 'parameter': str, title for the plot
        - 'matrix': np.ndarray, correlation matrix values
        - 'variables': list, y-axis labels (input variables)
        - 'properties': list, x-axis labels (output properties with units)
        - 'n_cases': int, number of cases analyzed
        - 'filter_name': str, filter name (optional)
        - 'constant_variables': list of tuples (optional)
        - 'skipped_variables': list (optional)

    outfile : str or Path, optional
        Path to save the figure. Supports formats: .png, .svg, .pdf, .jpg

    figsize : tuple, optional
        Figure size as (width, height) tuple (default: (12, 8))

    settings : dict, optional
        Dictionary to override default visual settings. Available keys:
        - figsize: tuple, figure dimensions (default: (12, 8))
        - dpi: int, resolution (default: 160)
        - title_fontsize: int (default: 15)
        - subtitle_fontsize: int (default: 10)
        - xlabel_fontsize: int (default: 9)
        - ylabel_fontsize: int (default: 9)
        - tick_fontsize: int (default: 8)
        - colorbar_fontsize: int (default: 9)
        - value_fontsize: int (default: 7)
        - figure_bg_color: str (default: "white")
        - plot_bg_color: str (default: "white")
        - text_color: str (default: "#1C2833")
        - grid_color: str (default: "#D5D8DC")
        - grid_linewidth: float (default: 0.5)
        - cmap_colors: list (default: ["#2E5BFF", "white", "#E74C3C"])
        - show_values: bool, display correlation values on cells (default: True)
        - value_threshold: float, min abs value to show (default: 0.0)
        - cell_aspect: str, 'auto' or 'equal' (default: 'auto')
        - colorbar_label: str (default: "Correlation Coefficient")
        - vmin: float, minimum correlation value (default: -1)
        - vmax: float, maximum correlation value (default: 1)

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object containing the heatmap
    saved : bool
        True if figure was saved to file, False otherwise

    Examples
    --------
    >>> from tornadopy import TornadoProcessor, correlation_plot
    >>> processor = TornadoProcessor("data.xlsx")
    >>> corr_data = processor.correlation_grid(parameter="Full_Uncertainty")
    >>> fig, ax, saved = correlation_plot(corr_data, outfile="correlation.png")

    >>> # Customize appearance
    >>> custom_settings = {
    ...     "figsize": (14, 10),
    ...     "show_values": False,
    ...     "cmap_colors": ["blue", "white", "red"]
    ... }
    >>> fig, ax, saved = correlation_plot(corr_data, settings=custom_settings)
    """

    # Default settings dictionary
    s = {
        # Figure dimensions and resolution
        "figsize": (12, 8),
        "dpi": 160,

        # Font sizes
        "title_fontsize": 15,
        "subtitle_fontsize": 10,
        "xlabel_fontsize": 8,
        "ylabel_fontsize": 8,
        "tick_fontsize": 8,
        "colorbar_fontsize": 9,
        "value_fontsize": 7,

        # Colors
        "figure_bg_color": "white",
        "plot_bg_color": "white",
        "text_color": "#1C2833",
        "grid_color": "#D5D8DC",
        "outline_color": "#2C3E50",

        # Color map (Blue -> White -> Red)
        "cmap_colors": ["#2E5BFF", "white", "#E74C3C"],  # Blue, White, Red

        # Grid and lines
        "grid_linewidth": 0.5,

        # Value display
        "show_values": True,
        "value_threshold": 0.0,  # Minimum absolute value to display

        # Layout
        "cell_aspect": "auto",
        "colorbar_label": "Correlation Coefficient",

        # Value range
        "vmin": -1,
        "vmax": 1,
    }

    # Merge with user-provided settings
    if settings:
        s.update(settings)

    # Apply direct figsize parameter (takes precedence over settings)
    if figsize is not None:
        s["figsize"] = figsize

    # Extract data from correlation_data dictionary
    parameter = correlation_data.get('parameter', 'Correlation Matrix')
    matrix = correlation_data['matrix']
    variables = correlation_data['variables']
    properties = correlation_data['properties']
    n_cases = correlation_data.get('n_cases')
    variable_ranges = correlation_data.get('variable_ranges', [])
    constant_variables = correlation_data.get('constant_variables', [])
    skipped_variables = correlation_data.get('skipped_variables')

    # Setup figure and axes
    plt.close("all")
    fig, ax = plt.subplots(figsize=s["figsize"], dpi=s["dpi"])

    # Configure background colors
    fig.patch.set_facecolor(s["figure_bg_color"])
    ax.set_facecolor(s["plot_bg_color"])

    # Create custom colormap (Blue -> White -> Red)
    cmap = LinearSegmentedColormap.from_list(
        "correlation", s["cmap_colors"], N=256
    )

    # Create the heatmap
    im = ax.imshow(
        matrix,
        cmap=cmap,
        aspect=s["cell_aspect"],
        vmin=s["vmin"],
        vmax=s["vmax"],
        interpolation='nearest'
    )

    # Add gridlines between cells
    for i in range(len(variables) + 1):
        ax.axhline(i - 0.5, color=s["grid_color"], linewidth=s["grid_linewidth"])
    for j in range(len(properties) + 1):
        ax.axvline(j - 0.5, color=s["grid_color"], linewidth=s["grid_linewidth"])

    # Set ticks and labels
    ax.set_xticks(np.arange(len(properties)))
    ax.set_yticks(np.arange(len(variables)))

    # Parse properties to extract names and units, replace underscores with spaces
    property_labels = []
    property_units = []
    for prop in properties:
        # Extract unit from brackets if present
        if '[' in prop and ']' in prop:
            name_part = prop[:prop.index('[')].strip().replace('_', ' ')
            # Remove brackets from unit
            unit_part = prop[prop.index('[')+1:prop.index(']')]
            property_labels.append(name_part)
            property_units.append(unit_part)
        else:
            property_labels.append(prop.replace('_', ' '))
            property_units.append('')

    # Replace underscores with spaces in variable names
    variables_display = [var.replace('_', ' ') for var in variables]

    # Set tick positions (labels will be added manually below)
    ax.set_xticklabels([])  # Clear default labels
    ax.set_yticklabels([])  # Clear default labels

    # Add x-axis labels at bottom (property names bold, no units) - horizontal, no rotation
    for j, label in enumerate(property_labels):
        ax.text(j, len(variables) - 0.5 + 0.15, label,
               ha='center', va='top',
               fontsize=s["xlabel_fontsize"],
               color=s["text_color"],
               fontweight='bold',
               rotation=0,
               transform=ax.transData)

    # Add y-axis labels on left (bold variable names with ranges) - outside the plot area
    for i, var_label in enumerate(variables_display):
        # Bold variable name
        ax.text(-0.5 - 0.15, i - 0.05, var_label,
               ha='right', va='center',
               fontsize=s["ylabel_fontsize"],
               color=s["text_color"],
               fontweight='bold',
               transform=ax.transData)

        # Add min-max range below in smaller, non-bold dark grey text
        if i < len(variable_ranges):
            var_min, var_max = variable_ranges[i]
            if var_min is not None and var_max is not None:
                # Determine decimal places based on the maximum absolute value
                max_abs = max(abs(var_min), abs(var_max))

                if max_abs >= 1000:
                    decimals = 0  # No decimals for >= 1000
                elif max_abs >= 100:
                    decimals = 1  # 1 decimal for >= 100
                else:
                    decimals = 2  # 2 decimals for < 100

                # Format both values with the same decimal places
                range_text = f"{var_min:.{decimals}f} | {var_max:.{decimals}f}"
                ax.text(-0.5 - 0.15, i + 0.15, range_text,
                       ha='right', va='center',
                       fontsize=s["ylabel_fontsize"] - 1,
                       color='#666666',  # Dark grey
                       fontweight='normal',
                       transform=ax.transData)

    # Add title at top with prefix (replace underscores with spaces)
    title_y = 1.08

    # Add "Correlation matrix: " prefix (smaller, non-bold)
    ax.text(0.5, title_y, "Correlation matrix: ",
            transform=ax.transAxes,
            fontsize=s["title_fontsize"] - 2,
            fontweight='normal',
            color=s["text_color"],
            ha='right',
            va='bottom')

    # Add parameter name (bold)
    ax.text(0.5, title_y, parameter.replace('_', ' '),
            transform=ax.transAxes,
            fontsize=s["title_fontsize"],
            fontweight='bold',
            color=s["text_color"],
            ha='left',
            va='bottom')

    # Add subtitle with metadata
    # Format: "<filter name>  |  n = <count>"
    filter_name = correlation_data.get('filter_name')
    subtitle_parts = []

    if filter_name:
        subtitle_parts.append(filter_name)

    if n_cases:
        subtitle_parts.append(f"n = {n_cases}")

    if subtitle_parts:
        subtitle = "  |  ".join(subtitle_parts)
        subtitle_y = 1.04
        ax.text(0.5, subtitle_y, subtitle,
                transform=ax.transAxes,
                fontsize=s["subtitle_fontsize"],
                color=s["text_color"],
                ha='center',
                va='top',
                style='italic')

    # Add correlation values to cells
    if s["show_values"]:
        for i in range(len(variables)):
            for j in range(len(properties)):
                value = matrix[i, j]

                # Only show values above threshold
                if abs(value) >= s["value_threshold"]:
                    # Choose text color based on correlation value
                    # Values between -0.8 and 0.8: dark colors (grey near 0, black towards ±0.8)
                    # Values outside -0.8 to 0.8: white

                    if -0.8 <= value <= 0.8:
                        # Dark colors for values in range [-0.8, 0.8]
                        # Grey near 0, black towards ±0.8
                        abs_value = abs(value)
                        # Map 0->0.8 to grey->black
                        # At 0: use grey (#808080)
                        # At ±0.8: use black (#000000)
                        grey_component = int(128 - (abs_value / 0.8) * 128)
                        text_color = f"#{grey_component:02x}{grey_component:02x}{grey_component:02x}"
                        path_effect = None
                    else:
                        # White text for strong correlations outside [-0.8, 0.8]
                        text_color = "white"
                        # Add subtle outline for better readability
                        path_effect = [patheffects.withStroke(
                            linewidth=1, foreground='black', alpha=0.3
                        )]

                    text = ax.text(j, i, f'{value:.2f}',
                                 ha="center", va="center",
                                 color=text_color,
                                 fontsize=s["value_fontsize"],
                                 fontweight='normal')

                    if path_effect:
                        text.set_path_effects(path_effect)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(s["colorbar_label"],
                   fontsize=s["colorbar_fontsize"],
                   color=s["text_color"])
    cbar.ax.tick_params(labelsize=s["tick_fontsize"],
                        colors=s["text_color"])

    # Style the colorbar outline
    cbar.outline.set_edgecolor(s["outline_color"])
    cbar.outline.set_linewidth(1.0)

    # Style spines (borders)
    for spine in ax.spines.values():
        spine.set_color(s["outline_color"])
        spine.set_linewidth(1.2)

    # Hide axes ticks (we're using custom labels)
    ax.tick_params(top=False, bottom=False, left=False, right=False,
                   labeltop=False, labelbottom=False,
                   labelleft=False, labelright=False,
                   colors=s["outline_color"])

    # Set plot limits to match grid boundaries (no gaps)
    ax.set_xlim(-0.5, len(properties) - 0.5)
    ax.set_ylim(len(variables) - 0.5, -0.5)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Add extra space for bottom labels, title, and left labels
    plt.subplots_adjust(bottom=0.15, top=0.90, left=0.20)

    # Save figure if output path is provided
    saved = False
    if outfile:
        outfile_path = Path(outfile)
        outfile_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile_path, bbox_inches="tight",
                   facecolor=s["figure_bg_color"], dpi=s["dpi"])
        saved = True

    return fig, ax, saved


if __name__ == "__main__":
    # Example usage with sample data
    sample_data = {
        'parameter': 'Full_Uncertainty',
        'matrix': np.array([
            [ 0.  ,  0.02,  0.04,  0.04,  0.03,  0.04,  0.03,  0.04,  0.03],
            [ 0.01, -0.02, -0.02, -0.  , -0.01, -0.  , -0.01, -0.  , -0.01],
            [ 0.01,  0.  ,  0.  ,  0.01, -0.01,  0.01, -0.01,  0.01, -0.01],
            [-0.  ,  0.  ,  0.  , -0.  , -0.03, -0.  , -0.03,  0.  , -0.03],
            [ 0.01,  0.  ,  0.  ,  0.33, -0.32,  0.33, -0.32,  0.33, -0.32],
            [-0.52, -0.19, -0.2 , -0.4 ,  0.  , -0.4 ,  0.  , -0.4 ,  0.  ],
            [ 0.02,  0.03,  0.03,  0.01,  0.  ,  0.01,  0.  ,  0.01,  0.  ],
            [ 0.01,  0.  ,  0.  ,  0.01,  0.01,  0.01,  0.01,  0.01,  0.01],
            [-0.01,  0.91,  0.9 ,  0.67,  0.75,  0.67,  0.75,  0.67,  0.75],
            [ 0.02,  0.01,  0.01,  0.31,  0.19,  0.31,  0.19,  0.31,  0.19],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ],
            [ 0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ,  0.  ]
        ]),
        'variables': [
            'LeveeAzi',
            'CrevasseAzi',
            'NTG_SEED',
            'POR_SEED',
            'GOCcase',
            'FWLcase',
            'Structure_seed',
            'Isochore_seed',
            'NTGcase',
            'SHFcase',
            'Structure_variation',
            'Isochore_variation'
        ],
        'properties': [
            'bulk volume [mcm]',
            'net volume [mcm]',
            'pore volume [mcm]',
            'hcpv oil [mcm]',
            'hcpv gas [mcm]',
            'stoiip (in oil) [mcm]',
            'stoiip (in gas) [mcm]',
            'giip (in oil) [bcm]',
            'giip (in gas) [bcm]'
        ],
        'n_cases': 3000,
        'constant_variables': [
            ('Structure_variation', 1.0),
            ('Isochore_variation', 1.0)
        ],
        'skipped_variables': None
    }

    # Create the plot
    fig, ax, saved = correlation_plot(sample_data)
    plt.show()
