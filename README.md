# TornadoPy

A Python library for tornado chart generation and analysis. TornadoPy provides tools for processing Excel-based tornado data and generating professional tornado charts for uncertainty analysis.

## Features

- **TornadoProcessor**: Process Excel files containing tornado analysis data
  - Parse multi-sheet Excel files with complex headers
  - Extract and compute statistics (p90p10, mean, median, minmax, percentiles)
  - Filter data by properties and dynamic fields
  - Case selection with weighted criteria
  - Batch processing for multiple parameters
  - Optimized for performance with native numpy operations

- **tornado_plot**: Generate professional tornado charts
  - Customizable colors, fonts, and styling
  - Support for p90/p10 ranges with automatic label placement
  - Reference case lines
  - Custom parameter ordering
  - Export to various image formats

- **distribution_plot**: Generate distribution histograms with cumulative curves
  - Beautiful bin sizing with round numbers
  - Cumulative distribution curve showing % of cases above value
  - P90/P50/P10 percentile markers and subtitle
  - Optional reference case line
  - Multiple color schemes available
  - Export to various image formats

## Installation

Install from PyPI:

```bash
pip install tornadopy
```

## Quick Start

### Processing Tornado Data

```python
from tornadopy import TornadoProcessor

# Load Excel file with tornado data
processor = TornadoProcessor("tornado_data.xlsb")

# Get available parameters
parameters = processor.get_parameters()
print(f"Parameters: {parameters}")

# Get properties for a parameter
properties = processor.get_properties(parameter="Parameter1")
print(f"Properties: {properties}")

# Compute statistics
result = processor.compute(
    stats="p90p10",
    parameter="Parameter1",
    filters={"property": "npv"},
    multiplier=1e-6  # Convert to millions
)
print(f"P90/P10: {result['p90p10']}")
```

### Generating Tornado Charts

```python
from tornadopy import TornadoProcessor, tornado_plot

# Get tornado data
processor = TornadoProcessor("tornado_data.xlsb")
tornado_data = processor.get_tornado_data(
    parameters="all",
    filters={"property": "npv"},
    multiplier=1e-6
)

# Convert to sections format for plotting
sections = []
for param, data in tornado_data.items():
    sections.append({
        "parameter": param,
        "minmax": [data["p10"], data["p90"]],
        "p90p10": [data["p10"], data["p90"]]
    })

# Generate tornado chart
fig, ax, saved = tornado_plot(
    sections=sections,
    title="NPV Tornado Chart",
    subtitle="Base case = 100.0 MM USD",
    base=100.0,
    unit="MM USD",
    outfile="tornado_chart.png"
)
```

### Generating Distribution Charts

```python
from tornadopy import TornadoProcessor, distribution_plot

# Get distribution data
processor = TornadoProcessor("tornado_data.xlsb")
distribution = processor.get_distribution(
    parameter="Parameter1",
    filters={"property": "npv"},
    multiplier=1e-6
)

# Generate distribution chart
fig, ax, saved = distribution_plot(
    distribution,
    title="NPV Distribution",
    unit="MM USD",
    color="blue",
    reference_case=100.0,
    outfile="npv_distribution.png"
)
```

### Advanced Usage

#### Multi-Zone Analysis with Batch Processing

Process multiple parameters at once with zone filtering and custom options:

```python
from tornadopy import TornadoProcessor, tornado_plot

processor = TornadoProcessor("reservoir_data.xlsb")

# Compute statistics for all parameters with zone filtering
results = processor.compute_batch(
    stats=["minmax", "p90p10"],
    parameters="all",
    filters={
        "zones": ["Zone A - Reservoir", "Zone B - Reservoir"],
        "property": "STOIIP"
    },
    multiplier=1e-3,  # Convert to thousands
    options={
        "p90p10_threshold": 150,  # Minimum cases required
        "skip": ["sources"]  # Skip source tracking for cleaner output
    }
)

# Convert results to tornado plot format
sections = []
for result in results:
    if "p90p10" in result and "errors" not in result:
        p10, p90 = result["p90p10"]
        sections.append({
            "parameter": result["parameter"],
            "minmax": result.get("minmax", [p10, p90]),
            "p90p10": [p10, p90]
        })

# Generate tornado chart
fig, ax, saved = tornado_plot(
    sections,
    title="STOIIP Tornado - Multi-Zone Analysis",
    base=14.5,  # Base case value
    reference_case=14.2,  # Reference case line
    unit="MM m³",
    outfile="stoiip_tornado.svg"
)
```

#### Distribution Plot with Custom Gridlines

Create distribution charts with percentile markers and custom grid settings:

```python
from tornadopy import TornadoProcessor, distribution_plot

processor = TornadoProcessor("reservoir_data.xlsb")

# Get distribution data for specific zones
distribution = processor.get_distribution(
    parameter="Uncertainty_Analysis",
    filters={
        "zones": ["Zone A - Reservoir", "Zone B - Reservoir"],
        "property": "STOIIP"
    },
    multiplier=1e-3  # Convert to thousands
)

# Generate distribution chart with custom settings
fig, ax, saved = distribution_plot(
    data=distribution,
    title="STOIIP Distribution - Uncertainty Analysis",
    unit="MM m³",
    color="blue",
    reference_case=14.5,
    target_bins=20,
    settings={
        "show_percentile_markers": True,  # Show P90/P50/P10 markers
        "marker_size": 8,
        "show_minor_grid": True,
        # Custom gridline intervals
        "x_major_interval": 5,   # Major x-gridlines every 5 units
        "x_minor_interval": 1,   # Minor x-gridlines every 1 unit
        "y_major_interval": 50,  # Major y-gridlines every 50 frequency
        "y_minor_interval": 10,  # Minor y-gridlines every 10 frequency
    },
    outfile="stoiip_distribution.svg"
)
```

#### Working with Multiple Properties

Analyze multiple properties simultaneously:

```python
# Compute statistics for multiple properties
result = processor.compute(
    stats=["p90p10", "mean", "median"],
    parameter="Reservoir_Model",
    filters={
        "zones": ["Main_Reservoir"],
        "property": ["STOIIP", "GIIP"]  # Multiple properties
    },
    multiplier=1e-6  # Convert to millions
)

# Access results by property
stoiip_p90, stoiip_p10 = result["p90p10"][0]  # First property (STOIIP)
giip_p90, giip_p10 = result["p90p10"][1]      # Second property (GIIP)

print(f"STOIIP P90/P10: {stoiip_p90:.2f} / {stoiip_p10:.2f} MM m³")
print(f"GIIP P90/P10: {giip_p90:.2f} / {giip_p10:.2f} bcm")
```

#### Case Selection with Weighted Criteria

Find specific cases that match target percentiles:

```python
# Find closest cases to p90/p10 with custom weights
result = processor.compute(
    stats="p90p10",
    parameter="Reservoir_Model",
    filters={
        "zones": ["Main_Reservoir"],
        "property": "STOIIP"
    },
    multiplier=1e-6,
    case_selection=True,  # Enable case selection
    selection_criteria={
        "weights": {"STOIIP": 0.6, "GIIP": 0.4}  # Weighted criteria
    }
)

# Access closest cases
for case in result["closest_cases"]:
    print(f"Case {case['case']}: index={case['idx']}, STOIIP={case['STOIIP']:.2f}")
    print(f"  Properties: {case['properties']}")
```

#### Skipping Specific Parameters

Exclude certain parameters from batch processing:

```python
# Process all parameters except specific ones
results = processor.compute_batch(
    stats="p90p10",
    parameters="all",
    filters={"property": "STOIIP"},
    multiplier=1e-3,
    options={
        "skip_parameters": ["Reference_Case", "Full_Uncertainty"],  # Skip these
        "skip": ["sources", "errors"]  # Skip these fields in output
    }
)
```

#### Custom Tornado Chart Styling

Full control over chart appearance:

```python
# Custom styling for professional reports
settings = {
    "figsize": (12, 8),
    "dpi": 200,
    "pos_dark": "#1E88E5",  # Blue for positive
    "neg_dark": "#D32F2F",  # Red for negative
    "show_values": ["min", "max", "p10", "p90"],
    "show_percentage_diff": True,
}

fig, ax, saved = tornado_plot(
    sections=sections,
    title="Reservoir Volume Sensitivity Analysis",
    subtitle="Base Case: 100 MM m³",
    base=100.0,
    reference_case=95.0,
    unit="MM m³",
    preferred_order=["Porosity", "NTG", "Area"],  # Custom parameter order
    settings=settings,
    outfile="sensitivity_analysis.png"
)
```

## Common Workflows

### Complete Reservoir Uncertainty Analysis

End-to-end workflow for reservoir analysis with tornado and distribution charts:

```python
from tornadopy import TornadoProcessor, tornado_plot, distribution_plot
import matplotlib.pyplot as plt

# Load data
processor = TornadoProcessor("reservoir_uncertainty.xlsb")

# Define common filters
zones = ["Main Reservoir - SST1", "Main Reservoir - SST2"]
multiplier = 1e-3  # Convert to thousands

# 1. Generate STOIIP Tornado Chart
stoiip_results = processor.compute_batch(
    stats=["minmax", "p90p10"],
    parameters="all",
    filters={
        "zones": zones,
        "property": "STOIIP"
    },
    multiplier=multiplier,
    options={
        "p90p10_threshold": 150,
        "skip_parameters": ["Reference_Case", "Full_Uncertainty"]
    }
)

# Convert to tornado format
sections = []
for result in stoiip_results:
    if "p90p10" in result and "errors" not in result:
        p10, p90 = result["p90p10"]
        min_val, max_val = result.get("minmax", [p10, p90])
        sections.append({
            "parameter": result["parameter"],
            "minmax": [min_val, max_val],
            "p90p10": [p10, p90]
        })

# Create tornado chart
fig1, ax1, saved1 = tornado_plot(
    sections,
    title="STOIIP Sensitivity Analysis",
    base=14.5,
    reference_case=14.2,
    unit="MM m³",
    outfile="stoiip_tornado.svg"
)

# 2. Generate Distribution Chart
distribution = processor.get_distribution(
    parameter="Full_Uncertainty",
    filters={
        "zones": zones,
        "property": "STOIIP"
    },
    multiplier=multiplier
)

fig2, ax2, saved2 = distribution_plot(
    data=distribution,
    title="STOIIP Distribution - Full Uncertainty",
    unit="MM m³",
    color="blue",
    reference_case=14.5,
    settings={
        "show_percentile_markers": True,
        "x_major_interval": 5,
        "x_minor_interval": 1,
    },
    outfile="stoiip_distribution.svg"
)

# Show both charts
plt.show()

print(f"Charts saved: {saved1}, {saved2}")
```

### Comparing Multiple Scenarios

Compare different reservoir scenarios side by side:

```python
from tornadopy import TornadoProcessor, distribution_plot
import matplotlib.pyplot as plt
import numpy as np

processor = TornadoProcessor("scenarios.xlsb")

# Define scenarios
scenarios = [
    {"name": "Base Case", "param": "Base_Case", "color": "blue"},
    {"name": "Optimistic", "param": "Optimistic", "color": "green"},
    {"name": "Pessimistic", "param": "Pessimistic", "color": "red"},
]

# Create subplots for comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for idx, scenario in enumerate(scenarios):
    dist = processor.get_distribution(
        parameter=scenario["param"],
        filters={"property": "NPV"},
        multiplier=1e-6
    )

    distribution_plot(
        data=dist,
        title=f"{scenario['name']} Scenario",
        unit="MM USD",
        color=scenario["color"],
        target_bins=15,
        outfile=None  # Don't save individual plots
    )

    # Move the plot to the subplot
    plt.close()

plt.tight_layout()
plt.savefig("scenario_comparison.png", dpi=200)
plt.show()
```

## Tips and Best Practices

### Working with Filters

**Zone Filtering:**
```python
# Single zone
filters = {"zones": "Main Reservoir", "property": "STOIIP"}

# Multiple zones (will sum values across zones)
filters = {"zones": ["Zone A", "Zone B"], "property": "STOIIP"}
```

**Property Filtering:**
```python
# Single property
filters = {"property": "STOIIP"}

# Multiple properties (returns separate results for each)
filters = {"property": ["STOIIP", "GIIP"]}
```

### Using Multipliers

Convert units easily with the multiplier parameter:

```python
# Convert to thousands (mcm → MM m³)
multiplier = 1e-3

# Convert to millions (m³ → MM m³)
multiplier = 1e-6

# Convert to billions (m³ → bcm)
multiplier = 1e-9
```

### Skipping Parameters

Exclude specific parameters from batch processing:

```python
options = {
    "skip_parameters": ["Reference_Case", "Full_Uncertainty"],  # Skip these parameters
    "skip": ["sources", "errors"]  # Skip these fields in results
}
```

### Handling Errors

```python
results = processor.compute_batch(
    stats="p90p10",
    parameters="all",
    filters={"property": "STOIIP"},
    options={"skip": ["errors"]}  # Hide error messages
)

# Check for errors in results
for result in results:
    if "errors" in result:
        print(f"Parameter {result['parameter']} had errors: {result['errors']}")
    elif "p90p10" in result:
        print(f"Parameter {result['parameter']}: P90/P10 = {result['p90p10']}")
```

### Performance Tips

1. **Use batch processing** for multiple parameters:
   ```python
   # Good: Single call for all parameters
   results = processor.compute_batch(stats="p90p10", parameters="all", ...)

   # Avoid: Multiple calls
   for param in parameters:
       result = processor.compute(stats="p90p10", parameter=param, ...)
   ```

2. **Skip unnecessary data**:
   ```python
   options = {
       "skip": ["sources", "errors"],  # Reduces memory usage
   }
   ```

3. **Set appropriate thresholds**:
   ```python
   options = {
       "p90p10_threshold": 150,  # Require minimum cases for reliable statistics
   }
   ```

## Excel File Format

TornadoPy expects Excel files with the following structure:

```
[Info rows - optional metadata]
Header Row 1    | Dynamic Field 1 | Dynamic Field 1 | ...
Header Row 2    | Value A         | Value B         | ...
Case            | Property 1      | Property 2      | ...
1               | 123.45          | 67.89           | ...
2               | 234.56          | 78.90           | ...
...
```

- Multiple header rows are supported and will be combined
- The "Case" row marks the start of data
- Dynamic fields in column A define metadata columns
- Property names are extracted from the last header row

## API Reference

### TornadoProcessor

#### Methods

- `get_parameters()`: Get list of available parameters (sheet names)
- `get_properties(parameter)`: Get available properties for a parameter
- `get_unique(field, parameter)`: Get unique values for a dynamic field
- `get_info(parameter)`: Get metadata for a parameter
- `get_case(index, parameter)`: Get data for a specific case
- `compute(stats, parameter, filters, multiplier, options, case_selection, selection_criteria)`: Compute statistics
- `compute_batch(...)`: Batch compute for multiple parameters
- `get_tornado_data(...)`: Get tornado chart formatted data

### tornado_plot

#### Parameters

- `sections`: List of section dictionaries with parameter data
- `title`: Chart title
- `subtitle`: Chart subtitle
- `outfile`: Output file path
- `base`: Base case value
- `reference_case`: Reference case line value
- `unit`: Unit label
- `preferred_order`: List of parameter names for custom ordering
- `settings`: Dictionary of visual settings

#### Returns

- `fig`: Matplotlib figure object
- `ax`: Matplotlib axes object
- `saved`: Path to saved file (if outfile specified)

### distribution_plot

#### Parameters

- `data`: Array-like data (numpy array, list, or from get_distribution)
- `title`: Chart title (default "Distribution")
- `unit`: Unit label for x-axis and subtitle
- `outfile`: Output file path (if specified, saves the figure)
- `target_bins`: Target number of bins for histogram (default 20)
- `color`: Color scheme - "red", "blue", "green", "orange", "purple", "fuchsia", "yellow"
- `reference_case`: Optional reference case value to plot as vertical line
- `settings`: Dictionary of visual settings to override defaults

#### Settings Options

Common settings for customizing distribution plots:

```python
settings = {
    # Layout
    "figsize": (10, 6),
    "dpi": 160,

    # Percentile markers
    "show_percentile_markers": True,  # Show P90/P50/P10 on cumulative curve
    "marker_size": 8,

    # Grid customization
    "show_minor_grid": True,
    "x_major_interval": 5,   # Major x-gridlines every 5 units
    "x_minor_interval": 1,   # Minor x-gridlines every 1 unit
    "y_major_interval": 50,  # Major y-gridlines every 50 frequency
    "y_minor_interval": 10,  # Minor y-gridlines every 10 frequency

    # Font sizes
    "title_fontsize": 15,
    "subtitle_fontsize": 11,
    "label_fontsize": 10,
}
```

#### Returns

- `fig`: Matplotlib figure object
- `ax`: Matplotlib axes object (primary)
- `saved`: Path to saved file (if outfile specified)

## Requirements

- Python >= 3.9
- numpy >= 1.20.0
- polars >= 0.18.0
- fastexcel >= 0.9.0
- matplotlib >= 3.5.0

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.
