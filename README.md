# TornadoPy

A Python library for generating fast tornado and distribution plots using static model results from uncertainty analysis run in SLB Petrel.

TornadoPy provides efficient data processing and visualization tools for analyzing sensitivity and uncertainty results from reservoir modeling workflows. It leverages Polars for fast data manipulation and Matplotlib for publication-quality charts.

## Features

- Fast processing of Excel-based uncertainty analysis results using Polars
- Generate tornado charts showing parameter sensitivities
- Create distribution plots with cumulative curves
- Support for complex filtering and data aggregation
- Statistical computations (P90/P10, mean, median, percentiles)
- Case selection for representative scenarios
- Batch processing for multiple parameters
- Highly configurable plot styling

## Installation

```bash
pip install tornadopy
```

## Quick Start

```python
from tornadopy import TornadoProcessor, tornado_plot, distribution_plot

# Load data from Excel file
processor = TornadoProcessor("uncertainty_results.xlsx", multiplier=1e-6)

# Generate tornado chart data
results = processor.tornado(filters={'property': 'stoiip'})

# Create tornado plot
fig, ax, saved = tornado_plot(
    results,
    title="STOIIP Sensitivity Analysis",
    unit="MM bbl",
    outfile="tornado.png"
)

# Generate distribution data
dist_data = processor.distribution(
    parameter="NetPay",
    filters={'property': 'stoiip'}
)

# Create distribution plot
fig, ax, saved = distribution_plot(
    dist_data,
    title="Net Pay Distribution",
    unit="MM bbl",
    outfile="distribution.png"
)
```

## Data Setup

### Excel File Structure

TornadoPy expects uncertainty analysis results stored in an Excel file with a specific layout:

1. **Multiple Sheets (Tabs)**: Each parameter should be stored in a separate sheet
2. **Single Row Output Table**: Generate single row output tables in Petrel
3. **Segmentation**: Split results by Zones, Segments, or Boundaries as preferred

### Sheet Layout

Each sheet should follow this structure:

```
Row 1-N:     [Metadata rows - optional]
             Key: Value
             Description: Additional info

Header rows: Zone   Segment  Property
             z1     seg1     stoiip    z1  seg2  stoiip    z2  seg1  stoiip
Case row:    Case   Case     Case      ...
Data rows:   Case1  123.4    456.7     ...
             Case2  125.1    458.2     ...
             Case3  ...
```

**Important Layout Rules:**

1. **"Case" Row**: Must contain the text "Case" in the first column. This marks where data begins.

2. **Header Block**: One or more rows above the "Case" row that define column structure:
   - First column contains field names (Zone, Segment, Boundary, etc.)
   - Remaining columns contain the values for each combination
   - Headers are automatically combined (e.g., "z1_seg1_stoiip")

3. **Data Block**: Starts immediately after the "Case" row:
   - Each row represents a different uncertainty case
   - Values should be numeric
   - First column can contain case identifiers (optional)

4. **Properties**: Each unique property (e.g., stoiip, giip, npv) should be clearly labeled in headers

5. **Base Case and Reference Case** (Optional): Can be stored in a separate sheet:
   - Row 0: Base case values
   - Row 1: Reference case values (optional)
   - Same column structure as other parameters

### Example Sheet Structure

```
Metadata:    Reservoir: North Field
             Date: 2024-01-15

Headers:     Zone    Zone    Zone    Zone
             North   North   South   South
             stoiip  giip    stoiip  giip
Case:        Case    Case    Case    Case
Data:        Case1   150.2   45.3    98.1    29.4
             Case2   155.8   46.1    102.3   30.2
             Case3   148.9   44.8    95.7    28.9
             ...     ...     ...     ...     ...
```

### Excel File Preparation Workflow

1. **In Petrel**:
   - Run your uncertainty analysis
   - For each parameter, create a single-row output table
   - Export results to Excel

2. **In Excel**:
   - Create a new workbook
   - Create one sheet per parameter (e.g., "NetPay", "Porosity", "NTG")
   - Optionally create a "BaseCases" sheet with base and reference case values
   - Paste Petrel results into each sheet following the layout above
   - Ensure the "Case" row is present
   - Save as `.xlsx` or `.xlsb` format

## Using the TornadoProcessor

### Initialization

```python
# Basic initialization
processor = TornadoProcessor("data.xlsx")

# With multiplier (e.g., convert to millions)
processor = TornadoProcessor("data.xlsx", multiplier=1e-6)

# With base case sheet
processor = TornadoProcessor(
    "data.xlsx",
    multiplier=1e-6,
    base_case="BaseCases"  # Sheet name containing base/reference values
)
```

### Exploring Your Data

```python
# List all available parameters (sheet names)
parameters = processor.parameters()
print(parameters)  # ['NetPay', 'Porosity', 'NTG', 'BaseCases']

# List all properties for a parameter
properties = processor.properties("NetPay")
print(properties)  # ['stoiip', 'giip', 'npv']

# Get unique values for dynamic fields (zones, segments, etc.)
zones = processor.unique_values("zone", parameter="NetPay")
segments = processor.unique_values("segment", parameter="NetPay")
```

### Extracting Statistics

```python
# Compute P90/P10 for a single property
result = processor.compute(
    stats='p90p10',
    parameter='NetPay',
    filters={'property': 'stoiip', 'zone': 'z1'}
)
print(result)
# {'parameter': 'NetPay', 'p90p10': [145.2, 182.7], 'sources': [...]}

# Compute multiple statistics
result = processor.compute(
    stats=['mean', 'median', 'p90p10'],
    filters={'property': 'stoiip'}
)
print(result)
# {'parameter': 'NetPay', 'mean': 163.5, 'median': 162.8, 'p90p10': [145.2, 182.7]}

# Multi-property computation
result = processor.compute(
    stats='mean',
    filters={'property': ['stoiip', 'giip']}
)
print(result)
# {'parameter': 'NetPay', 'mean': {'stoiip': 163.5, 'giip': 48.2}}
```

### Batch Processing

```python
# Process all parameters at once
results = processor.compute_batch(
    stats='p90p10',
    parameters='all',  # or specify list: ['NetPay', 'Porosity']
    filters={'property': 'stoiip', 'zone': ['z1', 'z2']}
)

# Results is a list of dictionaries, one per parameter
for result in results:
    print(f"{result['parameter']}: {result['p90p10']}")
```

### Using Filters

```python
# Simple filter
result = processor.compute(
    'mean',
    filters={'property': 'stoiip', 'zone': 'z1'}
)

# Multiple values (aggregates across zones)
result = processor.compute(
    'mean',
    filters={'property': 'stoiip', 'zone': ['z1', 'z2', 'z3']}
)

# Store filter presets for reuse
processor.set_filter('north_zones', {'zone': ['z1', 'z2', 'z3']})
processor.set_filter('south_zones', {'zone': ['z4', 'z5']})

# Use stored filter
result = processor.compute('mean', filters='north_zones')
```

### Working with Base and Reference Cases

```python
# Get base case value for a property
base_stoiip = processor.base_case('stoiip')

# Get all base case values
base_all = processor.base_case()

# Get reference case
ref_stoiip = processor.ref_case('stoiip')

# With filters
base_filtered = processor.base_case(
    'stoiip',
    filters={'zone': ['z1', 'z2']}
)

# With custom multiplier
base_mm = processor.base_case('stoiip', multiplier=1e-6)
```

### Generating Tornado Data

The `tornado()` method is a convenience function that computes both minmax and p90p10 statistics for all parameters:

```python
# Generate tornado data for all parameters
tornado_data = processor.tornado(
    filters={'property': 'stoiip'},
    multiplier=1e-6
)

# With options
tornado_data = processor.tornado(
    filters={'property': 'stoiip'},
    skip='sources',  # Don't include source columns
    options={'decimals': 2}
)

# The result is ready to pass directly to tornado_plot()
```

### Extracting Distribution Data

```python
# Get distribution for a parameter
dist = processor.distribution(
    parameter='NetPay',
    filters={'property': 'stoiip', 'zone': 'z1'},
    multiplier=1e-6
)

# dist is a numpy array of all case values
# Pass directly to distribution_plot()
```

### Case Selection

Find representative cases that best match statistical targets:

```python
# Find cases closest to mean with weighted properties
result = processor.compute(
    'mean',
    filters={'property': 'stoiip'},
    case_selection=True,
    selection_criteria={'weights': {'stoiip': 0.6, 'giip': 0.4}}
)

# Result includes closest_cases with full case details
print(result['closest_cases'])
# [{'idx': 42, 'stoiip': 163.2, 'case': 'mean', 'properties': {...}, ...}]
```

## Plotting

### Tornado Plot

Create publication-quality tornado charts:

```python
from tornadopy import tornado_plot

# Basic tornado plot
fig, ax, saved = tornado_plot(
    tornado_data,
    title="STOIIP Sensitivity Analysis",
    unit="MM bbl",
    outfile="tornado.png"
)

# With reference case line
fig, ax, saved = tornado_plot(
    tornado_data,
    title="STOIIP Sensitivity",
    base=150.0,
    reference_case=155.0,
    unit="MM bbl"
)

# With preferred parameter order
fig, ax, saved = tornado_plot(
    tornado_data,
    title="STOIIP Sensitivity",
    preferred_order=["NetPay", "Porosity", "NTG"],  # Show these first
    unit="MM bbl"
)
```

### Customizing Tornado Plots

Control the appearance with the `settings` parameter:

```python
custom_settings = {
    'figsize': (12, 8),
    'dpi': 200,
    'pos_light': '#A9CFF7',  # Light blue for positive bars
    'neg_light': '#F5B7B1',  # Light red for negative bars
    'pos_dark': '#2E5BFF',   # Dark blue for P90/P10 overlay
    'neg_dark': '#E74C3C',   # Dark red for P90/P10 overlay
    'show_values': ['min', 'p10', 'p90', 'max'],  # Which values to label
    'show_value_headers': True,
    'show_relative_values': False,  # Show absolute values
    'show_percentage_diff': True,   # Show % difference from base
    'value_format': '{:.1f}',
    'bar_height': 0.6,
    'label_fontsize': 9,
}

fig, ax, saved = tornado_plot(
    tornado_data,
    title="Custom Tornado Chart",
    unit="MM bbl",
    settings=custom_settings
)
```

**Key Tornado Plot Settings:**

- **Colors**: `pos_light`, `neg_light`, `pos_dark`, `neg_dark`, `baseline_color`, `reference_color`
- **Sizes**: `figsize`, `dpi`, `bar_height`, `bar_linewidth`
- **Values**: `show_values` (list), `show_value_headers` (bool), `value_format` (str)
- **Labels**: `show_relative_values` (bool), `show_percentage_diff` (bool)
- **Font sizes**: `title_fontsize`, `subtitle_fontsize`, `label_fontsize`, `value_fontsize`

### Distribution Plot

Create histograms with cumulative distribution curves:

```python
from tornadopy import distribution_plot

# Basic distribution
fig, ax, saved = distribution_plot(
    dist_data,
    title="Net Pay Distribution",
    unit="MM bbl",
    outfile="distribution.png"
)

# With reference case and custom bins
fig, ax, saved = distribution_plot(
    dist_data,
    title="Net Pay Distribution",
    unit="MM bbl",
    reference_case=150.0,
    target_bins=30,
    color="blue"
)
```

### Customizing Distribution Plots

```python
custom_settings = {
    'figsize': (12, 7),
    'dpi': 200,
    'bar_color': '#66C3EB',  # Light blue bars
    'bar_outline_color': '#0075A6',  # Dark blue outline
    'cumulative_color': '#BA2A19',  # Dark red cumulative line
    'cumulative_linewidth': 3.0,
    'show_percentile_markers': True,  # Show P90/P50/P10 markers
    'target_bins': 25,
    'show_minor_grid': True,
}

fig, ax, saved = distribution_plot(
    dist_data,
    title="Custom Distribution",
    unit="MM bbl",
    settings=custom_settings
)
```

**Available Color Schemes:**
- `"red"`, `"blue"`, `"green"`, `"orange"`, `"purple"`, `"fuchsia"`, `"yellow"`

**Key Distribution Plot Settings:**

- **Colors**: Color scheme name or custom `bar_color`, `bar_outline_color`, `cumulative_color`
- **Sizes**: `figsize`, `dpi`, `bar_linewidth`, `cumulative_linewidth`
- **Bins**: `target_bins` (int) - number of histogram bins
- **Grid**: `show_minor_grid` (bool), `grid_alpha`, `minor_grid_alpha`
- **Markers**: `show_percentile_markers` (bool), `marker_size`, `marker_color`
- **Font sizes**: `title_fontsize`, `subtitle_fontsize`, `label_fontsize`, `tick_fontsize`

## Complete Workflow Example

```python
from tornadopy import TornadoProcessor, tornado_plot, distribution_plot

# 1. Initialize processor
processor = TornadoProcessor(
    "uncertainty_analysis.xlsx",
    multiplier=1e-6,  # Convert to millions
    base_case="BaseCases"
)

# 2. Set up filter presets
processor.set_filter('main_zones', {
    'zone': ['Zone1', 'Zone2', 'Zone3']
})

# 3. Generate tornado chart
tornado_data = processor.tornado(
    filters={
        'property': 'stoiip',
        'zone': ['Zone1', 'Zone2', 'Zone3']
    },
    skip='sources',
    options={'decimals': 1}
)

fig, ax, saved = tornado_plot(
    tornado_data,
    title="STOIIP Tornado Chart",
    subtitle="Main Development Zones",
    unit="MM STB",
    preferred_order=["NetPay", "Porosity", "NTG", "Area"],
    outfile="stoiip_tornado.png"
)

# 4. Generate distribution plot for key parameter
dist_data = processor.distribution(
    parameter="NetPay",
    filters={'property': 'stoiip', 'zone': ['Zone1', 'Zone2', 'Zone3']}
)

fig, ax, saved = distribution_plot(
    dist_data,
    title="Net Pay Impact on STOIIP",
    unit="MM STB",
    reference_case=processor.ref_case('stoiip', filters='main_zones'),
    color="blue",
    outfile="netpay_distribution.png"
)

# 5. Compute statistics with case selection
result = processor.compute(
    stats=['mean', 'p90p10'],
    parameter='NetPay',
    filters='main_zones',
    case_selection=True,
    selection_criteria={'weights': {'stoiip': 0.7, 'giip': 0.3}}
)

print(f"Mean STOIIP: {result['mean']:.1f} MM STB")
print(f"P90-P10 Range: {result['p90p10']}")
print(f"Representative case: {result['closest_cases'][0]['idx']}")
```

## Advanced Features

### Multi-Property Analysis

```python
# Compute statistics across multiple properties simultaneously
result = processor.compute(
    stats='mean',
    filters={'property': ['stoiip', 'giip', 'npv']}
)

print(result['mean'])
# {'stoiip': 163.5, 'giip': 48.2, 'npv': 450.3}
```

### Complex Case Selection

```python
# Use weighted combinations for sophisticated case selection
selection_criteria = {
    'combinations': [
        {
            'filters': {'zone': 'z1'},
            'properties': {'stoiip': 0.5, 'giip': 0.3}
        },
        {
            'filters': {'zone': 'z2'},
            'properties': {'stoiip': 0.2}
        }
    ]
}

result = processor.compute(
    'p90p10',
    filters={'property': 'stoiip'},
    case_selection=True,
    selection_criteria=selection_criteria
)
```

### Custom Statistical Options

```python
# Compute arbitrary percentile
result = processor.compute(
    'percentile',
    filters={'property': 'stoiip'},
    options={'p': 75}  # P75
)

# Skip certain outputs
result = processor.compute(
    'mean',
    filters={'property': 'stoiip'},
    options={'skip': ['sources', 'errors'], 'decimals': 2}
)
```

## API Reference

### TornadoProcessor

**Initialization:**
- `TornadoProcessor(filepath, multiplier=1.0, base_case=None)`

**Data Exploration:**
- `parameters()` - List all parameter names
- `properties(parameter=None)` - List properties for a parameter
- `unique_values(field, parameter=None)` - Get unique values for a field
- `info(parameter=None)` - Get metadata for a parameter
- `case(index, parameter=None)` - Get data for a specific case

**Statistics:**
- `compute(stats, parameter=None, filters=None, multiplier=None, options=None, case_selection=False, selection_criteria=None)`
- `compute_batch(stats, parameters='all', filters=None, multiplier=None, options=None, ...)`
- `tornado(filters=None, multiplier=None, skip=None, options=None, ...)` - Convenience method
- `distribution(parameter=None, filters=None, multiplier=None, options=None)` - Convenience method

**Base Cases:**
- `base_case(property=None, filters=None, multiplier=None)` - Get base case value(s)
- `ref_case(property=None, filters=None, multiplier=None)` - Get reference case value(s)

**Filters:**
- `set_filter(name, filters)` - Store a filter preset
- `get_filter(name)` - Retrieve a filter preset
- `list_filters()` - List all filter presets

### Plotting Functions

**tornado_plot:**
```python
tornado_plot(sections, title="Tornado Chart", subtitle=None, outfile=None,
             base=None, reference_case=None, unit=None,
             preferred_order=None, settings=None)
```

**distribution_plot:**
```python
distribution_plot(data, title="Distribution", unit=None, outfile=None,
                  target_bins=20, color="blue", reference_case=None,
                  settings=None)
```

## Requirements

- Python >= 3.9
- numpy >= 1.20.0
- polars >= 0.18.0
- fastexcel >= 0.9.0
- matplotlib >= 3.5.0

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Issues

Report issues at: https://github.com/kkollsga/tornadopy/issues

## Author

Kristian dF Kollsgård (kkollsg@gmail.com)
