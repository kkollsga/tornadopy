# TornadoPy

A Python library for generating fast tornado and distribution plots using static model results from uncertainty analysis run in SLB Petrel.

TornadoPy provides efficient data processing and visualization tools for analyzing sensitivity and uncertainty results from reservoir modeling workflows. It leverages Polars for fast data manipulation and Matplotlib for publication-quality charts.

## Features

- Fast processing of Excel-based uncertainty analysis results using Polars
- Generate tornado charts showing parameter sensitivities
- Create distribution plots with cumulative curves
- Support for complex filtering and data aggregation
- Statistical computations (P90/P10, mean, median, percentiles)
- **Intelligent case selection** for representative scenarios
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

# Generate distribution plot
dist_data = processor.distribution(
    parameter="NetPay",
    filters={'property': 'stoiip'}
)

fig, ax, saved = distribution_plot(
    dist_data,
    title="Net Pay Distribution",
    unit="MM bbl",
    outfile="distribution.png"
)
```

## Data Setup

### Excel File Structure

TornadoPy expects uncertainty analysis results stored in an Excel file with this layout:

**Sheet Structure:**
```
Metadata rows (optional):
    Key: Value
    Description: Additional info

Header block:
    Zone     Segment   Property
    north    main      stoiip    north  flank  stoiip    south  main  stoiip

Case marker:
    Case     Case      Case      ...

Data rows:
    Case1    123.4     456.7     ...
    Case2    125.1     458.2     ...
    Case3    ...
```

**Important Rules:**

1. **"Case" Row**: Must contain the text "Case" in the first column - marks where data begins
2. **Headers**: One or more rows above "Case" defining column structure (automatically combined)
3. **Data Block**: Starts after "Case" row, each row is a different uncertainty case
4. **Properties**: Clearly labeled in headers (e.g., stoiip, giip, npv)
5. **Multiple Sheets**: Each parameter in a separate sheet
6. **Base Case Sheet** (Optional): Row 0 = base case, Row 1 = reference case

### Excel Preparation Workflow

1. **In Petrel**: Run uncertainty analysis, create single-row output tables, export to Excel
2. **In Excel**: Create one sheet per parameter, paste results, ensure "Case" row exists, save as `.xlsx`

## Using the TornadoProcessor

### Initialization

```python
# Basic initialization
processor = TornadoProcessor("data.xlsx")

# With multiplier and base case
processor = TornadoProcessor(
    "data.xlsx",
    multiplier=1e-6,           # Convert to millions
    base_case="BaseCases"      # Sheet with base/reference values
)
```

### Exploring Your Data

```python
# List all parameters (sheet names)
parameters = processor.parameters()

# List properties for a parameter
properties = processor.properties("NetPay")
# ['stoiip', 'giip', 'npv']

# Get unique values for dynamic fields
zones = processor.unique_values("zones", parameter="NetPay")
segments = processor.unique_values("segments", parameter="NetPay")
```

### Computing Statistics

```python
# Single statistic
result = processor.compute(
    stats='p90p10',
    parameter='NetPay',
    filters={'property': 'stoiip', 'zones': 'north'}
)
# {'parameter': 'NetPay', 'p90p10': [145.2, 182.7], ...}

# Multiple statistics
result = processor.compute(
    stats=['mean', 'median', 'p90p10'],
    filters={'property': 'stoiip'}
)

# Multi-property computation
result = processor.compute(
    stats='mean',
    filters={'property': ['stoiip', 'giip']}
)
# {'parameter': 'NetPay', 'mean': {'stoiip': 163.5, 'giip': 48.2}}
```

**Available Statistics:**
- `p90p10`, `minmax`, `mean`, `median`, `std`, `cv`, `sum`, `count`, `variance`, `range`
- `p1p99`, `p25p75`, `percentile` (with `options={'p': 75}`)
- `distribution` (returns full array)

### Working with Filters

```python
# Simple filter
result = processor.compute('mean', filters={'property': 'stoiip', 'zones': 'north'})

# Multiple zones (aggregates)
result = processor.compute('mean', filters={'zones': ['north', 'central', 'south']})

# Store filter presets for reuse
processor.set_filter('north_zones', {
    'zones': ['north_main', 'north_flank'],
    'property': 'stoiip'
})
processor.set_filter('south_zones', {
    'zones': ['south_main', 'south_flank'],
    'property': 'stoiip'
})

# Use stored filter by name
result = processor.compute('mean', filters='north_zones')
```

### Batch Processing

```python
# Process all parameters at once
results = processor.compute_batch(
    stats='p90p10',
    parameters='all',
    filters={'property': 'stoiip'}
)

# Results is a list sorted by sensitivity (largest range first)
for result in results:
    print(f"{result['parameter']}: {result['p90p10']}")
```

### Base and Reference Cases

```python
# Get base case value
base_stoiip = processor.base_case('stoiip')

# Get all base case values
base_all = processor.base_case()

# Get reference case
ref_stoiip = processor.ref_case('stoiip')

# With filters and custom multiplier
base_filtered = processor.base_case(
    'stoiip',
    filters={'zones': ['north_main', 'north_flank']},
    multiplier=1e-6
)
```

### Case Selection

Find representative cases that best match statistical targets using weighted property combinations.

#### Simple Case Selection

Use property names when all properties share the same filters:

```python
# Find P90/P10 cases weighted by STOIIP and GIIP
result = processor.compute(
    stats='p90p10',
    parameter='Full_Uncertainty',
    filters={'zones': ['north_main', 'north_flank'], 'property': 'stoiip'},
    case_selection=True,
    selection_criteria={'stoiip': 0.6, 'giip': 0.4}
)

# Output includes closest matching cases
print(result['closest_cases'])
# [
#     {
#         'reference': 'p10.Full_Uncertainty_1854',
#         'weights': {'stoiip': 0.6, 'giip': 0.4},
#         'weighted_distance': 0.000128,
#         'selection_values': {
#             'stoiip_actual': 145.3,
#             'stoiip_p10': 145.2,      # Target P10 for STOIIP
#             'giip_actual': 48.1,
#             'giip_p10': 48.0          # Target P10 for GIIP
#         },
#         'selection_method': 'weighted'
#     },
#     {...}  # P90 case
# ]
```

#### Advanced Case Selection with Different Filters

Use stored filter names when properties need different zone sets:

```python
# Define filters for different reservoir areas
processor.set_filters({
    'north_stoiip': {
        'zones': ['north_main', 'north_flank', 'north_terrace'],
        'property': 'stoiip'
    },
    'south_giip': {
        'zones': ['south_main', 'south_crest'],
        'property': 'giip'
    }
})

# Use filter names as keys - each property uses its own filter!
result = processor.compute(
    stats='p90p10',
    parameter='Full_Uncertainty',
    filters={'property': 'stoiip'},  # Main computation
    case_selection=True,
    selection_criteria={
        'north_stoiip': 0.6,  # Uses north zones for STOIIP
        'south_giip': 0.4     # Uses south zones for GIIP
    }
)

# Result: STOIIP weighted from north zones, GIIP from south zones
```

#### Mixed Case Selection

Combine property names (inherit main filter) with stored filters:

```python
result = processor.compute(
    stats='mean',
    filters={'zones': ['north_main'], 'property': 'stoiip'},
    case_selection=True,
    selection_criteria={
        'stoiip': 0.5,          # Uses north_main (main filter)
        'north_stoiip': 0.3,    # Uses north_main + north_flank + north_terrace
        'giip': 0.2             # Uses north_main (main filter)
    }
)
```

**How It Works:**

The processor intelligently resolves keys in `selection_criteria`:

1. **Property name** (e.g., `'stoiip'`) → Uses filters from main `compute()` call
2. **Stored filter name** (e.g., `'north_stoiip'`) → Uses that filter's zones and property
3. **Not found** → Helpful error showing available properties and filters

**Key Benefits:**

- **Flexible weighting** across different reservoir areas
- **Transparent selection** - see actual vs target values for all properties
- **Smart resolution** - no ambiguity between property and filter names

#### Complex Combinations (Advanced)

For maximum control, use the `combinations` syntax:

```python
result = processor.compute(
    stats='p90p10',
    filters={'property': 'stoiip'},
    case_selection=True,
    selection_criteria={
        'combinations': [
            {
                'filters': 'north_zones',  # Stored filter
                'properties': {'stoiip': 0.5, 'giip': 0.2}
            },
            {
                'filters': {'zones': ['south_main']},  # Inline filter
                'properties': {'stoiip': 0.3}
            }
        ]
    }
)
```

### Tornado Chart Data

```python
# Generate tornado data (minmax + p90p10 for all parameters)
tornado_data = processor.tornado(
    filters={'property': 'stoiip'},
    skip='filters',  # Cleaner output
    options={'decimals': 2}
)

# With case selection
tornado_data = processor.tornado(
    filters={'property': 'stoiip'},
    case_selection=True,
    selection_criteria={'stoiip': 0.7, 'giip': 0.3}
)

# Pass directly to tornado_plot()
fig, ax, saved = tornado_plot(tornado_data, title="Sensitivity", unit="MM bbl")
```

### Distribution Data

```python
# Get distribution array
dist = processor.distribution(
    parameter='NetPay',
    filters={'property': 'stoiip', 'zones': 'north'}
)

# dist is a numpy array ready for distribution_plot()
fig, ax, saved = distribution_plot(dist, title="Net Pay", unit="MM bbl")
```

## Plotting

### Tornado Plot

```python
from tornadopy import tornado_plot

# Basic tornado
fig, ax, saved = tornado_plot(
    tornado_data,
    title="STOIIP Sensitivity",
    unit="MM bbl",
    outfile="tornado.png"
)

# With reference case and custom order
fig, ax, saved = tornado_plot(
    tornado_data,
    title="STOIIP Sensitivity",
    base=150.0,
    reference_case=155.0,
    unit="MM bbl",
    preferred_order=["NetPay", "Porosity", "NTG"]
)
```

**Customization:**

```python
custom_settings = {
    'figsize': (12, 8),
    'dpi': 200,
    'pos_light': '#A9CFF7',      # Light blue positive bars
    'neg_light': '#F5B7B1',      # Light red negative bars
    'pos_dark': '#2E5BFF',       # Dark blue P90/P10
    'neg_dark': '#E74C3C',       # Dark red P90/P10
    'show_values': ['min', 'p10', 'p90', 'max'],
    'show_percentage_diff': True,
    'bar_height': 0.6,
}

fig, ax, saved = tornado_plot(
    tornado_data,
    title="Custom Tornado",
    unit="MM bbl",
    settings=custom_settings
)
```

**Key Settings:**
- **Colors**: `pos_light`, `neg_light`, `pos_dark`, `neg_dark`, `baseline_color`, `reference_color`
- **Sizes**: `figsize`, `dpi`, `bar_height`, `bar_linewidth`
- **Labels**: `show_values`, `show_value_headers`, `show_relative_values`, `show_percentage_diff`
- **Fonts**: `title_fontsize`, `label_fontsize`, `value_fontsize`

### Distribution Plot

```python
from tornadopy import distribution_plot

# Basic distribution
fig, ax, saved = distribution_plot(
    dist_data,
    title="Net Pay Distribution",
    unit="MM bbl",
    outfile="distribution.png"
)

# With reference and custom bins
fig, ax, saved = distribution_plot(
    dist_data,
    title="Net Pay Distribution",
    unit="MM bbl",
    reference_case=150.0,
    target_bins=30,
    color="blue"
)
```

**Available Colors:** `"red"`, `"blue"`, `"green"`, `"orange"`, `"purple"`, `"fuchsia"`, `"yellow"`

**Customization:**

```python
custom_settings = {
    'figsize': (12, 7),
    'dpi': 200,
    'bar_color': '#66C3EB',
    'bar_outline_color': '#0075A6',
    'cumulative_color': '#BA2A19',
    'cumulative_linewidth': 3.0,
    'show_percentile_markers': True,
    'target_bins': 25,
}

fig, ax, saved = distribution_plot(
    dist_data,
    title="Custom Distribution",
    unit="MM bbl",
    settings=custom_settings
)
```

## Complete Workflow Example

```python
from tornadopy import TornadoProcessor, tornado_plot, distribution_plot

# 1. Initialize
processor = TornadoProcessor(
    "uncertainty_analysis.xlsx",
    multiplier=1e-6,
    base_case="BaseCases"
)

# 2. Set up filters
processor.set_filters({
    'north_zones': {'zones': ['north_main', 'north_flank']},
    'south_zones': {'zones': ['south_main', 'south_flank']},
    'north_stoiip': {'zones': ['north_main', 'north_flank'], 'property': 'stoiip'}
})

# 3. Generate tornado with case selection
tornado_data = processor.tornado(
    filters='north_zones',
    case_selection=True,
    selection_criteria={'stoiip': 0.6, 'giip': 0.4},
    options={'decimals': 1}
)

fig, ax, saved = tornado_plot(
    tornado_data,
    title="STOIIP Tornado Chart",
    subtitle="North Zone Development",
    unit="MM STB",
    preferred_order=["NetPay", "Porosity", "NTG"],
    outfile="stoiip_tornado.png"
)

# 4. Generate distribution for key parameter
dist_data = processor.distribution(
    parameter="NetPay",
    filters='north_stoiip'
)

fig, ax, saved = distribution_plot(
    dist_data,
    title="Net Pay Impact on STOIIP",
    unit="MM STB",
    reference_case=processor.ref_case('stoiip', filters='north_zones'),
    color="blue",
    outfile="netpay_distribution.png"
)

# 5. Extract representative cases
result = processor.compute(
    stats='p90p10',
    parameter='NetPay',
    filters='north_zones',
    case_selection=True,
    selection_criteria={'north_stoiip': 0.6, 'south_zones': 0.4}
)

print(f"P10 Case: {result['closest_cases'][0]['reference']}")
print(f"P90 Case: {result['closest_cases'][1]['reference']}")
```

## API Quick Reference

### TornadoProcessor

**Initialization:**
```python
TornadoProcessor(filepath, multiplier=1.0, base_case=None)
```

**Data Exploration:**
```python
.parameters()                          # List all parameter names
.properties(parameter=None)            # List properties
.unique_values(field, parameter=None)  # Get unique field values
.case(index, parameter=None)           # Get specific case data
```

**Statistics:**
```python
.compute(stats, parameter=None, filters=None, multiplier=None,
         options=None, case_selection=False, selection_criteria=None)
.compute_batch(stats, parameters='all', filters=None, ...)
.tornado(filters=None, multiplier=None, ...)
.distribution(parameter=None, filters=None, ...)
```

**Base Cases:**
```python
.base_case(property=None, filters=None, multiplier=None)
.ref_case(property=None, filters=None, multiplier=None)
```

**Filters:**
```python
.set_filter(name, filters)      # Store filter preset
.set_filters(filters_dict)      # Store multiple presets
.get_filter(name)               # Retrieve filter
.list_filters()                 # List all filters
```

### Plotting Functions

**tornado_plot:**
```python
tornado_plot(sections, title="Tornado Chart", subtitle=None,
             outfile=None, base=None, reference_case=None,
             unit=None, preferred_order=None, settings=None)
```

**distribution_plot:**
```python
distribution_plot(data, title="Distribution", unit=None,
                  outfile=None, target_bins=20, color="blue",
                  reference_case=None, settings=None)
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

Contributions welcome! Submit a Pull Request at: https://github.com/kkollsga/tornadopy

## Issues

Report issues at: https://github.com/kkollsga/tornadopy/issues

## Author

Kristian dF Kollsgård (kkollsg@gmail.com)