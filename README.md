# TornadoPy

A Python library for tornado chart generation and analysis. TornadoPy provides tools for processing Excel-based tornado data and generating professional tornado charts for uncertainty analysis.

## Features

- **TornadoProcessor**: Process Excel files containing tornado analysis data
  - Parse multi-sheet Excel files with complex headers
  - Extract and compute statistics (p90p10, mean, median, minmax, percentiles)
  - Filter data by properties and dynamic fields
  - Case selection with weighted criteria
  - Batch processing for multiple parameters

- **tornado_plot**: Generate professional tornado charts
  - Customizable colors, fonts, and styling
  - Support for p90/p10 ranges with automatic label placement
  - Reference case lines
  - Custom parameter ordering
  - Export to various image formats

## Installation

### From source

```bash
git clone https://github.com/kkollsga/tornadopy.git
cd tornadopy
pip install -e .
```

### For development

```bash
pip install -e ".[dev]"
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

### Advanced Usage

#### Multi-property Analysis

```python
# Compute statistics for multiple properties
result = processor.compute(
    stats=["p90p10", "mean", "median"],
    parameter="Parameter1",
    filters={"property": ["npv", "irr"]}
)
print(f"NPV P90/P10: {result['p90p10'][0]}")
print(f"IRR P90/P10: {result['p90p10'][1]}")
```

#### Case Selection with Weighted Criteria

```python
# Find closest cases to p90/p10 with custom weights
result = processor.compute(
    stats="p90p10",
    parameter="Parameter1",
    filters={"property": "npv"},
    case_selection=True,
    selection_criteria={
        "weights": {"npv": 0.6, "irr": 0.4}
    }
)

# Access closest cases
for case in result["closest_cases"]:
    print(f"Case {case['case']}: idx={case['idx']}, value={case['npv']}")
```

#### Batch Processing

```python
# Process all parameters at once
results = processor.compute_batch(
    stats="p90p10",
    parameters="all",
    filters={"property": "npv"},
    multiplier=1e-6
)

for result in results:
    print(f"{result['parameter']}: {result['p90p10']}")
```

#### Custom Chart Styling

```python
# Create custom styled tornado chart
settings = {
    "figsize": (12, 8),
    "dpi": 200,
    "pos_dark": "#1E88E5",
    "neg_dark": "#D32F2F",
    "show_values": ["min", "max", "p10", "p90"],
    "show_percentage_diff": True,
}

fig, ax, saved = tornado_plot(
    sections=sections,
    title="Custom Styled Tornado Chart",
    base=100.0,
    reference_case=110.0,
    preferred_order=["Param1", "Param2", "Param3"],
    settings=settings,
    outfile="custom_tornado.png"
)
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

## Requirements

- Python >= 3.8
- numpy >= 1.20.0
- polars >= 0.18.0
- pyxlsb >= 1.0.9
- matplotlib >= 3.5.0

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Support

For issues and questions, please open an issue on GitHub.
