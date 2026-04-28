# TornadoPy

A Python library for fast tornado, distribution, and correlation plots from
uncertainty-analysis results exported from SLB Petrel.

TornadoPy uses Polars for data handling and Matplotlib for publication-quality
charts.

## Installation

```bash
pip install tornadopy
```

## Quick start

```python
from tornadopy import Dataset, tornado_plot, distribution_plot, correlation_plot

# 1. Load the Excel workbook into a dataset
ds = Dataset("uncertainty_results.xlsx")

# 2. (Optional) define reusable filter presets
#    A filter is a spatial selection only — never include 'property' here.
ds.set_filter("north", {"zone": ["north_main", "north_flank"]})

# 3. Plot — the plot function decides which parameter (sheet) and property to use
fig, ax, _ = tornado_plot(
    ds, property="stoiip", filters="north", title="STOIIP sensitivity", unit="MM bbl"
)
fig, ax, _ = distribution_plot(
    ds, parameter="NetPay", property="stoiip", filters="north"
)
fig, ax, _ = correlation_plot(
    ds, parameter="Full_Uncertainty", filters="north",
    variables=["NetPay", "Porosity", "NTG"],
)
```

## API mental model

```
Dataset                   the dataset
  └─ holds: data + filter presets + introspection
  └─ no opinions on which property or sheet to plot

tornado_plot / distribution_plot / correlation_plot
  └─ accept the dataset
  └─ accept property and (where relevant) parameter/sheet
  └─ accept filters as either a stored preset name or an inline dict
```

## Inspecting the dataset

```python
ds.parameters()                       # ['Full_Uncertainty', 'NetPay', ...]
ds.properties("Full_Uncertainty")     # ['stoiip', 'giip', ...]
ds.unique_values("zone", "Full_Uncertainty")

ds.show_filters("Full_Uncertainty")
# {'zone': ['north_main', 'north_flank', ...], 'contact_segment': [...]}

ds.show_parameters()
# {
#   'Full_Uncertainty': {
#       'n_cases': 1854,
#       'properties': ['stoiip', 'giip'],
#       'filters': {'zone': [...], 'contact_segment': [...]},
#       'is_base_case': False,
#   },
#   'NetPay': {...},
# }

ds.describe()                         # Pretty-printed overview + usage examples
```

## Filters

A filter is a dict of dynamic-field selections. The spatial fields (zones,
segments, boundaries) come from your Excel header rows. **The `property` key is
not allowed** — pass property to the plot or compute call instead.

```python
# Inline filter
tornado_plot(ds, property="stoiip", filters={"zone": "north_main"})

# Multiple values aggregate
distribution_plot(
    ds, parameter="NetPay", property="stoiip",
    filters={"zone": ["north_main", "north_flank"]},
)

# Stored presets — reuse by name
ds.set_filters({
    "north": {"zone": ["north_main", "north_flank"]},
    "south": {"zone": ["south_main", "south_flank"]},
})
ds.list_filters()        # ['north', 'south']
ds.get_filter("north")   # {'zone': [...]}
```

## Default parameter

`distribution_plot` and `correlation_plot` need a `parameter` (sheet). If you
omit it, the first sheet is used and a warning is printed listing all available
parameters. `tornado_plot` does not take `parameter` — a tornado chart is
inherently across all sheets.

## Base / reference cases

```python
ds.base_case("stoiip")
ds.base_case("stoiip", filters="north")
ds.ref_case("stoiip", filters="north")
```

The base / reference sheet is set at construction time (`base_case="Base_case"`
by default). Sheet 0 = base; sheet 1 = reference.

## Statistics (raw)

For numerical work without plotting, use `compute` and `compute_batch` directly.
Same rule: `property` is a kwarg, not a filter key.

```python
ds.compute("p90p10", parameter="NetPay", property="stoiip", filters="north")
ds.compute_batch("p90p10", property="stoiip", filters="north")  # all sheets
```

Available stats: `p90p10`, `minmax`, `p1p99`, `p25p75`, `mean`, `median`,
`std`, `cv`, `sum`, `count`, `variance`, `range`, `percentile`
(`options={"p": 75}`), `distribution`.

## Case selection

Find representative cases that best match statistical targets:

```python
fig, ax, _ = tornado_plot(
    ds, property="stoiip", filters="north",
    case_selection=True,
    selection_criteria={"stoiip": 0.6, "giip": 0.4},
)
```

`selection_criteria` keys can be:

- a property name → uses the call's main filter
- a stored-filter name → uses that filter's spatial fields plus its name as the
  property (the `'property'` ban applies; if you need different properties per
  zone set, use the explicit `combinations` form)

```python
ds.set_filter("north", {"zone": ["north_main", "north_flank"]})
ds.set_filter("south", {"zone": ["south_main"]})

tornado_plot(
    ds, property="stoiip", filters="north",
    case_selection=True,
    selection_criteria={
        "combinations": [
            {"filters": "north", "properties": {"stoiip": 0.5, "giip": 0.2}},
            {"filters": "south", "properties": {"stoiip": 0.3}},
        ]
    },
)
```

## Excel layout

Each parameter is one sheet:

```
Metadata rows (optional):
    Key: Value

Header block (one or more rows, combined automatically):
    Zone     Segment   Property
    north    main      stoiip   north  flank  stoiip   south  main  stoiip

Case marker:
    Case     Case      Case     ...

Data rows:
    Case1    123.4     456.7    ...
    Case2    125.1     458.2    ...
```

Rules:

1. The "Case" row's first column is the literal string `Case`.
2. Headers above it define columns; multiple header rows are combined.
3. The data block follows the Case row; one row per uncertainty case.
4. Each parameter is a separate sheet.
5. Base-case sheet (default `"Base_case"`): row 0 = base, row 1 = reference.

## Plot styling

Each plot function accepts a `settings` dict to override defaults — colors,
fonts, gridlines, etc. See the docstrings for keys.

```python
tornado_plot(
    ds, property="stoiip",
    settings={
        "figsize": (12, 8),
        "pos_dark": "#2E5BFF",
        "neg_dark": "#E74C3C",
        "show_percentage_diff": True,
    },
)
```

## Requirements

- Python ≥ 3.9
- numpy ≥ 1.20
- polars ≥ 0.18
- fastexcel ≥ 0.9
- matplotlib ≥ 3.5

## License

MIT — see LICENSE.

## Issues / contributions

https://github.com/kkollsga/tornadopy
