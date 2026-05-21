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

# Active filter — applied to every plot/compute call that doesn't pass `filters=`
# `filter()` is chainable: it sets and returns the dataset.
ds.filter({"contact_regions": ["cerisa main"]})   # set inline
ds.filter("north")                                # set from a stored preset
ds.filter(None)                                   # clear
ds.active_filter                                  # read current

tornado_plot(ds, property="stoiip")                              # uses the active filter
tornado_plot(ds, property="stoiip", filters="south")             # explicit override
distribution_plot(ds.filter({"zone": "x"}), property="stoiip")   # one-liner chain
```

## Default parameter

`distribution_plot` and `correlation_plot` need a `parameter` (sheet). If you
omit it, the first sheet is used and a warning is printed listing all available
parameters. `tornado_plot` does not take `parameter` — a tornado chart is
inherently across all sheets.

## Plot grids

`tornado_plot` and `distribution_plot` render a grid of subplots when you pass a
**list** for `property` (→ columns) and/or the rows (→ rows). Rows can be given
three ways — a list of `filters`, or a list of `FilteredDataset` views, or a
list of `Dataset`s as the first argument. A single dataset/filter with a scalar
`property` still produces one plot.

```python
# Rows from a filter list: filters -> rows, properties -> columns
fig, axes, _ = distribution_plot(
    ds, parameter="Full_Uncertainty",
    property=["stoiip", "giip"],
    filters=["Cerisa Main", "Cerisa West"],
    outfile="dist_grid.png",
)
# axes is a 2-D array: axes[row][col] — axes[0][0] is CMain / STOIIP

# Rows from a list of filtered views (handy when the views already exist)
fig, axes, _ = distribution_plot(
    [ds.filter("Cerisa Main"), ds.filter("Cerisa West")],
    parameter="Full_Uncertainty",
    property="stoiip",
    color=["blue", "red"],          # one colour per row
)

fig, axes, _ = tornado_plot(
    [ds.filter("Cerisa Main"), ds.filter("Cerisa West")],
    property=["stoiip", "giip"],
)
```

Each row is labelled with its filter name, each column with its property name.
Only the bottom row draws x-axis labels and only the left column draws y-labels,
so the subplots stay as large as possible. Grid margins and inter-cell gaps are
sized in inches, so padding stays tight no matter how large the grid is. The
figure auto-sizes to the grid; pass `figsize=(w, h)` for an explicit total size,
or tune the `settings` keys `grid_cell_width` / `grid_cell_height` /
`grid_col_gap` / `grid_row_gap` (the gaps are in inches).

`distribution_plot`'s `color` draws from a 17-family Tailwind-style palette:
`slate`, `zinc`, `stone`, `red`, `orange`, `amber`, `yellow`, `lime`, `green`,
`emerald`, `cyan`, `sky`, `blue`, `violet`, `purple`, `fuchsia`, `rose`. A
colour spec is a family name (`"blue"` — default shade 400), optionally a
shade (`"red-50"` — shades 50–950), and optionally an opacity suffix
(`"red-50:80"` = 80% opaque). Any literal matplotlib colour (hex, or a CSS
name like `"teal"`) also works. Pass a **flat list** for one colour per row,
or a **nested list** `color[row][col]` for per-cell colours — e.g.
`color=[["red-50", "blue"], ["green", "teal"]]`.

Single-plot mode still returns `(fig, ax, saved)`; in grid mode `ax` is the 2-D
array of axes. A row's label comes from a stored-preset name, a `title` key in
an inline filter dict, or a generated fallback. When `ds` is a list, each entry
carries its own filter, so `filters=` must be left as `None`.

## Base / reference cases

```python
ds.base_case()                      # full (unfiltered) base case as a Case
ds.base_case("north")               # volumes summed over the 'north' segments
ds.base_case("north", "stoiip")     # filter, then focus one property
ds.ref_case("north", "stoiip")      # same, for the reference case

bc = ds.base_case("north")
bc.properties()                     # raw m³ volumes — {'stoiip': ..., 'giip': ...}
bc["stoiip"]                        # a single volume (raw m³)
print(bc)                           # formatted for display (mcm/bcm)

ds.filter("north").base_case()      # a FilteredDataset applies its own filter
```

Signature is `base_case(filters=None, property=None)` — `filters` is the first
positional argument (a dict or stored-preset name); `property` is optional.
Called bare, `base_case()` / `ref_case()` return the full unfiltered case.
`.properties()` always returns raw m³ (filtered or not); `print()` applies
display units. The base / reference sheet is set at construction time
(`base_case="Base_case"` by default). Row 0 = base; row 1 = reference.

## Extracting a case by percentile

`extract_case` returns the **`Case`** whose property value is closest to a
percentile or summary statistic. The result is a real realisation from the
sheet — printable, and with variable/metadata access.

```python
# Single case — the realisation nearest the median stoiip
case = ds.extract_case("stoiip", parameter="NTGseed", percentile=50)

print(case)              # Case NTGseed_<idx> (p50) + stoiip, giip, ... + selection info
case.var("NTGseed")      # a $-prefixed variable value
case.variables()         # every variable on the case
case.properties()        # {'stoiip': ..., 'giip': ...}
case.idx, case.type      # row index, "p50"
case.selection_info      # {'selection_values': {'stoiip_target': ..., 'stoiip_actual': ...}, ...}

# Several at once — pass a list, get a list back
p10, p50, p90 = ds.extract_case("stoiip", parameter="NTGseed", percentile=[10, 50, 90])

# Named stats instead of a percentile
hi = ds.extract_case("stoiip", parameter="NTGseed", stat="max")
lo = ds.extract_case("stoiip", parameter="NTGseed", stat=["min", "mean"])
```

`percentile` is the literal percentile (`90` = high value), and the match is
the realisation nearest the interpolated target. `filters` scopes which
segments are summed before ranking. For multi-property weighted selection use
`compute(..., case_selection=True)` instead.

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
