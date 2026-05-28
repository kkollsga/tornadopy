"""JupyterLab interactive widgets for distribution_plot and tornado_plot.

ipywidgets is imported lazily so the package stays importable in non-Jupyter
environments and without the optional dependency.
"""

import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .processor import Dataset, FilteredDataset

# Palette families exposed by ``_colors._PALETTE``. Kept inline so the widget
# stays cheap to instantiate (no import of the colors module at widget-build
# time).
_PALETTE_FAMILIES = (
    "slate", "zinc", "stone", "red", "orange", "amber", "yellow", "lime",
    "green", "emerald", "cyan", "sky", "blue", "violet", "purple", "fuchsia",
    "rose",
)
# Color dropdown options: (label, value). ``None`` keeps the plot's default
# colour scheme rather than tinting from one hue.
_COLOR_OPTIONS = [("default", None)] + [(name, name) for name in _PALETTE_FAMILIES]


def _resolve_dataset(ds: Any) -> Dataset:
    """Pull the underlying Dataset out of a Dataset/FilteredDataset/list."""
    if isinstance(ds, list):
        if not ds:
            raise ValueError("Interactive mode needs at least one dataset.")
        first = ds[0]
        return first.dataset if isinstance(first, FilteredDataset) else first
    if isinstance(ds, FilteredDataset):
        return ds.dataset
    if isinstance(ds, Dataset):
        return ds
    raise TypeError(
        f"Interactive mode expects a Dataset, FilteredDataset, or list; "
        f"got {type(ds).__name__}."
    )


def _import_widgets():
    try:
        import ipywidgets as widgets  # noqa: F401
        from IPython.display import display  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Interactive mode requires ipywidgets. Install with "
            "`pip install ipywidgets` (JupyterLab also needs the corresponding "
            "lab extension on older setups)."
        ) from e
    return widgets, display


def _title_case(text: Any) -> str:
    """Title-case a string and treat underscores as word separators.

    ``contact_regions`` → ``Contact Regions``; ``sand 2`` → ``Sand 2``.
    """
    return str(text).replace("_", " ").title()


def _format_filter_summary(filters: Optional[Dict[str, Any]]) -> str:
    """Render an active-filter dict as a single human-readable line.

    Used as both the plot subtitle and (sanitised) the exported filename.
    Group labels and values are title-cased for display.
    """
    if not filters:
        return ""
    parts = []
    for k, v in filters.items():
        if k in ("title", "name"):
            continue
        key_disp = _title_case(k)
        if isinstance(v, (list, tuple)):
            vals = ", ".join(_title_case(x) for x in v)
            parts.append(f"{key_disp}: {vals}")
        else:
            parts.append(f"{key_disp}: {_title_case(v)}")
    return " | ".join(parts)


def _filename_safe(text: str) -> str:
    """Reduce a string to filesystem-safe characters."""
    cleaned = re.sub(r"[^\w\-]+", "_", text).strip("_")
    return cleaned or "plot"


def build_interactive(
    *,
    ds: Any,
    plot_fn: Callable,
    plot_label: str,
    default_property: Optional[str],
    base_kwargs: Dict[str, Any],
    pick_parameter: bool = True,
) -> Any:
    """Build a JupyterLab widget with property + filter pickers on top and the
    plot below, re-rendering on every change.

    The settings panel ends with a Title text field plus PNG/SVG export
    buttons; exported files are named ``{title}_{filter-summary}.{ext}`` in
    the current working directory.
    """
    widgets, display = _import_widgets()
    import matplotlib.pyplot as plt

    dataset = _resolve_dataset(ds)

    initial_param = base_kwargs.pop("parameter", None)
    # Title is sourced from the input widget below, not from base_kwargs.
    initial_title = base_kwargs.pop("title", None)

    params = dataset.parameters()
    if not params:
        raise ValueError("Dataset has no parameters to plot.")
    if initial_param is None:
        initial_param = next(
            (p for p in params if p != dataset.base_case_parameter), params[0]
        )

    # Defined later so all widgets share the same column width.
    _cell_width = "280px"

    param_widget = None
    if pick_parameter and len(params) > 1:
        param_widget = widgets.Dropdown(
            options=params, value=initial_param, description="Parameter:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width=_cell_width),
        )

    properties = sorted(dataset.properties(initial_param))
    if not properties:
        raise ValueError(f"Parameter '{initial_param}' has no properties.")
    if default_property not in properties:
        default_property = properties[0]
    prop_widget = widgets.Dropdown(
        options=properties, value=default_property, description="Property:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width=_cell_width),
    )

    # Color is pulled into a dropdown so it can be flipped from the widget.
    # ``base_kwargs`` may carry a caller-supplied color; use it as the initial
    # value when it matches a known palette family, else fall back to default.
    initial_color = base_kwargs.pop("color", None)
    color_value = initial_color if initial_color in (None, *_PALETTE_FAMILIES) else None
    color_widget = widgets.Dropdown(
        options=_COLOR_OPTIONS, value=color_value, description="Color:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width=_cell_width),
    )

    field_widgets: Dict[str, Any] = {}

    def _rebuild_field_widgets(parameter: str) -> None:
        field_widgets.clear()
        fields = dataset.dynamic_fields.get(parameter, [])
        # The parser leaves a spurious row for the case column whose values
        # echo the dynamic-field labels themselves (e.g. zones=='zones'). Drop
        # those — they aren't real selections.
        spurious = {f.replace("_", " ").lower() for f in fields} | set(fields)
        for field in fields:
            try:
                values = dataset.unique_values(field, parameter)
            except Exception:
                values = []
            values = [v for v in values if v not in spurious]
            if not values:
                continue
            field_widgets[field] = widgets.SelectMultiple(
                options=values, value=tuple(values),
                description=f"{field}:",
                style={"description_width": "initial"},
                rows=min(6, len(values)),
                layout=widgets.Layout(width=_cell_width),
            )

    _rebuild_field_widgets(initial_param)

    # Title input is one grid cell; the PNG/SVG buttons + status are another.
    # Buttons stack vertically so each one fits in the narrower cell width.
    _export_button_w = "110px"
    title_widget = widgets.Text(
        value=initial_title or "",
        description="Title:",
        placeholder="figure title (and export filename)",
        style={"description_width": "initial"},
        layout=widgets.Layout(width=_cell_width),
    )
    png_button = widgets.Button(
        description="Export PNG", button_style="primary",
        layout=widgets.Layout(width=_export_button_w),
    )
    svg_button = widgets.Button(
        description="Export SVG",
        layout=widgets.Layout(width=_export_button_w),
    )
    status_label = widgets.HTML(
        value="",
        layout=widgets.Layout(width=_cell_width),
    )
    export_cell = widgets.VBox(
        [
            widgets.HBox(
                [png_button, svg_button],
                layout=widgets.Layout(justify_content="flex-start"),
            ),
            status_label,
        ],
        layout=widgets.Layout(width=_cell_width),
    )

    header = widgets.HTML(
        f"<div style='font-weight:600;font-size:13px;margin-bottom:6px;'>"
        f"{plot_label} — interactive filters</div>"
    )
    output = widgets.Output()
    controls_box = widgets.VBox([])

    # Cache the latest render context so exports re-run the same plot with
    # outfile= set, without diverging from what the user is currently seeing.
    _state: Dict[str, Any] = {"filters": None, "title": "", "kwargs": {}}

    def _current_filters_dict() -> Dict[str, Any]:
        filters: Dict[str, Any] = {}
        for field, w in field_widgets.items():
            selected = list(w.value)
            all_opts = list(w.options)
            if selected and len(selected) < len(all_opts):
                filters[field] = selected
        return filters

    def _build_kwargs(active_filters: Dict[str, Any], title: str,
                      outfile: Optional[str] = None) -> Dict[str, Any]:
        parameter = param_widget.value if param_widget is not None else initial_param
        prop = prop_widget.value
        # Attach a 'title' to filters so the active selection bubbles into the
        # plot's subtitle via FilterManager.resolve_filter_title.
        filters_with_label: Optional[Dict[str, Any]] = None
        if active_filters:
            filters_with_label = dict(active_filters)
            summary = _format_filter_summary(active_filters)
            if summary:
                filters_with_label["title"] = summary
        kwargs = {
            **base_kwargs,
            "property": prop,
            "filters": filters_with_label,
            "interactive": False,
        }
        # "default" in the dropdown is mapped to None; only forward an
        # explicit colour so distribution_plot's own default kicks in.
        if color_widget.value is not None:
            kwargs["color"] = color_widget.value
        if pick_parameter:
            kwargs["parameter"] = parameter
        if title:
            kwargs["title"] = title
        if outfile is not None:
            kwargs["outfile"] = outfile
        return kwargs

    def _render(*_: Any) -> None:
        active = _current_filters_dict()
        title = title_widget.value.strip()
        _state["filters"] = active
        _state["title"] = title
        kwargs = _build_kwargs(active, title)
        _state["kwargs"] = kwargs
        with output:
            output.clear_output(wait=True)
            try:
                fig, _ax, _saved = plot_fn(ds, **kwargs)
                display(fig)
                plt.close(fig)
            except Exception as exc:
                msg = str(exc)
                # ``select_columns`` raises a verbose diagnostic when the
                # filter selection matches no columns. Surface a clean
                # "no data" panel in the widget for that case so picking a
                # value that doesn't apply to the current property (e.g.
                # ``facies=undefined`` for a property whose QC balances
                # perfectly) reads as empty rather than as a crash.
                no_data = (
                    "No columns match filters" in msg
                    or "No valid data points" in msg
                    or "No numeric data found" in msg
                    or "No data points remain after clipping" in msg
                )
                if no_data:
                    display(widgets.HTML(
                        "<div style='padding:28px;text-align:center;"
                        "color:#888;font-size:13px;border:1px dashed #ccc;"
                        "border-radius:4px;margin:8px 0;'>"
                        "No data matches the current filter selection."
                        "</div>"
                    ))
                else:
                    display(widgets.HTML(
                        f"<pre style='color:#b00;white-space:pre-wrap'>"
                        f"{type(exc).__name__}: {exc}</pre>"
                    ))

    def _export(ext: str) -> None:
        title_part = _filename_safe(_state["title"] or plot_label.lower())
        filter_summary = _format_filter_summary(_state["filters"])
        filter_part = _filename_safe(filter_summary) if filter_summary else "all"
        fname = f"{title_part}__{filter_part}.{ext}"
        kwargs = _build_kwargs(_state["filters"] or {}, _state["title"],
                               outfile=fname)
        try:
            fig, _ax, saved = plot_fn(ds, **kwargs)
            plt.close(fig)
            full = Path(saved).resolve()
            status_label.value = (
                f"<span style='color:#0a0;margin-left:8px;'>Saved: "
                f"{full.name}</span>"
            )
        except Exception as exc:
            status_label.value = (
                f"<span style='color:#b00;margin-left:8px;'>"
                f"Export failed: {type(exc).__name__}: {exc}</span>"
            )

    def _on_parameter_change(change: Any) -> None:
        new_param = change["new"]
        new_props = sorted(dataset.properties(new_param))
        if new_props:
            keep = prop_widget.value if prop_widget.value in new_props else new_props[0]
            prop_widget.options = new_props
            prop_widget.value = keep
        _rebuild_field_widgets(new_param)
        _wire_fields()
        _refresh_controls()
        _render()

    def _wire_fields() -> None:
        for w in field_widgets.values():
            w.observe(_render, names="value")

    def _refresh_controls() -> None:
        cells: List[Any] = []
        if param_widget is not None:
            cells.append(param_widget)
        cells.append(prop_widget)
        cells.append(color_widget)
        cells.extend(field_widgets.values())
        # Title input and export buttons each occupy their own cell.
        cells.append(title_widget)
        cells.append(export_cell)

        # Arrange cells in rows of 3.
        rows = [
            widgets.HBox(
                cells[i:i + 3],
                layout=widgets.Layout(
                    align_items="flex-start",
                    justify_content="flex-start",
                    margin="2px 0",
                ),
            )
            for i in range(0, len(cells), 3)
        ]
        controls_box.children = tuple(rows)

    if param_widget is not None:
        param_widget.observe(_on_parameter_change, names="value")
    prop_widget.observe(_render, names="value")
    color_widget.observe(_render, names="value")
    title_widget.observe(_render, names="value")
    png_button.on_click(lambda _: _export("png"))
    svg_button.on_click(lambda _: _export("svg"))
    _wire_fields()
    _refresh_controls()
    _render()

    return widgets.VBox([header, controls_box, output])
