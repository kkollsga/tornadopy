"""JupyterLab interactive widgets for distribution_plot and tornado_plot.

ipywidgets is imported lazily so the package stays importable in non-Jupyter
environments and without the optional dependency.
"""

from typing import Any, Callable, Dict, List, Optional

from .processor import Dataset, FilteredDataset


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

    Args:
        ds: Original first-arg passed to the plot function (Dataset,
            FilteredDataset, list of either, or pandas DataFrame already
            wrapped). The active Dataset is recovered with _resolve_dataset.
        plot_fn: The non-interactive plot callable (distribution_plot or
            tornado_plot).
        plot_label: Human-readable label for the widget header.
        default_property: Property to preselect (auto-detected by the caller).
        base_kwargs: Kwargs to forward to plot_fn on every render (excluding
            property, filters, parameter, interactive).
        pick_parameter: When True and the dataset has more than one parameter,
            adds a parameter dropdown. Tornado uses False (it spans all
            parameters intrinsically).
    """
    widgets, display = _import_widgets()
    import matplotlib.pyplot as plt

    dataset = _resolve_dataset(ds)

    initial_param = base_kwargs.pop("parameter", None)
    params = dataset.parameters()
    if not params:
        raise ValueError("Dataset has no parameters to plot.")
    if initial_param is None:
        initial_param = next(
            (p for p in params if p != dataset.base_case_parameter), params[0]
        )

    param_widget = None
    if pick_parameter and len(params) > 1:
        param_widget = widgets.Dropdown(
            options=params, value=initial_param, description="Parameter:",
            style={"description_width": "initial"},
            layout=widgets.Layout(width="320px"),
        )

    properties = sorted(dataset.properties(initial_param))
    if not properties:
        raise ValueError(f"Parameter '{initial_param}' has no properties.")
    if default_property not in properties:
        default_property = properties[0]
    prop_widget = widgets.Dropdown(
        options=properties, value=default_property, description="Property:",
        style={"description_width": "initial"},
        layout=widgets.Layout(width="320px"),
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
                layout=widgets.Layout(width="320px"),
            )

    _rebuild_field_widgets(initial_param)

    header = widgets.HTML(
        f"<div style='font-weight:600;font-size:13px;margin-bottom:6px;'>"
        f"{plot_label} — interactive filters</div>"
    )
    output = widgets.Output()
    controls_box = widgets.VBox([])

    def _render(*_: Any) -> None:
        parameter = param_widget.value if param_widget is not None else initial_param
        prop = prop_widget.value
        filters: Dict[str, Any] = {}
        for field, w in field_widgets.items():
            selected = list(w.value)
            all_opts = list(w.options)
            # Only filter when the user has narrowed below "all selected".
            if selected and len(selected) < len(all_opts):
                filters[field] = selected
        with output:
            output.clear_output(wait=True)
            kwargs = {
                **base_kwargs,
                "property": prop,
                "filters": filters or None,
                "interactive": False,
            }
            if pick_parameter:
                kwargs["parameter"] = parameter
            try:
                fig, _ax, _saved = plot_fn(ds, **kwargs)
                display(fig)
                plt.close(fig)
            except Exception as exc:  # surface inside the widget rather than tracebacking
                display(widgets.HTML(
                    f"<pre style='color:#b00;white-space:pre-wrap'>"
                    f"{type(exc).__name__}: {exc}</pre>"
                ))

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
        items: List[Any] = []
        if param_widget is not None:
            items.append(param_widget)
        items.append(prop_widget)
        items.extend(field_widgets.values())
        controls_box.children = tuple(items)

    if param_widget is not None:
        param_widget.observe(_on_parameter_change, names="value")
    prop_widget.observe(_render, names="value")
    _wire_fields()
    _refresh_controls()
    _render()

    return widgets.VBox([header, controls_box, output])
