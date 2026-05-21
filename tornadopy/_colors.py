"""Shared Tailwind-style colour palette and helpers for the plot modules."""

from typing import Any, Optional, Tuple

import matplotlib.colors as mcolors

# --- 17 families x 11 shades (50..950) ---
_SHADES = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950]
_SHADE_INDEX = {shade: i for i, shade in enumerate(_SHADES)}
_DEFAULT_SHADE_IDX = 4  # shade 400 — used when a bare family name is given

_PALETTE = {
    "slate":   ["#ECEEF0", "#E1E3E7", "#CBCFD5", "#B5BBC3", "#9099A5", "#697585", "#475569", "#353F4E", "#272E39", "#1C222A", "#11151A"],
    "zinc":    ["#EDEDEE", "#E3E3E4", "#CECED1", "#B9B9BD", "#97979C", "#72727A", "#52525B", "#3D3D44", "#2D2D32", "#202024", "#141416"],
    "stone":   ["#EEEDED", "#E5E3E2", "#D1CECD", "#BEB9B7", "#9E9694", "#7C726E", "#5E514D", "#463C39", "#332C2A", "#25201E", "#171413"],
    "red":     ["#FEEBE8", "#FEDFDB", "#FDC7C1", "#FCAFA6", "#FB877A", "#FA5D4B", "#F93822", "#BA2A19", "#881E12", "#63160D", "#3E0E08"],
    "orange":  ["#FFF0E7", "#FFE7D9", "#FFD5BC", "#FFC3A0", "#FFA571", "#FF863F", "#FF6A13", "#BF4F0E", "#8C3A0A", "#662A07", "#3F1A04"],
    "amber":   ["#FDF6E5", "#FCF1D6", "#FBE6B7", "#F9DC99", "#F7CB66", "#F4B930", "#F2A900", "#B57E00", "#855C00", "#604300", "#3C2A00"],
    "yellow":  ["#FEFBE5", "#FEF9D6", "#FEF5B7", "#FEF199", "#FEEA66", "#FEE330", "#FEDD00", "#BEA500", "#8B7900", "#655800", "#3F3700"],
    "lime":    ["#F9FAE5", "#F5F8D6", "#EEF3B7", "#E7EE99", "#DBE666", "#CFDD30", "#C4D600", "#93A000", "#6B7500", "#4E5500", "#313500"],
    "green":   ["#F1F8E8", "#E9F4DB", "#D9ECC0", "#C9E5A5", "#AED879", "#91CA4A", "#78BE20", "#5A8E18", "#426811", "#304C0C", "#1E2F08"],
    "emerald": ["#E9FAF1", "#DCF7EA", "#C2F1DA", "#A8ECCA", "#7CE2B0", "#4FD894", "#26D07C", "#1C9C5D", "#147244", "#0F5331", "#09341F"],
    "cyan":    ["#E5F8FA", "#D6F5F8", "#B7EDF3", "#99E6EE", "#66D9E5", "#30CCDC", "#00C1D5", "#00909F", "#006A75", "#004D55", "#003035"],
    "sky":     ["#E5F5FB", "#D6EFF9", "#B7E3F5", "#99D7F1", "#66C3EB", "#30AEE4", "#009CDE", "#0075A6", "#00557A", "#003E58", "#002737"],
    "blue":    ["#E5F0FA", "#D6E8F7", "#B7D7F1", "#99C6EB", "#66AAE1", "#308CD7", "#0072CE", "#00559A", "#003E71", "#002D52", "#001C33"],
    "violet":  ["#EFEEF9", "#E6E4F6", "#D4D1EF", "#C2BDE8", "#A49CDD", "#847AD1", "#685BC7", "#4E4495", "#39326D", "#29244F", "#1A1631"],
    "purple":  ["#F3EFFA", "#EDE6F7", "#DFD3F1", "#D2C0EB", "#BCA1E1", "#A580D6", "#9063CD", "#6C4A99", "#4F3670", "#392752", "#241833"],
    "fuchsia": ["#F9E9F7", "#F6DBF2", "#EFC1E9", "#E8A7DF", "#DD7BD0", "#D14DBF", "#C724B1", "#951B84", "#6D1361", "#4F0E46", "#31092C"],
    "rose":    ["#FCE5EC", "#FAD6E1", "#F7B7CB", "#F499B5", "#EE6690", "#E93069", "#E40046", "#AB0034", "#7D0026", "#5B001C", "#390011"],
}


def _darken(color: Any, factor: float = 0.6) -> Any:
    """Return a darker version of a colour (RGB channels scaled toward black)."""
    try:
        return tuple(ch * factor for ch in mcolors.to_rgb(color))
    except (ValueError, TypeError):
        return color


def _resolve_color(spec: Any) -> Tuple[Any, Any, Optional[float]]:
    """Resolve a colour spec to a ``(fill, outline, alpha)`` triple.

    Accepted spec forms (case-insensitive):
      - ``'blue'``         — a palette family at its default shade (400)
      - ``'red-50'``       — a palette family at an explicit shade (50..950)
      - ``'#FF8800'`` / ``'teal'`` — any literal matplotlib colour
      - any of the above + ``':N'`` — N% opacity (``'red-50:80'`` -> alpha 0.8)

    The outline is 3 shades darker than the fill. ``alpha`` is ``None`` when
    no ``:N`` suffix is given (the caller then uses its default).
    """
    alpha: Optional[float] = None
    if isinstance(spec, str) and ":" in spec:
        base, _, op_str = spec.rpartition(":")
        try:
            alpha = max(0.0, min(1.0, float(op_str) / 100.0))
            spec = base
        except ValueError:
            pass  # ':' was not an opacity suffix — keep the string intact

    if isinstance(spec, str):
        family, sep, shade_str = spec.strip().lower().partition("-")
        if family in _PALETTE:
            shades = _PALETTE[family]
            idx = _DEFAULT_SHADE_IDX
            if sep and shade_str.isdigit() and int(shade_str) in _SHADE_INDEX:
                idx = _SHADE_INDEX[int(shade_str)]
            fill = shades[idx]
            out_idx = min(idx + 3, len(shades) - 1)
            outline = shades[out_idx] if out_idx != idx else _darken(fill)
            return fill, outline, alpha

    try:
        mcolors.to_rgb(spec)
        return spec, _darken(spec), alpha
    except (ValueError, TypeError):
        shades = _PALETTE["blue"]
        return shades[_DEFAULT_SHADE_IDX], shades[_DEFAULT_SHADE_IDX + 3], alpha


def _tint_pair(spec: Any, light_shade: int, dark_shade: int) -> Tuple[Any, Any]:
    """Resolve a single colour spec to a ``(light, dark)`` tint pair.

    For a palette family, returns the family at the two given shade numbers
    (any ``-shade`` / ``:opacity`` suffix on the spec is ignored — tornado
    derives its own shades). For a literal matplotlib colour, returns the
    colour and a darkened copy of it.
    """
    if isinstance(spec, str):
        family = spec.strip().lower().split(":")[0].split("-")[0]
        if family in _PALETTE:
            shades = _PALETTE[family]
            li = _SHADE_INDEX.get(light_shade, _DEFAULT_SHADE_IDX)
            di = _SHADE_INDEX.get(dark_shade, 6)
            return shades[li], shades[di]
    try:
        mcolors.to_rgb(spec)
        return spec, _darken(spec)
    except (ValueError, TypeError):
        shades = _PALETTE["blue"]
        return (shades[_SHADE_INDEX.get(light_shade, _DEFAULT_SHADE_IDX)],
                shades[_SHADE_INDEX.get(dark_shade, 6)])


def _cell_color(color: Any, r: int, c: int) -> Any:
    """Pick the colour spec for grid cell (row r, column c).

    A scalar applies to every cell; a flat list is one colour per row; a
    nested list is indexed ``color[row][col]`` (per-cell). Indices cycle
    when an axis is shorter than the grid.
    """
    if not isinstance(color, list) or not color:
        return color
    entry = color[r % len(color)]
    if isinstance(entry, list):
        return entry[c % len(entry)] if entry else "blue"
    return entry


def _gap_fraction(gap_in: float, n: int, avail_in: float) -> float:
    """Convert an inter-axes gap in inches to a subplots_adjust w/hspace fraction."""
    if n <= 1:
        return 0.0
    axis_in = (avail_in - gap_in * (n - 1)) / n
    return max(0.02, gap_in / axis_in) if axis_in > 0 else 0.3
