import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
from fastexcel import read_excel


# ================================================================
# UNIT & DISPLAY FORMATTING MANAGER
# ================================================================

class UnitManager:
    """Manages unit conversions, normalization, and display formatting."""
    
    def __init__(self, display_formats: Dict[str, float] = None):
        """Initialize with display format settings.
        
        Args:
            display_formats: Property-specific display multipliers
        """
        self.display_formats: Dict[str, float] = display_formats or {
            'bulk volume': 1e-6,      # mcm
            'net volume': 1e-6,       # mcm
            'pore volume': 1e-6,      # mcm
            'hcpv oil': 1e-6,         # mcm
            'hcpv gas': 1e-6,         # mcm
            'stoiip': 1e-6,           # mcm
            'stoiip (in oil)': 1e-6,  # mcm
            'stoiip (in gas)': 1e-6,  # mcm
            'giip': 1e-9,             # bcm
            'giip (in oil)': 1e-9,    # bcm
            'giip (in gas)': 1e-9,    # bcm
        }
        
        self.property_units: Dict[str, str] = {}
        self.unit_shortnames: Dict[str, str] = {
            '[*10^3 m3]': 'kcm',
            '[*10^3 rm3]': 'kcm',
            '[*10^3 sm3]': 'kcm',
            '[*10^6 sm3]': 'mcm',
            '[*10^6 rm3]': 'mcm',
            '[*10^6 m3]': 'mcm',
            '[*10^9 sm3]': 'bcm',
            '[*10^9 rm3]': 'bcm',
            '[*10^9 m3]': 'bcm',
        }
    
    @lru_cache(maxsize=512)
    def parse_property_unit(self, property_name: str) -> Tuple[str, str]:
        """Parse property name to extract unit suffix."""
        match = re.match(r'^(.+?)(\[.*\])$', property_name.strip())
        if match:
            prop_clean = match.group(1).strip()
            unit = match.group(2)
            return prop_clean, unit
        return property_name, ''
    
    def get_normalization_factor(self, unit: str) -> float:
        """Get normalization factor to convert to base units (m³)."""
        unit_factors = {
            '[*10^3 m3]': 1e3,
            '[*10^3 rm3]': 1e3,
            '[*10^3 sm3]': 1e3,
            '[*10^6 sm3]': 1e6,
            '[*10^6 rm3]': 1e6,
            '[*10^6 m3]': 1e6,
            '[*10^9 sm3]': 1e9,
            '[*10^9 rm3]': 1e9,
            '[*10^9 m3]': 1e9,
        }
        return unit_factors.get(unit, 1.0)
    
    def is_volumetric_property(self, property_name: str) -> bool:
        """Check if property is volumetric and should be normalized."""
        prop_lower = property_name.lower()
        volumetric_keywords = ['volume', 'stoiip', 'giip', 'hcpv']
        return any(keyword in prop_lower for keyword in volumetric_keywords)
    
    def get_display_multiplier(self, property_name: str, override_multiplier: float = None) -> float:
        """Get display multiplier for a property."""
        if override_multiplier is not None:
            return override_multiplier
        
        prop_normalized = property_name.lower().strip()
        return self.display_formats.get(prop_normalized, 1.0)
    
    def get_display_unit(self, property_name: str) -> str:
        """Get display unit string for a property."""
        prop_normalized = property_name.lower().strip()
        multiplier = self.display_formats.get(prop_normalized, 1.0)
        
        if multiplier == 1e-6:
            return 'mcm'
        elif multiplier == 1e-9:
            return 'bcm'
        elif multiplier == 1e-3:
            return 'kcm'
        else:
            return ''
    
    def format_for_display(
        self, 
        property_name: str, 
        value: float, 
        decimals: int = 6, 
        override_multiplier: float = None
    ) -> float:
        """Format value for display using property-specific multiplier."""
        multiplier = self.get_display_multiplier(property_name, override_multiplier)
        return self._to_float(value * multiplier, decimals)
    
    def _to_float(self, value: Any, decimals: int = None) -> float:
        """Convert value to native Python float with optional rounding."""
        if value is None:
            return None
        val = float(value)
        return round(val, decimals) if decimals is not None else val
    
    def set_display_format(self, property: str, unit: str = 'mcm') -> None:
        """Set display format for a property."""
        unit_map = {'kcm': 1e-3, 'mcm': 1e-6, 'bcm': 1e-9}
        if unit not in unit_map:
            raise ValueError(f"Unit must be 'kcm', 'mcm', or 'bcm', got: {unit}")
        self.display_formats[property.lower()] = unit_map[unit]
    
    def get_property_units(self) -> Dict[str, str]:
        """Get dictionary of property-to-unit mappings."""
        return {
            prop: self.unit_shortnames.get(unit, unit)
            for prop, unit in self.property_units.items()
        }
    
    def get_normalization_info(self) -> Dict[str, Dict[str, Any]]:
        """Get normalization information for all properties."""
        result = {}
        
        for prop, original_unit in self.property_units.items():
            factor = self.get_normalization_factor(original_unit)
            unit_short = self.unit_shortnames.get(original_unit, original_unit)
            was_normalized = self.is_volumetric_property(prop) and factor != 1.0
            
            result[prop] = {
                'original_unit': original_unit,
                'unit_short': unit_short,
                'factor': factor,
                'was_normalized': was_normalized
            }
        
        return result


# ================================================================
# FILTER MANAGER
# ================================================================

class FilterManager:
    """Manages filter presets and resolution."""
    
    def __init__(self):
        self.stored_filters: Dict[str, Dict[str, Any]] = {}
    
    @staticmethod
    def normalize_property_for_matching(property_name: str) -> str:
        """Normalize property name for matching."""
        return property_name.lower().replace('(', '').replace(')', '').replace(' ', '-')
    
    def resolve_filter_preset(
        self, 
        filters: Union[Dict[str, Any], str],
        available_properties: List[str] = None
    ) -> Dict[str, Any]:
        """Resolve filter preset if string, otherwise return dict as-is."""
        if isinstance(filters, str):
            if '_' in filters:
                parts = filters.rsplit('_', 1)
                if len(parts) == 2:
                    base_filter_name, property_part = parts
                    
                    if (base_filter_name in self.stored_filters and 
                        property_part not in self.stored_filters):
                        
                        normalized_input = self.normalize_property_for_matching(property_part)
                        matched_property = property_part.replace('-', ' ')
                        
                        if available_properties:
                            for prop in available_properties:
                                normalized_prop = self.normalize_property_for_matching(prop)
                                if normalized_prop == normalized_input:
                                    matched_property = prop
                                    break
                        
                        base_filters = self.stored_filters[base_filter_name].copy()
                        base_filters['property'] = matched_property
                        return base_filters
            
            return self.get_filter(filters)
        
        return filters if filters is not None else {}
    
    def set_filter(self, name: str, filters: Dict[str, Any]) -> None:
        """Store a named filter preset for reuse."""
        self.stored_filters[name] = filters
    
    def set_filters(self, filters_dict: Dict[str, Dict[str, Any]]) -> None:
        """Store multiple named filter presets at once."""
        self.stored_filters.update(filters_dict)
    
    def get_filter(self, name: str) -> Dict[str, Any]:
        """Retrieve a stored filter preset."""
        if name not in self.stored_filters:
            raise KeyError(
                f"Filter preset '{name}' not found. "
                f"Available: {list(self.stored_filters.keys())}"
            )
        return self.stored_filters[name]
    
    def list_filters(self) -> List[str]:
        """List all stored filter preset names."""
        return list(self.stored_filters.keys())
    
    @staticmethod
    def merge_property_filter(
        filters: Dict[str, Any], 
        property: Union[str, List[str], bool, None]
    ) -> Dict[str, Any]:
        """Merge property parameter with filters dict."""
        merged = dict(filters) if filters else {}
        
        if property is None:
            return merged
        elif property is False:
            merged.pop('property', None)
            return merged
        else:
            merged['property'] = property
            return merged


# ================================================================
# EXCEL DATA LOADER
# ================================================================

class ExcelDataLoader:
    """Handles loading and parsing Excel files."""
    
    def __init__(self, unit_manager: UnitManager):
        self.unit_manager = unit_manager
    
    @staticmethod
    def load_sheets(filepath: Path) -> Dict[str, pl.DataFrame]:
        """Load all sheets from Excel file into Polars DataFrames."""
        sheets = {}
        excel_file = read_excel(str(filepath))
        
        for sheet_name in excel_file.sheet_names:
            df = excel_file.load_sheet_by_name(
                sheet_name,
                header_row=None,
                skip_rows=0
            ).to_polars()
            sheets[sheet_name] = df
        
        return sheets
    
    @staticmethod
    @lru_cache(maxsize=256)
    def normalize_fieldname(name: str) -> str:
        """Normalize field name to lowercase with underscores."""
        name = str(name).strip().lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        name = re.sub(r"_+$", "", name)
        return name or "property"
    
    @staticmethod
    @lru_cache(maxsize=512)
    def strip_units(property_name: str) -> str:
        """Strip unit annotations from property name."""
        cleaned = re.sub(r'\s*\[.*?\]\s*$', '', property_name)
        return cleaned.strip()
    
    def parse_sheet(
        self, 
        df: pl.DataFrame, 
        sheet_name: str = "unknown"
    ) -> Tuple[pl.DataFrame, pl.DataFrame, List[str], Dict]:
        """Parse individual sheet into data, metadata, dynamic fields, and info."""
        # Find "Case" row
        case_mask = df.select(
            pl.col(df.columns[0]).cast(pl.Utf8).str.strip_chars() == "Case"
        ).to_series()
        
        case_row_idx = case_mask.arg_true().to_list()
        if not case_row_idx:
            raise ValueError("No 'Case' row found in sheet")
        case_row = case_row_idx[0]
        
        # Extract metadata from rows above headers
        info_dict = {}
        if case_row > 0:
            info_block = df.slice(0, case_row)
            for row in info_block.iter_rows():
                key = str(row[0]).strip() if row[0] is not None else ""
                if key and key.lower() != "case":
                    values = [str(v).strip() for v in row[1:] if v is not None and str(v).strip()]
                    if values:
                        info_dict[key] = " ".join(values)
        
        # Find header start
        header_start = case_row - 1
        while header_start > 0:
            val = df[df.columns[0]][header_start]
            if val is None or str(val).strip() == "":
                break
            header_start -= 1
        
        header_block = df.slice(header_start, case_row - header_start + 1)
        data_block = df.slice(case_row + 1)
        
        # Extract dynamic field labels
        dynamic_labels = []
        for i in range(len(header_block) - 1):
            val = header_block[header_block.columns[0]][i]
            if val is not None and str(val).strip():
                dynamic_labels.append(self.normalize_fieldname(val))
        
        if not dynamic_labels:
            dynamic_labels = ["property"]
        
        # Build combined column headers
        header_rows = header_block.to_numpy().tolist()
        combined_headers = []
        
        for col_idx in range(len(header_rows[0])):
            labels = []
            for row in header_rows:
                val = row[col_idx]
                if val is not None and str(val).strip():
                    labels.append(str(val).strip())
            combined_headers.append("_".join(labels) if labels else "")
        
        if len(set(combined_headers)) < len(combined_headers):
            raise ValueError("Duplicate column headers detected")
        
        data_block.columns = combined_headers
        data_block = data_block.select([
            col for col in data_block.columns 
            if col and not col.startswith("_")
        ])
        
        if "Case" in data_block.columns:
            data_block = data_block.rename({"Case": "property"})
        
        # Build column metadata table
        metadata_rows = []
        for idx, col_name in enumerate(data_block.columns):
            if col_name.startswith("$") or col_name.lower().startswith("property"):
                continue
            
            parts = col_name.split("_")
            property_name_raw = parts[-1] if parts else col_name
            
            property_name, unit = self.unit_manager.parse_property_unit(property_name_raw)
            
            meta = {
                "column_name": col_name,
                "column_index": idx,
                "property": property_name.strip().lower()
            }
            
            for field_idx, field_name in enumerate(dynamic_labels):
                if field_idx < len(parts) - 1:
                    meta[field_name] = parts[field_idx].strip().lower()
                else:
                    meta[field_name] = None
            
            metadata_rows.append(meta)
        
        metadata_df = pl.DataFrame(metadata_rows) if metadata_rows else pl.DataFrame()
        
        return data_block, metadata_df, dynamic_labels, info_dict
    
    def extract_sheet_property_units(self, metadata: pl.DataFrame) -> Dict[str, str]:
        """Extract property->unit mapping from a sheet's metadata."""
        property_units = {}
        
        for row in metadata.iter_rows(named=True):
            col_name = row['column_name']
            prop = row['property']
            
            parts = col_name.split("_")
            property_name_raw = parts[-1] if parts else col_name
            _, unit = self.unit_manager.parse_property_unit(property_name_raw)
            
            if prop not in property_units and unit:
                property_units[prop] = unit
        
        return property_units
    
    def normalize_data_values(
        self, 
        df: pl.DataFrame, 
        metadata: pl.DataFrame, 
        sheet_name: str
    ) -> pl.DataFrame:
        """Normalize volumetric values to base units (m³)."""
        if metadata.is_empty():
            return df
        
        for row in metadata.iter_rows(named=True):
            col_name = row['column_name']
            property_name = row['property']
            
            if not self.unit_manager.is_volumetric_property(property_name):
                continue
            
            if col_name not in df.columns:
                continue
            
            if property_name in self.unit_manager.property_units:
                unit = self.unit_manager.property_units[property_name]
                factor = self.unit_manager.get_normalization_factor(unit)
                
                if factor != 1.0:
                    try:
                        df = df.with_columns(
                            (pl.col(col_name).cast(pl.Float64, strict=False) * factor).alias(col_name)
                        )
                    except Exception as e:
                        print(f"[!] Warning: Could not normalize column '{col_name}' in sheet '{sheet_name}': {e}")
        
        return df


# ================================================================
# DATA EXTRACTOR
# ================================================================

class DataExtractor:
    """Handles data extraction, column selection, and validation."""
    
    def __init__(self, unit_manager: UnitManager):
        self.unit_manager = unit_manager
        self._extraction_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        self._column_selection_cache: Dict[str, Tuple[List[str], List[str]]] = {}
    
    @staticmethod
    @lru_cache(maxsize=512)
    def normalize_filters_cached(filters_tuple: tuple) -> tuple:
        """Cached version of normalize_filters that works with tuples."""
        filters = dict(filters_tuple)
        normalized = {}
        
        for key, value in filters.items():
            key_norm = ExcelDataLoader.normalize_fieldname(key)
            
            if isinstance(value, str):
                value_norm = value.strip().lower()
            elif isinstance(value, list):
                value_norm = tuple(v.strip().lower() if isinstance(v, str) else v for v in value)
            else:
                value_norm = value
            
            normalized[key_norm] = value_norm
        
        return tuple(sorted(normalized.items()))
    
    def normalize_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize filter keys and string values to lowercase."""
        if not filters:
            return {}
        
        filters_tuple = tuple(sorted(filters.items()))
        normalized_tuple = self.normalize_filters_cached(filters_tuple)
        
        result = {}
        for key, value in normalized_tuple:
            if isinstance(value, tuple) and key in filters and isinstance(filters[key], list):
                result[key] = list(value)
            else:
                result[key] = value
        
        return result
    
    def select_columns(
        self,
        metadata: pl.DataFrame,
        dynamic_fields: List[str],
        filters: Dict[str, Any],
        parameter: str,
        cache_key: str
    ) -> Tuple[List[str], List[str]]:
        """Select columns matching filters and return column names and sources."""
        if cache_key in self._column_selection_cache:
            return self._column_selection_cache[cache_key]
        
        if metadata.is_empty():
            result = ([], [])
            self._column_selection_cache[cache_key] = result
            return result
        
        filters_norm = self.normalize_filters(filters)
        
        mask = pl.lit(True)
        
        for field, value in filters_norm.items():
            if value is None:
                continue
            
            if field not in metadata.columns:
                raise ValueError(
                    f"Field '{field}' not available. "
                    f"Available: {dynamic_fields}"
                )
            
            if isinstance(value, list):
                mask = mask & pl.col(field).is_in(value)
            else:
                mask = mask & (pl.col(field) == value)
        
        matched = metadata.filter(mask)
        
        if matched.is_empty():
            filter_desc = ", ".join(f"{k}={v}" for k, v in filters_norm.items())
            raise ValueError(f"No columns match filters: {filter_desc}")
        
        column_names = matched.select("column_name").to_series().to_list()
        result = (column_names, column_names)
        
        self._column_selection_cache[cache_key] = result
        return result
    
    def extract_values(
        self,
        data_df: pl.DataFrame,
        metadata: pl.DataFrame,
        dynamic_fields: List[str],
        parameter: str,
        filters: Dict[str, Any],
        cache_key: str
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract and sum values for columns matching filters."""
        if cache_key in self._extraction_cache:
            cached_values, cached_sources = self._extraction_cache[cache_key]
            return cached_values.copy(), cached_sources.copy()
        
        list_fields = {k: v for k, v in filters.items() if isinstance(v, list)}
        
        if list_fields:
            n_rows = len(data_df)
            combined = np.zeros(n_rows, dtype=np.float64)
            all_sources = []
            
            for field, values in list_fields.items():
                for value in values:
                    single_filters = {**filters, field: value}
                    cols, sources = self.select_columns(
                        metadata, dynamic_fields, single_filters, parameter, 
                        cache_key + f"_{field}_{value}"
                    )
                    
                    arr = (
                        data_df.select(cols)
                        .select(pl.sum_horizontal(pl.all().cast(pl.Float64, strict=False)))
                        .to_series()
                        .to_numpy()
                    )
                    combined += arr
                    all_sources.extend(sources)
            
            result = (combined, all_sources)
        else:
            cols, sources = self.select_columns(
                metadata, dynamic_fields, filters, parameter, cache_key
            )
            
            values = (
                data_df.select(cols)
                .select(pl.sum_horizontal(pl.all().cast(pl.Float64, strict=False)))
                .to_series()
                .to_numpy()
            )
            
            result = (values, sources)
        
        self._extraction_cache[cache_key] = result
        return result[0].copy(), result[1].copy()
    
    @staticmethod
    def validate_numeric(values: np.ndarray, description: str) -> np.ndarray:
        """Validate array contains finite numeric values."""
        if values.size == 0 or not np.isfinite(values).any():
            raise ValueError(f"No numeric data found for {description}")
        
        return values[np.isfinite(values)]
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear all caches and return statistics."""
        stats = {
            'extraction_cache': len(self._extraction_cache),
            'column_selection_cache': len(self._column_selection_cache),
        }
        
        self._extraction_cache.clear()
        self._column_selection_cache.clear()
        self.normalize_filters_cached.cache_clear()
        
        return stats


# ================================================================
# CASE CLASS
# ================================================================

class Case:
    """Represents a single case from a tornado analysis."""
    
    def __init__(
        self,
        data: Dict[str, Any],
        processor: 'TornadoProcessor',
        index: int = None,
        parameter: str = None,
        reference: str = None,
        case_type: str = None,
        selection_info: Dict[str, Any] = None
    ):
        """Initialize a Case object."""
        self._data = data
        self._processor = processor
        
        self.idx = index if index is not None else data.get('idx')
        self.tornado_parameter = parameter
        self.ref = reference
        self._selection_info = selection_info or {}
        
        if case_type:
            self.type = case_type
        elif 'case' in data:
            self.type = data['case']
        elif reference and '.' in reference:
            self.type = reference.split('.')[0]
        else:
            self.type = None
        
        self._properties = data.get('properties', {})
        self._variables = data.get('variables', {})
        
        if not self._properties:
            self._properties = {
                k: v for k, v in data.items()
                if not k.startswith('_') and k not in ['idx', 'parameter', 'case', 'multiplier', 'variables', 'properties']
            }
    
    @property
    def selection_info(self) -> Dict[str, Any]:
        """Get selection information if this case was selected via case_selection."""
        return self._selection_info.copy()
    
    def __repr__(self) -> str:
        """String representation of the case."""
        if self.ref:
            return f"Case({self.ref})"
        elif self.tornado_parameter and self.idx is not None:
            return f"Case({self.tornado_parameter}_{self.idx})"
        else:
            return f"Case(idx={self.idx})"
    
    def __str__(self) -> str:
        """Human-readable string representation with display formatting."""
        lines = []
        
        header = f"Case {self.ref if self.ref else f'{self.tornado_parameter}_{self.idx}'}"
        if self.type:
            header += f" ({self.type})"
        lines.append(header)
        lines.append("-" * len(header))
        
        numeric_props = {
            k: v for k, v in self._properties.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        
        if numeric_props:
            lines.append("")
            for prop, val in sorted(numeric_props.items())[:15]:
                display_val = self._processor.unit_manager.format_for_display(prop, val, decimals=2)
                unit = self._processor.unit_manager.get_display_unit(prop)
                lines.append(f"  {prop:.<30} {display_val:>12,.2f} {unit}")
            if len(numeric_props) > 15:
                lines.append(f"  ... {len(numeric_props) - 15} more properties")
        
        if self._selection_info:
            lines.append("")
            lines.append("Selection Info:")
            if 'selection_method' in self._selection_info:
                lines.append(f"  Method: {self._selection_info['selection_method']}")
            if 'weighted_distance' in self._selection_info:
                lines.append(f"  Distance: {self._selection_info['weighted_distance']:.4f}")
            if 'weights' in self._selection_info:
                lines.append(f"  Weights: {self._selection_info['weights']}")
            if 'selection_values' in self._selection_info and self._selection_info['selection_values']:
                lines.append("  Selection values:")
                for key, val in self._selection_info['selection_values'].items():
                    if isinstance(val, (int, float)):
                        lines.append(f"    {key}: {val:,.2f}")
                    else:
                        lines.append(f"    {key}: {val}")
        
        if self._variables and len(self._variables) <= 10:
            lines.append("")
            lines.append("Variables:")
            for var, val in list(self._variables.items())[:10]:
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    lines.append(f"  {var:.<30} {val:>12,.2f}")
                else:
                    lines.append(f"  {var:.<30} {val}")
        
        return "\n".join(lines)
    
    def __call__(
        self, 
        filter_name: str, 
        property: Union[str, List[str], None] = None,
        selection: bool = False
    ) -> None:
        """Print filtered results without modifying the case."""
        filters = self._processor.filter_manager.resolve_filter_preset(filter_name)
        
        if property is not None:
            filters = filters.copy()
            filters['property'] = property
        
        filtered_case = self._processor.case_manager.get_case(
            self.idx,
            parameter=self.tornado_parameter,
            filters=filters,
            as_dict=True
        )
        
        lines = []
        
        header = f"Case {self.ref if self.ref else f'{self.tornado_parameter}_{self.idx}'}"
        if self.type:
            header += f" ({self.type})"
        lines.append(header)
        lines.append("-" * len(header))
        
        lines.append("")
        lines.append(f"Filter: {filter_name}")
        
        if 'properties' in filtered_case:
            props = filtered_case['properties']
            
            if isinstance(props, dict):
                lines.append("")
                
                def flatten_props(d, prefix=''):
                    items = []
                    for k, v in d.items():
                        if isinstance(v, dict):
                            items.extend(flatten_props(v, f"{prefix}{k}."))
                        elif isinstance(v, (int, float)) and not isinstance(v, bool):
                            items.append((f"{prefix}{k}", v))
                    return items
                
                flat_props = flatten_props(props)
                
                for prop_name, val in flat_props[:15]:
                    base_prop = prop_name.split('.')[-1]
                    unit = self._processor.unit_manager.get_display_unit(base_prop)
                    lines.append(f"  {prop_name:.<35} {val:>12,.2f} {unit}")
                
                if len(flat_props) > 15:
                    lines.append(f"  ... {len(flat_props) - 15} more properties")
        
        if selection and self._selection_info:
            lines.append("")
            lines.append("Selection Info:")
            if 'selection_method' in self._selection_info:
                lines.append(f"  Method: {self._selection_info['selection_method']}")
            if 'weighted_distance' in self._selection_info:
                lines.append(f"  Distance: {self._selection_info['weighted_distance']:.4f}")
            if 'weights' in self._selection_info:
                lines.append(f"  Weights: {self._selection_info['weights']}")
            if 'selection_values' in self._selection_info and self._selection_info['selection_values']:
                lines.append("  Selection values:")
                for key, val in self._selection_info['selection_values'].items():
                    if isinstance(val, (int, float)):
                        lines.append(f"    {key}: {val:,.2f}")
                    else:
                        lines.append(f"    {key}: {val}")
        
        print("\n".join(lines))
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to properties."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if name in self._properties:
            return self._properties[name]
        
        if name in self._data:
            return self._data[name]
        
        raise AttributeError(f"Case has no property '{name}'. Available: {list(self._properties.keys())}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to properties."""
        key_lower = key.lower()
        
        if key_lower in self._properties:
            return self._properties[key_lower]
        
        if key in self._data:
            return self._data[key]
        
        raise KeyError(f"Case has no property '{key}'. Available: {list(self._properties.keys())}")
    
    def __contains__(self, key: str) -> bool:
        """Check if property exists."""
        key_lower = key.lower()
        return key_lower in self._properties or key in self._data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get property value with optional default."""
        try:
            return self[key]
        except KeyError:
            return default
    
    def properties(self, flat: bool = False) -> Dict[str, Any]:
        """Get all properties."""
        if flat and isinstance(self._properties, dict):
            def flatten(d, prefix=''):
                items = []
                for k, v in d.items():
                    new_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        items.extend(flatten(v, new_key))
                    else:
                        items.append((new_key, v))
                return items
            
            return dict(flatten(self._properties))
        
        return self._properties.copy()
    
    def var(self, name: str, default: Any = None) -> Any:
        """Get a variable value by name."""
        name = name.lstrip('$')
        return self._variables.get(name, default)
    
    def variables(
        self,
        names: Union[List[str], str, None] = None,
        use_defaults: bool = True
    ) -> Dict[str, Any]:
        """Get variables with optional filtering."""
        if names is None and use_defaults:
            names = self._processor.default_variables
        
        if names is None:
            return self._variables.copy()
        
        if isinstance(names, str):
            names = [names]
        
        result = {}
        for name in names:
            name_clean = name.lstrip('$')
            if name_clean in self._variables:
                result[name_clean] = self._variables[name_clean]
        
        return result
    
    def parameters(
        self, 
        decimals: int = None,
        filters: Union[Dict[str, Any], str] = None
    ) -> Dict[str, float]:
        """Calculate derived parameters from volumetric properties.
        
        Uses raw base unit values (m³) for all calculations.
        GRV is converted using display formatting (e.g., mcm).
        
        Args:
            decimals: Number of decimal places (None for unlimited/full precision)
            filters: Optional filters to apply when extracting properties.
                    Can be a filter name string or a filters dict.
                    If provided, recalculates using filtered data for this case.
        
        Returns:
            Dictionary with calculated parameters:
            - GRV: Gross rock volume (display units, e.g., mcm)
            - NTG: Net-to-Gross ratio
            - Por: Porosity
            - So: Oil saturation
            - Sg: Gas saturation
            - Bo: Oil formation volume factor (rm³/sm³)
            - Bg: Gas formation volume factor (rm³/sm³)
            - Rs: Solution gas-oil ratio (sm³/sm³)
            - Rv: Vaporized oil-gas ratio (sm³/sm³)
        """
        # If filters provided, get filtered data for this case
        if filters is not None:
            filters = self._processor.filter_manager.resolve_filter_preset(filters)
            
            # Get all properties we need
            property_names = [
                'bulk volume', 'net volume', 'pore volume',
                'hcpv oil', 'hcpv gas',
                'stoiip (in oil)', 'stoiip (in gas)',
                'giip (in oil)', 'giip (in gas)'
            ]
            
            props = {}
            non_property_filters = {k: v for k, v in filters.items() if k != 'property'}
            
            for prop_name in property_names:
                try:
                    prop_filters = {**non_property_filters, 'property': prop_name}
                    cache_key = self._processor._create_cache_key(
                        self.tornado_parameter, prop_filters
                    )
                    values, _ = self._processor.data_extractor.extract_values(
                        self._processor.data[self.tornado_parameter],
                        self._processor.metadata[self.tornado_parameter],
                        self._processor.dynamic_fields[self.tornado_parameter],
                        self.tornado_parameter,
                        prop_filters,
                        cache_key
                    )
                    if self.idx < len(values):
                        props[prop_name] = float(values[self.idx])
                    else:
                        props[prop_name] = 0.0
                except:
                    props[prop_name] = 0.0
        else:
            # Use existing properties from the case
            props = self._properties
        
        params = {}
        
        def get_prop(name):
            name_norm = name.lower().strip()
            return props.get(name_norm, 0.0)
        
        bulk_vol = get_prop('bulk volume')
        net_vol = get_prop('net volume')
        pore_vol = get_prop('pore volume')
        hcpv_oil = get_prop('hcpv oil')
        hcpv_gas = get_prop('hcpv gas')
        stoiip_oil = get_prop('stoiip (in oil)')
        stoiip_gas = get_prop('stoiip (in gas)')
        giip_oil = get_prop('giip (in oil)')
        giip_gas = get_prop('giip (in gas)')
        
        params['GRV'] = self._processor.unit_manager.format_for_display(
            'bulk volume', bulk_vol, decimals=decimals
        )
        
        params['NTG'] = net_vol / bulk_vol if bulk_vol > 0 else 0.0
        params['Por'] = pore_vol / net_vol if net_vol > 0 else 0.0
        params['So'] = hcpv_oil / pore_vol if pore_vol > 0 else 0.0
        params['Sg'] = hcpv_gas / pore_vol if pore_vol > 0 else 0.0
        params['Bo'] = hcpv_oil / stoiip_oil if stoiip_oil > 0 else 0.0
        params['Bg'] = hcpv_gas / giip_gas if giip_gas > 0 else 0.0
        params['Rs'] = giip_oil / stoiip_oil if stoiip_oil > 0 else 0.0
        params['Rv'] = stoiip_gas / giip_gas if giip_gas > 0 else 0.0
        
        if decimals is not None:
            for k in ['NTG', 'Por', 'So', 'Sg', 'Bo', 'Bg', 'Rs', 'Rv']:
                if k in params and isinstance(params[k], float):
                    params[k] = round(params[k], decimals)
        
        return params
    
    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Convert case to dictionary."""
        result = {}
        
        if include_metadata:
            if self.idx is not None:
                result['idx'] = self.idx
            if self.tornado_parameter:
                result['parameter'] = self.tornado_parameter
            if self.type:
                result['case'] = self.type
            if self.ref:
                result['reference'] = self.ref
        
        result['properties'] = self._properties.copy()
        
        if self._variables:
            result['variables'] = self._variables.copy()
        
        if self._selection_info:
            result['selection_info'] = self._selection_info.copy()
        
        return result


# ================================================================
# CASE MANAGER
# ================================================================

class CaseManager:
    """Handles case creation, references, and base/reference cases."""
    
    def __init__(
        self,
        processor: 'TornadoProcessor',
        unit_manager: UnitManager,
        data_extractor: DataExtractor
    ):
        self.processor = processor
        self.unit_manager = unit_manager
        self.data_extractor = data_extractor
        self.base_case_values: Dict[str, float] = {}
        self.reference_case_values: Dict[str, float] = {}
    
    @staticmethod
    def create_case_reference(parameter: str, index: int, tag: str = None) -> str:
        """Create a unique reference string for a case."""
        base_ref = f"{parameter}_{index}"
        return f"{tag}.{base_ref}" if tag else base_ref
    
    @staticmethod
    def parse_case_reference(reference: str) -> Tuple[str, int, str]:
        """Parse a case reference string."""
        tag = None
        if '.' in reference:
            tag, reference = reference.split('.', 1)
        
        last_underscore = reference.rfind('_')
        if last_underscore == -1:
            raise ValueError(f"Invalid case reference format: {reference}")
        
        parameter = reference[:last_underscore]
        try:
            index = int(reference[last_underscore + 1:])
        except ValueError:
            raise ValueError(
                f"Invalid case reference format: {reference} "
                "(index portion must be numeric)"
            )
        
        return parameter, index, tag
    
    def extract_case(
        self,
        parameter: str,
        case_index: int,
        filters: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Extract values for a specific case index."""
        if parameter not in self.processor.data:
            return {}
        
        case_df = self.processor.data[parameter]
        if len(case_df) <= case_index:
            return {}
        
        try:
            properties = self.processor.properties(parameter)
        except:
            properties = []
        
        base_filters = dict(filters) if filters else {}
        base_filters.pop("property", None)
        
        case_values = {}
        for prop in properties:
            try:
                prop_filters = {**base_filters, "property": prop}
                cache_key = self.processor._create_cache_key(parameter, prop_filters)
                values, _ = self.data_extractor.extract_values(
                    self.processor.data[parameter],
                    self.processor.metadata[parameter],
                    self.processor.dynamic_fields[parameter],
                    parameter,
                    prop_filters,
                    cache_key
                )
                if len(values) > case_index:
                    case_values[prop] = float(values[case_index])
            except:
                pass
        
        return case_values
    
    def extract_base_and_reference_cases(
        self,
        base_case_parameter: str,
        filters: Dict[str, Any] = None
    ):
        """Extract and cache base case (index 0) and reference case (index 1)."""
        if not base_case_parameter:
            return
        
        self.base_case_values = self.extract_case(
            base_case_parameter, 
            case_index=0, 
            filters=filters
        )
        
        self.reference_case_values = self.extract_case(
            base_case_parameter,
            case_index=1,
            filters=filters
        )
    
    def get_case_values(
        self,
        case_type: str,
        base_case_parameter: str,
        property: str = None,
        filters: Union[Dict[str, Any], str] = None,
        multiplier: float = None
    ) -> Union[float, Dict[str, float]]:
        """Get base or reference case values."""
        if filters and 'property' in filters:
            filters = filters.copy()
            property = filters.pop('property')
        
        if property:
            property = property.lower().strip()
        
        if filters:
            case_index = 0 if case_type == 'base' else 1
            case_values = self.extract_case(
                base_case_parameter,
                case_index=case_index,
                filters=filters
            )
        else:
            case_values = self.base_case_values if case_type == 'base' else self.reference_case_values
        
        if property:
            normalized_map = {
                FilterManager.normalize_property_for_matching(k): k 
                for k in case_values.keys()
            }
            normalized_property = FilterManager.normalize_property_for_matching(property)
            
            if normalized_property not in normalized_map:
                raise KeyError(
                    f"Property '{property}' not found in case. "
                    f"Available: {list(case_values.keys())}"
                )
            
            actual_property = normalized_map[normalized_property]
            raw_value = case_values[actual_property]
            return self.unit_manager.format_for_display(
                actual_property, raw_value, decimals=6, override_multiplier=multiplier
            )
        
        formatted = {}
        for prop, raw_value in case_values.items():
            formatted[prop] = self.unit_manager.format_for_display(
                prop, raw_value, decimals=6, override_multiplier=multiplier
            )
        
        return formatted
    
    def parse_case_to_hierarchy(
        self,
        case_data: Dict,
        parameter: str,
        decimals: int = 6
    ) -> Dict:
        """Parse flat case data into hierarchical structure."""
        if parameter not in self.processor.metadata or self.processor.metadata[parameter].is_empty():
            return {}
        
        metadata = self.processor.metadata[parameter]
        dynamic_field_names = self.processor.dynamic_fields.get(parameter, [])
        
        properties_agg = {}
        hierarchy = {}
        
        for row in metadata.iter_rows(named=True):
            col_name = row['column_name']
            prop = row['property']
            
            if col_name not in case_data:
                continue
                
            value = case_data[col_name]
            
            if value is not None:
                try:
                    value = self.unit_manager._to_float(value, decimals)
                except (TypeError, ValueError):
                    pass
            
            if value is not None and isinstance(value, (int, float)):
                if prop not in properties_agg:
                    properties_agg[prop] = 0.0
                properties_agg[prop] += value
            
            path_parts = []
            for field_name in dynamic_field_names:
                field_value = row.get(field_name)
                if field_value is not None and field_value != '':
                    path_parts.append(str(field_value))
            
            if path_parts:
                current_level = hierarchy
                for part in path_parts:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
                
                current_level[prop] = value
        
        if decimals is not None:
            properties_agg = {
                k: self.unit_manager._to_float(v, decimals) 
                for k, v in properties_agg.items()
            }
        
        result = {**properties_agg, **hierarchy}
        
        return result
    
    def get_case_details(
        self,
        index: int,
        parameter: str,
        filters: Dict[str, Any],
        value: float,
        decimals: int = 6,
        override_multiplier: float = None
    ) -> Dict:
        """Extract detailed information for a specific case."""
        case_data = self.get_case(index, parameter=parameter, _skip_filtering=True, as_dict=True)
        
        try:
            all_properties = self.processor.properties(parameter)
        except:
            all_properties = []
        
        variables_raw = {k: v for k, v in case_data.items() if k.startswith("$")}
        variables_dict = self._strip_variable_prefix(variables_raw)
        
        if filters:
            properties_dict = {}
            non_property_filters = {k: v for k, v in filters.items() if k != "property"}
            
            for prop in all_properties:
                try:
                    prop_filters = {**non_property_filters, "property": prop}
                    cache_key = self.processor._create_cache_key(parameter, prop_filters)
                    values, _ = self.data_extractor.extract_values(
                        self.processor.data[parameter],
                        self.processor.metadata[parameter],
                        self.processor.dynamic_fields[parameter],
                        parameter,
                        prop_filters,
                        cache_key
                    )
                    if index < len(values):
                        raw_value = values[index]
                        display_value = self.unit_manager.format_for_display(
                            prop, raw_value, decimals, override_multiplier
                        )
                        properties_dict[prop] = display_value
                except:
                    pass
            
            property_filter = filters.get("property")
            if isinstance(property_filter, list):
                property_key = property_filter[0] if property_filter else "value"
            else:
                property_key = property_filter if property_filter else "value"
            
            if value is None and property_key in properties_dict:
                main_value = properties_dict[property_key]
            else:
                main_value = self.unit_manager.format_for_display(
                    property_key, value, decimals, override_multiplier
                ) if value is not None else None
            
            details = {
                "idx": index,
                **{property_key: main_value},
                **{k: v for k, v in filters.items() if k != "property"},
                "properties": properties_dict,
                "variables": variables_dict
            }
        else:
            properties_dict = self.parse_case_to_hierarchy(case_data, parameter, decimals)
            
            def format_hierarchy(d):
                result = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        result[k] = format_hierarchy(v)
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        result[k] = self.unit_manager.format_for_display(
                            k, v, decimals, override_multiplier
                        )
                    else:
                        result[k] = v
                return result
            
            properties_dict = format_hierarchy(properties_dict)
            
            details = {
                "idx": index,
                "properties": properties_dict,
                "variables": variables_dict
            }
        
        return details
    
    def get_case(
        self, 
        index_or_reference: Union[int, str], 
        parameter: str = None,
        filters: Union[Dict[str, Any], str] = None,
        property: Union[str, List[str], bool, None] = None,
        as_dict: bool = False,
        _skip_filtering: bool = False
    ) -> Union[Case, Dict]:
        """Get data for a specific case by index or reference."""
        tag = None
        
        if filters is not None:
            filters = self.processor.filter_manager.resolve_filter_preset(filters)
        
        if filters is not None:
            filters = FilterManager.merge_property_filter(filters, property)
        elif property is not None and property is not False:
            filters = {'property': property}
        
        if isinstance(index_or_reference, str):
            param, index, tag = self.parse_case_reference(index_or_reference)
            reference = index_or_reference
        else:
            index = index_or_reference
            param = self.processor._resolve_parameter(parameter)
            reference = self.create_case_reference(param, index, tag=tag)
        
        if filters and not _skip_filtering:
            case_data = self.get_case_details(
                index, param, filters, None, decimals=6
            )
        else:
            df = self.processor.data[param]
            
            if index < 0 or index >= len(df):
                raise IndexError(f"Index {index} out of range (0–{len(df)-1})")
            
            case_data = df[index].to_dicts()[0]
            
            if not _skip_filtering:
                properties_dict = self.parse_case_to_hierarchy(case_data, param, decimals=6)
                case_data = {
                    'idx': index,
                    'properties': properties_dict,
                    **{k: v for k, v in case_data.items() if k.startswith('$')}
                }
        
        if tag:
            case_data['case'] = tag
        
        if _skip_filtering:
            if as_dict:
                return case_data
            return Case(
                data=case_data,
                processor=self.processor,
                index=index,
                parameter=param,
                reference=reference,
                case_type=tag
            )
        
        var_list = self.processor.default_variables
        if var_list is not None:
            self._filter_case_variables(case_data, var_list)
        else:
            if 'variables' not in case_data:
                vars_dict = {k: v for k, v in case_data.items() if k.startswith('$')}
                for k in list(vars_dict.keys()):
                    del case_data[k]
                case_data['variables'] = self._strip_variable_prefix(vars_dict)
            else:
                case_data['variables'] = self._strip_variable_prefix(case_data['variables'])
        
        if as_dict:
            return case_data
        
        return Case(
            data=case_data,
            processor=self.processor,
            index=index,
            parameter=param,
            reference=reference,
            case_type=tag
        )
    
    @staticmethod
    def _normalize_variable_name(var_name: str) -> str:
        """Ensure variable name has $ prefix."""
        return var_name if var_name.startswith('$') else f'${var_name}'
    
    @staticmethod
    def _strip_variable_prefix(variables_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove $ prefix from variable names in dict."""
        return {k.lstrip('$'): v for k, v in variables_dict.items()}
    
    def _filter_case_variables(self, case_data: Dict, var_list: List[str]) -> None:
        """Filter variables in case_data dict in-place."""
        normalized_vars = [self._normalize_variable_name(v) for v in var_list]
        
        if 'variables' in case_data:
            filtered_variables = {
                k: v for k, v in case_data['variables'].items()
                if k in normalized_vars
            }
            case_data['variables'] = self._strip_variable_prefix(filtered_variables)
        else:
            filtered_vars = {
                k: v for k, v in case_data.items() 
                if k.startswith('$') and k in normalized_vars
            }
            for k in list(case_data.keys()):
                if k.startswith('$'):
                    del case_data[k]
            if filtered_vars:
                case_data['variables'] = self._strip_variable_prefix(filtered_vars)


# ================================================================
# STATISTICS COMPUTER
# ================================================================

class StatisticsComputer:
    """Handles statistics computation and case selection."""
    
    def __init__(
        self,
        processor: 'TornadoProcessor',
        unit_manager: UnitManager,
        data_extractor: DataExtractor,
        case_manager: CaseManager
    ):
        self.processor = processor
        self.unit_manager = unit_manager
        self.data_extractor = data_extractor
        self.case_manager = case_manager
    
    def compute_all_stats(
        self,
        property_values: Dict[str, np.ndarray],
        stats: List[str],
        options: Dict[str, Any],
        decimals: int,
        skip: List[str],
        override_multiplier: float = None
    ) -> Dict:
        """Compute all requested statistics efficiently."""
        result = {}
        threshold = options.get("p90p10_threshold", 10)
        
        is_multi_property = len(property_values) > 1
        
        all_prop_stats = {}
        
        for prop, values in property_values.items():
            prop_stats = {}
            
            for stat in stats:
                try:
                    if stat == 'mean':
                        raw_val = np.mean(values)
                        prop_stats['mean'] = self.unit_manager.format_for_display(
                            prop, raw_val, decimals, override_multiplier
                        )
                    
                    elif stat == 'median':
                        raw_val = np.median(values)
                        prop_stats['median'] = self.unit_manager.format_for_display(
                            prop, raw_val, decimals, override_multiplier
                        )
                    
                    elif stat == 'std':
                        raw_val = np.std(values)
                        prop_stats['std'] = self.unit_manager.format_for_display(
                            prop, raw_val, decimals, override_multiplier
                        )
                    
                    elif stat == 'cv':
                        mean_val = np.mean(values)
                        if abs(mean_val) > 1e-10:
                            prop_stats['cv'] = self.unit_manager._to_float(
                                np.std(values) / mean_val, decimals
                            )
                        else:
                            prop_stats['cv'] = None
                    
                    elif stat == 'count':
                        prop_stats['count'] = len(values)
                    
                    elif stat == 'sum':
                        raw_val = np.sum(values)
                        prop_stats['sum'] = self.unit_manager.format_for_display(
                            prop, raw_val, decimals, override_multiplier
                        )
                    
                    elif stat == 'variance':
                        raw_val = np.var(values)
                        prop_stats['variance'] = self.unit_manager.format_for_display(
                            prop, raw_val, decimals, override_multiplier
                        )
                    
                    elif stat == 'range':
                        raw_val = np.max(values) - np.min(values)
                        prop_stats['range'] = self.unit_manager.format_for_display(
                            prop, raw_val, decimals, override_multiplier
                        )
                    
                    elif stat == 'minmax':
                        min_raw = np.min(values)
                        max_raw = np.max(values)
                        min_val = self.unit_manager.format_for_display(
                            prop, min_raw, decimals, override_multiplier
                        )
                        max_val = self.unit_manager.format_for_display(
                            prop, max_raw, decimals, override_multiplier
                        )
                        prop_stats['minmax'] = [min_val, max_val]
                    
                    elif stat == 'p90p10':
                        if threshold and len(values) < threshold:
                            if "errors" not in skip:
                                result.setdefault("errors", []).append(
                                    f"Too few cases ({len(values)}) for {prop} p90p10; threshold={threshold}"
                                )
                        else:
                            p10_raw, p90_raw = np.percentile(values, [10, 90])
                            p10 = self.unit_manager.format_for_display(
                                prop, p10_raw, decimals, override_multiplier
                            )
                            p90 = self.unit_manager.format_for_display(
                                prop, p90_raw, decimals, override_multiplier
                            )
                            prop_stats['p90p10'] = [p10, p90]
                    
                    elif stat == 'p1p99':
                        p1_raw, p99_raw = np.percentile(values, [1, 99])
                        p1 = self.unit_manager.format_for_display(
                            prop, p1_raw, decimals, override_multiplier
                        )
                        p99 = self.unit_manager.format_for_display(
                            prop, p99_raw, decimals, override_multiplier
                        )
                        prop_stats['p1p99'] = [p1, p99]
                    
                    elif stat == 'p25p75':
                        p25_raw, p75_raw = np.percentile(values, [25, 75])
                        p25 = self.unit_manager.format_for_display(
                            prop, p25_raw, decimals, override_multiplier
                        )
                        p75 = self.unit_manager.format_for_display(
                            prop, p75_raw, decimals, override_multiplier
                        )
                        prop_stats['p25p75'] = [p25, p75]
                    
                    elif stat == 'percentile':
                        p = options.get('p', 50)
                        perc_raw = np.percentile(values, p)
                        perc_val = self.unit_manager.format_for_display(
                            prop, perc_raw, decimals, override_multiplier
                        )
                        prop_stats[f'p{p}'] = perc_val
                    
                    elif stat == 'distribution':
                        formatted_values = [
                            self.unit_manager.format_for_display(
                                prop, v, decimals, override_multiplier
                            ) for v in values
                        ]
                        prop_stats['distribution'] = np.array(formatted_values)
                    
                    else:
                        raise ValueError(
                            f"Unknown stat '{stat}'. Valid: "
                            "['p90p10', 'mean', 'median', 'minmax', 'percentile', 'distribution', "
                            "'std', 'cv', 'count', 'sum', 'variance', 'range', 'p1p99', 'p25p75']"
                        )
                
                except Exception as e:
                    if "errors" not in skip:
                        result.setdefault("errors", []).append(
                            f"Failed to compute {stat} for {prop}: {e}"
                        )
            
            all_prop_stats[prop] = prop_stats
        
        if is_multi_property:
            for stat in stats:
                if stat == 'distribution':
                    result['distribution'] = {
                        prop: all_prop_stats[prop].get('distribution') 
                        for prop in property_values.keys() 
                        if 'distribution' in all_prop_stats[prop]
                    }
                elif stat in ['minmax', 'p90p10', 'p1p99', 'p25p75']:
                    stat_dict = {
                        prop: all_prop_stats[prop].get(stat) 
                        for prop in property_values.keys() 
                        if stat in all_prop_stats[prop]
                    }
                    if stat_dict:
                        result[stat] = [
                            {prop: val[0] for prop, val in stat_dict.items()},
                            {prop: val[1] for prop, val in stat_dict.items()}
                        ]
                else:
                    stat_dict = {
                        prop: all_prop_stats[prop].get(stat) 
                        for prop in property_values.keys() 
                        if stat in all_prop_stats[prop]
                    }
                    if stat_dict:
                        result[stat] = stat_dict
        else:
            prop = list(property_values.keys())[0]
            result.update(all_prop_stats[prop])
        
        return result
    
    def normalize_selection_criteria(
        self,
        criteria: Dict[str, Any],
        parameter: str,
        main_filters: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Normalize selection_criteria to unified format."""
        if not criteria:
            return [], 'empty'
        
        specs = []
        
        if 'combinations' in criteria:
            for combo in criteria['combinations']:
                combo_filters = combo.get('filters')
                
                if combo_filters is None:
                    combo_filters = main_filters
                elif isinstance(combo_filters, str):
                    combo_filters = self.processor.filter_manager.get_filter(combo_filters)
                
                combo_filters = combo_filters.copy()
                filter_property = combo_filters.pop('property', None)
                
                for prop, weight in combo['properties'].items():
                    prop_normalized = prop.lower()
                    specs.append({
                        'property': prop_normalized,
                        'weight': weight,
                        'filters': combo_filters
                    })
            
            return specs, 'combinations'
        
        else:
            properties = self.processor.properties(parameter)
            for key, weight in criteria.items():
                key_normalized = key.lower()
                
                if key_normalized in properties:
                    filters_resolved = main_filters
                    property_name = key_normalized
                    source = 'property'
                elif key in self.processor.filter_manager.stored_filters:
                    filter_def = self.processor.filter_manager.stored_filters[key].copy()
                    property_from_filter = filter_def.pop('property', None)
                    if property_from_filter is None:
                        raise ValueError(
                            f"Stored filter '{key}' must have 'property' key for use in selection_criteria"
                        )
                    if isinstance(property_from_filter, list):
                        if len(property_from_filter) != 1:
                            raise ValueError(
                                f"Stored filter '{key}' has multiple properties {property_from_filter}. "
                                f"Selection criteria requires single property per filter."
                            )
                        property_from_filter = property_from_filter[0]
                    
                    filters_resolved = filter_def
                    property_name = property_from_filter.lower()
                    source = 'stored_filter'
                else:
                    raise ValueError(
                        f"Selection key '{key}' not recognized.\n"
                        f"  Available properties: {properties}\n"
                        f"  Available filters: {list(self.processor.filter_manager.stored_filters.keys())}"
                    )
                
                filters_resolved = filters_resolved.copy()
                filters_resolved.pop('property', None)
                
                specs.append({
                    'property': property_name,
                    'weight': weight,
                    'filters': filters_resolved
                })
            
            return specs, 'simple'
    
    def extract_weighted_properties(
        self,
        specs: List[Dict[str, Any]],
        parameter: str,
        skip: List[str]
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """Extract all properties needed for weighted case selection."""
        property_values = {}
        errors = []
        
        for spec in specs:
            prop = spec['property']
            weight = spec['weight']
            filters = spec['filters']
            
            cache_key = f"{prop}:{hash(str(sorted(filters.items())))}"
            if cache_key in property_values:
                continue
            
            try:
                prop_filters = {**filters, 'property': prop}
                cache_key_full = self.processor._create_cache_key(parameter, prop_filters)
                prop_vals, _ = self.data_extractor.extract_values(
                    self.processor.data[parameter],
                    self.processor.metadata[parameter],
                    self.processor.dynamic_fields[parameter],
                    parameter,
                    prop_filters,
                    cache_key_full
                )
                prop_vals = self.data_extractor.validate_numeric(prop_vals, prop)
                property_values[prop] = prop_vals
            except Exception as e:
                if "errors" not in skip:
                    errors.append(
                        f"Failed to extract '{prop}' with filters {filters}: {e}"
                    )
        
        return property_values, errors
    
    def calculate_weighted_distance(
        self,
        property_values: Dict[str, np.ndarray],
        specs: List[Dict[str, Any]],
        targets: Dict[str, float]
    ) -> np.ndarray:
        """Calculate weighted normalized distance for each case."""
        n_cases = len(list(property_values.values())[0])
        distances = np.zeros(n_cases)
        
        for spec in specs:
            prop = spec['property']
            weight = spec['weight']
            
            if prop in property_values and prop in targets:
                p_vals = property_values[prop]
                target = targets[prop]
                
                prop_range = np.percentile(p_vals, 90) - np.percentile(p_vals, 10)
                if prop_range > 0:
                    distances += weight * np.abs(p_vals - target) / prop_range
                else:
                    distances += weight * np.abs(p_vals - target)
        
        return distances
    
    def create_selected_case(
        self,
        index: int,
        parameter: str,
        case_type: str,
        specs: List[Dict[str, Any]],
        weighted_distance: float,
        selection_values: Dict[str, float],
        selection_method: str,
        filters: Dict[str, Any],
        decimals: int,
        override_multiplier: float = None
    ) -> Case:
        """Create a Case object with selection info and full data."""
        reference = CaseManager.create_case_reference(parameter, index, tag=None)
        
        selection_info = {
            "reference": reference,
            "selection_method": selection_method
        }
        
        if specs:
            selection_info["weights"] = {spec['property']: spec['weight'] for spec in specs}
        
        if weighted_distance is not None:
            selection_info["weighted_distance"] = weighted_distance
        
        if selection_values:
            selection_info["selection_values"] = selection_values
        
        case_data = self.case_manager.get_case(
            index,
            parameter=parameter,
            filters=None,
            as_dict=True,
            _skip_filtering=False
        )
        
        if case_type:
            case_data['case'] = case_type
        
        return Case(
            data=case_data,
            processor=self.processor,
            index=index,
            parameter=parameter,
            reference=reference,
            case_type=case_type,
            selection_info=selection_info
        )
    
    def find_closest_cases(
        self,
        property_values: Dict[str, np.ndarray],
        specs: List[Dict[str, Any]],
        targets: Dict[str, Dict[str, float]],
        resolved: str,
        filters: Dict[str, Any],
        decimals: int,
        override_multiplier: float = None
    ) -> List[Case]:
        """Find closest cases to targets using weighted distance."""
        closest_cases = []
        
        for case_type, case_targets in targets.items():
            distances = self.calculate_weighted_distance(property_values, specs, case_targets)
            idx = np.argmin(distances)
            
            selection_values = {}
            for spec in specs:
                prop = spec['property']
                if prop in property_values:
                    actual_val = property_values[prop][idx]
                    display_val = self.unit_manager.format_for_display(
                        prop, actual_val, decimals, override_multiplier
                    )
                    selection_values[f"{prop}_actual"] = display_val
                    if prop in case_targets:
                        target_val = case_targets[prop]
                        display_target = self.unit_manager.format_for_display(
                            prop, target_val, decimals, override_multiplier
                        )
                        selection_values[f"{prop}_{case_type}"] = display_target
            
            case_obj = self.create_selected_case(
                index=int(idx),
                parameter=resolved,
                case_type=case_type,
                specs=specs,
                weighted_distance=self.unit_manager._to_float(distances[idx], decimals),
                selection_values=selection_values,
                selection_method="weighted",
                filters=filters,
                decimals=decimals,
                override_multiplier=override_multiplier
            )
            
            closest_cases.append(case_obj)
        
        return closest_cases
    
    def perform_case_selection(
        self,
        property_values: Dict[str, np.ndarray],
        stats: List[str],
        stats_result: Dict,
        selection_criteria: Dict[str, Any],
        resolved: str,
        filters: Dict[str, Any],
        skip: List[str],
        decimals: int,
        override_multiplier: float = None
    ) -> List[Case]:
        """Perform case selection for computed statistics."""
        closest_cases = []
        
        specs, mode = self.normalize_selection_criteria(
            selection_criteria,
            resolved,
            filters
        )
        
        if not specs:
            specs = [
                {'property': prop, 'weight': 1.0 / len(property_values), 'filters': filters}
                for prop in property_values.keys()
            ]
            mode = 'auto'
        
        weighted_property_values, errors = self.extract_weighted_properties(
            specs, resolved, skip
        )
        
        if not weighted_property_values:
            return closest_cases
        
        targets = {}
        
        for stat in stats:
            if stat == 'minmax':
                first_prop = list(property_values.keys())[0]
                
                idx_min = np.argmin(property_values[first_prop])
                case_min = self.create_selected_case(
                    index=int(idx_min),
                    parameter=resolved,
                    case_type="min",
                    specs=None,
                    weighted_distance=None,
                    selection_values={},
                    selection_method="exact",
                    filters=filters,
                    decimals=decimals,
                    override_multiplier=override_multiplier
                )
                closest_cases.append(case_min)
                
                idx_max = np.argmax(property_values[first_prop])
                case_max = self.create_selected_case(
                    index=int(idx_max),
                    parameter=resolved,
                    case_type="max",
                    specs=None,
                    weighted_distance=None,
                    selection_values={},
                    selection_method="exact",
                    filters=filters,
                    decimals=decimals,
                    override_multiplier=override_multiplier
                )
                closest_cases.append(case_max)
            
            elif stat in ['mean', 'median', 'p90p10']:
                if stat in stats_result:
                    stat_value = stats_result[stat]
                    
                    if stat == 'p90p10':
                        targets['p10'] = {}
                        targets['p90'] = {}
                        for prop in weighted_property_values.keys():
                            p_vals = weighted_property_values[prop]
                            p10_raw, p90_raw = np.percentile(p_vals, [10, 90])
                            targets['p10'][prop] = p10_raw
                            targets['p90'][prop] = p90_raw
                    else:
                        targets[stat] = {}
                        for prop in weighted_property_values.keys():
                            p_vals = weighted_property_values[prop]
                            if stat == 'mean':
                                targets[stat][prop] = np.mean(p_vals)
                            elif stat == 'median':
                                targets[stat][prop] = np.median(p_vals)
                    
                    for case_type in list(targets.keys()):
                        if isinstance(targets[case_type], dict):
                            for prop in weighted_property_values.keys():
                                if prop not in targets[case_type]:
                                    p_vals = weighted_property_values[prop]
                                    if case_type == 'p10':
                                        targets[case_type][prop] = np.percentile(p_vals, 10)
                                    elif case_type == 'p90':
                                        targets[case_type][prop] = np.percentile(p_vals, 90)
                                    elif case_type == 'mean':
                                        targets[case_type][prop] = np.mean(p_vals)
                                    elif case_type == 'median':
                                        targets[case_type][prop] = np.median(p_vals)
        
        if targets:
            found_cases = self.find_closest_cases(
                weighted_property_values, specs, targets,
                resolved, filters, decimals,
                override_multiplier=override_multiplier
            )
            closest_cases.extend(found_cases)
        
        return closest_cases
    
    def compute_correlation_grid(
        self,
        parameter: str,
        variables: List[str],
        filters: Dict[str, Any],
        multiplier: float,
        decimals: int
    ) -> Dict[str, Any]:
        """Compute Pearson correlation matrix between variables and properties.
        
        Args:
            parameter: Parameter name
            variables: List of variable names (without $ prefix)
            filters: Base filters (property key will be removed)
            multiplier: Optional display multiplier override
            decimals: Decimal places for correlation coefficients
            
        Returns:
            Dictionary with correlation matrix and labels
        """
        # Define volumetric properties in logical order (same as Case.parameters)
        volumetric_properties = [
            'bulk volume',      # GRV
            'net volume',       # NTG
            'pore volume',      # Por
            'hcpv oil',         # So
            'hcpv gas',         # Sg
            'stoiip (in oil)',  # Bo, Rs
            'stoiip (in gas)',  # Rs
            'giip (in oil)',    # Rs
            'giip (in gas)'     # Bg, Rv
        ]
        
        # Remove property from filters
        base_filters = {k: v for k, v in filters.items() if k != 'property'}
        
        # Get data for this parameter
        df = self.processor.data[parameter]
        n_cases = len(df)
        
        # Extract all properties
        property_data = []
        property_labels = []
        
        for prop in volumetric_properties:
            try:
                prop_filters = {**base_filters, 'property': prop}
                cache_key = self.processor._create_cache_key(parameter, prop_filters)
                values, _ = self.data_extractor.extract_values(
                    self.processor.data[parameter],
                    self.processor.metadata[parameter],
                    self.processor.dynamic_fields[parameter],
                    parameter,
                    prop_filters,
                    cache_key
                )
                
                # Format for display
                formatted_values = np.array([
                    self.unit_manager.format_for_display(
                        prop, v, decimals=6, override_multiplier=multiplier
                    ) for v in values
                ])
                
                property_data.append(formatted_values)
                
                # Create label with unit
                unit = self.unit_manager.get_display_unit(prop)
                label = f"{prop} [{unit}]" if unit else prop
                property_labels.append(label)
                
            except Exception:
                # Skip properties that can't be extracted
                pass
        
        if not property_data:
            raise ValueError(
                f"No volumetric properties could be extracted for parameter '{parameter}'. "
                f"Available properties: {self.processor.properties(parameter)}"
            )
        
        # Extract all variables (only numeric ones for correlation)
        variable_data = []
        variable_labels = []
        constant_variables = []  # Track zero-variance variables
        skipped_variables = []
        
        for var in variables:
            var_col = f'${var}'
            if var_col in df.columns:
                # Try to cast to float - if it works, it's numeric
                try:
                    values = df.select(pl.col(var_col).cast(pl.Float64, strict=False)).to_series().to_numpy()
                    
                    # Check if we actually got numeric data (not all NaN)
                    if not np.isfinite(values).any():
                        skipped_variables.append((var, "no valid numeric data"))
                        continue
                    
                    # Check for zero variance (constant values)
                    # Include them but track for reporting
                    variance = np.var(values[np.isfinite(values)])
                    if variance < 1e-10:  # Essentially zero
                        unique_vals = np.unique(values[np.isfinite(values)])
                        if len(unique_vals) == 1:
                            constant_variables.append((var, float(unique_vals[0])))
                    
                    variable_data.append(values)
                    variable_labels.append(var)
                    
                except Exception as e:
                    # If casting fails, it's probably a string column
                    skipped_variables.append((var, "non-numeric (string) data"))
            else:
                skipped_variables.append((var, "variable not found"))
        
        if not variable_data:
            available_vars = [c for c in df.columns if c.startswith('$')]
            error_msg = f"No valid numeric variables could be extracted. "
            if skipped_variables:
                error_msg += f"\nSkipped variables: {[f'{v} ({reason})' for v, reason in skipped_variables]}"
            error_msg += f"\nRequested: {variables}\n"
            error_msg += f"Available: {[v.lstrip('$') for v in available_vars]}"
            raise ValueError(error_msg)

        # Calculate min/max ranges for each variable
        variable_ranges = []
        for var_values in variable_data:
            # Get finite values only (exclude NaN)
            finite_values = var_values[np.isfinite(var_values)]
            if len(finite_values) > 0:
                var_min = np.min(finite_values)
                var_max = np.max(finite_values)
                variable_ranges.append((float(var_min), float(var_max)))
            else:
                # No finite values
                variable_ranges.append((None, None))

        # Compute correlation matrix: rows = variables, cols = properties
        n_vars = len(variable_data)
        n_props = len(property_data)
        corr_matrix = np.zeros((n_vars, n_props))
        
        # Suppress warnings for invalid values (expected for constant variables)
        with np.errstate(divide='ignore', invalid='ignore'):
            for i, var_values in enumerate(variable_data):
                for j, prop_values in enumerate(property_data):
                    # Compute Pearson correlation coefficient
                    corr = np.corrcoef(var_values, prop_values)[0, 1]
                    
                    # Handle NaN (occurs when variable or property has zero variance)
                    # Set to 0.0 for display purposes - indicates no correlation possible
                    if np.isnan(corr):
                        corr = 0.0
                    
                    corr_matrix[i, j] = np.round(corr, decimals)
        
        return {
            'parameter': parameter,
            'matrix': corr_matrix,
            'variables': variable_labels,
            'properties': property_labels,
            'n_cases': n_cases,
            'variable_ranges': variable_ranges,
            'constant_variables': constant_variables if constant_variables else None,
            'skipped_variables': skipped_variables if skipped_variables else None
        }


# ================================================================
# MAIN TORNADO PROCESSOR
# ================================================================

class TornadoProcessor:
    """Main orchestrator for tornado analysis."""
    
    def __init__(
        self, 
        filepath: str, 
        display_formats: Dict[str, float] = None,
        base_case: str = None
    ):
        """Initialize processor with Excel file path and display formatting."""
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        # Initialize managers
        self.unit_manager = UnitManager(display_formats)
        self.filter_manager = FilterManager()
        self.data_extractor = DataExtractor(self.unit_manager)
        self.data_loader = ExcelDataLoader(self.unit_manager)
        
        # Load data
        try:
            self.sheets_raw = self.data_loader.load_sheets(self.filepath)
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")
        
        # Storage
        self.data: Dict[str, pl.DataFrame] = {}
        self.metadata: Dict[str, pl.DataFrame] = {}
        self.info: Dict[str, Dict] = {}
        self.dynamic_fields: Dict[str, List[str]] = {}
        self.default_variables: List[str] = None
        self.base_case_parameter: str = base_case
        
        # Parse all sheets
        try:
            self._parse_all_sheets()
        except Exception as e:
            print(f"[!] Warning: some sheets failed to parse: {e}")
        
        # Initialize case manager (needs processor to be set up first)
        self.case_manager = CaseManager(self, self.unit_manager, self.data_extractor)
        
        # Initialize statistics computer
        self.statistics_computer = StatisticsComputer(
            self, self.unit_manager, self.data_extractor, self.case_manager
        )
        
        # Extract base/reference cases if specified
        if base_case:
            try:
                self.case_manager.extract_base_and_reference_cases(base_case)
            except Exception as e:
                print(f"[!] Warning: failed to extract base/reference case from '{base_case}': {e}")
    
    def _parse_all_sheets(self):
        """Parse all loaded sheets and store results."""
        first_sheet = True
        first_sheet_name = None
        
        for sheet_name, df_raw in self.sheets_raw.items():
            try:
                data, metadata, fields, info = self.data_loader.parse_sheet(df_raw, sheet_name)
                
                sheet_units = self.data_loader.extract_sheet_property_units(metadata)
                
                if first_sheet:
                    self.unit_manager.property_units.update(sheet_units)
                    first_sheet = False
                    first_sheet_name = sheet_name
                else:
                    for prop, unit in sheet_units.items():
                        if prop in self.unit_manager.property_units:
                            if self.unit_manager.property_units[prop] != unit:
                                stored_short = self.unit_manager.unit_shortnames.get(
                                    self.unit_manager.property_units[prop], 
                                    self.unit_manager.property_units[prop]
                                )
                                current_short = self.unit_manager.unit_shortnames.get(unit, unit)
                                raise ValueError(
                                    f"Unit mismatch in sheet '{sheet_name}' for property '{prop}':\n"
                                    f"  Sheet '{first_sheet_name}' uses: {stored_short}\n"
                                    f"  Sheet '{sheet_name}' uses: {current_short}"
                                )
                
                data = self.data_loader.normalize_data_values(data, metadata, sheet_name)
                
                self.data[sheet_name] = data
                self.metadata[sheet_name] = metadata
                self.dynamic_fields[sheet_name] = fields
                self.info[sheet_name] = info
                
            except Exception as e:
                print(f"[!] Skipped sheet '{sheet_name}': {e}")
    
    def _resolve_parameter(self, parameter: str = None) -> str:
        """Resolve parameter name, defaulting to first if None."""
        if parameter is None:
            return list(self.data.keys())[0]
        return parameter
    
    @staticmethod
    def _create_cache_key(parameter: str, filters: Dict[str, Any], *args) -> str:
        """Create a hashable cache key."""
        import json
        
        sorted_filters = dict(sorted(filters.items()))
        json_filters = {}
        for k, v in sorted_filters.items():
            if isinstance(v, list):
                json_filters[k] = tuple(v)
            else:
                json_filters[k] = v
        
        key_parts = [parameter, json.dumps(json_filters, sort_keys=True)]
        key_parts.extend(str(arg) for arg in args)
        
        return ":".join(key_parts)
    
    # ================================================================
    # PUBLIC API - INFORMATION ACCESS
    # ================================================================
    
    def parameters(self) -> List[str]:
        """Get list of all available parameter names."""
        return list(self.data.keys())
    
    @lru_cache(maxsize=128)
    def properties(self, parameter: str = None) -> List[str]:
        """Get list of unique properties for a parameter."""
        resolved = self._resolve_parameter(parameter)
        
        if resolved not in self.metadata or self.metadata[resolved].is_empty():
            raise ValueError(f"No properties found in sheet '{resolved}'")
        
        return (
            self.metadata[resolved]
            .select("property")
            .unique()
            .sort("property")
            .to_series()
            .to_list()
        )
    
    @lru_cache(maxsize=256)
    def unique_values(self, field: str, parameter: str = None) -> List[str]:
        """Get unique values for a dynamic field."""
        resolved = self._resolve_parameter(parameter)
        field_norm = ExcelDataLoader.normalize_fieldname(field)
        
        if resolved not in self.dynamic_fields:
            raise ValueError(f"No dynamic fields for '{resolved}'")
        
        available = self.dynamic_fields[resolved]
        if field_norm not in available:
            raise ValueError(f"'{field}' not found. Available: {available}")
        
        if self.metadata[resolved].is_empty():
            return []
        
        return (
            self.metadata[resolved]
            .select(field_norm)
            .filter(pl.col(field_norm).is_not_null())
            .unique()
            .sort(field_norm)
            .to_series()
            .to_list()
        )
    
    def info(self, parameter: str = None) -> Dict:
        """Get metadata info for a parameter."""
        resolved = self._resolve_parameter(parameter)
        return self.info.get(resolved, {})
    
    # ================================================================
    # PUBLIC API - CASE ACCESS
    # ================================================================
    
    def case(self, *args, **kwargs) -> Union[Case, Dict]:
        """Get data for a specific case (delegates to CaseManager)."""
        return self.case_manager.get_case(*args, **kwargs)
    
    def case_variables(self, index_or_reference: Union[int, str], parameter: str = None, variables: List[str] = None) -> Dict[str, Any]:
        """Get only the variables for a specific case."""
        if variables is None:
            variables = self.default_variables
        
        if isinstance(index_or_reference, str):
            param, index, tag = CaseManager.parse_case_reference(index_or_reference)
            resolved = param
        else:
            index = index_or_reference
            resolved = self._resolve_parameter(parameter)
        
        df = self.data[resolved]
        
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} out of range (0–{len(df)-1})")
        
        case_data = df[index].to_dicts()[0]
        
        all_variables = {k: v for k, v in case_data.items() if k.startswith('$')}
        
        if variables is not None:
            normalized_vars = [CaseManager._normalize_variable_name(v) for v in variables]
            all_variables = {k: v for k, v in all_variables.items() if k in normalized_vars}
        
        return CaseManager._strip_variable_prefix(all_variables)
    
    # ================================================================
    # PUBLIC API - BASE & REFERENCE CASE
    # ================================================================
    
    def base_case(self) -> Case:
        """Get base case as a Case object."""
        return self.case_manager.get_case(0, self.base_case_parameter)
    
    def ref_case(self) -> Case:
        """Get reference case as a Case object."""
        return self.case_manager.get_case(1, self.base_case_parameter)
    
    # ================================================================
    # PUBLIC API - STATISTICS COMPUTATION
    # ================================================================
    
    def compute(
        self,
        stats: Union[str, List[str]],
        parameter: str = None,
        filters: Union[Dict[str, Any], str] = None,
        property: Union[str, List[str], bool, None] = None,
        multiplier: float = None,
        options: Dict[str, Any] = None,
        case_selection: bool = False,
        selection_criteria: Dict[str, Any] = None
    ) -> Union[Dict, Tuple[Dict, List[Case]]]:
        """Compute statistics for a single parameter with filters."""
        resolved = self._resolve_parameter(parameter)
        
        filters = self.filter_manager.resolve_filter_preset(
            filters,
            self.properties(resolved)
        )
        filters = FilterManager.merge_property_filter(filters, property)
        
        options = options or {}
        selection_criteria = selection_criteria or {}
        skip = options.get("skip", [])
        decimals = options.get("decimals", 6)
        
        property_filter = filters.get("property")
        is_multi_property = isinstance(property_filter, list)
        
        if isinstance(stats, str):
            stats = [stats]
        
        result = {"parameter": resolved}
        
        property_values = {}
        
        if is_multi_property:
            non_property_filters = {k: v for k, v in filters.items() if k != "property"}
            
            for prop in property_filter:
                try:
                    prop_filters = {**non_property_filters, "property": prop}
                    cache_key = self._create_cache_key(resolved, prop_filters)
                    prop_vals, _ = self.data_extractor.extract_values(
                        self.data[resolved],
                        self.metadata[resolved],
                        self.dynamic_fields[resolved],
                        resolved,
                        prop_filters,
                        cache_key
                    )
                    prop_vals = self.data_extractor.validate_numeric(prop_vals, prop)
                    property_values[prop] = prop_vals
                except Exception as e:
                    if "errors" not in skip:
                        result["errors"] = result.get("errors", [])
                        result["errors"].append(f"Failed to extract {prop}: {e}")
        else:
            prop = filters.get("property", "value")
            try:
                cache_key = self._create_cache_key(resolved, filters)
                values, _ = self.data_extractor.extract_values(
                    self.data[resolved],
                    self.metadata[resolved],
                    self.dynamic_fields[resolved],
                    resolved,
                    filters,
                    cache_key
                )
                values = self.data_extractor.validate_numeric(values, prop)
                property_values[prop] = values
            except Exception as e:
                if "errors" not in skip:
                    result["errors"] = result.get("errors", [])
                    result["errors"].append(f"Failed to extract {prop}: {e}")
                return result
        
        if not property_values:
            if "errors" not in skip and "errors" not in result:
                result["errors"] = ["No data could be extracted for any property"]
            return result
        
        stats_result = self.statistics_computer.compute_all_stats(
            property_values,
            stats,
            options,
            decimals,
            skip,
            override_multiplier=multiplier
        )
        result.update(stats_result)
        
        selected_cases = []
        if case_selection and "closest_case" not in skip:
            selected_cases = self.statistics_computer.perform_case_selection(
                property_values,
                stats,
                stats_result,
                selection_criteria,
                resolved,
                filters,
                skip,
                decimals,
                override_multiplier=multiplier
            )
        
        if "filters" not in skip:
            result["filters"] = filters.copy() if filters else {}
        
        if selected_cases:
            return result, selected_cases
        else:
            return result
    
    def compute_batch(
        self,
        stats: Union[str, List[str]],
        parameters: Union[str, List[str]] = "all",
        filters: Union[Dict[str, Any], str] = None,
        property: Union[str, List[str], bool, None] = None,
        multiplier: float = None,
        options: Dict[str, Any] = None,
        case_selection: bool = False,
        selection_criteria: Dict[str, Any] = None,
        include_base_case: bool = True,
        include_reference_case: bool = True,
        sort_by_range: bool = True
    ) -> Union[Dict, List[Dict], Tuple[List[Dict], List[Case]]]:
        """Compute statistics for multiple parameters."""
        filters = self.filter_manager.resolve_filter_preset(filters)
        filters = FilterManager.merge_property_filter(filters, property)
        
        if parameters == "all":
            param_list = list(self.data.keys())
            if self.base_case_parameter and self.base_case_parameter in param_list:
                param_list = [p for p in param_list if p != self.base_case_parameter]
        elif isinstance(parameters, str):
            param_list = [parameters]
        else:
            param_list = parameters
        
        options = options or {}
        skip = options.get("skip", [])
        skip_parameters = options.get("skip_parameters", [])
        
        all_param_names = set(self.data.keys())
        param_list = [
            p for p in param_list 
            if p not in skip_parameters and p not in (skip if p in all_param_names else [])
        ]
        
        results = []
        all_selected_cases = []
        
        if self.base_case_parameter and (include_base_case or include_reference_case):
            case_entry = {"parameter": self.base_case_parameter}

            # Add metadata if present in options
            if options:
                if 'filter_name' in options:
                    case_entry['filter_name'] = options['filter_name']
                if 'property_name' in options:
                    case_entry['property_name'] = options['property_name']
                if 'unit' in options:
                    case_entry['unit'] = options['unit']

            # Extract property from filters
            prop_to_use = None
            non_property_filters = None
            if filters:
                non_property_filters = {k: v for k, v in filters.items() if k != "property"}
                if "property" in filters:
                    prop_filter = filters["property"]
                    if isinstance(prop_filter, str):
                        prop_to_use = prop_filter
                    elif isinstance(prop_filter, list) and len(prop_filter) > 0:
                        prop_to_use = prop_filter[0]

            if include_base_case:
                try:
                    # Get base case with filters applied
                    base_case_obj = self.case_manager.get_case(
                        0,
                        self.base_case_parameter,
                        filters=non_property_filters
                    )
                    if prop_to_use:
                        # Use __getitem__ to raise KeyError if property not found
                        base_val = base_case_obj[prop_to_use]
                        # Apply multiplier if provided
                        if multiplier is not None:
                            base_val = base_val * multiplier
                        case_entry["base_case"] = base_val
                    else:
                        # No property specified - get first numeric property
                        props = base_case_obj.properties()
                        if props:
                            base_val = next(iter(props.values()))
                            if multiplier is not None:
                                base_val = base_val * multiplier
                            case_entry["base_case"] = base_val
                except Exception as e:
                    if "errors" not in skip:
                        case_entry["errors"] = case_entry.get("errors", [])
                        case_entry["errors"].append(f"Failed to extract base_case: {e}")

            if include_reference_case:
                try:
                    # Get reference case with filters applied
                    ref_case_obj = self.case_manager.get_case(
                        1,
                        self.base_case_parameter,
                        filters=non_property_filters
                    )
                    if prop_to_use:
                        # Use __getitem__ to raise KeyError if property not found
                        ref_val = ref_case_obj[prop_to_use]
                        # Apply multiplier if provided
                        if multiplier is not None:
                            ref_val = ref_val * multiplier
                        case_entry["reference_case"] = ref_val
                    else:
                        # No property specified - get first numeric property
                        props = ref_case_obj.properties()
                        if props:
                            ref_val = next(iter(props.values()))
                            if multiplier is not None:
                                ref_val = ref_val * multiplier
                            case_entry["reference_case"] = ref_val
                except Exception as e:
                    error_msg = str(e)
                    if "errors" not in skip and "Available: []" not in error_msg:
                        case_entry["errors"] = case_entry.get("errors", [])
                        case_entry["errors"].append(f"Failed to extract reference_case: {e}")
            
            results.append(case_entry)
        
        for param in param_list:
            try:
                compute_result = self.compute(
                    stats=stats,
                    parameter=param,
                    filters=filters,
                    multiplier=multiplier,
                    options=options,
                    case_selection=case_selection,
                    selection_criteria=selection_criteria
                )
                
                if isinstance(compute_result, tuple):
                    result_dict, cases = compute_result
                    results.append(result_dict)
                    all_selected_cases.extend(cases)
                else:
                    results.append(compute_result)
                    
            except Exception as e:
                if "errors" not in skip:
                    result = {"parameter": param, "errors": [str(e)]}
                    results.append(result)
        
        if sort_by_range and len(results) > 1:
            ref_case_entry = None
            other_results = []
            
            for result in results:
                param_name = result.get("parameter", "")
                if param_name == self.base_case_parameter:
                    ref_case_entry = result
                else:
                    other_results.append(result)
            
            def get_sort_key(result):
                if "p90p10" in result and "errors" not in result:
                    p90p10 = result["p90p10"]
                    if isinstance(p90p10, list) and len(p90p10) == 2:
                        if isinstance(p90p10[0], dict):
                            first_prop = list(p90p10[0].keys())[0]
                            return p90p10[1][first_prop] - p90p10[0][first_prop]
                        else:
                            return p90p10[1] - p90p10[0]
                
                if "minmax" in result:
                    minmax = result["minmax"]
                    if isinstance(minmax, list) and len(minmax) == 2:
                        if isinstance(minmax[0], dict):
                            first_prop = list(minmax[0].keys())[0]
                            return minmax[1][first_prop] - minmax[0][first_prop]
                        else:
                            return minmax[1] - minmax[0]
                
                return -float('inf')
            
            other_results.sort(key=get_sort_key, reverse=True)
            
            results = []
            if ref_case_entry:
                results.append(ref_case_entry)
            results.extend(other_results)
        
        if case_selection and all_selected_cases:
            return results, all_selected_cases
        else:
            return results[0] if len(results) == 1 else results
    
    # ================================================================
    # PUBLIC API - CORRELATION GRID
    # ================================================================
    
    def correlation_grid(
        self,
        parameter: str = None,
        filters: Union[Dict[str, Any], str] = None,
        variables: List[str] = None,
        multiplier: float = None,
        decimals: int = 2
    ) -> Dict[str, Any]:
        """Generate Pearson correlation matrix for correlation plots.
        
        Computes correlations between variables (y-axis) and volumetric properties (x-axis).
        Properties are ordered logically based on Case.parameters() calculation sequence.
        
        Pearson correlation coefficient ranges from -1 to +1:
        - +1.0 = perfect positive correlation (variable ↑ → property ↑)
        - -1.0 = perfect negative correlation (variable ↑ → property ↓)
        -  0.0 = no linear correlation
        
        Args:
            parameter: Tornado parameter name (defaults to first parameter).
                      REQUIRED: Must specify which parameter to analyze.
            filters: Optional filters to apply ONLY to property extraction (NOT to variables).
                    Variables always come directly from the raw data.
                    Can be a filter name string or dict.
                    Note: 'property' key in filters is ignored - all volumetric properties are used.
            variables: List of variable names to use (with or without $ prefix).
                      Defaults to self.default_variables if set.
                      REQUIRED if default_variables is not set.
            multiplier: Optional display multiplier override for properties (e.g., 1e-6 for mcm)
            decimals: Number of decimal places for correlation coefficients (default: 2)
        
        Returns:
            Dictionary with:
                - 'parameter': Parameter name (for plot title)
                - 'matrix': 2D numpy array of correlation coefficients (rows=variables, cols=properties)
                - 'variables': List of variable names (y-axis labels) - includes constant variables
                - 'properties': List of property names with units (x-axis labels)
                - 'n_cases': Number of cases used in correlation calculation
                - 'variable_ranges': List of (min, max) tuples for each variable
                - 'constant_variables': List of (variable, value) tuples for zero-variance variables (None if none found)
                - 'skipped_variables': List of (variable, reason) tuples for non-numeric variables (None if all included)
        
        Raises:
            ValueError: If no variables are available (neither passed nor in default_variables)
            ValueError: If no volumetric properties could be extracted
            ValueError: If no numeric variables could be found (all variables are strings)
        
        Note:
            - Only numeric variables are included. String variables are automatically skipped.
            - Constant variables (zero variance) are INCLUDED but their correlations are set to 0.0.
              This is important for QC - knowing a variable doesn't vary is valuable information.
            - Warnings about division by zero are suppressed for constant variables.
        
        Example:
            >>> # Minimal usage (requires default_variables to be set)
            >>> processor.default_variables = ['Porosity', 'NTG', 'Sw']
            >>> corr = processor.correlation_grid('Full_Uncertainty')
            >>> 
            >>> # Check for constant variables (important for QC!)
            >>> if corr['constant_variables']:
            ...     print("⚠️  Constant variables (no variance):")
            ...     for var, value in corr['constant_variables']:
            ...         print(f"   - {var} = {value} (correlation set to 0.0)")
            >>> # Output: Structure_variation = 1.0 (correlation set to 0.0)
            >>> 
            >>> # Check if any variables were skipped
            >>> if corr['skipped_variables']:
            ...     print(f"Skipped: {corr['skipped_variables']}")
            >>> # Output: Skipped: [('WellName', 'non-numeric (string) data')]
            >>> 
            >>> # With explicit variables (mixing numeric, string, and constant)
            >>> corr = processor.correlation_grid(
            ...     parameter='Full_Uncertainty',
            ...     variables=['Porosity', 'NTG', 'WellName', 'Structure_variation']
            ... )
            >>> print(f"Included: {corr['variables']}")  
            >>> # ['Porosity', 'NTG', 'Structure_variation']
            >>> # Note: Structure_variation included even though constant!
            >>> 
            >>> # With filters (applied only to properties)
            >>> corr = processor.correlation_grid(
            ...     parameter='Full_Uncertainty',
            ...     filters={'zone': 'zone1', 'fluid': 'oil'}
            ... )
            >>> 
            >>> # Print as table
            >>> import pandas as pd
            >>> df = pd.DataFrame(corr['matrix'], 
            ...                   index=corr['variables'], 
            ...                   columns=corr['properties'])
            >>> print(df)
            >>> 
            >>> # Create heatmap (constant variables will show as 0.0)
            >>> import seaborn as sns
            >>> sns.heatmap(corr['matrix'], 
            ...            xticklabels=corr['properties'],
            ...            yticklabels=corr['variables'],
            ...            annot=True, fmt='.2f', cmap='coolwarm', 
            ...            center=0, vmin=-1, vmax=1, square=True)
            >>> plt.title(f"Correlation Matrix - {corr['parameter']}")
        """
        # Resolve parameter
        resolved = self._resolve_parameter(parameter)
        
        # Resolve variables
        if variables is None:
            if self.default_variables is None:
                raise ValueError(
                    "No variables specified. Either pass 'variables' parameter or "
                    "set default_variables on the processor."
                )
            variables = self.default_variables
        
        # Normalize variable names (strip $ prefix)
        variables_normalized = [v.lstrip('$') for v in variables]
        
        # Resolve filters
        if filters is not None:
            filters = self.filter_manager.resolve_filter_preset(filters)
        else:
            filters = {}
        
        # Delegate to StatisticsComputer
        return self.statistics_computer.compute_correlation_grid(
            parameter=resolved,
            variables=variables_normalized,
            filters=filters,
            multiplier=multiplier,
            decimals=decimals
        )
    
    # ================================================================
    # PUBLIC API - CONVENIENCE METHODS
    # ================================================================
    
    def tornado(
        self,
        filters: Union[Dict[str, Any], str] = None,
        property: Union[str, List[str], bool, None] = None,
        multiplier: float = None,
        skip: Union[str, List[str]] = None,
        options: Dict[str, Any] = None,
        case_selection: bool = False,
        selection_criteria: Dict[str, Any] = None,
        include_base_case: bool = True,
        include_reference_case: bool = True,
        sort_by_range: bool = True
    ) -> Union[Dict, List[Dict], Tuple[List[Dict], List[Case]]]:
        """Compute tornado chart statistics."""
        merged_options = options.copy() if options else {}

        if skip is not None:
            skip_list = [skip] if isinstance(skip, str) else skip
            existing_skip = merged_options.get('skip', [])
            if isinstance(existing_skip, str):
                existing_skip = [existing_skip]
            merged_options['skip'] = list(set(existing_skip + skip_list))

        # Extract filter name and property for tornado plot metadata
        filter_name = None
        property_name = None
        original_filter_str = filters if isinstance(filters, str) else None
        if original_filter_str:
            if '_' in original_filter_str:
                parts = original_filter_str.rsplit('_', 1)
                if len(parts) == 2:
                    base_filter_name, prop_part = parts
                    if base_filter_name in self.filter_manager.stored_filters:
                        filter_name = base_filter_name
                        property_name = prop_part.replace('-', ' ')
            elif original_filter_str in self.filter_manager.stored_filters:
                filter_name = original_filter_str

        # Resolve filters to extract property if not already extracted
        if property_name is None:
            resolved_filters = self.filter_manager.resolve_filter_preset(filters)
            resolved_filters = FilterManager.merge_property_filter(resolved_filters, property)
            if resolved_filters and 'property' in resolved_filters:
                prop_filter = resolved_filters['property']
                if isinstance(prop_filter, str):
                    property_name = prop_filter
                elif isinstance(prop_filter, list) and len(prop_filter) > 0:
                    property_name = prop_filter[0]

        # Get unit for property
        unit = None
        if property_name:
            unit = self.unit_manager.get_display_unit(property_name)

        # Store metadata in options to pass through
        if filter_name:
            merged_options['filter_name'] = filter_name
        if property_name:
            merged_options['property_name'] = property_name
        if unit:
            merged_options['unit'] = unit

        return self.compute_batch(
            stats=['minmax', 'p90p10'],
            parameters="all",
            filters=filters,
            property=property,
            multiplier=multiplier,
            options=merged_options,
            case_selection=case_selection,
            selection_criteria=selection_criteria,
            include_base_case=include_base_case,
            include_reference_case=include_reference_case,
            sort_by_range=sort_by_range
        )
    
    def distribution(
        self,
        parameter: str = None,
        filters: Union[Dict[str, Any], str] = None,
        property: Union[str, List[str], bool, None] = None,
        multiplier: float = None,
        options: Dict[str, Any] = None
    ) -> Union[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """Get full distribution of values with metadata.

        Returns:
            For single property: Dict with keys:
                - 'data': np.ndarray of distribution values
                - 'title': str title for the distribution (format: "{parameter} distribution", underscores replaced with spaces)
                - 'property': str property name
                - 'unit': str unit string (e.g., 'mcm', 'bcm')
                - 'parameter': str tornado parameter name (original with underscores)
                - 'filter_name': str filter name if extracted from filter preset
            For multiple properties: Dict mapping property names to the above dict structure
        """
        # Store original filter string before resolution
        original_filter_str = filters if isinstance(filters, str) else None

        # Resolve parameter
        resolved_parameter = self._resolve_parameter(parameter)

        compute_result = self.compute(
            stats="distribution",
            parameter=parameter,
            filters=filters,
            property=property,
            multiplier=multiplier,
            options=options
        )

        if isinstance(compute_result, tuple):
            result_dict, _ = compute_result
        else:
            result_dict = compute_result

        distribution_data = result_dict["distribution"]

        # If multiple properties, return dict of dicts
        if isinstance(distribution_data, dict):
            result = {}
            for prop_name, prop_data in distribution_data.items():
                result[prop_name] = self._create_distribution_metadata(
                    prop_data, prop_name, original_filter_str, multiplier, resolved_parameter
                )
            return result

        # Single property - extract property name
        resolved_filters = self.filter_manager.resolve_filter_preset(
            filters,
            self.properties(resolved_parameter)
        )
        resolved_filters = FilterManager.merge_property_filter(resolved_filters, property)
        property_name = resolved_filters.get('property', 'Value')

        return self._create_distribution_metadata(
            distribution_data, property_name, original_filter_str, multiplier, resolved_parameter
        )

    def _create_distribution_metadata(
        self,
        data: np.ndarray,
        property_name: str,
        filter_str: str = None,
        _multiplier: float = None,
        parameter: str = None
    ) -> Dict[str, Any]:
        """Create distribution metadata dict."""
        # Extract filter name if filter string provided
        filter_name = None
        if filter_str and isinstance(filter_str, str):
            if '_' in filter_str:
                parts = filter_str.rsplit('_', 1)
                if len(parts) == 2:
                    base_filter_name, _ = parts
                    # Check if this looks like a filter_property pattern
                    if base_filter_name in self.filter_manager.stored_filters:
                        filter_name = base_filter_name
            elif filter_str in self.filter_manager.stored_filters:
                filter_name = filter_str

        # Get unit string
        unit = self.unit_manager.get_display_unit(property_name)

        # Create title - format: "{parameter} distribution"
        # Replace underscores with spaces in parameter name
        if parameter:
            parameter_display = parameter.replace('_', ' ')
            title = f"{parameter_display} distribution"
        else:
            title = "Distribution"

        return {
            'data': data,
            'title': title,
            'property': property_name,
            'unit': unit,
            'parameter': parameter,
            'filter_name': filter_name
        }
    
    # ================================================================
    # PUBLIC API - FILTER & CACHE MANAGEMENT
    # ================================================================
    
    def set_filter(self, name: str, filters: Dict[str, Any]) -> None:
        """Store a named filter preset."""
        self.filter_manager.set_filter(name, filters)
    
    def set_filters(self, filters_dict: Dict[str, Dict[str, Any]]) -> None:
        """Store multiple named filter presets."""
        self.filter_manager.set_filters(filters_dict)
    
    def get_filter(self, name: str) -> Dict[str, Any]:
        """Retrieve a stored filter preset."""
        return self.filter_manager.get_filter(name)
    
    def list_filters(self) -> List[str]:
        """List all stored filter preset names."""
        return self.filter_manager.list_filters()
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear all performance caches."""
        stats = self.data_extractor.clear_cache()
        
        ExcelDataLoader.normalize_fieldname.cache_clear()
        ExcelDataLoader.strip_units.cache_clear()
        self.unit_manager.parse_property_unit.cache_clear()
        
        if hasattr(self.properties, 'cache_clear'):
            self.properties.cache_clear()
        if hasattr(self.unique_values, 'cache_clear'):
            self.unique_values.cache_clear()
        
        stats['lru_caches_cleared'] = True
        
        return stats
    
    # ================================================================
    # PUBLIC API - UNIT & DISPLAY MANAGEMENT
    # ================================================================
    
    def get_property_units(self) -> Dict[str, str]:
        """Get dictionary of property-to-unit mappings."""
        return self.unit_manager.get_property_units()
    
    def normalization_multipliers(self) -> None:
        """Print normalization multipliers."""
        print("\n=== Normalization Multipliers ===")
        print("(Applied during Excel parsing to convert to base m³)\n")
        
        volumetric_props = {}
        for prop, unit in sorted(self.unit_manager.property_units.items()):
            if self.unit_manager.is_volumetric_property(prop):
                factor = self.unit_manager.get_normalization_factor(unit)
                if factor != 1.0:
                    unit_short = self.unit_manager.unit_shortnames.get(unit, unit)
                    volumetric_props[prop] = (unit_short, factor)
        
        if volumetric_props:
            for prop, (unit_short, factor) in volumetric_props.items():
                print(f"  {prop:.<30} {unit_short:>6}  ×{factor:>12,.0f}  → m³")
        else:
            print("  (no normalization applied)")
        
        print("\nNote: All internal values are stored in base m³.")
        print("      Display formatting is applied separately when outputting results.\n")
    
    def get_normalization_info(self) -> Dict[str, Dict[str, Any]]:
        """Get normalization information for all properties."""
        return self.unit_manager.get_normalization_info()
    
    def print_normalization_summary(self) -> None:
        """Print a summary of normalization applied."""
        info = self.get_normalization_info()
        
        print("\n=== Normalization Summary ===")
        print("\nNormalized Properties (converted to base m³):")
        normalized = [(prop, details) for prop, details in info.items() if details['was_normalized']]
        if normalized:
            for prop, details in normalized:
                print(f"  {prop:.<30} {details['unit_short']:>6} (×{details['factor']:,.0f})")
        else:
            print("  (none)")
        
        print("\nNon-Normalized Properties (kept as-is):")
        non_normalized = [(prop, details) for prop, details in info.items() if not details['was_normalized']]
        if non_normalized:
            for prop, details in non_normalized:
                unit_display = details['original_unit'] if details['original_unit'] else '(no unit)'
                print(f"  {prop:.<30} {unit_display}")
        else:
            print("  (none)")
    
    def get_display_formats(self) -> Dict[str, Dict[str, Any]]:
        """Get display format information for all properties."""
        result = {}
        for prop in self.unit_manager.property_units.keys():
            multiplier = self.unit_manager.display_formats.get(prop, 1.0)
            unit = self.unit_manager.get_display_unit(prop)
            result[prop] = {
                'multiplier': multiplier,
                'unit': unit,
                'original_unit': self.unit_manager.property_units[prop]
            }
        return result
    
    def set_display_format(self, property: str, unit: str = 'mcm') -> None:
        """Set display format for a property."""
        self.unit_manager.set_display_format(property, unit)
    
    def print_display_formats(self) -> None:
        """Print current display format settings."""
        formats = self.get_display_formats()
        
        print("\n=== Display Format Settings ===")
        print("(How values are shown in output)")
        
        for prop, details in sorted(formats.items()):
            if details['multiplier'] != 1.0:
                print(f"  {prop:.<30} {details['unit']:>6}")
            else:
                print(f"  {prop:.<30} (raw m³)")