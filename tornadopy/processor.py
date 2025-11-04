import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
from fastexcel import read_excel


class Case:
    """Represents a single case from a tornado analysis with convenient access methods.
    
    Simplified interface for accessing properties, variables, and calculated parameters.
    All volumetric properties are stored in base units (m³) internally.
    """
    
    def __init__(
        self,
        data: Dict[str, Any],
        processor: 'TornadoProcessor',
        index: int = None,
        parameter: str = None,
        reference: str = None,
        case_type: str = None
    ):
        """Initialize a Case object.
        
        Args:
            data: Raw case data dictionary
            processor: Parent TornadoProcessor instance
            index: Case index
            parameter: Parameter name
            reference: Case reference string
            case_type: Case type tag
        """
        self._data = data
        self._processor = processor
        self.idx = index if index is not None else data.get('idx')
        self.parameter = parameter
        self.reference = reference
        
        # Extract case_type from data or reference if not provided
        if case_type:
            self.case_type = case_type
        elif 'case' in data:
            self.case_type = data['case']
        elif reference and '.' in reference:
            self.case_type = reference.split('.')[0]
        else:
            self.case_type = None
        
        # Extract properties and variables from data
        self._properties = data.get('properties', {})
        self._variables = data.get('variables', {})
        
        # For backward compatibility with flat structure
        if not self._properties:
            self._properties = {
                k: v for k, v in data.items()
                if not k.startswith('_') and k not in ['idx', 'parameter', 'case', 'multiplier', 'variables', 'properties']
            }
    
    def __repr__(self) -> str:
        """String representation of the case."""
        if self.reference:
            return f"Case({self.reference})"
        elif self.parameter and self.idx is not None:
            return f"Case({self.parameter}_{self.idx})"
        else:
            return f"Case(idx={self.idx})"
    
    def __str__(self) -> str:
        """Human-readable string representation with display formatting."""
        lines = []
        
        # Header with reference and type
        header = f"Case {self.reference if self.reference else f'{self.parameter}_{self.idx}'}"
        if self.case_type:
            header += f" ({self.case_type})"
        lines.append(header)
        lines.append("-" * len(header))
        
        # Show top-level numeric properties (not nested) with display formatting
        numeric_props = {
            k: v for k, v in self._properties.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        
        if numeric_props:
            for prop, val in sorted(numeric_props.items())[:8]:
                # Apply display multiplier and get unit
                display_val = self._processor._format_for_display(prop, val, decimals=2)
                unit = self._processor._get_display_unit(prop)
                lines.append(f"  {prop:.<20} {display_val:>12,.2f} {unit}")
            if len(numeric_props) > 8:
                lines.append(f"  ... {len(numeric_props) - 8} more properties")
        
        # Show variables compactly
        if self._variables:
            lines.append("")
            lines.append("Variables:")
            for var, val in list(self._variables.items())[:5]:
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    lines.append(f"  {var:.<20} {val:>12,.2f}")
                else:
                    lines.append(f"  {var:.<20} {val}")
            if len(self._variables) > 5:
                lines.append(f"  ... {len(self._variables) - 5} more")
        
        return "\n".join(lines)
    
    def __call__(self, filter_name: str) -> None:
        """Print filtered results without modifying the case.
        
        Args:
            filter_name: Name of stored filter to apply
            
        Examples:
            case('Cerisa_STOIIP')  # Prints filtered case
        """
        filtered_case = self._processor.case(
            self.idx,
            parameter=self.parameter,
            filters=filter_name
        )
        print(filtered_case)
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to properties.
        
        Examples:
            case.stoiip  # Access stoiip property (in base units)
            case.giip    # Access giip property (in base units)
        """
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        if name in self._properties:
            return self._properties[name]
        
        if name in self._data:
            return self._data[name]
        
        raise AttributeError(f"Case has no property '{name}'. Available: {list(self._properties.keys())}")
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to properties.
        
        Examples:
            case['stoiip']  # Access stoiip property (in base units)
            case['GIIP']    # Access giip property (in base units)
        """
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
        """Get property value with optional default.
        
        Args:
            key: Property name
            default: Default value if property not found
            
        Returns:
            Property value (in base units) or default
        """
        try:
            return self[key]
        except KeyError:
            return default
    
    def properties(self, flat: bool = False) -> Dict[str, Any]:
        """Get all properties (in base units).
        
        Args:
            flat: If True, flatten nested hierarchy; if False, return as-is
            
        Returns:
            Dictionary of properties (values in base units m³)
        """
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
        """Get a variable value by name (without $ prefix).
        
        Args:
            name: Variable name ($ prefix optional)
            default: Default value if variable not found
            
        Returns:
            Variable value or default
        """
        name = name.lstrip('$')
        return self._variables.get(name, default)
    
    def variables(
        self,
        names: Union[List[str], str, None] = None,
        use_defaults: bool = True
    ) -> Dict[str, Any]:
        """Get variables with optional filtering.
        
        Args:
            names: Specific variable names to get (optional)
            use_defaults: If True and names is None, use processor's default_variables
            
        Returns:
            Dictionary of variable names to values (without $ prefix)
        """
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
    
    def parameters(self) -> Dict[str, float]:
        """Calculate derived parameters from volumetric properties.
        
        Uses raw base unit values (m³) for all calculations.
        
        Returns:
            Dictionary with calculated parameters:
            - GRV: Gross rock volume (m³)
            - NTG: Net-to-Gross ratio
            - Por: Porosity
            - So: Oil saturation
            - Sg: Gas saturation
            - Bo: Oil formation volume factor (rm³/sm³)
            - Bg: Gas formation volume factor (rm³/sm³)
            - Rs: Solution gas-oil ratio (sm³/sm³)
            - Rv: Vaporized oil-gas ratio (sm³/sm³)
        """
        props = self._properties
        params = {}
        
        # Helper to get normalized property value
        def get_prop(name):
            name_norm = name.lower().strip()
            return props.get(name_norm, 0.0)
        
        # Get base properties (all in m³)
        bulk_vol = get_prop('bulk volume')
        net_vol = get_prop('net volume')
        pore_vol = get_prop('pore volume')
        hcpv_oil = get_prop('hcpv oil')
        hcpv_gas = get_prop('hcpv gas')
        stoiip_oil = get_prop('stoiip (in oil)')
        stoiip_gas = get_prop('stoiip (in gas)')
        giip_oil = get_prop('giip (in oil)')
        giip_gas = get_prop('giip (in gas)')
        
        # Calculate parameters
        params['GRV'] = bulk_vol
        params['NTG'] = net_vol / bulk_vol if bulk_vol > 0 else 0.0
        params['Por'] = pore_vol / net_vol if net_vol > 0 else 0.0
        params['So'] = hcpv_oil / pore_vol if pore_vol > 0 else 0.0
        params['Sg'] = hcpv_gas / pore_vol if pore_vol > 0 else 0.0
        params['Bo'] = hcpv_oil / stoiip_oil if stoiip_oil > 0 else 0.0
        params['Bg'] = hcpv_gas / giip_gas if giip_gas > 0 else 0.0
        params['Rs'] = giip_oil / stoiip_oil if stoiip_oil > 0 else 0.0
        params['Rv'] = stoiip_gas / giip_gas if giip_gas > 0 else 0.0
        
        return params
    
    def to_dict(self, include_metadata: bool = True) -> Dict[str, Any]:
        """Convert case to dictionary.
        
        Args:
            include_metadata: Include idx, parameter, case_type
            
        Returns:
            Dictionary representation (properties in base units)
        """
        result = {}
        
        if include_metadata:
            if self.idx is not None:
                result['idx'] = self.idx
            if self.parameter:
                result['parameter'] = self.parameter
            if self.case_type:
                result['case'] = self.case_type
            if self.reference:
                result['reference'] = self.reference
        
        result['properties'] = self._properties.copy()
        
        if self._variables:
            result['variables'] = self._variables.copy()
        
        return result


class TornadoProcessor:
    def __init__(
        self, 
        filepath: str, 
        display_formats: Dict[str, float] = None,
        base_case: str = None
    ):
        """Initialize processor with Excel file path and display formatting.
        
        Args:
            filepath: Path to Excel file
            display_formats: Property-specific display multipliers (default: mcm for oil, bcm for gas)
            base_case: Name of sheet containing base/reference case data
            
        Attributes:
            display_formats: Property-to-multiplier mapping for output formatting
            default_variables: Default list of variables to include
            stored_filters: Dictionary of named filter presets
            base_case_parameter: Name of base case sheet
            property_units: Dictionary mapping property names to their units
            unit_shortnames: Dictionary mapping full unit strings to shorthand codes
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        # Set default display formats (property-specific multipliers)
        self.display_formats: Dict[str, float] = display_formats or {
            'bulk volume': 1e-6,      # mcm
            'net volume': 1e-6,       # mcm
            'pore volume': 1e-6,      # mcm
            'hcpv oil': 1e-6,         # mcm
            'hcpv gas': 1e-9,         # bcm
            'stoiip': 1e-6,           # mcm
            'stoiip (in oil)': 1e-6,  # mcm
            'stoiip (in gas)': 1e-6,  # mcm
            'giip': 1e-9,             # bcm
            'giip (in oil)': 1e-9,    # bcm
            'giip (in gas)': 1e-9,    # bcm
        }

        try:
            self.sheets_raw = self._load_sheets()
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

        self.data: Dict[str, pl.DataFrame] = {}
        self.metadata: Dict[str, pl.DataFrame] = {}
        self.info: Dict[str, Dict] = {}
        self.dynamic_fields: Dict[str, List[str]] = {}
        self.stored_filters: Dict[str, Dict[str, Any]] = {}
        self.default_variables: List[str] = None
        self.base_case_parameter: str = base_case
        self.base_case_values: Dict[str, float] = {}
        self.reference_case_values: Dict[str, float] = {}
        
        # Performance optimization caches
        self._extraction_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}
        self._column_selection_cache: Dict[str, Tuple[List[str], List[str]]] = {}
        
        # Property unit tracking
        self.property_units: Dict[str, str] = {}
        self._unit_validation_done: bool = False
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

        try:
            self._parse_all_sheets()
        except Exception as e:
            print(f"[!] Warning: some sheets failed to parse: {e}")

        if base_case:
            try:
                self._extract_base_and_reference_cases()
            except Exception as e:
                print(f"[!] Warning: failed to extract base/reference case from '{base_case}': {e}")
    
    # ================================================================
    # UTILITY HELPERS
    # ================================================================
    
    def _to_float(self, value: Any, decimals: int = None) -> float:
        """Convert value to native Python float with optional rounding."""
        if value is None:
            return None
        val = float(value)
        return round(val, decimals) if decimals is not None else val
    
    def _normalize_variable_name(self, var_name: str) -> str:
        """Ensure variable name has $ prefix."""
        return var_name if var_name.startswith('$') else f'${var_name}'
    
    def _strip_variable_prefix(self, variables_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove $ prefix from variable names in dict."""
        return {k.lstrip('$'): v for k, v in variables_dict.items()}
    
    @lru_cache(maxsize=512)
    def _parse_property_unit(self, property_name: str) -> Tuple[str, str]:
        """Parse property name to extract unit suffix."""
        match = re.match(r'^(.+?)(\[.*\])$', property_name.strip())
        if match:
            prop_clean = match.group(1).strip()
            unit = match.group(2)
            return prop_clean, unit
        return property_name, ''
    
    def _get_normalization_factor(self, unit: str) -> float:
        """Get normalization factor to convert to base units (m³).
        
        Args:
            unit: Unit string (e.g., '[*10^6 sm3]')
            
        Returns:
            Multiplication factor (1, 1e3, 1e6, 1e9)
        """
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
    
    def _is_volumetric_property(self, property_name: str) -> bool:
        """Check if property is volumetric and should be normalized."""
        prop_lower = property_name.lower()
        volumetric_keywords = [
            'volume', 'stoiip', 'giip', 'hcpv'
        ]
        return any(keyword in prop_lower for keyword in volumetric_keywords)
    
    def _get_display_multiplier(self, property_name: str, override_multiplier: float = None) -> float:
        """Get display multiplier for a property.
        
        Args:
            property_name: Property name
            override_multiplier: Optional override (takes precedence)
            
        Returns:
            Display multiplier to apply
        """
        if override_multiplier is not None:
            return override_multiplier
        
        prop_normalized = property_name.lower().strip()
        return self.display_formats.get(prop_normalized, 1.0)
    
    def _get_display_unit(self, property_name: str) -> str:
        """Get display unit string for a property.
        
        Args:
            property_name: Property name
            
        Returns:
            Unit string like 'mcm', 'bcm', or ''
        """
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
    
    def _format_for_display(self, property_name: str, value: float, decimals: int = 6, override_multiplier: float = None) -> float:
        """Format value for display using property-specific multiplier.
        
        Args:
            property_name: Property name (determines multiplier)
            value: Value in base units (m³)
            decimals: Decimal places for rounding
            override_multiplier: Optional override
            
        Returns:
            Formatted value
        """
        multiplier = self._get_display_multiplier(property_name, override_multiplier)
        return self._to_float(value * multiplier, decimals)
    
    def _merge_property_filter(
        self, 
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
    
    def _create_cache_key(self, parameter: str, filters: Dict[str, Any], *args) -> str:
        """Create a hashable cache key from parameter, filters, and optional args."""
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
    # INITIALIZATION & PARSING
    # ================================================================
    
    def _load_sheets(self) -> Dict[str, pl.DataFrame]:
        """Load all sheets from Excel file into Polars DataFrames."""
        sheets = {}
        excel_file = read_excel(str(self.filepath))
        
        for sheet_name in excel_file.sheet_names:
            df = excel_file.load_sheet_by_name(
                sheet_name,
                header_row=None,
                skip_rows=0
            ).to_polars()
            sheets[sheet_name] = df
        
        return sheets
    
    @lru_cache(maxsize=256)
    def _normalize_fieldname(self, name: str) -> str:
        """Normalize field name to lowercase with underscores."""
        name = str(name).strip().lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        name = re.sub(r"_+$", "", name)
        return name or "property"
    
    @lru_cache(maxsize=512)
    def _strip_units(self, property_name: str) -> str:
        """Strip unit annotations from property name."""
        cleaned = re.sub(r'\s*\[.*?\]\s*$', '', property_name)
        return cleaned.strip()
    
    def _normalize_data_values(self, df: pl.DataFrame, metadata: pl.DataFrame, sheet_name: str) -> pl.DataFrame:
        """Normalize volumetric values to base units (m³).
        
        Uses canonical property units from first sheet (stored in self.property_units).
        Works even if individual column headers don't have unit suffixes.
        
        Args:
            df: Data DataFrame with raw values
            metadata: Metadata DataFrame with column-to-property mapping
            sheet_name: Sheet name (for logging)
            
        Returns:
            DataFrame with normalized values
        """
        if metadata.is_empty():
            return df
        
        for row in metadata.iter_rows(named=True):
            col_name = row['column_name']
            property_name = row['property']
            
            # Skip if not volumetric
            if not self._is_volumetric_property(property_name):
                continue
            
            # Skip if column doesn't exist
            if col_name not in df.columns:
                continue
            
            # Get canonical unit from first sheet
            if property_name in self.property_units:
                unit = self.property_units[property_name]
                factor = self._get_normalization_factor(unit)
                
                # Apply normalization to this column
                if factor != 1.0:
                    try:
                        # Cast to float first (strict=False handles non-numeric gracefully)
                        df = df.with_columns(
                            (pl.col(col_name).cast(pl.Float64, strict=False) * factor).alias(col_name)
                        )
                    except Exception as e:
                        print(f"[!] Warning: Could not normalize column '{col_name}' in sheet '{sheet_name}': {e}")
        
        return df
    
    def _parse_sheet(self, df: pl.DataFrame, sheet_name: str = "unknown") -> Tuple[pl.DataFrame, pl.DataFrame, List[str], Dict]:
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
        
        # Extract dynamic field labels from column A
        dynamic_labels = []
        for i in range(len(header_block) - 1):
            val = header_block[header_block.columns[0]][i]
            if val is not None and str(val).strip():
                dynamic_labels.append(self._normalize_fieldname(val))
        
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
            
            # Parse unit (but don't validate here - validation happens at sheet level)
            property_name, unit = self._parse_property_unit(property_name_raw)
            
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
        
        # Note: Data normalization happens in _parse_all_sheets after unit validation
        
        return data_block, metadata_df, dynamic_labels, info_dict
    
    def _extract_sheet_property_units(self, metadata: pl.DataFrame) -> Dict[str, str]:
        """Extract property->unit mapping from a sheet's metadata.
        
        Scans all columns, takes first non-empty unit for each property.
        
        Args:
            metadata: Metadata DataFrame for a sheet
            
        Returns:
            Dictionary mapping property names to their unit strings
        """
        property_units = {}
        
        for row in metadata.iter_rows(named=True):
            col_name = row['column_name']
            prop = row['property']
            
            # Parse unit from this column's property name
            parts = col_name.split("_")
            property_name_raw = parts[-1] if parts else col_name
            _, unit = self._parse_property_unit(property_name_raw)
            
            # If this property doesn't have a unit yet, and this column has one, store it
            if prop not in property_units and unit:
                property_units[prop] = unit
        
        return property_units
    
    def _parse_all_sheets(self):
        """Parse all loaded sheets and store results.
        
        First sheet defines canonical property units. Subsequent sheets are validated
        against the first sheet's units.
        """
        first_sheet = True
        first_sheet_name = None
        
        for sheet_name, df_raw in self.sheets_raw.items():
            try:
                data, metadata, fields, info = self._parse_sheet(df_raw, sheet_name)
                
                # Extract property units from this sheet
                sheet_units = self._extract_sheet_property_units(metadata)
                
                if first_sheet:
                    # First sheet defines the canonical units
                    self.property_units.update(sheet_units)
                    first_sheet = False
                    first_sheet_name = sheet_name
                else:
                    # Validate against first sheet's units
                    for prop, unit in sheet_units.items():
                        if prop in self.property_units:
                            if self.property_units[prop] != unit:
                                stored_short = self.unit_shortnames.get(
                                    self.property_units[prop], 
                                    self.property_units[prop]
                                )
                                current_short = self.unit_shortnames.get(unit, unit)
                                raise ValueError(
                                    f"Unit mismatch in sheet '{sheet_name}' for property '{prop}':\n"
                                    f"  Sheet '{first_sheet_name}' uses: {stored_short}\n"
                                    f"  Sheet '{sheet_name}' uses: {current_short}"
                                )
                
                # Normalize data using canonical units
                data = self._normalize_data_values(data, metadata, sheet_name)
                
                self.data[sheet_name] = data
                self.metadata[sheet_name] = metadata
                self.dynamic_fields[sheet_name] = fields
                self.info[sheet_name] = info
                
            except Exception as e:
                print(f"[!] Skipped sheet '{sheet_name}': {e}")
        
        self._unit_validation_done = True
    
    # ================================================================
    # BASE & REFERENCE CASE EXTRACTION
    # ================================================================
    
    def _extract_case(
        self,
        parameter: str,
        case_index: int,
        filters: Dict[str, Any] = None
    ) -> Dict[str, float]:
        """Extract values for a specific case index from a parameter."""
        if parameter not in self.data:
            return {}
        
        case_df = self.data[parameter]
        if len(case_df) <= case_index:
            return {}
        
        try:
            properties = self.properties(parameter)
        except:
            properties = []
        
        base_filters = dict(filters) if filters else {}
        base_filters.pop("property", None)
        
        case_values = {}
        for prop in properties:
            try:
                prop_filters = {**base_filters, "property": prop}
                values, _ = self._extract_values(parameter, prop_filters)
                if len(values) > case_index:
                    case_values[prop] = float(values[case_index])
            except:
                pass
        
        return case_values
    
    def _extract_base_and_reference_cases(self, filters: Dict[str, Any] = None):
        """Extract and cache base case (index 0) and reference case (index 1)."""
        if not self.base_case_parameter:
            return
        
        self.base_case_values = self._extract_case(
            self.base_case_parameter, 
            case_index=0, 
            filters=filters
        )
        
        self.reference_case_values = self._extract_case(
            self.base_case_parameter,
            case_index=1,
            filters=filters
        )
    
    # ================================================================
    # CASE REFERENCE MANAGEMENT
    # ================================================================
    
    def _create_case_reference(self, parameter: str, index: int, tag: str = None) -> str:
        """Create a unique reference string for a case."""
        base_ref = f"{parameter}_{index}"
        return f"{tag}.{base_ref}" if tag else base_ref
    
    def _parse_case_reference(self, reference: str) -> Tuple[str, int, str]:
        """Parse a case reference string into parameter, index, and optional tag."""
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
    
    # ================================================================
    # FILTER MANAGEMENT
    # ================================================================
    
    def _resolve_filter_preset(self, filters: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """Resolve filter preset if string, otherwise return dict as-is."""
        if isinstance(filters, str):
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
            raise KeyError(f"Filter preset '{name}' not found. Available: {list(self.stored_filters.keys())}")
        return self.stored_filters[name]

    def list_filters(self) -> List[str]:
        """List all stored filter preset names."""
        return list(self.stored_filters.keys())
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear all performance caches and return statistics."""
        stats = {
            'extraction_cache': len(self._extraction_cache),
            'column_selection_cache': len(self._column_selection_cache),
        }
        
        self._extraction_cache.clear()
        self._column_selection_cache.clear()
        
        self._normalize_fieldname.cache_clear()
        self._parse_property_unit.cache_clear()
        self._strip_units.cache_clear()
        self._normalize_filters_cached.cache_clear()
        
        if hasattr(self.properties, 'cache_clear'):
            self.properties.cache_clear()
        if hasattr(self.unique_values, 'cache_clear'):
            self.unique_values.cache_clear()
        
        stats['lru_caches_cleared'] = True
        
        return stats
    
    def get_property_units(self) -> Dict[str, str]:
        """Get dictionary of property-to-unit mappings."""
        return {
            prop: self.unit_shortnames.get(unit, unit)
            for prop, unit in self.property_units.items()
        }
    
    def normalization_multipliers(self) -> None:
        """Print normalization multipliers applied during Excel parsing.
        
        Shows which properties were normalized to base units (m³) and by what factor.
        """
        print("\n=== Normalization Multipliers ===")
        print("(Applied during Excel parsing to convert to base m³)\n")
        
        # Get volumetric properties with their units
        volumetric_props = {}
        for prop, unit in sorted(self.property_units.items()):
            if self._is_volumetric_property(prop):
                factor = self._get_normalization_factor(unit)
                if factor != 1.0:
                    unit_short = self.unit_shortnames.get(unit, unit)
                    volumetric_props[prop] = (unit_short, factor)
        
        if volumetric_props:
            for prop, (unit_short, factor) in volumetric_props.items():
                print(f"  {prop:.<30} {unit_short:>6}  ×{factor:>12,.0f}  → m³")
        else:
            print("  (no normalization applied)")
        
        print("\nNote: All internal values are stored in base m³.")
        print("      Display formatting is applied separately when outputting results.\n")
    
    def get_normalization_info(self) -> Dict[str, Dict[str, Any]]:
        """Get normalization information for all properties.
        
        Returns:
            Dict mapping property names to their normalization details:
            {
                'net volume': {
                    'original_unit': '[*10^3 m3]',
                    'unit_short': 'kcm',
                    'factor': 1000.0,
                    'was_normalized': True
                },
                'stoiip': {
                    'original_unit': '[*10^6 sm3]',
                    'unit_short': 'mcm', 
                    'factor': 1000000.0,
                    'was_normalized': True
                },
                'porosity': {
                    'original_unit': '',
                    'unit_short': '',
                    'factor': 1.0,
                    'was_normalized': False
                }
            }
        """
        result = {}
        
        # Get all properties from metadata
        all_properties = set()
        for metadata_df in self.metadata.values():
            if not metadata_df.is_empty():
                all_properties.update(metadata_df.select("property").unique().to_series().to_list())
        
        for prop in sorted(all_properties):
            original_unit = self.property_units.get(prop, '')
            factor = self._get_normalization_factor(original_unit)
            unit_short = self.unit_shortnames.get(original_unit, original_unit)
            was_normalized = self._is_volumetric_property(prop) and factor != 1.0
            
            result[prop] = {
                'original_unit': original_unit,
                'unit_short': unit_short,
                'factor': factor,
                'was_normalized': was_normalized
            }
        
        return result
    
    def print_normalization_summary(self) -> None:
        """Print a summary of normalization applied during parsing."""
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
        """Get display format information for all properties.
        
        Returns:
            Dict with property names and their display info:
            {
                'stoiip': {
                    'multiplier': 1e-6,
                    'unit': 'mcm',
                    'original_unit': '[*10^6 sm3]'
                }
            }
        """
        result = {}
        for prop in self.property_units.keys():
            multiplier = self.display_formats.get(prop, 1.0)
            unit = self._get_display_unit(prop)
            result[prop] = {
                'multiplier': multiplier,
                'unit': unit,
                'original_unit': self.property_units[prop]
            }
        return result
    
    def set_display_format(self, property: str, unit: str = 'mcm') -> None:
        """Set display format for a property.
        
        Args:
            property: Property name
            unit: 'kcm', 'mcm', or 'bcm'
        """
        unit_map = {'kcm': 1e-3, 'mcm': 1e-6, 'bcm': 1e-9}
        if unit not in unit_map:
            raise ValueError(f"Unit must be 'kcm', 'mcm', or 'bcm', got: {unit}")
        self.display_formats[property.lower()] = unit_map[unit]
    
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

    
    # ================================================================
    # DATA EXTRACTION & VALIDATION
    # ================================================================
    
    def _resolve_parameter(self, parameter: str = None) -> str:
        """Resolve parameter name, defaulting to first if None."""
        if parameter is None:
            return list(self.data.keys())[0]
        return parameter
    
    @lru_cache(maxsize=512)
    def _normalize_filters_cached(self, filters_tuple: tuple) -> tuple:
        """Cached version of _normalize_filters that works with tuples."""
        filters = dict(filters_tuple)
        normalized = {}
        
        for key, value in filters.items():
            key_norm = self._normalize_fieldname(key)
            
            if isinstance(value, str):
                value_norm = value.strip().lower()
            elif isinstance(value, list):
                value_norm = tuple(v.strip().lower() if isinstance(v, str) else v for v in value)
            else:
                value_norm = value
            
            normalized[key_norm] = value_norm
        
        return tuple(sorted(normalized.items()))
    
    def _normalize_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize filter keys and string values to lowercase."""
        if not filters:
            return {}
        
        filters_tuple = tuple(sorted(filters.items()))
        normalized_tuple = self._normalize_filters_cached(filters_tuple)
        
        result = {}
        for key, value in normalized_tuple:
            if isinstance(value, tuple) and key in filters and isinstance(filters[key], list):
                result[key] = list(value)
            else:
                result[key] = value
        
        return result
    
    def _select_columns(self, parameter: str, filters: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Select columns matching filters and return column names and sources."""
        cache_key = self._create_cache_key(parameter, filters)
        if cache_key in self._column_selection_cache:
            return self._column_selection_cache[cache_key]
        
        if parameter not in self.metadata or self.metadata[parameter].is_empty():
            result = ([], [])
            self._column_selection_cache[cache_key] = result
            return result
        
        metadata = self.metadata[parameter]
        filters_norm = self._normalize_filters(filters)
        
        mask = pl.lit(True)
        
        for field, value in filters_norm.items():
            if value is None:
                continue
            
            if field not in metadata.columns:
                raise ValueError(
                    f"Field '{field}' not available. "
                    f"Available: {self.dynamic_fields.get(parameter, [])}"
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
    
    def _extract_values(
        self,
        parameter: str,
        filters: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract and sum values for columns matching filters.
        
        Returns values in base units (m³) - no multiplier applied.
        """
        cache_key = self._create_cache_key(parameter, filters)
        if cache_key in self._extraction_cache:
            cached_values, cached_sources = self._extraction_cache[cache_key]
            return cached_values.copy(), cached_sources.copy()
        
        list_fields = {k: v for k, v in filters.items() if isinstance(v, list)}
        
        if list_fields:
            df = self.data[parameter]
            n_rows = len(df)
            combined = np.zeros(n_rows, dtype=np.float64)
            all_sources = []
            
            for field, values in list_fields.items():
                for value in values:
                    single_filters = {**filters, field: value}
                    cols, sources = self._select_columns(parameter, single_filters)
                    
                    arr = (
                        df.select(cols)
                        .select(pl.sum_horizontal(pl.all().cast(pl.Float64, strict=False)))
                        .to_series()
                        .to_numpy()
                    )
                    combined += arr
                    all_sources.extend(sources)
            
            result = (combined, all_sources)
        else:
            cols, sources = self._select_columns(parameter, filters)
            df = self.data[parameter]
            
            values = (
                df.select(cols)
                .select(pl.sum_horizontal(pl.all().cast(pl.Float64, strict=False)))
                .to_series()
                .to_numpy()
            )
            
            result = (values, sources)
        
        self._extraction_cache[cache_key] = result
        return result[0].copy(), result[1].copy()
    
    def _validate_numeric(self, values: np.ndarray, description: str) -> np.ndarray:
        """Validate array contains finite numeric values."""
        if values.size == 0 or not np.isfinite(values).any():
            raise ValueError(f"No numeric data found for {description}")
        
        return values[np.isfinite(values)]
    
    # ================================================================
    # CASE SELECTION HELPERS
    # ================================================================
    
    def _resolve_selection_item(
        self,
        key: str,
        parameter: str,
        main_filters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], str, str]:
        """Resolve a selection_criteria key to (filters, property, source)."""
        properties = self.properties(parameter)
        key_normalized = key.lower()
        
        if key_normalized in properties:
            return main_filters, key_normalized, 'property'
        
        if key in self.stored_filters:
            filter_def = self.stored_filters[key].copy()
            
            property_from_filter = filter_def.pop('property', None)
            if property_from_filter is None:
                raise ValueError(
                    f"Stored filter '{key}' must have 'property' key for use in selection_criteria"
                )
            
            if isinstance(property_from_filter, list):
                if len(property_from_filter) != 1:
                    raise ValueError(
                        f"Stored filter '{key}' has multiple properties {property_from_filter}. "
                        f"Selection criteria requires single property per filter. "
                        f"Use 'combinations' syntax for multi-property filters."
                    )
                property_from_filter = property_from_filter[0]
            
            return filter_def, property_from_filter.lower(), 'stored_filter'
        
        raise ValueError(
            f"Selection key '{key}' not recognized.\n"
            f"  Available properties: {properties}\n"
            f"  Available filters: {list(self.stored_filters.keys())}"
        )
    
    def _normalize_selection_criteria(
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
                    combo_filters = self.get_filter(combo_filters)
                
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
            for key, weight in criteria.items():
                filters_resolved, property_name, source = self._resolve_selection_item(
                    key, parameter, main_filters
                )
                
                filters_resolved = filters_resolved.copy()
                filters_resolved.pop('property', None)
                
                specs.append({
                    'property': property_name,
                    'weight': weight,
                    'filters': filters_resolved
                })
            
            return specs, 'simple'
    
    def _extract_weighted_properties(
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
                prop_vals, _ = self._extract_values(parameter, prop_filters)
                prop_vals = self._validate_numeric(prop_vals, prop)
                property_values[prop] = prop_vals
            except Exception as e:
                if "errors" not in skip:
                    errors.append(f"Failed to extract '{prop}' with filters {filters}: {e}")
        
        return property_values, errors
    
    def _calculate_weighted_distance(
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
    
    def _get_case_reference_info(
        self,
        index: int,
        parameter: str,
        case_type: str = None,
        specs: List[Dict[str, Any]] = None,
        weighted_distance: float = None,
        selection_values: Dict[str, float] = None,
        selection_method: str = "weighted"
    ) -> Dict:
        """Create a lightweight case reference info dictionary."""
        reference = self._create_case_reference(parameter, index, tag=case_type)
        
        info = {
            "reference": reference
        }
        
        if specs:
            weights = {spec['property']: spec['weight'] for spec in specs}
            info["weights"] = weights
        if weighted_distance is not None:
            info["weighted_distance"] = weighted_distance
        if selection_values:
            info["selection_values"] = selection_values
        if selection_method:
            info["selection_method"] = selection_method
            
        return info
    
    def _find_closest_cases(
        self,
        property_values: Dict[str, np.ndarray],
        specs: List[Dict[str, Any]],
        targets: Dict[str, Dict[str, float]],
        resolved: str,
        filters: Dict[str, Any],
        decimals: int,
        return_references: bool = True,
        override_multiplier: float = None
    ) -> List[Dict]:
        """Find closest cases to targets using weighted distance."""
        closest_cases = []
        first_prop = list(property_values.keys())[0]
        
        for case_type, case_targets in targets.items():
            distances = self._calculate_weighted_distance(property_values, specs, case_targets)
            idx = np.argmin(distances)
            
            if return_references:
                selection_values = {}
                for spec in specs:
                    prop = spec['property']
                    if prop in property_values:
                        actual_val = property_values[prop][idx]
                        display_val = self._format_for_display(prop, actual_val, decimals, override_multiplier)
                        selection_values[f"{prop}_actual"] = display_val
                        if prop in case_targets:
                            target_val = case_targets[prop]
                            display_target = self._format_for_display(prop, target_val, decimals, override_multiplier)
                            selection_values[f"{prop}_{case_type}"] = display_target
                
                case_info = self._get_case_reference_info(
                    int(idx),
                    resolved,
                    case_type=case_type,
                    specs=specs,
                    weighted_distance=self._to_float(distances[idx], decimals),
                    selection_values=selection_values,
                    selection_method="weighted"
                )
            else:
                case_info = self._get_case_details(
                    int(idx), resolved, filters,
                    property_values[first_prop][idx], decimals, override_multiplier
                )
                case_info["case"] = case_type
                case_info["weights"] = {spec['property']: spec['weight'] for spec in specs}
                case_info["weighted_distance"] = self._to_float(distances[idx], decimals)
                
                selection_values = {}
                for spec in specs:
                    prop = spec['property']
                    if prop in property_values:
                        actual_val = property_values[prop][idx]
                        display_val = self._format_for_display(prop, actual_val, decimals, override_multiplier)
                        selection_values[f"{prop}_actual"] = display_val
                        if prop in case_targets:
                            target_val = case_targets[prop]
                            display_target = self._format_for_display(prop, target_val, decimals, override_multiplier)
                            selection_values[f"{prop}_{case_type}"] = display_target
                case_info["selection_values"] = selection_values
            
            closest_cases.append(case_info)
        
        return closest_cases
    
    def _parse_case_to_hierarchy(
        self,
        case_data: Dict,
        parameter: str,
        decimals: int = 6
    ) -> Dict:
        """Parse flat case data into hierarchical structure."""
        if parameter not in self.metadata or self.metadata[parameter].is_empty():
            return {}
        
        metadata = self.metadata[parameter]
        dynamic_field_names = self.dynamic_fields.get(parameter, [])
        
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
                    value = self._to_float(value, decimals)
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
            properties_agg = {k: self._to_float(v, decimals) for k, v in properties_agg.items()}
        
        result = {**properties_agg, **hierarchy}
        
        return result
    
    def _get_case_details(
        self,
        index: int,
        parameter: str,
        filters: Dict[str, Any],
        value: float,
        decimals: int = 6,
        override_multiplier: float = None
    ) -> Dict:
        """Extract detailed information for a specific case with display formatting."""
        case_data = self.case(index, parameter=parameter, _skip_filtering=True)
        
        try:
            all_properties = self.properties(parameter)
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
                    values, _ = self._extract_values(parameter, prop_filters)
                    if index < len(values):
                        raw_value = values[index]
                        display_value = self._format_for_display(prop, raw_value, decimals, override_multiplier)
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
                main_value = self._format_for_display(property_key, value, decimals, override_multiplier) if value is not None else None
            
            details = {
                "idx": index,
                **{property_key: main_value},
                **{k: v for k, v in filters.items() if k != "property"},
                "properties": properties_dict,
                "variables": variables_dict
            }
        else:
            properties_dict = self._parse_case_to_hierarchy(case_data, parameter, decimals)
            
            # Apply display formatting to all numeric properties in hierarchy
            def format_hierarchy(d):
                result = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        result[k] = format_hierarchy(v)
                    elif isinstance(v, (int, float)) and not isinstance(v, bool):
                        result[k] = self._format_for_display(k, v, decimals, override_multiplier)
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
    
    # ================================================================
    # STATISTICS COMPUTATION
    # ================================================================
    
    def _compute_all_stats(
        self,
        property_values: Dict[str, np.ndarray],
        stats: List[str],
        options: Dict[str, Any],
        decimals: int,
        skip: List[str],
        override_multiplier: float = None
    ) -> Dict:
        """Compute all requested statistics efficiently in a single pass."""
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
                        prop_stats['mean'] = self._format_for_display(prop, raw_val, decimals, override_multiplier)
                    
                    elif stat == 'median':
                        raw_val = np.median(values)
                        prop_stats['median'] = self._format_for_display(prop, raw_val, decimals, override_multiplier)
                    
                    elif stat == 'std':
                        raw_val = np.std(values)
                        prop_stats['std'] = self._format_for_display(prop, raw_val, decimals, override_multiplier)
                    
                    elif stat == 'cv':
                        mean_val = np.mean(values)
                        if abs(mean_val) > 1e-10:
                            prop_stats['cv'] = self._to_float(np.std(values) / mean_val, decimals)
                        else:
                            prop_stats['cv'] = None
                    
                    elif stat == 'count':
                        prop_stats['count'] = len(values)
                    
                    elif stat == 'sum':
                        raw_val = np.sum(values)
                        prop_stats['sum'] = self._format_for_display(prop, raw_val, decimals, override_multiplier)
                    
                    elif stat == 'variance':
                        raw_val = np.var(values)
                        prop_stats['variance'] = self._format_for_display(prop, raw_val, decimals, override_multiplier)
                    
                    elif stat == 'range':
                        raw_val = np.max(values) - np.min(values)
                        prop_stats['range'] = self._format_for_display(prop, raw_val, decimals, override_multiplier)
                    
                    elif stat == 'minmax':
                        min_raw = np.min(values)
                        max_raw = np.max(values)
                        min_val = self._format_for_display(prop, min_raw, decimals, override_multiplier)
                        max_val = self._format_for_display(prop, max_raw, decimals, override_multiplier)
                        prop_stats['minmax'] = [min_val, max_val]
                    
                    elif stat == 'p90p10':
                        if threshold and len(values) < threshold:
                            if "errors" not in skip:
                                result.setdefault("errors", []).append(
                                    f"Too few cases ({len(values)}) for {prop} p90p10; threshold={threshold}"
                                )
                        else:
                            p10_raw, p90_raw = np.percentile(values, [10, 90])
                            p10 = self._format_for_display(prop, p10_raw, decimals, override_multiplier)
                            p90 = self._format_for_display(prop, p90_raw, decimals, override_multiplier)
                            prop_stats['p90p10'] = [p10, p90]
                    
                    elif stat == 'p1p99':
                        p1_raw, p99_raw = np.percentile(values, [1, 99])
                        p1 = self._format_for_display(prop, p1_raw, decimals, override_multiplier)
                        p99 = self._format_for_display(prop, p99_raw, decimals, override_multiplier)
                        prop_stats['p1p99'] = [p1, p99]
                    
                    elif stat == 'p25p75':
                        p25_raw, p75_raw = np.percentile(values, [25, 75])
                        p25 = self._format_for_display(prop, p25_raw, decimals, override_multiplier)
                        p75 = self._format_for_display(prop, p75_raw, decimals, override_multiplier)
                        prop_stats['p25p75'] = [p25, p75]
                    
                    elif stat == 'percentile':
                        p = options.get('p', 50)
                        perc_raw = np.percentile(values, p)
                        perc_val = self._format_for_display(prop, perc_raw, decimals, override_multiplier)
                        prop_stats[f'p{p}'] = perc_val
                    
                    elif stat == 'distribution':
                        formatted_values = [self._format_for_display(prop, v, decimals, override_multiplier) for v in values]
                        prop_stats['distribution'] = np.array(formatted_values)
                    
                    else:
                        raise ValueError(
                            f"Unknown stat '{stat}'. Valid: "
                            "['p90p10', 'mean', 'median', 'minmax', 'percentile', 'distribution', "
                            "'std', 'cv', 'count', 'sum', 'variance', 'range', 'p1p99', 'p25p75']"
                        )
                
                except Exception as e:
                    if "errors" not in skip:
                        result.setdefault("errors", []).append(f"Failed to compute {stat} for {prop}: {e}")
            
            all_prop_stats[prop] = prop_stats
        
        if is_multi_property:
            for stat in stats:
                if stat == 'distribution':
                    result['distribution'] = {prop: all_prop_stats[prop].get('distribution') 
                                             for prop in property_values.keys() 
                                             if 'distribution' in all_prop_stats[prop]}
                elif stat in ['minmax', 'p90p10', 'p1p99', 'p25p75']:
                    stat_dict = {prop: all_prop_stats[prop].get(stat) 
                                for prop in property_values.keys() 
                                if stat in all_prop_stats[prop]}
                    if stat_dict:
                        result[stat] = [
                            {prop: val[0] for prop, val in stat_dict.items()},
                            {prop: val[1] for prop, val in stat_dict.items()}
                        ]
                else:
                    stat_dict = {prop: all_prop_stats[prop].get(stat) 
                                for prop in property_values.keys() 
                                if stat in all_prop_stats[prop]}
                    if stat_dict:
                        result[stat] = stat_dict
        else:
            prop = list(property_values.keys())[0]
            result.update(all_prop_stats[prop])
        
        return result
    
    def _perform_case_selection(
        self,
        property_values: Dict[str, np.ndarray],
        stats: List[str],
        stats_result: Dict,
        selection_criteria: Dict[str, Any],
        resolved: str,
        filters: Dict[str, Any],
        skip: List[str],
        decimals: int,
        return_references: bool = True,
        override_multiplier: float = None
    ) -> List[Dict]:
        """Perform case selection for computed statistics."""
        closest_cases = []
        
        specs, mode = self._normalize_selection_criteria(
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
        
        weighted_property_values, errors = self._extract_weighted_properties(
            specs, resolved, skip
        )
        
        if errors and "errors" not in skip:
            for error in errors:
                closest_cases.append({"error": error})
        
        if not weighted_property_values:
            return closest_cases
        
        targets = {}
        
        for stat in stats:
            if stat == 'minmax':
                first_prop = list(property_values.keys())[0]
                
                idx_min = np.argmin(property_values[first_prop])
                if return_references:
                    case_min = self._get_case_reference_info(
                        int(idx_min),
                        resolved,
                        case_type="min",
                        selection_method="exact"
                    )
                else:
                    case_min = self._get_case_details(
                        int(idx_min), resolved, filters,
                        property_values[first_prop][idx_min], decimals, override_multiplier
                    )
                    case_min["case"] = "min"
                    case_min["selection_method"] = "exact"
                closest_cases.append(case_min)
                
                idx_max = np.argmax(property_values[first_prop])
                if return_references:
                    case_max = self._get_case_reference_info(
                        int(idx_max),
                        resolved,
                        case_type="max",
                        selection_method="exact"
                    )
                else:
                    case_max = self._get_case_details(
                        int(idx_max), resolved, filters,
                        property_values[first_prop][idx_max], decimals, override_multiplier
                    )
                    case_max["case"] = "max"
                    case_max["selection_method"] = "exact"
                closest_cases.append(case_max)
            
            elif stat in ['mean', 'median', 'p90p10']:
                if stat in stats_result:
                    stat_value = stats_result[stat]
                    
                    if stat == 'p90p10':
                        if isinstance(stat_value, list) and len(stat_value) == 2:
                            if isinstance(stat_value[0], dict):
                                targets['p10'] = stat_value[0].copy()
                                targets['p90'] = stat_value[1].copy()
                            else:
                                targets['p10'] = {}
                                targets['p90'] = {}
                                
                                for prop in weighted_property_values.keys():
                                    p_vals = weighted_property_values[prop]
                                    p10_raw, p90_raw = np.percentile(p_vals, [10, 90])
                                    targets['p10'][prop] = p10_raw
                                    targets['p90'][prop] = p90_raw
                        else:
                            targets['p10'] = {}
                            targets['p90'] = {}
                    else:
                        if isinstance(stat_value, dict):
                            # Need to convert back from display to raw for calculations
                            # This is tricky - formatted values can't be used for distance calc
                            # We need raw values for weighted distance calculation
                            targets[stat] = {}
                            for prop in weighted_property_values.keys():
                                p_vals = weighted_property_values[prop]
                                if stat == 'mean':
                                    targets[stat][prop] = np.mean(p_vals)
                                elif stat == 'median':
                                    targets[stat][prop] = np.median(p_vals)
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
            found_cases = self._find_closest_cases(
                weighted_property_values, specs, targets,
                resolved, filters, decimals,
                return_references=return_references,
                override_multiplier=override_multiplier
            )
            closest_cases.extend(found_cases)
        
        return closest_cases
    
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
        field_norm = self._normalize_fieldname(field)
        
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
    
    def case(
        self, 
        index_or_reference: Union[int, str], 
        parameter: str = None,
        filters: Union[Dict[str, Any], str] = None,
        property: Union[str, List[str], bool, None] = None,
        as_dict: bool = False,
        _skip_filtering: bool = False
    ) -> Union['Case', Dict]:
        """Get data for a specific case by index or reference.
        
        Returns values in base units (m³) internally.
        Display formatting is applied by Case.__str__() and other output methods.
        """
        tag = None
        
        if filters is not None:
            filters = self._resolve_filter_preset(filters)
        
        if filters is not None:
            filters = self._merge_property_filter(filters, property)
        elif property is not None and property is not False:
            filters = {'property': property}
        
        if isinstance(index_or_reference, str):
            param, index, tag = self._parse_case_reference(index_or_reference)
            reference = index_or_reference
        else:
            index = index_or_reference
            param = self._resolve_parameter(parameter)
            reference = self._create_case_reference(param, index, tag=tag)
        
        if filters and not _skip_filtering:
            case_data = self._get_case_details(
                index, param, filters, None, decimals=6
            )
        else:
            df = self.data[param]
            
            if index < 0 or index >= len(df):
                raise IndexError(f"Index {index} out of range (0–{len(df)-1})")
            
            case_data = df[index].to_dicts()[0]
            
            if not _skip_filtering:
                properties_dict = self._parse_case_to_hierarchy(case_data, param, decimals=6)
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
                processor=self,
                index=index,
                parameter=param,
                reference=reference,
                case_type=tag
            )
        
        var_list = self.default_variables
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
            processor=self,
            index=index,
            parameter=param,
            reference=reference,
            case_type=tag
        )
    
    def _filter_case_variables(self, case_data: Dict, var_list: List[str]) -> None:
        """Filter variables in case_data dict in-place and strip $ from keys."""
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
    
    def case_variables(
        self,
        index_or_reference: Union[int, str],
        parameter: str = None,
        variables: List[str] = None
    ) -> Dict[str, Any]:
        """Get only the variables for a specific case."""
        if variables is None:
            variables = self.default_variables
        
        if isinstance(index_or_reference, str):
            param, index, tag = self._parse_case_reference(index_or_reference)
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
            normalized_vars = [self._normalize_variable_name(v) for v in variables]
            all_variables = {k: v for k, v in all_variables.items() if k in normalized_vars}
        
        return self._strip_variable_prefix(all_variables)
    
    # ================================================================
    # PUBLIC API - BASE & REFERENCE CASE
    # ================================================================
    
    def _get_case_values(
        self,
        case_type: str,
        property: str = None,
        filters: Union[Dict[str, Any], str] = None,
        multiplier: float = None
    ) -> Union[float, Dict[str, float]]:
        """Shared logic for base_case and ref_case methods."""
        filters = self._resolve_filter_preset(filters)
        
        if filters and 'property' in filters:
            filters = filters.copy()
            property = filters.pop('property')
        
        if property:
            property = self._normalize_fieldname(property)
        
        if filters:
            case_index = 0 if case_type == 'base' else 1
            case_values = self._extract_case(
                self.base_case_parameter,
                case_index=case_index,
                filters=filters
            )
        else:
            case_values = self.base_case_values if case_type == 'base' else self.reference_case_values
        
        # Apply display formatting
        if property:
            if property not in case_values:
                raise KeyError(
                    f"Property '{property}' not found in case. "
                    f"Available: {list(case_values.keys())}"
                )
            raw_value = case_values[property]
            return self._format_for_display(property, raw_value, decimals=6, override_multiplier=multiplier)
        
        # Format all values
        formatted = {}
        for prop, raw_value in case_values.items():
            formatted[prop] = self._format_for_display(prop, raw_value, decimals=6, override_multiplier=multiplier)
        
        return formatted
    
    def base_case(
        self,
        property: str = None,
        filters: Union[Dict[str, Any], str] = None,
        multiplier: float = None
    ) -> Union[float, Dict[str, float]]:
        """Get base case value(s) from first row of base_case parameter."""
        return self._get_case_values('base', property, filters, multiplier)
    
    def ref_case(
        self,
        property: str = None,
        filters: Union[Dict[str, Any], str] = None,
        multiplier: float = None
    ) -> Union[float, Dict[str, float]]:
        """Get reference case value(s) from second row of base_case parameter."""
        return self._get_case_values('reference', property, filters, multiplier)
    
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
        selection_criteria: Dict[str, Any] = None,
        return_case_references: bool = True
    ) -> Dict:
        """Compute statistics for a single parameter with filters.
        
        Returns values formatted with display multipliers (property-specific or override).
        """
        resolved = self._resolve_parameter(parameter)
        
        filters = self._resolve_filter_preset(filters)
        filters = self._merge_property_filter(filters, property)

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
                    prop_vals, _ = self._extract_values(resolved, prop_filters)
                    prop_vals = self._validate_numeric(prop_vals, prop)
                    property_values[prop] = prop_vals
                except Exception as e:
                    if "errors" not in skip:
                        result["errors"] = result.get("errors", [])
                        result["errors"].append(f"Failed to extract {prop}: {e}")
        else:
            prop = filters.get("property", "value")
            try:
                values, _ = self._extract_values(resolved, filters)
                values = self._validate_numeric(values, prop)
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
        
        stats_result = self._compute_all_stats(
            property_values,
            stats,
            options,
            decimals,
            skip,
            override_multiplier=multiplier
        )
        result.update(stats_result)
        
        if case_selection and "closest_case" not in skip:
            closest_cases = self._perform_case_selection(
                property_values,
                stats,
                stats_result,
                selection_criteria,
                resolved,
                filters,
                skip,
                decimals,
                return_references=return_case_references,
                override_multiplier=multiplier
            )
            if closest_cases:
                result["closest_cases"] = closest_cases
        
        if "filters" not in skip:
            result["filters"] = filters.copy() if filters else {}
        
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
        sort_by_range: bool = True,
        return_case_references: bool = True
    ) -> Union[Dict, List[Dict]]:
        """Compute statistics for multiple parameters."""
        filters = self._resolve_filter_preset(filters)
        filters = self._merge_property_filter(filters, property)
            
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
        param_list = [p for p in param_list if p not in skip_parameters and p not in (skip if p in all_param_names else [])]
        
        results = []

        if self.base_case_parameter and (include_base_case or include_reference_case):
            case_entry = {"parameter": self.base_case_parameter}
            
            prop_to_use = None
            if filters and "property" in filters:
                prop_filter = filters["property"]
                if isinstance(prop_filter, str):
                    prop_to_use = prop_filter
                elif isinstance(prop_filter, list) and len(prop_filter) > 0:
                    prop_to_use = prop_filter[0]
            
            if include_base_case:
                try:
                    base_val = self.base_case(property=prop_to_use, filters=filters, multiplier=multiplier)
                    if isinstance(base_val, dict):
                        case_entry["base_case"] = next(iter(base_val.values()))
                    else:
                        case_entry["base_case"] = base_val
                except Exception as e:
                    if "errors" not in skip:
                        case_entry["errors"] = case_entry.get("errors", [])
                        case_entry["errors"].append(f"Failed to extract base_case: {e}")
            
            if include_reference_case:
                try:
                    ref_val = self.ref_case(property=prop_to_use, filters=filters, multiplier=multiplier)
                    if isinstance(ref_val, dict):
                        case_entry["reference_case"] = next(iter(ref_val.values()))
                    else:
                        case_entry["reference_case"] = ref_val
                except Exception as e:
                    error_msg = str(e)
                    if "errors" not in skip and "Available: []" not in error_msg:
                        case_entry["errors"] = case_entry.get("errors", [])
                        case_entry["errors"].append(f"Failed to extract reference_case: {e}")
            
            results.append(case_entry)

        for param in param_list:
            try:
                result = self.compute(
                    stats=stats,
                    parameter=param,
                    filters=filters,
                    multiplier=multiplier,
                    options=options,
                    case_selection=case_selection,
                    selection_criteria=selection_criteria,
                    return_case_references=return_case_references
                )
                results.append(result)
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

        return results[0] if len(results) == 1 else results
    
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
        sort_by_range: bool = True,
        return_case_references: bool = True
    ) -> Union[Dict, List[Dict]]:
        """Compute tornado chart statistics (minmax and p90p10) for all parameters."""
        merged_options = options.copy() if options else {}
        
        if skip is not None:
            skip_list = [skip] if isinstance(skip, str) else skip
            existing_skip = merged_options.get('skip', [])
            if isinstance(existing_skip, str):
                existing_skip = [existing_skip]
            merged_options['skip'] = list(set(existing_skip + skip_list))
        
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
            sort_by_range=sort_by_range,
            return_case_references=return_case_references
        )
    
    def distribution(
        self,
        parameter: str = None,
        filters: Union[Dict[str, Any], str] = None,
        property: Union[str, List[str], bool, None] = None,
        multiplier: float = None,
        options: Dict[str, Any] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Get full distribution of values (with display formatting)."""
        result = self.compute(
            stats="distribution",
            parameter=parameter,
            filters=filters,
            property=property,
            multiplier=multiplier,
            options=options
        )

        return result["distribution"]