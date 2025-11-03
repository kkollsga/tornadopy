import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import polars as pl
from fastexcel import read_excel


class TornadoProcessor:
    def __init__(self, filepath: str, multiplier: float = 1.0, base_case: str = None):
        """Initialize processor with Excel file path and optional parameters.
        
        Args:
            filepath: Path to Excel file
            multiplier: Default multiplier to apply to all operations (default 1.0)
            base_case: Name of sheet containing base/reference case data
            
        Attributes:
            default_multiplier: Default multiplier for all operations
            default_variables: Default list of variables to include in case() and case_variables() 
                              calls when variables=True (default None = all variables). Set this
                              to filter which variables are returned by default. The $ prefix is
                              optional - both 'NPV' and '$NPV' work the same.
            stored_filters: Dictionary of named filter presets
            base_case_parameter: Name of base case sheet
            property_units: Dictionary mapping property names to their units (auto-populated during parsing)
            unit_shortnames: Dictionary mapping full unit strings to shorthand codes (e.g., '[*10^6 sm3]' -> 'mcm')
            
        Unit Handling:
            Units in square brackets (e.g., "[*10^3 m3]") are automatically stripped from property names
            during parsing. The processor validates that each property uses consistent units across all
            sheets and raises an error if mismatches are detected.
            
        Examples:
            # Initialize with default multiplier
            processor = TornadoProcessor('data.xlsx', multiplier=1e-6)
            
            # Set default variables to filter ($ prefix optional)
            processor.default_variables = ['NPV', 'Recovery', 'EUR']
            
            # Check which units are being used
            units = processor.get_property_units()
            # Returns: {'bulk volume': 'kcm', 'stoiip': 'mcm'}
        """
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")

        try:
            self.sheets_raw = self._load_sheets()
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")

        self.data: Dict[str, pl.DataFrame] = {}
        self.metadata: Dict[str, pl.DataFrame] = {}
        self.info: Dict[str, Dict] = {}
        self.dynamic_fields: Dict[str, List[str]] = {}
        self.default_multiplier: float = multiplier
        self.stored_filters: Dict[str, Dict[str, Any]] = {}
        self.default_variables: List[str] = None
        self.base_case_parameter: str = base_case
        self.base_case_values: Dict[str, float] = {}
        self.reference_case_values: Dict[str, float] = {}
        
        # Performance optimization caches
        self._extraction_cache: Dict[str, Tuple[np.ndarray, List[str]]] = {}  # Cache for _extract_values
        self._column_selection_cache: Dict[str, Tuple[List[str], List[str]]] = {}  # Cache for _select_columns
        
        # Property unit tracking for consistency validation
        self.property_units: Dict[str, str] = {}  # Maps property name -> unit string
        self._unit_validation_done: bool = False  # Flag to skip redundant validation
        self.unit_shortnames: Dict[str, str] = {
            # Reservoir volume units
            '[*10^3 m3]': 'kcm',
            '[*10^3 rm3]': 'kcm',
            '[*10^6 sm3]': 'mcm',
            '[*10^6 rm3]': 'mcm',
            '[*10^9 sm3]': 'bcm',
            '[*10^9 rm3]': 'bcm',
            # Add more unit mappings as needed
        }

        try:
            self._parse_all_sheets()
        except Exception as e:
            print(f"[!] Warning: some sheets failed to parse: {e}")

        # Extract base case and reference case if specified
        if base_case:
            try:
                self._extract_base_and_reference_cases()
            except Exception as e:
                print(f"[!] Warning: failed to extract base/reference case from '{base_case}': {e}")
    
    # ================================================================
    # UTILITY HELPERS
    # ================================================================
    
    def _to_float(self, value: Any, decimals: int = None) -> float:
        """Convert value to native Python float with optional rounding.
        
        Args:
            value: Value to convert (can be numpy type, int, float, etc.)
            decimals: Optional number of decimal places to round to
            
        Returns:
            Native Python float, or None if value is None
        """
        if value is None:
            return None
        val = float(value)
        return round(val, decimals) if decimals is not None else val
    
    def _normalize_variable_name(self, var_name: str) -> str:
        """Ensure variable name has $ prefix.
        
        Args:
            var_name: Variable name with or without $ prefix
            
        Returns:
            Variable name with $ prefix
            
        Examples:
            _normalize_variable_name('NPV') -> '$NPV'
            _normalize_variable_name('$NPV') -> '$NPV'
        """
        return var_name if var_name.startswith('$') else f'${var_name}'
    
    def _strip_variable_prefix(self, variables_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Remove $ prefix from variable names in dict.
        
        Args:
            variables_dict: Dictionary with $ prefixed keys
            
        Returns:
            Dictionary with $ stripped from keys
            
        Examples:
            {'$NPV': 123.4} -> {'NPV': 123.4}
        """
        return {k.lstrip('$'): v for k, v in variables_dict.items()}
    
    @lru_cache(maxsize=512)
    def _parse_property_unit(self, property_name: str) -> Tuple[str, str]:
        """Parse property name to extract unit suffix.
        
        Units are expected to be in square brackets at the end of the property name.
        Cached for performance - same property names are parsed repeatedly.
        
        Args:
            property_name: Property name potentially with unit (e.g., "bulk volume[*10^3 m3]")
            
        Returns:
            Tuple of (property_without_unit, unit_string)
            If no unit found, returns (property_name, '')
            
        Examples:
            "bulk volume[*10^3 m3]" -> ("bulk volume", "[*10^3 m3]")
            "stoiip" -> ("stoiip", "")
        """
        # Match pattern: anything followed by [...] at the end
        match = re.match(r'^(.+?)(\[.*\])$', property_name.strip())
        if match:
            prop_clean = match.group(1).strip()
            unit = match.group(2)
            return prop_clean, unit
        return property_name, ''
    
    def _validate_property_unit(self, property_name: str, unit: str, sheet_name: str) -> None:
        """Validate property unit consistency across sheets.
        
        Uses lazy validation - after first pass through all sheets, validation is skipped
        for performance.
        
        Args:
            property_name: Property name (without unit)
            unit: Unit string (e.g., "[*10^3 m3]")
            sheet_name: Name of sheet being processed (for error messages)
            
        Raises:
            ValueError: If property has inconsistent units across sheets
        """
        if not unit:
            # No unit provided, skip validation
            return
        
        # Skip validation if already completed (performance optimization)
        if self._unit_validation_done:
            return
        
        if property_name in self.property_units:
            # Property seen before, check consistency
            stored_unit = self.property_units[property_name]
            if stored_unit != unit:
                # Get shortnames for better error message
                stored_short = self.unit_shortnames.get(stored_unit, stored_unit)
                current_short = self.unit_shortnames.get(unit, unit)
                raise ValueError(
                    f"Unit mismatch for property '{property_name}' in sheet '{sheet_name}': "
                    f"expected {stored_short} but found {current_short}"
                )
        else:
            # First time seeing this property, store the unit
            self.property_units[property_name] = unit
    
    def _merge_property_filter(
        self, 
        filters: Dict[str, Any], 
        property: Union[str, List[str], bool, None]
    ) -> Dict[str, Any]:
        """Merge property parameter with filters dict.
        
        Args:
            filters: Existing filters dict (will be copied, not modified)
            property: Property parameter to merge:
                - None: Use property from filters (no change)
                - False: Remove property from filters
                - str or List[str]: Override property in filters
        
        Returns:
            New filters dict with property merged
            
        Examples:
            _merge_property_filter({'zones': ['z1']}, 'stoiip')
            -> {'zones': ['z1'], 'property': 'stoiip'}
            
            _merge_property_filter({'property': 'giip', 'zones': ['z1']}, 'stoiip')
            -> {'zones': ['z1'], 'property': 'stoiip'}
            
            _merge_property_filter({'property': 'giip', 'zones': ['z1']}, False)
            -> {'zones': ['z1']}
        """
        # Copy filters to avoid modifying original
        merged = dict(filters) if filters else {}
        
        if property is None:
            # No change - use existing property filter if present
            return merged
        elif property is False:
            # Explicitly remove property filter
            merged.pop('property', None)
            return merged
        else:
            # Override property filter (string or list)
            merged['property'] = property
            return merged
    
    def _create_cache_key(self, parameter: str, filters: Dict[str, Any], *args) -> str:
        """Create a hashable cache key from parameter, filters, and optional args.
        
        Args:
            parameter: Parameter name
            filters: Filters dictionary
            *args: Additional arguments to include in key (e.g., multiplier)
            
        Returns:
            String cache key suitable for dictionary lookup
        """
        import json
        
        # Sort filters for consistent ordering
        sorted_filters = dict(sorted(filters.items()))
        
        # Convert lists to tuples for JSON serialization
        json_filters = {}
        for k, v in sorted_filters.items():
            if isinstance(v, list):
                json_filters[k] = tuple(v)
            else:
                json_filters[k] = v
        
        # Create key from parameter, filters, and args
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
        """Normalize field name to lowercase with underscores.
        
        Cached for performance - same field names are normalized repeatedly.
        """
        name = str(name).strip().lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        name = re.sub(r"_+$", "", name)
        return name or "property"
    
    @lru_cache(maxsize=512)
    def _strip_units(self, property_name: str) -> str:
        """Strip unit annotations from property name.
        
        Removes anything in square brackets at the end of the property name.
        Cached for performance.
        
        Args:
            property_name: Property name possibly containing units (e.g., "bulk volume[*10^3 m3]")
            
        Returns:
            Property name without units (e.g., "bulk volume")
            
        Examples:
            _strip_units("bulk volume[*10^3 m3]") -> "bulk volume"
            _strip_units("stoiip [MMstb]") -> "stoiip"
            _strip_units("porosity") -> "porosity"
        """
        # Remove anything in square brackets at the end, including the brackets
        # Pattern: [ followed by anything until ]
        cleaned = re.sub(r'\s*\[.*?\]\s*$', '', property_name)
        return cleaned.strip()
    
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
            
            # Parse and validate units
            property_name, unit = self._parse_property_unit(property_name_raw)
            self._validate_property_unit(property_name, unit, sheet_name)
            
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
    
    def _parse_all_sheets(self):
        """Parse all loaded sheets and store results."""
        for sheet_name, df_raw in self.sheets_raw.items():
            try:
                data, metadata, fields, info = self._parse_sheet(df_raw, sheet_name)
                self.data[sheet_name] = data
                self.metadata[sheet_name] = metadata
                self.dynamic_fields[sheet_name] = fields
                self.info[sheet_name] = info
            except Exception as e:
                print(f"[!] Skipped sheet '{sheet_name}': {e}")
        
        # Mark unit validation as complete for performance optimization
        self._unit_validation_done = True
    
    # ================================================================
    # BASE & REFERENCE CASE EXTRACTION
    # ================================================================
    
    def _extract_case(
        self,
        parameter: str,
        case_index: int,
        filters: Dict[str, Any] = None,
        multiplier: float = None
    ) -> Dict[str, float]:
        """Extract values for a specific case index from a parameter.
        
        Args:
            parameter: Parameter name
            case_index: Index of case to extract (0 for base, 1 for reference)
            filters: Optional filters to apply (zones, etc.)
            multiplier: Multiplier to apply (defaults to instance default_multiplier)
        
        Returns:
            Dictionary mapping property names to values for that case
        """
        if parameter not in self.data:
            return {}
        
        if multiplier is None:
            multiplier = self.default_multiplier
        
        case_df = self.data[parameter]
        if len(case_df) <= case_index:
            return {}
        
        # Get all properties for this parameter
        try:
            properties = self.properties(parameter)
        except:
            properties = []
        
        # Prepare filters (remove property key if present, we'll add it per property)
        base_filters = dict(filters) if filters else {}
        base_filters.pop("property", None)
        
        # Extract value for each property at the given case index
        case_values = {}
        for prop in properties:
            try:
                prop_filters = {**base_filters, "property": prop}
                values, _ = self._extract_values(parameter, prop_filters, multiplier)
                if len(values) > case_index:
                    case_values[prop] = float(values[case_index])
            except:
                pass
        
        return case_values
    
    def _extract_base_and_reference_cases(self, filters: Dict[str, Any] = None):
        """Extract and cache base case (index 0) and reference case (index 1) from base_case parameter.
        
        This method is primarily used during initialization to populate the cached
        base_case_values and reference_case_values dictionaries with default_multiplier.
        For runtime extraction with custom filters/multipliers, use the public 
        base_case() and ref_case() methods instead.
        
        Args:
            filters: Optional filters to apply when extracting (zones, etc.)
        """
        if not self.base_case_parameter:
            return
        
        # Extract base case (index 0)
        self.base_case_values = self._extract_case(
            self.base_case_parameter, 
            case_index=0, 
            filters=filters,
            multiplier=self.default_multiplier
        )
        
        # Extract reference case (index 1) if it exists
        self.reference_case_values = self._extract_case(
            self.base_case_parameter,
            case_index=1,
            filters=filters,
            multiplier=self.default_multiplier
        )
    
    # ================================================================
    # CASE REFERENCE MANAGEMENT
    # ================================================================
    
    def _create_case_reference(self, parameter: str, index: int, tag: str = None) -> str:
        """Create a unique reference string for a case.
        
        Args:
            parameter: Parameter name
            index: Case index
            tag: Optional tag prefix (e.g., 'p10', 'p90', 'mean')
            
        Returns:
            Reference string in format "tag.parameter_index" if tag provided,
            otherwise "parameter_index"
            
        Examples:
            _create_case_reference('NTG', 42) -> 'NTG_42'
            _create_case_reference('NTG', 42, 'p10') -> 'p10.NTG_42'
        """
        base_ref = f"{parameter}_{index}"
        return f"{tag}.{base_ref}" if tag else base_ref
    
    def _parse_case_reference(self, reference: str) -> Tuple[str, int, str]:
        """Parse a case reference string into parameter, index, and optional tag.
        
        Searches from right to find first underscore separator, allowing
        parameter names to contain underscores. If a dot prefix exists,
        treats it as a tag.
        
        Args:
            reference: Case reference string (e.g., "NTG_42" or "p10.NTG_42")
            
        Returns:
            Tuple of (parameter_name, case_index, tag)
            tag will be None if no prefix present
            
        Raises:
            ValueError: If reference format is invalid
            
        Examples:
            _parse_case_reference('NTG_42') -> ('NTG', 42, None)
            _parse_case_reference('p10.NTG_42') -> ('NTG', 42, 'p10')
            _parse_case_reference('p10.NET_TO_GROSS_42') -> ('NET_TO_GROSS', 42, 'p10')
        """
        # Check for tag prefix
        tag = None
        if '.' in reference:
            tag, reference = reference.split('.', 1)
        
        # Find the last underscore
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
        """Store a named filter preset for reuse.

        Args:
            name: Name for the filter preset
            filters: Dictionary of filters to store
            
        Examples:
            processor.set_filter('north_zones', {'zones': ['z1', 'z2', 'z3']})
            processor.set_filter('stoiip_only', {'property': 'stoiip'})
        """
        self.stored_filters[name] = filters
    
    def set_filters(self, filters_dict: Dict[str, Dict[str, Any]]) -> None:
        """Store multiple named filter presets at once.
        
        Args:
            filters_dict: Dictionary where keys are filter names and values are filter definitions
            
        Examples:
            processor.set_filters({
                'north_zones': {'zones': ['z1', 'z2', 'z3']},
                'south_zones': {'zones': ['z4', 'z5']},
                'stoiip_only': {'property': 'stoiip'},
                'giip_north': {'property': 'giip', 'zones': ['z1', 'z2', 'z3']}
            })
        """
        self.stored_filters.update(filters_dict)

    def get_filter(self, name: str) -> Dict[str, Any]:
        """Retrieve a stored filter preset.

        Args:
            name: Name of the filter preset

        Returns:
            Dictionary of filters

        Raises:
            KeyError: If filter name not found
        """
        if name not in self.stored_filters:
            raise KeyError(f"Filter preset '{name}' not found. Available: {list(self.stored_filters.keys())}")
        return self.stored_filters[name]

    def list_filters(self) -> List[str]:
        """List all stored filter preset names.

        Returns:
            List of filter preset names
        """
        return list(self.stored_filters.keys())
    
    def clear_cache(self) -> Dict[str, int]:
        """Clear all performance caches and return statistics.
        
        Useful for freeing memory or forcing recomputation. Caches are automatically
        rebuilt on subsequent operations.
        
        Returns:
            Dictionary with cache sizes before clearing
            
        Examples:
            # Clear all caches and see what was cached
            stats = processor.clear_cache()
            print(f"Cleared {stats['extraction_cache']} extraction results")
            print(f"Cleared {stats['column_selection_cache']} column selections")
        """
        stats = {
            'extraction_cache': len(self._extraction_cache),
            'column_selection_cache': len(self._column_selection_cache),
        }
        
        self._extraction_cache.clear()
        self._column_selection_cache.clear()
        
        # Also clear lru_cache decorators
        self._normalize_fieldname.cache_clear()
        self._parse_property_unit.cache_clear()
        self._strip_units.cache_clear()
        self._normalize_filters_cached.cache_clear()
        
        # Clear method caches if they exist
        if hasattr(self.properties, 'cache_clear'):
            self.properties.cache_clear()
        if hasattr(self.unique_values, 'cache_clear'):
            self.unique_values.cache_clear()
        
        stats['lru_caches_cleared'] = True
        
        return stats
    
    def get_property_units(self) -> Dict[str, str]:
        """Get dictionary of property-to-unit mappings.
        
        Returns:
            Dictionary mapping property names to their units (with shortnames where available)
            
        Examples:
            units = processor.get_property_units()
            # Returns: {
            #     'bulk volume': 'kcm',
            #     'stoiip': 'mcm',
            #     'recoverable': 'bcm'
            # }
        """
        # Return with shortnames where available
        return {
            prop: self.unit_shortnames.get(unit, unit)
            for prop, unit in self.property_units.items()
        }
    
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
        """Cached version of _normalize_filters that works with tuples.
        
        Args:
            filters_tuple: Tuple of (key, value) pairs from filters dict
            
        Returns:
            Tuple of normalized (key, value) pairs
        """
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
        """Normalize filter keys and string values to lowercase.
        
        Uses caching internally for performance.
        """
        if not filters:
            return {}
        
        # Convert to tuple for caching
        filters_tuple = tuple(sorted(filters.items()))
        normalized_tuple = self._normalize_filters_cached(filters_tuple)
        
        # Convert back to dict, handling tuples back to lists
        result = {}
        for key, value in normalized_tuple:
            if isinstance(value, tuple) and key in filters and isinstance(filters[key], list):
                # Convert tuple back to list if original was list
                result[key] = list(value)
            else:
                result[key] = value
        
        return result
    
    def _select_columns(self, parameter: str, filters: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Select columns matching filters and return column names and sources.
        
        Uses caching for performance - identical filter combinations return cached results.
        """
        # Check cache first
        cache_key = self._create_cache_key(parameter, filters)
        if cache_key in self._column_selection_cache:
            return self._column_selection_cache[cache_key]
        
        # Cache miss - compute result
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
        
        # Cache the result
        self._column_selection_cache[cache_key] = result
        return result
    
    def _extract_values(
        self,
        parameter: str,
        filters: Dict[str, Any],
        multiplier: float = 1.0
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract and sum values for columns matching filters.
        
        Optimized with pre-allocated arrays, in-place accumulation, and result caching.
        Identical queries return cached results for massive speedup.
        """
        # Check cache first
        cache_key = self._create_cache_key(parameter, filters, multiplier)
        if cache_key in self._extraction_cache:
            # Return cached result (numpy arrays are mutable, so return a copy)
            cached_values, cached_sources = self._extraction_cache[cache_key]
            return cached_values.copy(), cached_sources.copy()
        
        # Cache miss - compute result
        list_fields = {k: v for k, v in filters.items() if isinstance(v, list)}
        
        if list_fields:
            # Pre-allocate result array for efficiency
            df = self.data[parameter]
            n_rows = len(df)
            combined = np.zeros(n_rows, dtype=np.float64)
            all_sources = []
            
            # Accumulate directly into combined array (avoids list growth + vstack)
            for field, values in list_fields.items():
                for value in values:
                    single_filters = {**filters, field: value}
                    cols, sources = self._select_columns(parameter, single_filters)
                    
                    # Extract and accumulate in-place
                    arr = (
                        df.select(cols)
                        .select(pl.sum_horizontal(pl.all().cast(pl.Float64, strict=False)))
                        .to_series()
                        .to_numpy()
                    )
                    combined += arr  # In-place addition
                    all_sources.extend(sources)
            
            combined *= multiplier
            result = (combined, all_sources)
        else:
            cols, sources = self._select_columns(parameter, filters)
            df = self.data[parameter]
            
            values = (
                df.select(cols)
                .select(pl.sum_horizontal(pl.all().cast(pl.Float64, strict=False)))
                .to_series()
                .to_numpy()
            ) * multiplier
            
            result = (values, sources)
        
        # Cache the result
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
    
    def _prepare_weighted_case_selection(
        self,
        property_values: Dict[str, np.ndarray],
        selection_criteria: Dict[str, Any],
        resolved: str,
        filters: Dict[str, Any],
        multiplier: float,
        skip: List[str]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], List[str]]:
        """Extract weighted properties and compute weights for case selection.
        
        This eliminates code duplication across mean/median/percentile compute methods.
        
        Args:
            property_values: Already extracted property values
            selection_criteria: Criteria containing weights
            resolved: Resolved parameter name
            filters: Applied filters
            multiplier: Applied multiplier
            skip: Skip list for error handling
            
        Returns:
            Tuple of (weighted_property_values, weights, errors)
        """
        errors = []
        
        # Get or create default weights
        weights = selection_criteria.get("weights")
        if not weights:
            weights = {prop: 1.0 / len(property_values) for prop in property_values.keys()}
        
        # Extract additional properties if needed for weighting
        weighted_property_values = dict(property_values)
        non_property_filters = {k: v for k, v in filters.items() if k != "property"}
        
        for prop in weights.keys():
            if prop not in weighted_property_values:
                try:
                    prop_filters = {**non_property_filters, "property": prop}
                    prop_vals, _ = self._extract_values(resolved, prop_filters, multiplier)
                    prop_vals = self._validate_numeric(prop_vals, prop)
                    weighted_property_values[prop] = prop_vals
                except Exception as e:
                    if "errors" not in skip:
                        errors.append(f"Failed to extract {prop} for weighting: {e}")
        
        return weighted_property_values, weights, errors
    
    def _calculate_weighted_distance(
        self,
        property_values: Dict[str, np.ndarray],
        targets: Dict[str, float],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """Calculate weighted normalized distance for each case.
        
        Args:
            property_values: Dictionary of property name to array of values
            targets: Dictionary of property name to target value
            weights: Dictionary of property name to weight
            
        Returns:
            Array of distances for each case
        """
        n_cases = len(list(property_values.values())[0])
        distances = np.zeros(n_cases)
        
        for prop, weight in weights.items():
            if prop in property_values and prop in targets:
                p_vals = property_values[prop]
                target = targets[prop]
                
                # Normalize by range to make scale-independent
                prop_range = np.percentile(p_vals, 90) - np.percentile(p_vals, 10)
                if prop_range > 0:
                    distances += weight * np.abs(p_vals - target) / prop_range
                else:
                    distances += weight * np.abs(p_vals - target)
        
        return distances
    
    def _calculate_multi_combination_distance(
        self,
        weighted_combinations: List[Dict],
        resolved: str,
        base_filters: Dict[str, Any],
        multiplier: float,
        case_type: str,
        skip: List[str]
    ) -> Tuple[np.ndarray, Dict]:
        """Calculate weighted distance across multiple filter+property combinations.
        
        Args:
            weighted_combinations: List of dicts with 'filters' and 'properties' keys
            resolved: Resolved parameter name
            base_filters: Base filters (will be merged with combination filters)
            multiplier: Multiplier to apply
            case_type: 'p10' or 'p90' or other percentile
            skip: Skip list for error handling
            
        Returns:
            Tuple of (distances array, metadata dict with targets and weights)
        """
        n_cases = None
        total_distances = None
        all_targets = {}
        all_weights = {}
        
        for combo in weighted_combinations:
            combo_filters = combo.get("filters", {})
            property_weights = combo.get("properties", {})
            
            # Merge base filters with combination filters (combination overrides base)
            merged_filters = {**base_filters, **combo_filters}
            # Remove property from merged filters as we'll specify it per property
            merged_filters_no_prop = {k: v for k, v in merged_filters.items() if k != "property"}
            
            for prop, weight in property_weights.items():
                try:
                    prop_filters = {**merged_filters_no_prop, "property": prop}
                    prop_vals, _ = self._extract_values(resolved, prop_filters, multiplier)
                    prop_vals = self._validate_numeric(prop_vals, prop)
                    
                    # Initialize on first successful extraction
                    if n_cases is None:
                        n_cases = len(prop_vals)
                        total_distances = np.zeros(n_cases)
                    
                    # Calculate percentile target
                    if case_type == "p10":
                        target = np.percentile(prop_vals, 10)
                    elif case_type == "p90":
                        target = np.percentile(prop_vals, 90)
                    elif case_type == "median":
                        target = np.median(prop_vals)
                    elif case_type.startswith("p"):
                        p = int(case_type[1:])
                        target = np.percentile(prop_vals, p)
                    else:
                        target = np.median(prop_vals)
                    
                    # Create unique key for this combo+property
                    combo_key = f"{prop}_{hash(str(combo_filters))}"
                    all_targets[combo_key] = target
                    all_weights[combo_key] = weight
                    
                    # Calculate normalized distance
                    prop_range = np.percentile(prop_vals, 90) - np.percentile(prop_vals, 10)
                    if prop_range > 0:
                        total_distances += weight * np.abs(prop_vals - target) / prop_range
                    else:
                        total_distances += weight * np.abs(prop_vals - target)
                        
                except Exception as e:
                    if "errors" not in skip:
                        print(f"Warning: Failed to process {prop} with filters {combo_filters}: {e}")
        
        metadata = {
            "targets": all_targets,
            "weights": all_weights,
            "combinations": weighted_combinations
        }
        
        return total_distances, metadata
    
    def _get_case_reference_info(
        self,
        index: int,
        parameter: str,
        case_type: str = None,
        weights: Dict[str, float] = None,
        weighted_distance: float = None,
        selection_values: Dict[str, float] = None,
        selection_method: str = "weighted"
    ) -> Dict:
        """Create a lightweight case reference info dictionary.
        
        Args:
            index: Case index
            parameter: Parameter name
            case_type: Type of case (e.g., 'p10', 'p90', 'mean', 'min', 'max')
            weights: Weights used for selection
            weighted_distance: Distance metric for weighted selection
            selection_values: Values used in selection
            selection_method: Method used for selection
            
        Returns:
            Dictionary with case reference and metadata (simplified format)
        """
        # Create reference with tag if case_type provided
        reference = self._create_case_reference(parameter, index, tag=case_type)
        
        info = {
            "reference": reference
        }
        
        if weights:
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
        targets: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        resolved: str,
        filters: Dict[str, Any],
        multiplier: float,
        decimals: int,
        return_references: bool = True
    ) -> List[Dict]:
        """Find closest cases to targets using weighted distance.
        
        Args:
            property_values: Dictionary of property name to array of values
            targets: Dictionary of case type to {property: target_value}
            weights: Dictionary of property name to weight
            resolved: Resolved parameter name
            filters: Applied filters
            multiplier: Applied multiplier
            decimals: Number of decimal places
            return_references: If True, return lightweight references; if False, return full details
            
        Returns:
            List of case detail dictionaries or reference dictionaries
        """
        closest_cases = []
        first_prop = list(property_values.keys())[0]
        
        for case_type, case_targets in targets.items():
            distances = self._calculate_weighted_distance(property_values, case_targets, weights)
            idx = np.argmin(distances)
            
            if return_references:
                # Return lightweight reference with compact format
                selection_values = {}
                for prop in weights.keys():
                    if prop in property_values:
                        # Use 'actual' for the case value
                        selection_values[f"{prop}_actual"] = self._to_float(property_values[prop][idx], decimals)
                        if prop in case_targets:
                            # Use case type as prefix for target (e.g., 'p90', 'mean')
                            selection_values[f"{prop}_{case_type}"] = self._to_float(case_targets[prop], decimals)
                
                case_info = self._get_case_reference_info(
                    int(idx),
                    resolved,
                    case_type=case_type,
                    weights=weights,
                    weighted_distance=self._to_float(distances[idx], decimals),
                    selection_values=selection_values,
                    selection_method="weighted"
                )
            else:
                # Return full details
                case_info = self._get_case_details(
                    int(idx), resolved, filters, multiplier,
                    property_values[first_prop][idx], decimals
                )
                case_info["case"] = case_type
                case_info["weights"] = weights
                case_info["weighted_distance"] = self._to_float(distances[idx], decimals)
                
                # Add selection_values showing actual values and targets for this case
                selection_values = {}
                for prop in weights.keys():
                    if prop in property_values:
                        selection_values[f"{prop}_actual"] = self._to_float(property_values[prop][idx], decimals)
                        if prop in case_targets:
                            selection_values[f"{prop}_{case_type}"] = self._to_float(case_targets[prop], decimals)
                case_info["selection_values"] = selection_values
            
            closest_cases.append(case_info)
        
        return closest_cases
    
    def _parse_case_to_hierarchy(
        self,
        case_data: Dict,
        parameter: str,
        decimals: int = 6
    ) -> Dict:
        """Parse flat case data into hierarchical structure based on multi-headers.
        
        Converts column names like "Cerisa_giip (in oil)" into nested structure:
        {
            'giip (in oil)': 2648.0,  # Aggregated across all zones
            'Cerisa': {
                'giip (in oil)': 2648.0
            }
        }
        
        Args:
            case_data: Flat dictionary with column names as keys
            parameter: Parameter name for accessing metadata
            decimals: Decimal places for rounding
            
        Returns:
            Hierarchical dictionary organized by dynamic fields and properties
        """
        if parameter not in self.metadata or self.metadata[parameter].is_empty():
            return {}
        
        metadata = self.metadata[parameter]
        dynamic_field_names = self.dynamic_fields.get(parameter, [])
        
        # Structure to hold results
        properties_agg = {}  # Top level: aggregated by property
        hierarchy = {}  # Nested: by dynamic fields
        
        # Single pass through metadata - build both aggregation and hierarchy
        for row in metadata.iter_rows(named=True):
            col_name = row['column_name']
            prop = row['property']
            
            # Skip if column not in case data
            if col_name not in case_data:
                continue
                
            value = case_data[col_name]
            
            # Convert to float if needed using helper
            if value is not None:
                try:
                    value = self._to_float(value, decimals)
                except (TypeError, ValueError):
                    pass
            
            # Add to property aggregation (sum)
            if value is not None and isinstance(value, (int, float)):
                if prop not in properties_agg:
                    properties_agg[prop] = 0.0
                properties_agg[prop] += value
            
            # Build path through dynamic fields for hierarchy
            path_parts = []
            for field_name in dynamic_field_names:
                field_value = row.get(field_name)
                if field_value is not None and field_value != '':
                    path_parts.append(str(field_value))
            
            # If we have a path (zone, region, etc.), create nested structure
            if path_parts:
                # Navigate/create nested structure
                current_level = hierarchy
                for part in path_parts:
                    if part not in current_level:
                        current_level[part] = {}
                    current_level = current_level[part]
                
                # Set the property value at the deepest level
                current_level[prop] = value
        
        # Round aggregated properties
        if decimals is not None:
            properties_agg = {k: self._to_float(v, decimals) for k, v in properties_agg.items()}
        
        # Merge: top-level aggregated properties + hierarchical breakdown
        result = {**properties_agg, **hierarchy}
        
        return result
    
    def _get_case_details(
        self,
        index: int,
        parameter: str,
        filters: Dict[str, Any],
        multiplier: float,
        value: float,
        decimals: int = 6
    ) -> Dict:
        """Extract detailed information for a specific case with filters applied.
        
        Args:
            index: Case index
            parameter: Parameter name
            filters: Applied filters (zones, properties, etc.)
            multiplier: Applied multiplier
            value: Computed value for main property (can be None, will be calculated)
            decimals: Number of decimal places for rounding
            
        Returns:
            Dictionary with case details including idx, property values (filtered), 
            filters, multiplier, and properties (hierarchical with filters applied).
            Variables are included with $ stripped from keys.
        """
        # Get raw case data without filtering (internal call)
        case_data = self.case(index, parameter=parameter, _skip_filtering=True)
        
        # Get all available properties for this parameter
        try:
            all_properties = self.properties(parameter)
        except:
            all_properties = []
        
        # Extract variables and strip $ prefix
        variables_raw = {k: v for k, v in case_data.items() if k.startswith("$")}
        variables_dict = self._strip_variable_prefix(variables_raw)
        
        # If filters provided, calculate each property with filters applied
        # Otherwise just parse the hierarchy from raw data
        if filters:
            properties_dict = {}
            non_property_filters = {k: v for k, v in filters.items() if k != "property"}
            
            # Calculate all properties with the given filters for this specific case
            for prop in all_properties:
                try:
                    prop_filters = {**non_property_filters, "property": prop}
                    values, _ = self._extract_values(parameter, prop_filters, multiplier)
                    if index < len(values):
                        properties_dict[prop] = self._to_float(values[index], decimals)
                except:
                    # Skip properties that can't be calculated
                    pass
            
            # Handle property filter to determine main key
            property_filter = filters.get("property")
            if isinstance(property_filter, list):
                property_key = property_filter[0] if property_filter else "value"
            else:
                property_key = property_filter if property_filter else "value"
            
            # Get or calculate main value
            if value is None and property_key in properties_dict:
                main_value = properties_dict[property_key]
            else:
                main_value = self._to_float(value, decimals)
            
            details = {
                "idx": index,
                **{property_key: main_value},
                **{k: v for k, v in filters.items() if k != "property"},
                "multiplier": multiplier,
                "properties": properties_dict,
                "variables": variables_dict
            }
        else:
            # No filters - just parse hierarchy from raw case data
            properties_dict = self._parse_case_to_hierarchy(case_data, parameter, decimals)
            
            details = {
                "idx": index,
                "multiplier": multiplier,
                "properties": properties_dict,
                "variables": variables_dict
            }
        
        return details
    
    # ================================================================
    # STATISTICS COMPUTATION - EFFICIENT ENGINE
    # ================================================================
    
    def _compute_all_stats(
        self,
        property_values: Dict[str, np.ndarray],
        stats: List[str],
        options: Dict[str, Any],
        decimals: int,
        skip: List[str]
    ) -> Dict:
        """Compute all requested statistics efficiently in a single pass.
        
        This replaces individual _compute_mean(), _compute_median(), etc. methods
        with a single efficient implementation that processes all stats at once.
        
        Args:
            property_values: Dict of property name to array of values
            stats: List of statistics to compute
            options: Options dict (p90p10_threshold, p for percentile, etc.)
            decimals: Number of decimal places
            skip: List of fields to skip in output
            
        Returns:
            Dict with computed statistics
        """
        result = {}
        threshold = options.get("p90p10_threshold", 10)
        
        # For single property, we'll store results directly
        # For multi-property, we'll group by stat type
        is_multi_property = len(property_values) > 1
        
        # Collect all stat values per property
        all_prop_stats = {}
        
        for prop, values in property_values.items():
            prop_stats = {}
            
            for stat in stats:
                try:
                    if stat == 'mean':
                        prop_stats['mean'] = self._to_float(np.mean(values), decimals)
                    
                    elif stat == 'median':
                        prop_stats['median'] = self._to_float(np.median(values), decimals)
                    
                    elif stat == 'std':
                        prop_stats['std'] = self._to_float(np.std(values), decimals)
                    
                    elif stat == 'cv':
                        mean_val = np.mean(values)
                        if abs(mean_val) > 1e-10:  # Avoid division by zero
                            prop_stats['cv'] = self._to_float(np.std(values) / mean_val, decimals)
                        else:
                            prop_stats['cv'] = None
                    
                    elif stat == 'count':
                        prop_stats['count'] = len(values)
                    
                    elif stat == 'sum':
                        prop_stats['sum'] = self._to_float(np.sum(values), decimals)
                    
                    elif stat == 'variance':
                        prop_stats['variance'] = self._to_float(np.var(values), decimals)
                    
                    elif stat == 'range':
                        prop_stats['range'] = self._to_float(np.max(values) - np.min(values), decimals)
                    
                    elif stat == 'minmax':
                        min_val = self._to_float(np.min(values), decimals)
                        max_val = self._to_float(np.max(values), decimals)
                        prop_stats['minmax'] = [min_val, max_val]
                    
                    elif stat == 'p90p10':
                        if threshold and len(values) < threshold:
                            if "errors" not in skip:
                                result.setdefault("errors", []).append(
                                    f"Too few cases ({len(values)}) for {prop} p90p10; threshold={threshold}"
                                )
                        else:
                            p10, p90 = np.percentile(values, [10, 90])
                            prop_stats['p90p10'] = [self._to_float(p10, decimals), self._to_float(p90, decimals)]
                    
                    elif stat == 'p1p99':
                        p1, p99 = np.percentile(values, [1, 99])
                        prop_stats['p1p99'] = [self._to_float(p1, decimals), self._to_float(p99, decimals)]
                    
                    elif stat == 'p25p75':
                        p25, p75 = np.percentile(values, [25, 75])
                        prop_stats['p25p75'] = [self._to_float(p25, decimals), self._to_float(p75, decimals)]
                    
                    elif stat == 'percentile':
                        p = options.get('p', 50)
                        perc_val = np.percentile(values, p)
                        prop_stats[f'p{p}'] = self._to_float(perc_val, decimals)
                    
                    elif stat == 'distribution':
                        prop_stats['distribution'] = np.round(values, decimals)
                    
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
        
        # Format results based on single vs multi-property
        if is_multi_property:
            # Group by stat type: {stat_name: {prop1: val1, prop2: val2}}
            for stat in stats:
                if stat == 'distribution':
                    # Distribution stays as dict of arrays
                    result['distribution'] = {prop: all_prop_stats[prop].get('distribution') 
                                             for prop in property_values.keys() 
                                             if 'distribution' in all_prop_stats[prop]}
                elif stat in ['minmax', 'p90p10', 'p1p99', 'p25p75']:
                    # Convert to [dict_low, dict_high] format
                    stat_dict = {prop: all_prop_stats[prop].get(stat) 
                                for prop in property_values.keys() 
                                if stat in all_prop_stats[prop]}
                    if stat_dict:
                        result[stat] = [
                            {prop: val[0] for prop, val in stat_dict.items()},
                            {prop: val[1] for prop, val in stat_dict.items()}
                        ]
                else:
                    # Single value stats: {prop1: val1, prop2: val2}
                    stat_dict = {prop: all_prop_stats[prop].get(stat) 
                                for prop in property_values.keys() 
                                if stat in all_prop_stats[prop]}
                    if stat_dict:
                        result[stat] = stat_dict
        else:
            # Single property: flatten results
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
        multiplier: float,
        skip: List[str],
        decimals: int,
        return_references: bool = True
    ) -> List[Dict]:
        """Perform case selection for computed statistics.
        
        Args:
            property_values: Dict of property arrays
            stats: List of computed stats
            stats_result: Results from _compute_all_stats
            selection_criteria: Criteria with weights or combinations
            resolved: Parameter name
            filters: Applied filters
            multiplier: Applied multiplier
            skip: Skip list
            decimals: Decimal places
            return_references: If True, return references; if False, return full details
            
        Returns:
            List of closest case details or references
        """
        _round = lambda x: round(x, decimals)
        closest_cases = []
        
        # Check if using combinations (complex) or simple weights
        combinations = selection_criteria.get("combinations")
        
        if combinations:
            # Handle weighted combinations for p90p10 only
            if 'p90p10' in stats:
                # Calculate distances for p10 and p90
                distances_p10, _ = self._calculate_multi_combination_distance(
                    combinations, resolved, filters, multiplier, "p10", skip
                )
                distances_p90, _ = self._calculate_multi_combination_distance(
                    combinations, resolved, filters, multiplier, "p90", skip
                )
                
                if distances_p10 is not None and distances_p90 is not None:
                    first_prop = list(property_values.keys())[0]
                    
                    # P10 case
                    idx_p10 = np.argmin(distances_p10)
                    if return_references:
                        case_p10 = self._get_case_reference_info(
                            int(idx_p10),
                            resolved,
                            case_type="p10",
                            weighted_distance=self._to_float(distances_p10[idx_p10], decimals),
                            selection_method="weighted_combinations"
                        )
                    else:
                        case_p10 = self._get_case_details(
                            int(idx_p10), resolved, filters, multiplier,
                            property_values[first_prop][idx_p10], decimals
                        )
                        case_p10["case"] = "p10"
                        case_p10["selection_method"] = "weighted_combinations"
                        case_p10["weighted_distance"] = self._to_float(distances_p10[idx_p10], decimals)
                    closest_cases.append(case_p10)
                    
                    # P90 case
                    idx_p90 = np.argmin(distances_p90)
                    if return_references:
                        case_p90 = self._get_case_reference_info(
                            int(idx_p90),
                            resolved,
                            case_type="p90",
                            weighted_distance=self._to_float(distances_p90[idx_p90], decimals),
                            selection_method="weighted_combinations"
                        )
                    else:
                        case_p90 = self._get_case_details(
                            int(idx_p90), resolved, filters, multiplier,
                            property_values[first_prop][idx_p90], decimals
                        )
                        case_p90["case"] = "p90"
                        case_p90["selection_method"] = "weighted_combinations"
                        case_p90["weighted_distance"] = self._to_float(distances_p90[idx_p90], decimals)
                    closest_cases.append(case_p90)
            
            return closest_cases
        
        # Simple weights-based selection
        weighted_property_values, weights, errors = self._prepare_weighted_case_selection(
            property_values, selection_criteria, resolved, filters, multiplier, skip
        )
        
        if not weighted_property_values:
            return []
        
        # Build targets dict for each stat that supports case selection
        targets = {}
        
        for stat in stats:
            if stat == 'minmax':
                # Minmax uses exact matching
                first_prop = list(property_values.keys())[0]
                
                # Min case
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
                        int(idx_min), resolved, filters, multiplier,
                        property_values[first_prop][idx_min], decimals
                    )
                    case_min["case"] = "min"
                    case_min["selection_method"] = "exact"
                closest_cases.append(case_min)
                
                # Max case
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
                        int(idx_max), resolved, filters, multiplier,
                        property_values[first_prop][idx_max], decimals
                    )
                    case_max["case"] = "max"
                    case_max["selection_method"] = "exact"
                closest_cases.append(case_max)
            
            elif stat in ['mean', 'median', 'p90p10']:
                # Extract target values from stats_result
                if stat in stats_result:
                    stat_value = stats_result[stat]
                    
                    if stat == 'p90p10':
                        # p90p10 has [p10_dict, p90_dict] or [p10, p90]
                        if isinstance(stat_value, list) and len(stat_value) == 2:
                            if isinstance(stat_value[0], dict):
                                # Multi-property
                                targets['p10'] = stat_value[0]
                                targets['p90'] = stat_value[1]
                            else:
                                # Single property
                                prop = list(weighted_property_values.keys())[0]
                                targets['p10'] = {prop: stat_value[0]}
                                targets['p90'] = {prop: stat_value[1]}
                    else:
                        # mean or median
                        if isinstance(stat_value, dict):
                            targets[stat] = stat_value
                        else:
                            prop = list(weighted_property_values.keys())[0]
                            targets[stat] = {prop: stat_value}
        
        # Find closest cases for accumulated targets
        if targets:
            found_cases = self._find_closest_cases(
                weighted_property_values, targets, weights,
                resolved, filters, multiplier, decimals,
                return_references=return_references
            )
            closest_cases.extend(found_cases)
        
        return closest_cases
    
    # ================================================================
    # PUBLIC API - INFORMATION ACCESS
    # ================================================================
    
    def parameters(self) -> List[str]:
        """Get list of all available parameter names.
        
        Returns:
            List of parameter names (sheet names from Excel file)
        """
        return list(self.data.keys())
    
    @lru_cache(maxsize=128)
    def properties(self, parameter: str = None) -> List[str]:
        """Get list of unique properties for a parameter.
        
        Results are cached for performance.
        
        Args:
            parameter: Parameter name (defaults to first if only one)
            
        Returns:
            Sorted list of property names
        """
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
        """Get unique values for a dynamic field (e.g., zones, regions).
        
        Results are cached for performance.
        
        Args:
            field: Field name to get unique values for
            parameter: Parameter name (defaults to first if only one)
            
        Returns:
            Sorted list of unique values for the field
        """
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
        """Get metadata info for a parameter.
        
        Args:
            parameter: Parameter name (defaults to first if only one)
            
        Returns:
            Dictionary of metadata extracted from rows above the data table
        """
        resolved = self._resolve_parameter(parameter)
        return self.info.get(resolved, {})
    
    def case(
        self, 
        index_or_reference: Union[int, str], 
        parameter: str = None,
        filters: Union[Dict[str, Any], str] = None,
        property: Union[str, List[str], bool, None] = None,
        multiplier: float = None,
        decimals: int = 6,
        variables: Union[bool, List[str]] = False,
        _skip_filtering: bool = False
    ) -> Dict:
        """Get data for a specific case by index or reference.
        
        Args:
            index_or_reference: Case index (int) or case reference string (e.g., 'NTG_42' or 'p10.NTG_42')
            parameter: Parameter name (required if using index, ignored if using reference)
            filters: Filters dict, stored filter name, or None. When provided, recalculates volumes
                    based on filters (e.g., specific zones, regions). Use stored filter name to
                    apply preset filters.
            property: Property override (see compute() for details)
            multiplier: Multiplier to apply (defaults to instance default_multiplier)
            decimals: Number of decimal places for rounding
            variables: Control variable inclusion (default False):
                - False (default): No variables included
                - True: Include variables filtered by default_variables (or all if default_variables is None)
                - List[str]: Include only specified variables ($ prefix optional, e.g., ['NPV', 'Recovery'])
            
        Returns:
            Dictionary of volumetric values for that case. If filters provided, properties are
            recalculated based on those filters. By default only includes properties.
            Variables are returned without $ prefix in keys.
            If reference has a tag prefix (e.g., 'p10.NTG_42'), adds 'case' key to result.
            
        Raises:
            IndexError: If index out of range
            ValueError: If reference format is invalid
            
        Examples:
            # Get case with property override
            case_data = processor.case(42, parameter='NTG', 
                                      filters={'zones': ['z1']}, property='stoiip')
            
            # Remove property filter
            case_data = processor.case('p10.NTG_42', 
                                      filters={'zones': ['z1']}, property=False)
        """
        tag = None
        
        # Resolve filter preset if string provided
        if filters is not None:
            filters = self._resolve_filter_preset(filters)
        
        # Merge property parameter with filters
        if filters is not None:
            filters = self._merge_property_filter(filters, property)
        elif property is not None and property is not False:
            # No filters but property specified, create filters dict
            filters = {'property': property}
        
        # Determine if using index or reference
        if isinstance(index_or_reference, str):
            # Reference mode
            param, index, tag = self._parse_case_reference(index_or_reference)
        else:
            # Index mode
            index = index_or_reference
            param = self._resolve_parameter(parameter)
        
        # Set defaults
        if multiplier is None:
            multiplier = self.default_multiplier
        
        # Check if we need to recalculate with filters
        if filters and not _skip_filtering:
            # Filters provided - recalculate all properties for this case
            case_data = self._get_case_details(
                index, param, filters, multiplier, None, decimals
            )
        else:
            # No filters - return raw case data
            df = self.data[param]
            
            if index < 0 or index >= len(df):
                raise IndexError(f"Index {index} out of range (0–{len(df)-1})")
            
            case_data = df[index].to_dicts()[0]
            
            # For raw data without filters, still create hierarchical properties
            if not _skip_filtering:
                properties_dict = self._parse_case_to_hierarchy(case_data, param, decimals)
                case_data = {
                    'idx': index,
                    'properties': properties_dict,
                    **{k: v for k, v in case_data.items() if k.startswith('$')}
                }
        
        # Add tag as 'case' key if present
        if tag:
            case_data['case'] = tag
        
        # Skip further filtering if this is an internal call
        if _skip_filtering:
            return case_data
        
        # Handle variables inclusion/filtering
        if variables is False or variables is None:
            # Remove all variables (default behavior)
            if 'variables' in case_data:
                del case_data['variables']
            else:
                # For index mode, remove $ prefixed keys
                for k in list(case_data.keys()):
                    if k.startswith('$'):
                        del case_data[k]
        elif variables is True:
            # Include variables, filtered by default_variables if set
            var_list = self.default_variables
            if var_list is not None:
                # Filter and strip $
                self._filter_case_variables(case_data, var_list)
            else:
                # Include all variables, strip $ from keys
                if 'variables' not in case_data:
                    vars_dict = {k: v for k, v in case_data.items() if k.startswith('$')}
                    for k in list(vars_dict.keys()):
                        del case_data[k]
                    case_data['variables'] = self._strip_variable_prefix(vars_dict)
                else:
                    # Already in variables dict, just strip $
                    case_data['variables'] = self._strip_variable_prefix(case_data['variables'])
        elif isinstance(variables, list):
            # Include only specified variables
            # First ensure variables are in 'variables' dict
            if 'variables' not in case_data:
                vars_dict = {k: v for k, v in case_data.items() if k.startswith('$')}
                for k in list(vars_dict.keys()):
                    del case_data[k]
                case_data['variables'] = vars_dict
            # Filter and strip $
            self._filter_case_variables(case_data, variables)
        
        return case_data
    
    def _filter_case_variables(self, case_data: Dict, var_list: List[str]) -> None:
        """Filter variables in case_data dict in-place and strip $ from keys.
        
        Args:
            case_data: Case data dictionary to filter
            var_list: List of variable names (with or without $ prefix)
        """
        # Normalize variable names (ensure $ prefix for checking)
        normalized_vars = [self._normalize_variable_name(v) for v in var_list]
        
        if 'variables' in case_data:
            # Reference mode: variables are in a 'variables' dict
            filtered_variables = {
                k: v for k, v in case_data['variables'].items()
                if k in normalized_vars
            }
            # Strip $ prefix from keys in output
            case_data['variables'] = self._strip_variable_prefix(filtered_variables)
        else:
            # Index mode: variables are directly in case_data with $ prefix
            # Keep only requested variables, then strip $ and move to 'variables' dict
            filtered_vars = {
                k: v for k, v in case_data.items() 
                if k.startswith('$') and k in normalized_vars
            }
            # Remove $ prefixed keys from case_data
            for k in list(case_data.keys()):
                if k.startswith('$'):
                    del case_data[k]
            # Add back as 'variables' dict without $ prefix
            if filtered_vars:
                case_data['variables'] = self._strip_variable_prefix(filtered_vars)
    
    def case_variables(
        self,
        index_or_reference: Union[int, str],
        parameter: str = None,
        variables: List[str] = None
    ) -> Dict[str, Any]:
        """Get only the variables for a specific case.
        
        Variable names in the returned dict will not have the $ prefix.
        
        Args:
            index_or_reference: Case index (int) or case reference string (e.g., 'NTG_42' or 'p10.NTG_42')
            parameter: Parameter name (required if using index, ignored if using reference)
            variables: List of variable names to include (with or without $ prefix).
                      If None, uses default_variables. If default_variables is also None, 
                      returns all variables.
            
        Returns:
            Dictionary with only variable values (keys without $ prefix)
            
        Examples:
            # Get all variables
            vars = processor.case_variables(42, parameter='NTG')
            # Returns: {'NPV': 123.4, 'Recovery': 0.45, 'EUR': 89.2}
            
            # Get filtered variables (using default_variables)
            processor.default_variables = ['NPV', 'Recovery']  # $ optional
            vars = processor.case_variables('p10.NTG_42')
            # Returns: {'NPV': 123.4, 'Recovery': 0.45}
            
            # Get specific variables ($ prefix optional)
            vars = processor.case_variables('NTG_42', variables=['EUR', 'NPV'])
            # Returns: {'EUR': 89.2, 'NPV': 123.4}
        """
        # Use default_variables if not specified
        if variables is None:
            variables = self.default_variables
        
        # Parse reference if string
        if isinstance(index_or_reference, str):
            param, index, tag = self._parse_case_reference(index_or_reference)
            resolved = param
        else:
            index = index_or_reference
            resolved = self._resolve_parameter(parameter)
        
        # Get raw case data
        df = self.data[resolved]
        
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} out of range (0–{len(df)-1})")
        
        case_data = df[index].to_dicts()[0]
        
        # Extract only variables ($ prefixed keys)
        all_variables = {k: v for k, v in case_data.items() if k.startswith('$')}
        
        # Filter if variable list provided (normalize names first)
        if variables is not None:
            normalized_vars = [self._normalize_variable_name(v) for v in variables]
            all_variables = {k: v for k, v in all_variables.items() if k in normalized_vars}
        
        # Strip $ prefix from keys
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
        """Shared logic for base_case and ref_case methods.
        
        Args:
            case_type: Either 'base' or 'reference' to identify which case to extract
            property: Property name (if None, returns all values)
            filters: Filters to apply or name of stored filter preset
            multiplier: Multiplier to apply (defaults to instance default_multiplier)
        
        Returns:
            Single value if property specified, dict of all values otherwise
        """
        # Resolve filter preset if string provided
        filters = self._resolve_filter_preset(filters)
        
        # Extract property from filters if present (takes precedence)
        if filters and 'property' in filters:
            filters = filters.copy()
            property = filters.pop('property')
        
        # Normalize property name to lowercase
        if property:
            property = self._normalize_fieldname(property)
        
        # Use default multiplier if not specified
        if multiplier is None:
            multiplier = self.default_multiplier
        
        # Extract case values - either with filters or from cached values
        if filters or multiplier != self.default_multiplier:
            # Extract directly with specified filters and multiplier
            case_index = 0 if case_type == 'base' else 1
            case_values = self._extract_case(
                self.base_case_parameter,
                case_index=case_index,
                filters=filters,
                multiplier=multiplier
            )
        else:
            # Use cached values (no filters, default multiplier)
            case_values = self.base_case_values if case_type == 'base' else self.reference_case_values
        
        # Return specific property or all values
        if property:
            if property not in case_values:
                raise KeyError(
                    f"Property '{property}' not found in case. "
                    f"Available: {list(case_values.keys())}"
                )
            return case_values[property]
        
        return case_values.copy()
    
    def base_case(
        self,
        property: str = None,
        filters: Union[Dict[str, Any], str] = None,
        multiplier: float = None
    ) -> Union[float, Dict[str, float]]:
        """Get base case value(s) from first row of base_case parameter.
        
        Args:
            property: Property name (if None, returns all base case values)
            filters: Filters dict or stored filter name
            multiplier: Override default multiplier
            
        Returns:
            Single value if property specified, dict of all values otherwise
            
        Examples:
            # Get all base case values
            base = processor.base_case()
            
            # Get specific property
            stoiip_base = processor.base_case('stoiip')
            
            # With filters
            stoiip_base = processor.base_case('stoiip', filters={'zones': ['z1', 'z2']})
            
            # With stored filter
            stoiip_base = processor.base_case('stoiip', filters='north_zones')
            
            # Override multiplier
            stoiip_base = processor.base_case('stoiip', multiplier=1e-6)
        """
        return self._get_case_values('base', property, filters, multiplier)
    
    def ref_case(
        self,
        property: str = None,
        filters: Union[Dict[str, Any], str] = None,
        multiplier: float = None
    ) -> Union[float, Dict[str, float]]:
        """Get reference case value(s) from second row of base_case parameter.
        
        Args:
            property: Property name (if None, returns all reference case values)
            filters: Filters dict or stored filter name
            multiplier: Override default multiplier
            
        Returns:
            Single value if property specified, dict of all values otherwise
            
        Examples:
            # Get all reference case values
            ref = processor.ref_case()
            
            # Get specific property
            stoiip_ref = processor.ref_case('stoiip')
            
            # With filters
            stoiip_ref = processor.ref_case('stoiip', filters={'zones': ['z1', 'z2']})
            
            # With stored filter
            stoiip_ref = processor.ref_case('stoiip', filters='north_zones')
            
            # Override multiplier
            stoiip_ref = processor.ref_case('stoiip', multiplier=1e-6)
        """
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

        Args:
            stats: Statistic(s) to compute ('p90p10', 'mean', 'median', 'minmax', 'percentile', 'distribution')
            parameter: Parameter name (defaults to first if only one available)
            filters: Filters dict or stored filter name
            property: Property override:
                - None (default): Use property from filters
                - str or List[str]: Override property in filters
                - False: Remove property filter (compute across all properties)
            multiplier: Override default multiplier
            options: Stats-specific options:
                - decimals: Number of decimal places (default 6)
                - p90p10_threshold: Minimum case count for p90p10 (default 10)
                - p: Percentile value for 'percentile' stat (default 50)
                - skip: List of keys to skip in output (e.g., ['errors', 'filters'])
                  Note: 'filters' includes multiplier
            case_selection: Whether to find closest matching cases (default False)
            selection_criteria: Criteria for selecting cases (only used if case_selection=True):
                - weights: dict of property weights (e.g., {'stoiip': 0.6, 'giip': 0.4})
                - combinations: list of filter+property combinations for complex weighting
            return_case_references: If True (default), return lightweight references instead of full details

        Returns:
            Dictionary with computed statistics and applied filters (including multiplier). 
            Optionally includes closest_cases (as references or full details).
            
        Examples:
            # Simple mean computation
            result = processor.compute('mean', filters={'property': 'stoiip'})
            # Returns: {
            #     'parameter': 'NTG',
            #     'mean': 1234.5,
            #     'filters': {
            #         'property': 'stoiip',
            #         'multiplier': 1.0
            #     }
            # }
            
            # With custom multiplier
            result = processor.compute('mean', 
                                      filters={'property': 'stoiip'}, 
                                      multiplier=1e-6)
            # Returns: {
            #     'parameter': 'NTG',
            #     'mean': 1.2345,  # In MMsm3
            #     'filters': {
            #         'property': 'stoiip',
            #         'multiplier': 1e-6
            #     }
            # }
            
            # Override property
            result = processor.compute('mean', filters={'zones': ['z1']}, property='stoiip')
            
            # Remove property filter
            result = processor.compute('mean', filters={'zones': ['z1']}, property=False)
            
            # With case selection (returns references)
            result = processor.compute(
                'mean',
                filters={'property': 'stoiip'},
                case_selection=True,
                selection_criteria={'stoiip': 0.6, 'giip': 0.4}
            )
        """
        resolved = self._resolve_parameter(parameter)
        
        # Resolve filter preset if string provided
        filters = self._resolve_filter_preset(filters)
        
        # Merge property parameter with filters
        filters = self._merge_property_filter(filters, property)
        
        # Use default multiplier if not specified
        if multiplier is None:
            multiplier = self.default_multiplier

        options = options or {}
        selection_criteria = selection_criteria or {}
        skip = options.get("skip", [])
        decimals = options.get("decimals", 6)
        
        # Check if property is a list (multi-property mode)
        property_filter = filters.get("property")
        is_multi_property = isinstance(property_filter, list)
        
        if isinstance(stats, str):
            stats = [stats]
        
        result = {"parameter": resolved}
        
        # Helper to round
        def _round(val):
            return round(val, decimals)
        
        # Extract values for all properties
        property_values = {}
        
        if is_multi_property:
            non_property_filters = {k: v for k, v in filters.items() if k != "property"}
            
            for prop in property_filter:
                try:
                    prop_filters = {**non_property_filters, "property": prop}
                    prop_vals, _ = self._extract_values(resolved, prop_filters, multiplier)
                    prop_vals = self._validate_numeric(prop_vals, prop)
                    property_values[prop] = prop_vals
                except Exception as e:
                    if "errors" not in skip:
                        result["errors"] = result.get("errors", [])
                        result["errors"].append(f"Failed to extract {prop}: {e}")
        else:
            # Single property mode
            prop = filters.get("property", "value")
            try:
                values, _ = self._extract_values(resolved, filters, multiplier)
                values = self._validate_numeric(values, prop)
                property_values[prop] = values
            except Exception as e:
                if "errors" not in skip:
                    result["errors"] = result.get("errors", [])
                    result["errors"].append(f"Failed to extract {prop}: {e}")
                return result
        
        # If no properties were successfully extracted, return early
        if not property_values:
            if "errors" not in skip and "errors" not in result:
                result["errors"] = ["No data could be extracted for any property"]
            return result
        
        # Compute ALL stats efficiently in one pass using the new engine
        stats_result = self._compute_all_stats(
            property_values,
            stats,
            options,
            decimals,
            skip
        )
        result.update(stats_result)
        
        # Handle case selection separately (only if requested)
        if case_selection and "closest_case" not in skip:
            closest_cases = self._perform_case_selection(
                property_values,
                stats,
                stats_result,
                selection_criteria,
                resolved,
                filters,
                multiplier,
                skip,
                decimals,
                return_references=return_case_references
            )
            if closest_cases:
                result["closest_cases"] = closest_cases
        
        # Add filters with multiplier for full transparency (unless skipped)
        if "filters" not in skip:
            filters_output = filters.copy() if filters else {}
            filters_output['multiplier'] = multiplier
            result["filters"] = filters_output
        
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
        """Compute statistics for multiple parameters.

        Args:
            stats: Statistic(s) to compute
            parameters: Parameter name(s) or "all"
            filters: Filters dict or stored filter name
            property: Property override (see compute() for details)
            multiplier: Override default multiplier
            options: Stats-specific options (see compute() docstring)
            case_selection: Whether to find closest matching cases
            selection_criteria: Criteria for selecting cases (see compute() docstring)
            include_base_case: If True, add base case values to results (default True)
            include_reference_case: If True, add reference case values to results (default True)
            sort_by_range: If True, sort by p90p10 or minmax range (default True)
            return_case_references: If True (default), return references instead of full details

        Returns:
            List of result dictionaries with statistics and filters (including multiplier)
            (or single dict if only one parameter)
            
        Examples:
            # Compute for all parameters with property override
            results = processor.compute_batch(
                'p90p10',
                filters={'zones': ['z1']},
                property='stoiip'
            )
            # Returns: [
            #     {
            #         'parameter': 'NTG',
            #         'p90p10': [1100, 1400],
            #         'filters': {
            #             'zones': ['z1'], 
            #             'property': 'stoiip',
            #             'multiplier': 1.0
            #         }
            #     },
            #     ...
            # ]
        """
        # Resolve filter preset if string provided
        filters = self._resolve_filter_preset(filters)
        
        # Merge property parameter with filters
        filters = self._merge_property_filter(filters, property)

        # Use default multiplier if not specified
        if multiplier is None:
            multiplier = self.default_multiplier
            
        if parameters == "all":
            param_list = list(self.data.keys())
            # Automatically skip base_case parameter when using "all"
            if self.base_case_parameter and self.base_case_parameter in param_list:
                param_list = [p for p in param_list if p != self.base_case_parameter]
        elif isinstance(parameters, str):
            param_list = [parameters]
        else:
            param_list = parameters

        options = options or {}
        skip = options.get("skip", [])
        skip_parameters = options.get("skip_parameters", [])
        
        # Filter out parameters in skip_parameters list
        all_param_names = set(self.data.keys())
        param_list = [p for p in param_list if p not in skip_parameters and p not in (skip if p in all_param_names else [])]
        
        results = []

        # Add base/reference case as first entry if available
        # Use the public API methods which properly handle filters and multipliers
        if self.base_case_parameter and (include_base_case or include_reference_case):
            case_entry = {"parameter": self.base_case_parameter}
            
            # Determine which property value to use
            prop_to_use = None
            if filters and "property" in filters:
                prop_filter = filters["property"]
                if isinstance(prop_filter, str):
                    prop_to_use = prop_filter
                elif isinstance(prop_filter, list) and len(prop_filter) > 0:
                    prop_to_use = prop_filter[0]
            
            # Add base case value using public API
            if include_base_case:
                try:
                    base_val = self.base_case(property=prop_to_use, filters=filters, multiplier=multiplier)
                    if isinstance(base_val, dict):
                        # Multiple properties - use first one
                        case_entry["base_case"] = next(iter(base_val.values()))
                    else:
                        case_entry["base_case"] = base_val
                except Exception as e:
                    if "errors" not in skip:
                        case_entry["errors"] = case_entry.get("errors", [])
                        case_entry["errors"].append(f"Failed to extract base_case: {e}")
            
            # Add reference case value using public API
            if include_reference_case:
                try:
                    ref_val = self.ref_case(property=prop_to_use, filters=filters, multiplier=multiplier)
                    if isinstance(ref_val, dict):
                        # Multiple properties - use first one
                        case_entry["reference_case"] = next(iter(ref_val.values()))
                    else:
                        case_entry["reference_case"] = ref_val
                except Exception as e:
                    # Only report error if it's not about empty/missing reference case
                    error_msg = str(e)
                    if "errors" not in skip and "Available: []" not in error_msg:
                        case_entry["errors"] = case_entry.get("errors", [])
                        case_entry["errors"].append(f"Failed to extract reference_case: {e}")
            
            # Always add the entry (even if extraction failed, we'll show errors)
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

        # Sort results by range if requested
        if sort_by_range and len(results) > 1:
            # Separate ref_case entry (always first) from other results
            ref_case_entry = None
            other_results = []
            
            for result in results:
                param_name = result.get("parameter", "")
                if param_name == self.base_case_parameter:
                    ref_case_entry = result
                else:
                    other_results.append(result)
            
            # Calculate sort key for each result
            def get_sort_key(result):
                # Try p90p10 first (preferred)
                if "p90p10" in result and "errors" not in result:
                    p90p10 = result["p90p10"]
                    if isinstance(p90p10, list) and len(p90p10) == 2:
                        # Single property: [p10, p90]
                        return p90p10[1] - p90p10[0]
                    elif isinstance(p90p10, list) and len(p90p10) == 2 and isinstance(p90p10[0], dict):
                        # Multi-property: [{prop: p10_val}, {prop: p90_val}]
                        # Use first property's range
                        first_prop = list(p90p10[0].keys())[0]
                        return p90p10[1][first_prop] - p90p10[0][first_prop]
                
                # Fall back to minmax
                if "minmax" in result:
                    minmax = result["minmax"]
                    if isinstance(minmax, list) and len(minmax) == 2:
                        # Single property: [min, max]
                        return minmax[1] - minmax[0]
                    elif isinstance(minmax, list) and len(minmax) == 2 and isinstance(minmax[0], dict):
                        # Multi-property: [{prop: min_val}, {prop: max_val}]
                        # Use first property's range
                        first_prop = list(minmax[0].keys())[0]
                        return minmax[1][first_prop] - minmax[0][first_prop]
                
                # No sortable stat - use very small number to keep at end
                return -float('inf')
            
            # Sort by range (descending - largest range first)
            other_results.sort(key=get_sort_key, reverse=True)
            
            # Reconstruct results list with ref_case first
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
        """Compute tornado chart statistics (minmax and p90p10) for all parameters.
        
        This is a convenience wrapper around compute_batch() with stats=['minmax', 'p90p10']
        and parameters="all" predefined. Perfect for quickly generating tornado chart data.
        
        Args:
            filters: Filters dict or stored filter name
            property: Property override (see compute() for details)
            multiplier: Override default multiplier
            skip: Single field or list of fields to skip in output (e.g., 'filters' or ['filters', 'errors'])
                  Note: 'filters' includes multiplier
            options: Additional stats-specific options (merged with skip)
            case_selection: Whether to find closest matching cases
            selection_criteria: Criteria for selecting cases
            include_base_case: Include base case values (default True)
            include_reference_case: Include reference case values (default True)
            sort_by_range: Sort by range (default True)
            return_case_references: Return references instead of full details (default True)
            
        Returns:
            List of result dictionaries with minmax, p90p10 statistics, and filters 
            (including multiplier) for all parameters
            
        Examples:
            # Simple tornado with property override
            results = processor.tornado(
                filters={'zones': ['z1']},
                property='stoiip'
            )
            # Returns: [
            #     {
            #         'parameter': 'NTG',
            #         'minmax': [1000, 1500],
            #         'p90p10': [1100, 1400],
            #         'filters': {
            #             'zones': ['z1'], 
            #             'property': 'stoiip',
            #             'multiplier': 1.0
            #         }
            #     },
            #     ...
            # ]
            
            # Skip filters in output
            results = processor.tornado(
                property='stoiip',
                skip='filters'
            )
        """
        # Merge skip into options
        merged_options = options.copy() if options else {}
        
        if skip is not None:
            # Convert string to list if needed
            skip_list = [skip] if isinstance(skip, str) else skip
            
            # Merge with existing skip in options if present
            existing_skip = merged_options.get('skip', [])
            if isinstance(existing_skip, str):
                existing_skip = [existing_skip]
            
            # Combine and deduplicate
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
        """Get full distribution of values (convenience method).
        
        Args:
            parameter: Parameter name (defaults to first)
            filters: Filters dict or stored filter name
            property: Property override (see compute() for details)
            multiplier: Override default multiplier
            options: Options dict (decimals, skip, etc.)
            
        Returns:
            Array of values or dict of arrays (for multi-property)
            
        Examples:
            # Get distribution with property override
            values = processor.distribution(filters={'zones': ['z1']}, property='stoiip')
            
            # Remove property filter
            values = processor.distribution(filters={'zones': 'z1'}, property=False)
        """
        result = self.compute(
            stats="distribution",
            parameter=parameter,
            filters=filters,
            property=property,
            multiplier=multiplier,
            options=options
        )

        return result["distribution"]