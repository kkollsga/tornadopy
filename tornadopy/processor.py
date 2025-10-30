import re
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
        self.base_case_parameter: str = base_case
        self.base_case_values: Dict[str, float] = {}
        self.reference_case_values: Dict[str, float] = {}

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
    
    def _normalize_fieldname(self, name: str) -> str:
        """Normalize field name to lowercase with underscores."""
        name = str(name).strip().lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        name = re.sub(r"_+$", "", name)
        return name or "property"
    
    def _parse_sheet(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame, List[str], Dict]:
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
            property_name = parts[-1] if parts else col_name
            
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
                data, metadata, fields, info = self._parse_sheet(df_raw)
                self.data[sheet_name] = data
                self.metadata[sheet_name] = metadata
                self.dynamic_fields[sheet_name] = fields
                self.info[sheet_name] = info
            except Exception as e:
                print(f"[!] Skipped sheet '{sheet_name}': {e}")
    
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
    
    # ================================================================
    # DATA EXTRACTION & VALIDATION
    # ================================================================
    
    def _resolve_parameter(self, parameter: str = None) -> str:
        """Resolve parameter name, defaulting to first if None."""
        if parameter is None:
            return list(self.data.keys())[0]
        return parameter
    
    def _normalize_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize filter keys and string values to lowercase."""
        normalized = {}
        
        for key, value in filters.items():
            key_norm = self._normalize_fieldname(key)
            
            if isinstance(value, str):
                value_norm = value.strip().lower()
            elif isinstance(value, list):
                value_norm = [v.strip().lower() if isinstance(v, str) else v for v in value]
            else:
                value_norm = value
            
            normalized[key_norm] = value_norm
        
        return normalized
    
    def _select_columns(self, parameter: str, filters: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Select columns matching filters and return column names and sources."""
        if parameter not in self.metadata or self.metadata[parameter].is_empty():
            return [], []
        
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
        return column_names, column_names
    
    def _extract_values(
        self,
        parameter: str,
        filters: Dict[str, Any],
        multiplier: float = 1.0
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract and sum values for columns matching filters."""
        list_fields = {k: v for k, v in filters.items() if isinstance(v, list)}
        
        if list_fields:
            arrays = []
            all_sources = []
            
            for field, values in list_fields.items():
                for value in values:
                    single_filters = {**filters, field: value}
                    cols, sources = self._select_columns(parameter, single_filters)
                    
                    df = self.data[parameter]
                    arr = (
                        df.select(cols)
                        .select(pl.sum_horizontal(pl.all().cast(pl.Float64, strict=False)))
                        .to_series()
                        .to_numpy()
                    )
                    arrays.append(arr)
                    all_sources.extend(sources)
            
            combined = np.sum(np.vstack(arrays), axis=0) * multiplier
            return combined, all_sources
        
        cols, sources = self._select_columns(parameter, filters)
        df = self.data[parameter]
        
        values = (
            df.select(cols)
            .select(pl.sum_horizontal(pl.all().cast(pl.Float64, strict=False)))
            .to_series()
            .to_numpy()
        ) * multiplier
        
        return values, sources
    
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
    
    def _find_closest_cases(
        self,
        property_values: Dict[str, np.ndarray],
        targets: Dict[str, Dict[str, float]],
        weights: Dict[str, float],
        resolved: str,
        filters: Dict[str, Any],
        multiplier: float,
        decimals: int
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
            
        Returns:
            List of case detail dictionaries
        """
        closest_cases = []
        first_prop = list(property_values.keys())[0]
        
        for case_type, case_targets in targets.items():
            distances = self._calculate_weighted_distance(property_values, case_targets, weights)
            idx = np.argmin(distances)
            
            case_details = self._get_case_details(
                int(idx), resolved, filters, multiplier,
                property_values[first_prop][idx], decimals
            )
            case_details["case"] = case_type
            case_details["weights"] = weights
            case_details["weighted_distance"] = round(distances[idx], decimals)
            
            # Add selection_values showing actual values and targets for this case
            selection_values = {}
            for prop in weights.keys():
                if prop in property_values:
                    # Add actual case value
                    selection_values[prop] = round(property_values[prop][idx], decimals)
                    # Add target value
                    if prop in case_targets:
                        selection_values[f"{case_type}_{prop}"] = round(case_targets[prop], decimals)
            case_details["selection_values"] = selection_values
            
            closest_cases.append(case_details)
        
        return closest_cases
    
    def _get_case_details(
        self,
        index: int,
        parameter: str,
        filters: Dict[str, Any],
        multiplier: float,
        value: float,
        decimals: int = 6
    ) -> Dict:
        """Extract detailed information for a specific case.
        
        Args:
            index: Case index
            parameter: Parameter name
            filters: Applied filters
            multiplier: Applied multiplier
            value: Computed value for this case (already with multiplier applied)
            decimals: Number of decimal places for rounding
            
        Returns:
            Dictionary with case details including idx, property value, filters, multiplier, properties, and variables
        """
        case_data = self.case(index, parameter=parameter)
        
        # Get all available properties for this parameter
        try:
            all_properties = self.properties(parameter)
        except:
            all_properties = []
        
        # Calculate all property values for this case WITHOUT multiplier
        properties_dict = {}
        non_property_filters = {k: v for k, v in filters.items() if k != "property"}
        
        for prop in all_properties:
            try:
                prop_filters = {**non_property_filters, "property": prop}
                values, _ = self._extract_values(parameter, prop_filters, multiplier=1.0)  # No multiplier
                if index < len(values):
                    properties_dict[prop] = round(values[index], decimals)
            except:
                # Skip properties that can't be calculated
                pass
        
        # Handle property filter - if it's a list, use first property; if string, use it
        property_filter = filters.get("property")
        if isinstance(property_filter, list):
            property_key = property_filter[0] if property_filter else "value"
        else:
            property_key = property_filter if property_filter else "value"
        
        details = {
            "idx": index,
            **{property_key: round(value, decimals)},
            **{k: v for k, v in filters.items() if k != "property"},
            "multiplier": multiplier,
            "properties": properties_dict,
            "variables": {k: v for k, v in case_data.items() if k.startswith("$")}
        }
        
        return details
    
    # ================================================================
    # STATISTICS COMPUTATION
    # ================================================================
    
    def _compute_p90p10(
        self,
        property_values: Dict[str, np.ndarray],
        resolved: str,
        filters: Dict[str, Any],
        multiplier: float,
        options: Dict[str, Any],
        case_selection: bool,
        selection_criteria: Dict[str, Any],
        skip: List[str],
        decimals: int,
        _round
    ) -> Dict:
        """Compute p10 and p90 percentiles with optional case selection."""
        result = {}
        threshold = options.get("p90p10_threshold", 10)
        
        # Calculate percentiles for each property in property_values
        p10_dict = {}
        p90_dict = {}
        
        for prop, prop_vals in property_values.items():
            if threshold and len(prop_vals) < threshold:
                if "errors" not in skip:
                    result["errors"] = result.get("errors", [])
                    result["errors"].append(
                        f"Too few cases ({len(prop_vals)}) for {prop} p90p10; threshold={threshold}"
                    )
            else:
                p10_val, p90_val = np.percentile(prop_vals, [10, 90])
                p10_dict[prop] = _round(float(p10_val))
                p90_dict[prop] = _round(float(p90_val))
        
        if not p10_dict or not p90_dict:
            return result
        
        # Format result based on single vs multi-property
        if len(property_values) == 1:
            prop = list(property_values.keys())[0]
            result["p90p10"] = [p10_dict[prop], p90_dict[prop]]
        else:
            result["p90p10"] = [p10_dict, p90_dict]
        
        # Handle case selection
        if case_selection and "closest_case" not in skip:
            combinations = selection_criteria.get("combinations")
            
            if combinations:
                # Use weighted combinations approach
                closest_cases = []
                
                # Calculate distances for p10 and p90
                distances_p10, metadata_p10 = self._calculate_multi_combination_distance(
                    combinations, resolved, filters, multiplier, "p10", skip
                )
                distances_p90, metadata_p90 = self._calculate_multi_combination_distance(
                    combinations, resolved, filters, multiplier, "p90", skip
                )
                
                idx_p10 = np.argmin(distances_p10)
                idx_p90 = np.argmin(distances_p90)
                
                # Get case details
                first_prop = list(property_values.keys())[0]
                case_p10 = self._get_case_details(
                    int(idx_p10), resolved, filters, multiplier,
                    property_values[first_prop][idx_p10], decimals
                )
                case_p10["case"] = "p10"
                case_p10["selection_method"] = "weighted_combinations"
                case_p10["weighted_distance"] = _round(distances_p10[idx_p10])
                
                # Add selection_values as a list corresponding to combinations
                selection_values_list = []
                for combo in combinations:
                    combo_filters = combo.get("filters", {})
                    property_weights = combo.get("properties", {})
                    merged_filters = {**{k: v for k, v in filters.items() if k != "property"}, **combo_filters}
                    
                    combo_values = {}
                    for prop in property_weights.keys():
                        try:
                            prop_filters = {**merged_filters, "property": prop}
                            prop_vals, _ = self._extract_values(resolved, prop_filters, multiplier)
                            if idx_p10 < len(prop_vals):
                                # Add actual case value
                                combo_values[prop] = _round(prop_vals[idx_p10])
                                # Add target p10 value
                                p10_target = np.percentile(prop_vals, 10)
                                combo_values[f"p10_{prop}"] = _round(p10_target)
                        except:
                            pass
                    selection_values_list.append(combo_values)
                
                case_p10["selection_values"] = selection_values_list
                closest_cases.append(case_p10)
                
                case_p90 = self._get_case_details(
                    int(idx_p90), resolved, filters, multiplier,
                    property_values[first_prop][idx_p90], decimals
                )
                case_p90["case"] = "p90"
                case_p90["selection_method"] = "weighted_combinations"
                case_p90["weighted_distance"] = _round(distances_p90[idx_p90])
                
                # Add selection_values as a list corresponding to combinations
                selection_values_list = []
                for combo in combinations:
                    combo_filters = combo.get("filters", {})
                    property_weights = combo.get("properties", {})
                    merged_filters = {**{k: v for k, v in filters.items() if k != "property"}, **combo_filters}
                    
                    combo_values = {}
                    for prop in property_weights.keys():
                        try:
                            prop_filters = {**merged_filters, "property": prop}
                            prop_vals, _ = self._extract_values(resolved, prop_filters, multiplier)
                            if idx_p90 < len(prop_vals):
                                # Add actual case value
                                combo_values[prop] = _round(prop_vals[idx_p90])
                                # Add target p90 value
                                p90_target = np.percentile(prop_vals, 90)
                                combo_values[f"p90_{prop}"] = _round(p90_target)
                        except:
                            pass
                    selection_values_list.append(combo_values)
                
                case_p90["selection_values"] = selection_values_list
                closest_cases.append(case_p90)
                
                result["closest_cases"] = closest_cases
            else:
                # Use simple property weights
                weighted_property_values, weights, errors = self._prepare_weighted_case_selection(
                    property_values, selection_criteria, resolved, filters, multiplier, skip
                )
                
                if errors and "errors" not in skip:
                    result["errors"] = result.get("errors", [])
                    result["errors"].extend(errors)
                
                # Calculate percentiles for weighted properties
                weighted_p10_dict = {}
                weighted_p90_dict = {}
                
                for prop in weights.keys():
                    if prop in weighted_property_values:
                        if prop in p10_dict:
                            weighted_p10_dict[prop] = p10_dict[prop]
                            weighted_p90_dict[prop] = p90_dict[prop]
                        else:
                            p10_val, p90_val = np.percentile(weighted_property_values[prop], [10, 90])
                            weighted_p10_dict[prop] = p10_val
                            weighted_p90_dict[prop] = p90_val
                
                targets = {
                    "p10": weighted_p10_dict,
                    "p90": weighted_p90_dict
                }
                
                result["closest_cases"] = self._find_closest_cases(
                    weighted_property_values, targets, weights, resolved, filters, multiplier, decimals
                )
        
        return result
    
    def _compute_mean(
        self,
        property_values: Dict[str, np.ndarray],
        resolved: str,
        filters: Dict[str, Any],
        multiplier: float,
        options: Dict[str, Any],
        case_selection: bool,
        selection_criteria: Dict[str, Any],
        skip: List[str],
        decimals: int,
        _round
    ) -> Dict:
        """Compute mean with optional case selection."""
        mean_dict = {prop: _round(float(np.mean(vals))) for prop, vals in property_values.items()}
        
        result = {}
        if len(property_values) == 1:
            result["mean"] = mean_dict[list(property_values.keys())[0]]
        else:
            result["mean"] = mean_dict
        
        # Handle case selection
        if case_selection and "closest_case" not in skip:
            weighted_property_values, weights, errors = self._prepare_weighted_case_selection(
                property_values, selection_criteria, resolved, filters, multiplier, skip
            )
            
            if errors and "errors" not in skip:
                result["errors"] = result.get("errors", [])
                result["errors"].extend(errors)
            
            # Calculate means for weighted properties
            weighted_mean_dict = {}
            for prop in weights.keys():
                if prop in weighted_property_values:
                    if prop in mean_dict:
                        weighted_mean_dict[prop] = mean_dict[prop]
                    else:
                        weighted_mean_dict[prop] = _round(np.mean(weighted_property_values[prop]))
            
            targets = {"mean": weighted_mean_dict}
            
            result["closest_cases"] = self._find_closest_cases(
                weighted_property_values, targets, weights, resolved, filters, multiplier, decimals
            )
        
        return result
    
    def _compute_median(
        self,
        property_values: Dict[str, np.ndarray],
        resolved: str,
        filters: Dict[str, Any],
        multiplier: float,
        options: Dict[str, Any],
        case_selection: bool,
        selection_criteria: Dict[str, Any],
        skip: List[str],
        decimals: int,
        _round
    ) -> Dict:
        """Compute median with optional case selection."""
        median_dict = {prop: _round(float(np.median(vals))) for prop, vals in property_values.items()}
        
        result = {}
        if len(property_values) == 1:
            result["median"] = median_dict[list(property_values.keys())[0]]
        else:
            result["median"] = median_dict
        
        # Handle case selection
        if case_selection and "closest_case" not in skip:
            weighted_property_values, weights, errors = self._prepare_weighted_case_selection(
                property_values, selection_criteria, resolved, filters, multiplier, skip
            )
            
            if errors and "errors" not in skip:
                result["errors"] = result.get("errors", [])
                result["errors"].extend(errors)
            
            # Calculate medians for weighted properties
            weighted_median_dict = {}
            for prop in weights.keys():
                if prop in weighted_property_values:
                    if prop in median_dict:
                        weighted_median_dict[prop] = median_dict[prop]
                    else:
                        weighted_median_dict[prop] = _round(np.median(weighted_property_values[prop]))
            
            targets = {"median": weighted_median_dict}
            
            result["closest_cases"] = self._find_closest_cases(
                weighted_property_values, targets, weights, resolved, filters, multiplier, decimals
            )
        
        return result
    
    def _compute_minmax(
        self,
        property_values: Dict[str, np.ndarray],
        resolved: str,
        filters: Dict[str, Any],
        multiplier: float,
        case_selection: bool,
        selection_criteria: Dict[str, Any],
        skip: List[str],
        decimals: int,
        _round
    ) -> Dict:
        """Compute min and max with exact case matching."""
        min_dict = {prop: _round(float(np.min(vals))) for prop, vals in property_values.items()}
        max_dict = {prop: _round(float(np.max(vals))) for prop, vals in property_values.items()}

        if len(property_values) == 1:
            prop = list(property_values.keys())[0]
            result = {"minmax": [min_dict[prop], max_dict[prop]]}
        else:
            result = {"minmax": [min_dict, max_dict]}
        
        # Minmax always finds exact matching cases (no complex selection criteria needed)
        if case_selection and "closest_case" not in skip:
            closest_cases = []
            first_prop = list(property_values.keys())[0]
            
            # Find exact min case
            idx_min = np.argmin(property_values[first_prop])
            min_value = _round(property_values[first_prop][idx_min])
            case_min = self._get_case_details(
                int(idx_min), resolved, filters, multiplier,
                property_values[first_prop][idx_min], decimals
            )
            case_min["case"] = "min"
            case_min["selection_method"] = "exact"
            
            # Add selection_values with target (for minmax, actual = target)
            selection_values = {}
            for prop in property_values.keys():
                actual_val = _round(property_values[prop][idx_min])
                selection_values[prop] = actual_val
                selection_values[f"min_{prop}"] = actual_val  # Target is same as actual
            case_min["selection_values"] = selection_values
            closest_cases.append(case_min)
            
            # Find exact max case
            idx_max = np.argmax(property_values[first_prop])
            max_value = _round(property_values[first_prop][idx_max])
            case_max = self._get_case_details(
                int(idx_max), resolved, filters, multiplier,
                property_values[first_prop][idx_max], decimals
            )
            case_max["case"] = "max"
            case_max["selection_method"] = "exact"
            
            # Add selection_values with target
            selection_values = {}
            for prop in property_values.keys():
                actual_val = _round(property_values[prop][idx_max])
                selection_values[prop] = actual_val
                selection_values[f"max_{prop}"] = actual_val  # Target is same as actual
            case_max["selection_values"] = selection_values
            closest_cases.append(case_max)
            
            result["closest_cases"] = closest_cases
        
        return result
    
    def _compute_percentile(
        self,
        property_values: Dict[str, np.ndarray],
        resolved: str,
        filters: Dict[str, Any],
        multiplier: float,
        options: Dict[str, Any],
        case_selection: bool,
        selection_criteria: Dict[str, Any],
        skip: List[str],
        decimals: int,
        _round
    ) -> Dict:
        """Compute arbitrary percentile with optional case selection."""
        p = options.get("p", 50)
        perc_dict = {prop: _round(float(np.percentile(vals, p))) for prop, vals in property_values.items()}
        
        result = {}
        if len(property_values) == 1:
            result[f"p{p}"] = perc_dict[list(property_values.keys())[0]]
        else:
            result[f"p{p}"] = perc_dict
        
        # Handle case selection
        if case_selection and "closest_case" not in skip:
            weighted_property_values, weights, errors = self._prepare_weighted_case_selection(
                property_values, selection_criteria, resolved, filters, multiplier, skip
            )
            
            if errors and "errors" not in skip:
                result["errors"] = result.get("errors", [])
                result["errors"].extend(errors)
            
            # Calculate percentiles for weighted properties
            weighted_perc_dict = {}
            for prop in weights.keys():
                if prop in weighted_property_values:
                    if prop in perc_dict:
                        weighted_perc_dict[prop] = perc_dict[prop]
                    else:
                        weighted_perc_dict[prop] = _round(np.percentile(weighted_property_values[prop], p))
            
            targets = {f"p{p}": weighted_perc_dict}
            
            result["closest_cases"] = self._find_closest_cases(
                weighted_property_values, targets, weights, resolved, filters, multiplier, decimals
            )
        
        return result
    
    def _compute_distribution(
        self,
        property_values: Dict[str, np.ndarray],
        decimals: int,
        _round
    ) -> Dict:
        """Return full distribution of values (no aggregation).
        
        Returns numpy arrays directly for better performance.
        """
        # Use numpy's vectorized rounding instead of Python's round in a loop
        dist_dict = {prop: np.round(vals, decimals) for prop, vals in property_values.items()}

        if len(property_values) == 1:
            return {"distribution": dist_dict[list(property_values.keys())[0]]}
        else:
            return {"distribution": dist_dict}
    
    # ================================================================
    # PUBLIC API - INFORMATION ACCESS
    # ================================================================
    
    def parameters(self) -> List[str]:
        """Get list of all available parameter names.
        
        Returns:
            List of parameter names (sheet names from Excel file)
        """
        return list(self.data.keys())
    
    def properties(self, parameter: str = None) -> List[str]:
        """Get list of unique properties for a parameter.
        
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
    
    def unique_values(self, field: str, parameter: str = None) -> List[str]:
        """Get unique values for a dynamic field (e.g., zones, regions).
        
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
    
    def case(self, index: int, parameter: str = None) -> Dict:
        """Get data for a specific case by index.
        
        Args:
            index: Case index (row number, 0-based)
            parameter: Parameter name (defaults to first if only one)
            
        Returns:
            Dictionary of all values for that case
            
        Raises:
            IndexError: If index out of range
        """
        resolved = self._resolve_parameter(parameter)
        df = self.data[resolved]
        
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} out of range (0–{len(df)-1})")
        
        return df[index].to_dicts()[0]
    
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
        multiplier: float = None,
        options: Dict[str, Any] = None,
        case_selection: bool = False,
        selection_criteria: Dict[str, Any] = None
    ) -> Dict:
        """Compute statistics for a single parameter with filters.

        Args:
            stats: Statistic(s) to compute ('p90p10', 'mean', 'median', 'minmax', 'percentile', 'distribution')
            parameter: Parameter name (defaults to first if only one available)
            filters: Filters dict or stored filter name
            multiplier: Override default multiplier
            options: Stats-specific options:
                - decimals: Number of decimal places (default 6)
                - p90p10_threshold: Minimum case count for p90p10 (default 10)
                - p: Percentile value for 'percentile' stat (default 50)
                - skip: List of keys to skip in output (e.g., ['errors', 'sources'])
            case_selection: Whether to find closest matching cases (default False)
            selection_criteria: Criteria for selecting cases (only used if case_selection=True):
                - weights: dict of property weights (e.g., {'stoiip': 0.6, 'giip': 0.4})
                - combinations: list of filter+property combinations for complex weighting

        Returns:
            Dictionary with computed statistics and optionally closest_cases
            
        Examples:
            # Simple mean computation
            result = processor.compute('mean', filters={'property': 'stoiip'})
            
            # Multiple stats with stored filter
            result = processor.compute(['mean', 'p90p10'], filters='north_zones')
            
            # With custom multiplier
            result = processor.compute('p90p10', filters='north_zones', multiplier=1e-6)
            
            # With case selection
            result = processor.compute(
                'mean',
                filters={'property': 'stoiip'},
                case_selection=True,
                selection_criteria={'weights': {'stoiip': 0.6, 'giip': 0.4}}
            )
        """
        resolved = self._resolve_parameter(parameter)
        
        # Resolve filter preset if string provided
        filters = self._resolve_filter_preset(filters)
        
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
        property_sources = {}
        
        if is_multi_property:
            non_property_filters = {k: v for k, v in filters.items() if k != "property"}
            
            for prop in property_filter:
                try:
                    prop_filters = {**non_property_filters, "property": prop}
                    prop_vals, prop_sources = self._extract_values(resolved, prop_filters, multiplier)
                    prop_vals = self._validate_numeric(prop_vals, prop)
                    property_values[prop] = prop_vals
                    property_sources[prop] = prop_sources
                except Exception as e:
                    if "errors" not in skip:
                        result["errors"] = result.get("errors", [])
                        result["errors"].append(f"Failed to extract {prop}: {e}")
        else:
            # Single property mode
            prop = filters.get("property", "value")
            try:
                values, sources = self._extract_values(resolved, filters, multiplier)
                values = self._validate_numeric(values, prop)
                property_values[prop] = values
                property_sources[prop] = sources
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
        
        # Calculate each stat using helper functions
        for stat in stats:
            if stat == "p90p10":
                stat_result = self._compute_p90p10(
                    property_values, resolved, filters, multiplier, 
                    options, case_selection, selection_criteria, skip, decimals, _round
                )
                result.update(stat_result)
            
            elif stat == "mean":
                stat_result = self._compute_mean(
                    property_values, resolved, filters, multiplier,
                    options, case_selection, selection_criteria, skip, decimals, _round
                )
                result.update(stat_result)
            
            elif stat == "median":
                stat_result = self._compute_median(
                    property_values, resolved, filters, multiplier,
                    options, case_selection, selection_criteria, skip, decimals, _round
                )
                result.update(stat_result)
            
            elif stat == "minmax":
                stat_result = self._compute_minmax(
                    property_values, resolved, filters, multiplier,
                    case_selection, selection_criteria, skip, decimals, _round
                )
                result.update(stat_result)
            
            elif stat == "percentile":
                stat_result = self._compute_percentile(
                    property_values, resolved, filters, multiplier,
                    options, case_selection, selection_criteria, skip, decimals, _round
                )
                result.update(stat_result)
            
            elif stat == "distribution":
                stat_result = self._compute_distribution(property_values, decimals, _round)
                result.update(stat_result)
            
            else:
                raise ValueError(
                    f"Unknown stat '{stat}'. Valid: "
                    "['p90p10', 'mean', 'median', 'minmax', 'percentile', 'distribution']"
                )
        
        if "sources" not in skip:
            if is_multi_property:
                result["sources"] = property_sources
            else:
                result["sources"] = property_sources[list(property_sources.keys())[0]]
        
        return result
    
    def compute_batch(
        self,
        stats: Union[str, List[str]],
        parameters: Union[str, List[str]] = "all",
        filters: Union[Dict[str, Any], str] = None,
        multiplier: float = None,
        options: Dict[str, Any] = None,
        case_selection: bool = False,
        selection_criteria: Dict[str, Any] = None,
        include_base_case: bool = True,
        include_reference_case: bool = True
    ) -> Union[Dict, List[Dict]]:
        """Compute statistics for multiple parameters.

        Args:
            stats: Statistic(s) to compute
            parameters: Parameter name(s) or "all"
            filters: Filters dict or stored filter name
            multiplier: Override default multiplier
            options: Stats-specific options (see compute() docstring)
            case_selection: Whether to find closest matching cases
            selection_criteria: Criteria for selecting cases (see compute() docstring)
            include_base_case: If True, add base case values to results (default True)
            include_reference_case: If True, add reference case values to results (default True)

        Returns:
            List of result dictionaries (or single dict if only one parameter)
            
        Examples:
            # Compute for all parameters
            results = processor.compute_batch('p90p10', filters='north_zones')
            
            # Compute for specific parameters
            results = processor.compute_batch(
                ['mean', 'p90p10'],
                parameters=['param1', 'param2'],
                filters={'property': 'stoiip'}
            )
            
            # With custom multiplier
            results = processor.compute_batch(
                'mean',
                filters='north_zones',
                multiplier=1e-6
            )
        """
        # Resolve filter preset if string provided
        filters = self._resolve_filter_preset(filters)

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
                    selection_criteria=selection_criteria
                )
                results.append(result)
            except Exception as e:
                if "errors" not in skip:
                    result = {"parameter": param, "errors": [str(e)]}
                    results.append(result)

        return results[0] if len(results) == 1 else results
    
    # ================================================================
    # PUBLIC API - CONVENIENCE METHODS
    # ================================================================
    
    def distribution(
        self,
        parameter: str = None,
        filters: Union[Dict[str, Any], str] = None,
        multiplier: float = None,
        options: Dict[str, Any] = None
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Get full distribution of values (convenience method).
        
        Args:
            parameter: Parameter name (defaults to first)
            filters: Filters dict or stored filter name
            multiplier: Override default multiplier
            options: Options dict (decimals, skip, etc.)
            
        Returns:
            Array of values or dict of arrays (for multi-property)
            
        Examples:
            # Get distribution with stored filter
            values = processor.distribution(filters='my_zones')
            
            # Get distribution for specific property
            values = processor.distribution(filters={'property': 'stoiip', 'zones': 'z1'})
            
            # With custom multiplier
            values = processor.distribution(filters='my_zones', multiplier=1e-6)
        """
        result = self.compute(
            stats="distribution",
            parameter=parameter,
            filters=filters,
            multiplier=multiplier,
            options=options
        )

        return result["distribution"]