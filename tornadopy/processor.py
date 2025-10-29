import re
from pathlib import Path
from typing import Any, Union

import numpy as np
import polars as pl
from pyxlsb import read_excel


class TornadoProcessor:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"File not found: {self.filepath}")
        
        try:
            self.sheets_raw = self._load_sheets()
        except Exception as e:
            raise ValueError(f"Error reading Excel file: {e}")
        
        self.data: dict[str, pl.DataFrame] = {}
        self.metadata: dict[str, pl.DataFrame] = {}
        self.info: dict[str, dict] = {}
        self.dynamic_fields: dict[str, list[str]] = {}
        
        try:
            self._parse_all_sheets()
        except Exception as e:
            print(f"[!] Warning: some sheets failed to parse: {e}")
    
    def _load_sheets(self) -> dict[str, pl.DataFrame]:
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
        name = str(name).strip().lower()
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        name = re.sub(r"_+$", "", name)
        return name or "property"
    
    def _parse_sheet(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, list[str], dict]:
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
        for sheet_name, df_raw in self.sheets_raw.items():
            try:
                data, metadata, fields, info = self._parse_sheet(df_raw)
                self.data[sheet_name] = data
                self.metadata[sheet_name] = metadata
                self.dynamic_fields[sheet_name] = fields
                self.info[sheet_name] = info
            except Exception as e:
                print(f"[!] Skipped sheet '{sheet_name}': {e}")
    
    def get_parameters(self) -> list[str]:
        return list(self.data.keys())
    
    def get_properties(self, parameter: str = None) -> list[str]:
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
    
    def get_unique(self, field: str, parameter: str = None) -> list[str]:
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
    
    def get_distribution(
        self,
        parameter: str = None,
        filters: dict[str, Any] = None,
        multiplier: float = 1.0,
        options: dict[str, Any] = None
    ) -> list[float]:
        result = self.compute(
            stats="distribution",
            parameter=parameter,
            filters=filters,
            multiplier=multiplier,
            options=options
        )
        
        return result["distribution"]
    
    def get_info(self, parameter: str = None) -> dict:
        resolved = self._resolve_parameter(parameter)
        return self.info.get(resolved, {})
    
    def get_case(self, index: int, parameter: str = None) -> dict:
        resolved = self._resolve_parameter(parameter)
        df = self.data[resolved]
        
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} out of range (0–{len(df)-1})")
        
        return df[index].to_dicts()[0]
    
    def _resolve_parameter(self, parameter: str = None) -> str:
        if parameter is None:
            # Default to first parameter if not specified
            return list(self.data.keys())[0]
        return parameter
    
    def _normalize_filters(self, filters: dict[str, Any]) -> dict[str, Any]:
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
    
    def _select_columns(self, parameter: str, filters: dict[str, Any]) -> tuple[list[str], list[str]]:
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
        filters: dict[str, Any],
        multiplier: float = 1.0
    ) -> tuple[np.ndarray, list[str]]:
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
    
    def _calculate_weighted_distance(
        self,
        property_values: dict[str, np.ndarray],
        targets: dict[str, float],
        weights: dict[str, float]
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
        weighted_combinations: list[dict],
        resolved: str,
        base_filters: dict[str, Any],
        multiplier: float,
        case_type: str,
        skip: list[str]
    ) -> tuple[np.ndarray, dict]:
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
        property_values: dict[str, np.ndarray],
        targets: dict[str, dict[str, float]],
        weights: dict[str, float],
        resolved: str,
        filters: dict[str, Any],
        multiplier: float,
        decimals: int
    ) -> list[dict]:
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
    
    def _compute_p90p10(
        self,
        property_values: dict[str, np.ndarray],
        resolved: str,
        filters: dict[str, Any],
        multiplier: float,
        options: dict[str, Any],
        case_selection: bool,
        selection_criteria: dict[str, Any],
        skip: list[str],
        decimals: int,
        _round
    ) -> dict:
        """Compute p90p10 for single or multiple properties."""
        result = {}
        threshold = options.get("p90p10_threshold")
        
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
                p10_dict[prop] = _round(p10_val)
                p90_dict[prop] = _round(p90_val)
        
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
                weights = selection_criteria.get("weights")
                if not weights:
                    # Default: equal weights for all properties in property_values
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
                                result["errors"] = result.get("errors", [])
                                result["errors"].append(f"Failed to extract {prop} for weighting: {e}")
                
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
        property_values: dict[str, np.ndarray],
        resolved: str,
        filters: dict[str, Any],
        multiplier: float,
        options: dict[str, Any],
        case_selection: bool,
        selection_criteria: dict[str, Any],
        skip: list[str],
        decimals: int,
        _round
    ) -> dict:
        """Compute mean for single or multiple properties."""
        mean_dict = {prop: _round(np.mean(vals)) for prop, vals in property_values.items()}
        
        result = {}
        if len(property_values) == 1:
            result["mean"] = mean_dict[list(property_values.keys())[0]]
        else:
            result["mean"] = mean_dict
        
        # Handle case selection
        if case_selection and "closest_case" not in skip:
            weights = selection_criteria.get("weights")
            if not weights:
                weights = {prop: 1.0 / len(property_values) for prop in property_values.keys()}
            
            # Extract additional properties if needed
            weighted_property_values = dict(property_values)
            non_property_filters = {k: v for k, v in filters.items() if k != "property"}
            
            for prop in weights.keys():
                if prop not in weighted_property_values:
                    try:
                        prop_filters = {**non_property_filters, "property": prop}
                        prop_vals, _ = self._extract_values(resolved, prop_filters, multiplier)
                        prop_vals = self._validate_numeric(prop_vals, prop)
                        weighted_property_values[prop] = prop_vals
                    except:
                        pass
            
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
        property_values: dict[str, np.ndarray],
        resolved: str,
        filters: dict[str, Any],
        multiplier: float,
        options: dict[str, Any],
        case_selection: bool,
        selection_criteria: dict[str, Any],
        skip: list[str],
        decimals: int,
        _round
    ) -> dict:
        """Compute median for single or multiple properties."""
        median_dict = {prop: _round(np.median(vals)) for prop, vals in property_values.items()}
        
        result = {}
        if len(property_values) == 1:
            result["median"] = median_dict[list(property_values.keys())[0]]
        else:
            result["median"] = median_dict
        
        # Handle case selection
        if case_selection and "closest_case" not in skip:
            weights = selection_criteria.get("weights")
            if not weights:
                weights = {prop: 1.0 / len(property_values) for prop in property_values.keys()}
            
            # Extract additional properties if needed
            weighted_property_values = dict(property_values)
            non_property_filters = {k: v for k, v in filters.items() if k != "property"}
            
            for prop in weights.keys():
                if prop not in weighted_property_values:
                    try:
                        prop_filters = {**non_property_filters, "property": prop}
                        prop_vals, _ = self._extract_values(resolved, prop_filters, multiplier)
                        prop_vals = self._validate_numeric(prop_vals, prop)
                        weighted_property_values[prop] = prop_vals
                    except:
                        pass
            
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
        property_values: dict[str, np.ndarray],
        resolved: str,
        filters: dict[str, Any],
        multiplier: float,
        case_selection: bool,
        selection_criteria: dict[str, Any],
        skip: list[str],
        decimals: int,
        _round
    ) -> dict:
        """Compute minmax for single or multiple properties."""
        min_dict = {prop: _round(np.min(vals)) for prop, vals in property_values.items()}
        max_dict = {prop: _round(np.max(vals)) for prop, vals in property_values.items()}
        
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
        property_values: dict[str, np.ndarray],
        resolved: str,
        filters: dict[str, Any],
        multiplier: float,
        options: dict[str, Any],
        case_selection: bool,
        selection_criteria: dict[str, Any],
        skip: list[str],
        decimals: int,
        _round
    ) -> dict:
        """Compute percentile for single or multiple properties."""
        p = options.get("p", 50)
        perc_dict = {prop: _round(np.percentile(vals, p)) for prop, vals in property_values.items()}
        
        result = {}
        if len(property_values) == 1:
            result[f"p{p}"] = perc_dict[list(property_values.keys())[0]]
        else:
            result[f"p{p}"] = perc_dict
        
        # Handle case selection
        if case_selection and "closest_case" not in skip:
            weights = selection_criteria.get("weights")
            if not weights:
                weights = {prop: 1.0 / len(property_values) for prop in property_values.keys()}
            
            # Extract additional properties if needed
            weighted_property_values = dict(property_values)
            non_property_filters = {k: v for k, v in filters.items() if k != "property"}
            
            for prop in weights.keys():
                if prop not in weighted_property_values:
                    try:
                        prop_filters = {**non_property_filters, "property": prop}
                        prop_vals, _ = self._extract_values(resolved, prop_filters, multiplier)
                        prop_vals = self._validate_numeric(prop_vals, prop)
                        weighted_property_values[prop] = prop_vals
                    except:
                        pass
            
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
        property_values: dict[str, np.ndarray],
        decimals: int,
        _round
    ) -> dict:
        """Compute distribution for single or multiple properties."""
        dist_dict = {prop: [_round(v) for v in vals.tolist()] for prop, vals in property_values.items()}
        
        if len(property_values) == 1:
            return {"distribution": dist_dict[list(property_values.keys())[0]]}
        else:
            return {"distribution": dist_dict}
    
    def _validate_numeric(self, values: np.ndarray, description: str) -> np.ndarray:
        if values.size == 0 or not np.isfinite(values).any():
            raise ValueError(f"No numeric data found for {description}")
        
        return values[np.isfinite(values)]
    
    def _get_case_details(
        self,
        index: int,
        parameter: str,
        filters: dict[str, Any],
        multiplier: float,
        value: float,
        decimals: int = 6
    ) -> dict:
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
        case_data = self.get_case(index, parameter=parameter)
        
        # Get all available properties for this parameter
        try:
            all_properties = self.get_properties(parameter)
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
    
    def compute(
        self,
        stats: Union[str, list[str]],
        parameter: str = None,
        filters: dict[str, Any] = None,
        multiplier: float = 1.0,
        options: dict[str, Any] = None,
        case_selection: bool = False,
        selection_criteria: dict[str, Any] = None
    ) -> dict:
        """Compute statistics for given parameter and filters.
        
        Args:
            stats: Statistic(s) to compute ('p90p10', 'mean', 'median', 'minmax', 'percentile', 'distribution')
            parameter: Parameter name (defaults to first if only one available)
            filters: Filters to apply (zones, property, etc.)
            multiplier: Multiplier to apply to values
            options: Stats-specific options:
                - decimals: Number of decimal places (default 6)
                - p90p10_threshold: Minimum case count for p90p10
                - p: Percentile value for 'percentile' stat (default 50)
                - skip: List of keys to skip in output (e.g., ['errors', 'sources'])
            case_selection: Whether to find closest matching cases (default False)
            selection_criteria: Criteria for selecting cases (only used if case_selection=True):
                - weights: dict of property weights (e.g., {'stoiip': 0.6, 'giip': 0.4})
                - combinations: list of filter+property combinations for complex weighting:
                  [
                      {
                          "filters": {"zones": [...]},
                          "properties": {"stoiip": 0.3, "giip": 0.2}
                      },
                      ...
                  ]
        
        Returns:
            Dictionary with computed statistics and optionally closest_cases
        """
        resolved = self._resolve_parameter(parameter)
        filters = filters or {}
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
            # Single property mode - wrap in try/except to handle errors gracefully
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
                # Return early with error if no data could be extracted
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
        stats: Union[str, list[str]],
        parameters: Union[str, list[str]] = "all",
        filters: dict[str, Any] = None,
        multiplier: float = 1.0,
        options: dict[str, Any] = None,
        case_selection: bool = False,
        selection_criteria: dict[str, Any] = None
    ) -> Union[dict, list[dict]]:
        """Compute statistics for multiple parameters.
        
        Args:
            stats: Statistic(s) to compute
            parameters: Parameter name(s) or "all"
            filters: Filters to apply
            multiplier: Multiplier to apply
            options: Stats-specific options (see compute() docstring)
            case_selection: Whether to find closest matching cases
            selection_criteria: Criteria for selecting cases (see compute() docstring)
        
        Returns:
            List of result dictionaries (or single dict if only one parameter)
        """
        if parameters == "all":
            param_list = list(self.data.keys())
        elif isinstance(parameters, str):
            param_list = [parameters]
        else:
            param_list = parameters
        
        options = options or {}
        skip = options.get("skip", [])
        skip_parameters = options.get("skip_parameters", [])
        
        # Filter out parameters in skip_parameters list (for backward compatibility, also check skip)
        # Only filter if the skip item is actually a parameter name
        all_param_names = set(self.data.keys())
        param_list = [p for p in param_list if p not in skip_parameters and p not in (skip if p in all_param_names else [])]
        
        results = []
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
                # Only add error results if "errors" not in skip
                if "errors" not in skip:
                    result = {"parameter": param, "errors": [str(e)]}
                    results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def get_tornado_data(
        self,
        parameters: Union[str, list[str]] = "all",
        filters: dict[str, Any] = None,
        multiplier: float = 1.0,
        options: dict[str, Any] = None
    ) -> dict:
        """Get tornado chart data formatted for easy plotting.
        
        Returns a dictionary with parameter names as keys and their p90p10 ranges,
        plus a reference case value if available.
        
        Args:
            parameters: Parameters to include (default "all")
            filters: Filters to apply
            multiplier: Multiplier to apply to values
            options: Options including skip list, p90p10_threshold, etc.
            
        Returns:
            Dictionary with structure:
            {
                'parameter_name': {
                    'p10': value,
                    'p90': value,
                    'range': p90 - p10,
                    'midpoint': (p90 + p10) / 2
                },
                ...
            }
        """
        results = self.compute_batch(
            stats='p90p10',
            parameters=parameters,
            filters=filters,
            multiplier=multiplier,
            options=options
        )
        
        if not isinstance(results, list):
            results = [results]
        
        tornado_data = {}
        
        for result in results:
            param = result.get("parameter")
            if "p90p10" in result and "errors" not in result:
                p10, p90 = result["p90p10"]
                
                # Handle multi-property case
                if isinstance(p10, dict):
                    # Use first property for tornado display
                    first_prop = list(p10.keys())[0]
                    p10_val = p10[first_prop]
                    p90_val = p90[first_prop]
                else:
                    p10_val = p10
                    p90_val = p90
                
                tornado_data[param] = {
                    'p10': p10_val,
                    'p90': p90_val,
                    'range': p90_val - p10_val,
                    'midpoint': (p90_val + p10_val) / 2
                }
        
        return tornado_data