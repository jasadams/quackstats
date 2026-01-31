use duckdb::vtab::BindInfo;
use libduckdb_sys as ffi;
use std::ffi::CStr;

/// Parsed and validated parameters for the forecast() table function.
pub struct ForecastParams {
    pub table_name: String,
    pub timestamp_col: String,
    pub value_col: String,
    pub group_by: Vec<String>,
    pub horizon: i64,
    pub confidence_level: f64,
    pub model: String,
}

impl ForecastParams {
    /// Extract and validate parameters from the VTab bind info.
    pub fn from_bind_info(bind: &BindInfo) -> Result<Self, Box<dyn std::error::Error>> {
        // Positional parameter: table name
        let table_name = bind.get_parameter(0).to_string();

        // Named parameters with defaults
        let timestamp_col = get_string_param(bind, "timestamp")
            .unwrap_or_else(|| "timestamp".to_string());

        let value_col = get_string_param(bind, "value")
            .unwrap_or_else(|| "value".to_string());

        let horizon = bind
            .get_named_parameter("horizon")
            .map(|v| v.to_int64())
            .unwrap_or(3);

        let confidence_level = match bind.get_named_parameter("confidence_level") {
            Some(v) => {
                let s = v.to_string();
                s.parse::<f64>().map_err(|_| {
                    format!("confidence_level must be a valid number, got '{}'", s)
                })?
            }
            None => 0.95,
        };

        let model = get_string_param(bind, "model")
            .unwrap_or_else(|| "auto".to_string());

        // Parse group_by LIST parameter via the raw C API.
        // The duckdb crate's Value type doesn't expose list accessors, so we
        // use libduckdb_sys::duckdb_get_list_size / duckdb_get_list_child.
        let group_by = match bind.get_named_parameter("group_by") {
            Some(v) => extract_list_strings(&v)?,
            None => Vec::new(),
        };

        // Validate
        if table_name.is_empty() {
            return Err("table name cannot be empty".into());
        }
        if horizon <= 0 {
            return Err("horizon must be a positive integer".into());
        }
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err("confidence_level must be between 0.0 and 1.0 (exclusive)".into());
        }

        let valid_models = ["auto", "ets", "linear"];
        if !valid_models.contains(&model.as_str()) {
            return Err(format!(
                "Invalid model '{}'. Valid models: auto, ets, linear",
                model
            )
            .into());
        }

        Ok(ForecastParams {
            table_name,
            timestamp_col,
            value_col,
            group_by,
            horizon,
            confidence_level,
            model,
        })
    }
}

/// Helper to extract a named string parameter.
fn get_string_param(bind: &BindInfo, name: &str) -> Option<String> {
    bind.get_named_parameter(name).map(|v| v.to_string())
}

/// Extract a Vec<String> from a DuckDB LIST Value using the raw C API.
///
/// The duckdb crate's `Value` only exposes `to_int64()` and `Display`. For LIST
/// parameters we need `duckdb_get_list_size` and `duckdb_get_list_child` from
/// libduckdb_sys.
///
/// # Safety assumption (version-pinned)
/// This relies on `duckdb::vtab::Value` being `repr(C)` with a single field
/// `ptr: duckdb_value` (verified for duckdb crate =1.4.4). The crate version
/// is pinned in Cargo.toml with `=1.4.4`, so this layout is stable. When
/// upgrading duckdb-rs, verify the Value layout or replace with a safe API
/// if one becomes available (see: https://github.com/duckdb/duckdb-rs/issues).
fn extract_list_strings(
    value: &duckdb::vtab::Value,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // Transmute to read the raw duckdb_value pointer from Value's internal field.
    // Value is repr(C) with layout: struct Value { ptr: duckdb_value }
    let raw_ptr: ffi::duckdb_value = unsafe {
        let ptr_ref: *const duckdb::vtab::Value = value;
        *(ptr_ref as *const ffi::duckdb_value)
    };

    let size = unsafe { ffi::duckdb_get_list_size(raw_ptr) } as usize;
    let mut result = Vec::with_capacity(size);

    for i in 0..size {
        let child_ptr = unsafe { ffi::duckdb_get_list_child(raw_ptr, i as u64) };
        if child_ptr.is_null() {
            return Err(format!(
                "group_by list element at index {} is null",
                i
            )
            .into());
        }

        // Get the string value from the child
        let varchar = unsafe { ffi::duckdb_get_varchar(child_ptr) };
        if varchar.is_null() {
            unsafe { ffi::duckdb_destroy_value(&mut { child_ptr }) };
            return Err(format!(
                "group_by list element at index {} could not be read as string",
                i
            )
            .into());
        }

        let s = unsafe { CStr::from_ptr(varchar).to_string_lossy().to_string() };
        unsafe {
            ffi::duckdb_free(varchar as *mut std::ffi::c_void);
            ffi::duckdb_destroy_value(&mut { child_ptr });
        }

        if s.is_empty() {
            return Err("group_by column name cannot be empty".into());
        }
        result.push(s);
    }

    Ok(result)
}
