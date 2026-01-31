use duckdb::vtab::BindInfo;
use libduckdb_sys as ffi;
use std::ffi::CStr;

/// Helper to extract a named string parameter from VTab bind info.
pub fn get_string_param(bind: &BindInfo, name: &str) -> Option<String> {
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
pub fn extract_list_strings(
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
