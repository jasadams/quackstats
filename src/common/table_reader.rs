use super::types::TimeSeries;
use libduckdb_sys as ffi;
use std::ffi::{CStr, CString};
use std::mem;

/// Read a time series from a DuckDB table using the raw C API.
///
/// This executes `SELECT timestamp_col, value_col FROM table_name ORDER BY timestamp_col`
/// using the provided connection handle.
///
/// # Safety
/// The `con` pointer must be a valid duckdb_connection handle.
pub unsafe fn read_time_series(
    con: ffi::duckdb_connection,
    table_name: &str,
    timestamp_col: &str,
    value_col: &str,
) -> Result<TimeSeries, String> {
    if con.is_null() {
        return Err("Internal error: connection handle is null".to_string());
    }

    // Build the query - use double-quoting for identifiers to handle reserved words
    let sql = format!(
        "SELECT \"{}\", \"{}\" FROM \"{}\" ORDER BY \"{}\"",
        escape_identifier(timestamp_col),
        escape_identifier(value_col),
        escape_identifier(table_name),
        escape_identifier(timestamp_col),
    );

    let c_sql =
        CString::new(sql.as_str()).map_err(|e| format!("Invalid SQL string: {}", e))?;

    let mut result: ffi::duckdb_result = mem::zeroed();
    let rc = ffi::duckdb_query(con, c_sql.as_ptr(), &mut result);

    if rc != ffi::duckdb_state_DuckDBSuccess {
        let err_msg = {
            let c_err = ffi::duckdb_result_error(&mut result);
            if c_err.is_null() {
                format!("Query failed: {}", sql)
            } else {
                CStr::from_ptr(c_err).to_string_lossy().to_string()
            }
        };
        ffi::duckdb_destroy_result(&mut result);
        return Err(err_msg);
    }

    // Extract rows
    let row_count = ffi::duckdb_row_count(&mut result) as usize;
    let col_count = ffi::duckdb_column_count(&mut result) as usize;

    if col_count < 2 {
        ffi::duckdb_destroy_result(&mut result);
        return Err(format!(
            "Expected at least 2 columns, got {}",
            col_count
        ));
    }

    // Validate that column 0 is a DATE type
    let col0_type = ffi::duckdb_column_type(&mut result, 0);
    if col0_type != ffi::DUCKDB_TYPE_DUCKDB_TYPE_DATE {
        ffi::duckdb_destroy_result(&mut result);
        return Err(format!(
            "Timestamp column '{}' must be DATE type (use CAST or ::DATE to convert)",
            timestamp_col
        ));
    }

    let mut timestamps = Vec::with_capacity(row_count);
    let mut values = Vec::with_capacity(row_count);

    for row in 0..row_count {
        // Check for NULL values before reading
        if ffi::duckdb_value_is_null(&mut result, 0, row as u64)
            || ffi::duckdb_value_is_null(&mut result, 1, row as u64)
        {
            continue;
        }

        // Column 0: timestamp (DATE) - DuckDB DATE is days since epoch
        let date_val = ffi::duckdb_value_date(&mut result, 0, row as u64);
        let days = date_val.days;

        // Column 1: value (numeric) - read as double
        let val = ffi::duckdb_value_double(&mut result, 1, row as u64);

        // Skip NaN values (distinct from NULL, already handled above)
        if val.is_nan() {
            continue;
        }

        timestamps.push(days);
        values.push(val);
    }

    ffi::duckdb_destroy_result(&mut result);

    if timestamps.is_empty() {
        return Err(format!(
            "No valid data found in table '{}' (columns '{}', '{}')",
            table_name, timestamp_col, value_col
        ));
    }

    Ok(TimeSeries { timestamps, values })
}

/// Escape a SQL identifier by doubling any internal double quotes.
fn escape_identifier(name: &str) -> String {
    name.replace('"', "\"\"")
}
