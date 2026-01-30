use super::grouping::GroupedTimeSeries;
use super::types::TimeSeries;
use libduckdb_sys as ffi;
use std::collections::BTreeMap;
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

/// Read grouped time series from a DuckDB table using the raw C API.
///
/// Executes a query like:
/// `SELECT "region", "product", "ds", "revenue" FROM "sales" ORDER BY "region", "product", "ds"`
///
/// Rows are split into separate `GroupedTimeSeries` by unique combinations of
/// the group column values. Each group's time series is sorted by timestamp.
///
/// # Safety
/// The `con` pointer must be a valid duckdb_connection handle.
pub unsafe fn read_grouped_time_series(
    con: ffi::duckdb_connection,
    table_name: &str,
    timestamp_col: &str,
    value_col: &str,
    group_cols: &[String],
) -> Result<Vec<GroupedTimeSeries>, String> {
    if con.is_null() {
        return Err("Internal error: connection handle is null".to_string());
    }

    if group_cols.is_empty() {
        // No group columns - delegate to single-series reader and wrap
        let series = read_time_series(con, table_name, timestamp_col, value_col)?;
        return Ok(vec![GroupedTimeSeries {
            group_key: vec![],
            group_columns: vec![],
            series,
        }]);
    }

    // Build SELECT: group_cols..., timestamp_col, value_col
    // ORDER BY: group_cols..., timestamp_col
    let group_select: Vec<String> = group_cols
        .iter()
        .map(|c| format!("\"{}\"", escape_identifier(c)))
        .collect();
    let group_order = group_select.clone();

    let sql = format!(
        "SELECT {}, \"{}\", \"{}\" FROM \"{}\" ORDER BY {}, \"{}\"",
        group_select.join(", "),
        escape_identifier(timestamp_col),
        escape_identifier(value_col),
        escape_identifier(table_name),
        group_order.join(", "),
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

    let row_count = ffi::duckdb_row_count(&mut result) as usize;
    let col_count = ffi::duckdb_column_count(&mut result) as usize;
    let num_group_cols = group_cols.len();
    let expected_cols = num_group_cols + 2; // group cols + timestamp + value

    if col_count < expected_cols {
        ffi::duckdb_destroy_result(&mut result);
        return Err(format!(
            "Expected at least {} columns, got {}",
            expected_cols, col_count
        ));
    }

    // Validate that the timestamp column (after group cols) is DATE type
    let ts_col_idx = num_group_cols as u64;
    let col_type = ffi::duckdb_column_type(&mut result, ts_col_idx);
    if col_type != ffi::DUCKDB_TYPE_DUCKDB_TYPE_DATE {
        ffi::duckdb_destroy_result(&mut result);
        return Err(format!(
            "Timestamp column '{}' must be DATE type (use CAST or ::DATE to convert)",
            timestamp_col
        ));
    }

    // Use BTreeMap keyed by Vec<String> for deterministic group ordering.
    // Vec<String> compares element-wise, so no key collision risk.
    let mut groups: BTreeMap<Vec<String>, (Vec<i32>, Vec<f64>)> = BTreeMap::new();

    let val_col_idx = (num_group_cols + 1) as u64;

    for row in 0..row_count {
        let row_u64 = row as u64;

        // Check for NULL in timestamp or value columns
        if ffi::duckdb_value_is_null(&mut result, ts_col_idx, row_u64)
            || ffi::duckdb_value_is_null(&mut result, val_col_idx, row_u64)
        {
            continue;
        }

        // Read group column values as strings
        let mut key = Vec::with_capacity(num_group_cols);
        let mut skip_row = false;
        for g in 0..num_group_cols {
            if ffi::duckdb_value_is_null(&mut result, g as u64, row_u64) {
                skip_row = true;
                break;
            }
            let raw = ffi::duckdb_value_varchar(&mut result, g as u64, row_u64);
            if raw.is_null() {
                skip_row = true;
                break;
            }
            let s = CStr::from_ptr(raw).to_string_lossy().to_string();
            ffi::duckdb_free(raw as *mut std::ffi::c_void);
            key.push(s);
        }
        if skip_row {
            continue;
        }

        // Read timestamp (DATE)
        let date_val = ffi::duckdb_value_date(&mut result, ts_col_idx, row_u64);
        let days = date_val.days;

        // Read value (DOUBLE)
        let val = ffi::duckdb_value_double(&mut result, val_col_idx, row_u64);
        if val.is_nan() {
            continue;
        }

        let entry = groups.entry(key).or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(days);
        entry.1.push(val);
    }

    ffi::duckdb_destroy_result(&mut result);

    // Convert to Vec<GroupedTimeSeries>
    let group_columns: Vec<String> = group_cols.to_vec();
    let result_vec: Vec<GroupedTimeSeries> = groups
        .into_iter()
        .map(|(key, (timestamps, values))| GroupedTimeSeries {
            group_key: key,
            group_columns: group_columns.clone(),
            series: TimeSeries { timestamps, values },
        })
        .collect();

    Ok(result_vec)
}

/// Escape a SQL identifier by doubling any internal double quotes.
fn escape_identifier(name: &str) -> String {
    name.replace('"', "\"\"")
}
