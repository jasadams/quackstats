pub mod models;
pub mod params;

use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab},
    Result,
};
use std::cell::UnsafeCell;
use std::error::Error;

use crate::common::grouping::GroupForecastRow;
use crate::common::table_reader;
use crate::common::types::{ForecastResult, MIN_DATA_POINTS};
use crate::ConnHandle;
use params::ForecastParams;

/// Bind data for the forecast table function.
/// Stores the parsed parameters from the SQL call.
#[repr(C)]
pub struct ForecastBindData {
    params: ForecastParams,
}

// Safety: ForecastBindData is read-only after creation and contains only owned data.
unsafe impl Send for ForecastBindData {}
unsafe impl Sync for ForecastBindData {}

/// Mutable state for the forecast init data.
/// Wrapped in UnsafeCell because the VTab trait provides &self in func()
/// but we need to mutate position across calls.
struct ForecastState {
    /// Pre-computed rows for output.
    rows: Vec<GroupForecastRow>,
    /// Current position into `rows` for chunked emission.
    position: usize,
    /// Whether the computation has been performed.
    computed: bool,
}

/// Init data for the forecast table function.
///
/// Uses UnsafeCell for interior mutability because the VTab func() receives
/// &self but we need to track position across multiple calls. The DuckDB VTab
/// contract guarantees single-threaded access per query execution.
#[repr(C)]
pub struct ForecastInitData {
    state: UnsafeCell<ForecastState>,
}

// Safety: DuckDB guarantees single-threaded access per query. The UnsafeCell
// is only mutated within func() which is called sequentially by one thread.
unsafe impl Send for ForecastInitData {}
unsafe impl Sync for ForecastInitData {}

pub struct ForecastVTab;

impl VTab for ForecastVTab {
    type InitData = ForecastInitData;
    type BindData = ForecastBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let params = ForecastParams::from_bind_info(bind)?;

        // Add group columns first (all as VARCHAR)
        for col_name in &params.group_by {
            bind.add_result_column(col_name, LogicalTypeHandle::from(LogicalTypeId::Varchar));
        }

        // Add the standard forecast output columns
        bind.add_result_column("forecast_timestamp", LogicalTypeHandle::from(LogicalTypeId::Date));
        bind.add_result_column("forecast", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("lower_bound", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("upper_bound", LogicalTypeHandle::from(LogicalTypeId::Double));

        Ok(ForecastBindData { params })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(ForecastInitData {
            state: UnsafeCell::new(ForecastState {
                rows: Vec::new(),
                position: 0,
                computed: false,
            }),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();
        let bind_data = func.get_bind_data();
        let params = &bind_data.params;

        // Safety: DuckDB VTab contract guarantees single-threaded func() calls
        // per query execution. UnsafeCell provides the interior mutability we
        // need to track state across calls.
        let state = unsafe { &mut *init_data.state.get() };

        // Get the connection handle from extra info (stored during extension init)
        let conn_handle_ptr = func.get_extra_info::<ConnHandle>();
        if conn_handle_ptr.is_null() {
            return Err("Internal error: connection handle not available".into());
        }
        let con = unsafe { (*conn_handle_ptr).0 };

        // Compute results on first call
        if !state.computed {
            state.computed = true;

            if params.group_by.is_empty() {
                state.rows = compute_single_forecast(con, params)?;
            } else {
                state.rows = compute_grouped_forecast(con, params)?;
            }
        }

        // Emit rows from the pre-computed results
        let remaining = state.rows.len() - state.position;
        if remaining == 0 {
            output.set_len(0);
            return Ok(());
        }

        // DuckDB vector size is typically 2048
        let chunk_size = unsafe { libduckdb_sys::duckdb_vector_size() } as usize;
        let emit_count = remaining.min(chunk_size);
        let start = state.position;
        let end = start + emit_count;
        let num_group_cols = params.group_by.len();

        // Validate group column count matches expectations
        if !state.rows.is_empty() && num_group_cols > 0 {
            let actual = state.rows[start].group_values.len();
            if actual != num_group_cols {
                return Err(format!(
                    "Internal error: expected {} group columns, got {}",
                    num_group_cols, actual
                )
                .into());
            }
        }

        // Write group columns (VARCHAR)
        for g in 0..num_group_cols {
            let vec = output.flat_vector(g);
            for (row_offset, row) in state.rows[start..end].iter().enumerate() {
                vec.insert(row_offset, row.group_values[g].as_str());
            }
        }

        // Write forecast_timestamp (DATE as i32)
        let ts_col = num_group_cols;
        let mut date_vector = output.flat_vector(ts_col);
        let date_slice = date_vector.as_mut_slice::<i32>();
        for (row_offset, row) in state.rows[start..end].iter().enumerate() {
            date_slice[row_offset] = row.timestamp;
        }

        // Write forecast (DOUBLE)
        let fc_col = num_group_cols + 1;
        let mut fc_vector = output.flat_vector(fc_col);
        let fc_slice = fc_vector.as_mut_slice::<f64>();
        for (row_offset, row) in state.rows[start..end].iter().enumerate() {
            fc_slice[row_offset] = row.forecast;
        }

        // Write lower_bound (DOUBLE)
        let lb_col = num_group_cols + 2;
        let mut lb_vector = output.flat_vector(lb_col);
        let lb_slice = lb_vector.as_mut_slice::<f64>();
        for (row_offset, row) in state.rows[start..end].iter().enumerate() {
            lb_slice[row_offset] = row.lower_bound;
        }

        // Write upper_bound (DOUBLE)
        let ub_col = num_group_cols + 3;
        let mut ub_vector = output.flat_vector(ub_col);
        let ub_slice = ub_vector.as_mut_slice::<f64>();
        for (row_offset, row) in state.rows[start..end].iter().enumerate() {
            ub_slice[row_offset] = row.upper_bound;
        }

        state.position = end;
        output.set_len(emit_count);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![LogicalTypeHandle::from(LogicalTypeId::Varchar)])
    }

    fn named_parameters() -> Option<Vec<(String, LogicalTypeHandle)>> {
        Some(vec![
            (
                "timestamp".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "value".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
            (
                "group_by".to_string(),
                LogicalTypeHandle::list(&LogicalTypeHandle::from(LogicalTypeId::Varchar)),
            ),
            (
                "horizon".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Integer),
            ),
            (
                "confidence_level".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Double),
            ),
            (
                "model".to_string(),
                LogicalTypeHandle::from(LogicalTypeId::Varchar),
            ),
        ])
    }
}

/// Route to the correct model based on the model parameter string.
///
/// Wraps the call in `catch_unwind` to prevent panics from unwinding through
/// the DuckDB VTab `extern "C"` boundary, which would abort the WASM process.
fn run_forecast(
    series: &crate::common::types::TimeSeries,
    horizon: usize,
    confidence_level: f64,
    model: &str,
) -> Result<ForecastResult, String> {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let result = catch_unwind(AssertUnwindSafe(|| match model {
        "ets" => models::forecast_ets(series, horizon, confidence_level),
        "linear" => models::forecast_linear(series, horizon, confidence_level),
        "exponential" => models::forecast_exponential(series, horizon, confidence_level),
        "logistic" => models::forecast_logistic(series, horizon, confidence_level),
        "auto" => models::forecast_auto(series, horizon, confidence_level),
        _ => Err(format!(
            "Unknown model '{}'. Valid models: 'auto', 'ets', 'linear', 'exponential', 'logistic'",
            model
        )),
    }));

    match result {
        Ok(inner) => inner,
        Err(panic_info) => {
            let msg = if let Some(s) = panic_info.downcast_ref::<&str>() {
                format!("Model '{}' panicked: {}", model, s)
            } else if let Some(s) = panic_info.downcast_ref::<String>() {
                format!("Model '{}' panicked: {}", model, s)
            } else {
                format!("Model '{}' panicked (no message available)", model)
            };
            eprintln!("quackstats: {}", msg);
            Err(msg)
        }
    }
}

/// Compute forecast for a single (non-grouped) time series.
/// Returns the results as GroupForecastRow with empty group_values for
/// uniform handling in the output path.
fn compute_single_forecast(
    con: libduckdb_sys::duckdb_connection,
    params: &ForecastParams,
) -> Result<Vec<GroupForecastRow>, Box<dyn Error>> {
    let series = unsafe {
        table_reader::read_time_series(
            con,
            &params.table_name,
            &params.timestamp_col,
            &params.value_col,
        )
    }
    .map_err(|e: String| -> Box<dyn Error> { e.into() })?;

    let anchor = last_observation(&series);

    let result: ForecastResult = run_forecast(
        &series,
        params.horizon as usize,
        params.confidence_level,
        &params.model,
    )
    .map_err(|e: String| -> Box<dyn Error> { e.into() })?;

    Ok(forecast_result_to_rows(&[], anchor.as_ref(), &result))
}

/// Compute independent forecasts for each group and concatenate the results.
/// Groups with insufficient data (< MIN_DATA_POINTS) are silently skipped.
fn compute_grouped_forecast(
    con: libduckdb_sys::duckdb_connection,
    params: &ForecastParams,
) -> Result<Vec<GroupForecastRow>, Box<dyn Error>> {
    let grouped_series = unsafe {
        table_reader::read_grouped_time_series(
            con,
            &params.table_name,
            &params.timestamp_col,
            &params.value_col,
            &params.group_by,
        )
    }
    .map_err(|e: String| -> Box<dyn Error> { e.into() })?;

    let mut all_rows = Vec::new();

    for group in &grouped_series {
        // Skip groups with insufficient data instead of failing
        if group.series.len() < MIN_DATA_POINTS {
            eprintln!(
                "quackstats: skipping group {:?} - insufficient data ({} points, need {})",
                group.group_key, group.series.len(), MIN_DATA_POINTS
            );
            continue;
        }

        let anchor = last_observation(&group.series);

        let result = match run_forecast(
            &group.series,
            params.horizon as usize,
            params.confidence_level,
            &params.model,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "quackstats: skipping group {:?} - model fitting failed: {}",
                    group.group_key, e
                );
                continue;
            }
        };

        let rows = forecast_result_to_rows(&group.group_key, anchor.as_ref(), &result);
        all_rows.extend(rows);
    }

    Ok(all_rows)
}

/// Extract the last observation from a time series as an anchor point.
/// Returns (timestamp, value) for the last data point, or None if empty.
fn last_observation(series: &crate::common::types::TimeSeries) -> Option<(i32, f64)> {
    if series.timestamps.is_empty() {
        return None;
    }
    let last = series.timestamps.len() - 1;
    Some((series.timestamps[last], series.values[last]))
}

/// Convert a ForecastResult into a vec of GroupForecastRow.
/// If `anchor` is provided, prepends the last historical observation so the
/// forecast line connects seamlessly with the real data (no visual gap).
fn forecast_result_to_rows(
    group_key: &[String],
    anchor: Option<&(i32, f64)>,
    result: &ForecastResult,
) -> Vec<GroupForecastRow> {
    let num_rows = result.timestamps.len();
    let capacity = num_rows + if anchor.is_some() { 1 } else { 0 };
    let mut rows = Vec::with_capacity(capacity);

    // Anchor row: last historical value with zero uncertainty
    if let Some(&(ts, val)) = anchor {
        rows.push(GroupForecastRow {
            group_values: group_key.to_vec(),
            timestamp: ts,
            forecast: val,
            lower_bound: val,
            upper_bound: val,
        });
    }

    for i in 0..num_rows {
        rows.push(GroupForecastRow {
            group_values: group_key.to_vec(),
            timestamp: result.timestamps[i],
            forecast: result.forecasts[i],
            lower_bound: result.lower_bounds[i],
            upper_bound: result.upper_bounds[i],
        });
    }

    rows
}
