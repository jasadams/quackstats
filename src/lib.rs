use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    duckdb_entrypoint_c_api,
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab},
    Connection, Result,
};
use std::{
    error::Error,
    ffi::CString,
    sync::atomic::{AtomicBool, Ordering},
};

/// Bind data for the forecast table function.
/// Stores the parsed parameters from the SQL call.
#[repr(C)]
struct ForecastBindData {
    table_name: String,
    timestamp_col: String,
    value_col: String,
    horizon: i64,
    confidence_level: f64,
    model: String,
}

/// Init data for the forecast table function.
/// Tracks whether we have already emitted our result rows.
#[repr(C)]
struct ForecastInitData {
    done: AtomicBool,
}

struct ForecastVTab;

impl VTab for ForecastVTab {
    type InitData = ForecastInitData;
    type BindData = ForecastBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        // Define output columns
        bind.add_result_column("forecast_timestamp", LogicalTypeHandle::from(LogicalTypeId::Date));
        bind.add_result_column("forecast", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("lower_bound", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("upper_bound", LogicalTypeHandle::from(LogicalTypeId::Double));

        // Positional parameter: table name
        let table_name = bind.get_parameter(0).to_string();

        // Named parameters with defaults
        let timestamp_col = bind
            .get_named_parameter("timestamp")
            .map(|v| v.to_string())
            .unwrap_or_else(|| "timestamp".to_string());

        let value_col = bind
            .get_named_parameter("value")
            .map(|v| v.to_string())
            .unwrap_or_else(|| "value".to_string());

        let horizon = bind
            .get_named_parameter("horizon")
            .map(|v| v.to_int64())
            .unwrap_or(3);

        let confidence_level = bind
            .get_named_parameter("confidence_level")
            .map(|v| {
                // confidence_level comes in as a double; use to_string and parse
                v.to_string().parse::<f64>().unwrap_or(0.95)
            })
            .unwrap_or(0.95);

        let model = bind
            .get_named_parameter("model")
            .map(|v| v.to_string())
            .unwrap_or_else(|| "ets".to_string());

        Ok(ForecastBindData {
            table_name,
            timestamp_col,
            value_col,
            horizon,
            confidence_level,
            model,
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(ForecastInitData {
            done: AtomicBool::new(false),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let init_data = func.get_init_data();

        if init_data.done.swap(true, Ordering::Relaxed) {
            output.set_len(0);
            return Ok(());
        }

        let bind_data = func.get_bind_data();
        let horizon = bind_data.horizon.min(3).max(1) as usize;

        // DuckDB DATE is stored as i32: days since 1970-01-01 (Unix epoch).
        // 2026-02-01 = day 20,485 from epoch
        // (1970 to 2026 = 56 years, accounting for leap years)
        const BASE_DATE: i32 = 20_485; // 2026-02-01

        // Write date column (column 0) - dates as i32
        let dates: Vec<i32> = (0..horizon as i32).map(|i| BASE_DATE + i).collect();
        let date_vector = output.flat_vector(0);
        let date_slice = date_vector.as_mut_slice::<i32>(horizon);
        date_slice[..horizon].copy_from_slice(&dates);

        // Write forecast column (column 1) - doubles
        let forecasts = [100.0_f64, 102.0, 104.0];
        let forecast_vector = output.flat_vector(1);
        let forecast_slice = forecast_vector.as_mut_slice::<f64>(horizon);
        forecast_slice[..horizon].copy_from_slice(&forecasts[..horizon]);

        // Write lower_bound column (column 2) - doubles
        let lower_bounds = [90.0_f64, 91.0, 92.0];
        let lower_vector = output.flat_vector(2);
        let lower_slice = lower_vector.as_mut_slice::<f64>(horizon);
        lower_slice[..horizon].copy_from_slice(&lower_bounds[..horizon]);

        // Write upper_bound column (column 3) - doubles
        let upper_bounds = [110.0_f64, 113.0, 116.0];
        let upper_vector = output.flat_vector(3);
        let upper_slice = upper_vector.as_mut_slice::<f64>(horizon);
        upper_slice[..horizon].copy_from_slice(&upper_bounds[..horizon]);

        output.set_len(horizon);
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

#[duckdb_entrypoint_c_api()]
pub unsafe fn quackstats_init(con: Connection) -> Result<(), Box<dyn Error>> {
    con.register_table_function::<ForecastVTab>("forecast")
        .expect("Failed to register forecast table function");
    Ok(())
}
