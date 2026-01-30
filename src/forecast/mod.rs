pub mod models;
pub mod params;

use duckdb::{
    core::{DataChunkHandle, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab},
    Result,
};
use std::{
    error::Error,
    sync::atomic::{AtomicBool, Ordering},
};

use crate::common::table_reader;
use crate::common::types::ForecastResult;
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

/// Init data for the forecast table function.
/// Tracks whether we have already emitted our result rows.
#[repr(C)]
pub struct ForecastInitData {
    done: AtomicBool,
}

pub struct ForecastVTab;

impl VTab for ForecastVTab {
    type InitData = ForecastInitData;
    type BindData = ForecastBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        // Define output columns
        bind.add_result_column("forecast_timestamp", LogicalTypeHandle::from(LogicalTypeId::Date));
        bind.add_result_column("forecast", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("lower_bound", LogicalTypeHandle::from(LogicalTypeId::Double));
        bind.add_result_column("upper_bound", LogicalTypeHandle::from(LogicalTypeId::Double));

        let params = ForecastParams::from_bind_info(bind)?;

        Ok(ForecastBindData { params })
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
        let params = &bind_data.params;

        // Get the connection handle from extra info (stored during extension init)
        let conn_handle_ptr = func.get_extra_info::<ConnHandle>();
        if conn_handle_ptr.is_null() {
            return Err("Internal error: connection handle not available".into());
        }
        let con = unsafe { (*conn_handle_ptr).0 };

        // Read the source table data
        let series = unsafe {
            table_reader::read_time_series(
                con,
                &params.table_name,
                &params.timestamp_col,
                &params.value_col,
            )
        }
        .map_err(|e| -> Box<dyn Error> { e.into() })?;

        // Run ETS forecasting
        let result: ForecastResult = models::forecast_ets(
            &series,
            params.horizon as usize,
            params.confidence_level,
        )
        .map_err(|e| -> Box<dyn Error> { e.into() })?;

        let num_rows = result.timestamps.len();
        if num_rows == 0 {
            output.set_len(0);
            return Ok(());
        }

        // Write forecast_timestamp column (column 0) - DATE as i32
        let mut date_vector = output.flat_vector(0);
        let date_slice = date_vector.as_mut_slice::<i32>();
        date_slice[..num_rows].copy_from_slice(&result.timestamps);

        // Write forecast column (column 1) - DOUBLE
        let mut forecast_vector = output.flat_vector(1);
        let forecast_slice = forecast_vector.as_mut_slice::<f64>();
        forecast_slice[..num_rows].copy_from_slice(&result.forecasts);

        // Write lower_bound column (column 2) - DOUBLE
        let mut lower_vector = output.flat_vector(2);
        let lower_slice = lower_vector.as_mut_slice::<f64>();
        lower_slice[..num_rows].copy_from_slice(&result.lower_bounds);

        // Write upper_bound column (column 3) - DOUBLE
        let mut upper_vector = output.flat_vector(3);
        let upper_slice = upper_vector.as_mut_slice::<f64>();
        upper_slice[..num_rows].copy_from_slice(&result.upper_bounds);

        output.set_len(num_rows);
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
