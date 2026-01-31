pub mod detection;
pub mod params;

use duckdb::{
    core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId},
    vtab::{BindInfo, InitInfo, TableFunctionInfo, VTab},
    Result,
};
use std::cell::UnsafeCell;
use std::error::Error;

use crate::common::grouping::GroupSeasonalityRow;
use crate::common::table_reader;
use crate::ConnHandle;
use detection::{detect_seasonality, MIN_SEASONALITY_POINTS};
use params::SeasonalityParams;

/// Bind data for the detect_seasonality table function.
/// Stores the parsed parameters from the SQL call.
#[repr(C)]
pub struct SeasonalityBindData {
    params: SeasonalityParams,
}

// Safety: SeasonalityBindData is read-only after creation and contains only owned data.
unsafe impl Send for SeasonalityBindData {}
unsafe impl Sync for SeasonalityBindData {}

/// Mutable state for the seasonality init data.
struct SeasonalityState {
    /// Pre-computed rows for output.
    rows: Vec<GroupSeasonalityRow>,
    /// Current position into `rows` for chunked emission.
    position: usize,
    /// Whether the computation has been performed.
    computed: bool,
}

/// Init data for the detect_seasonality table function.
///
/// Uses UnsafeCell for interior mutability because the VTab func() receives
/// &self but we need to track position across multiple calls. The DuckDB VTab
/// contract guarantees single-threaded access per query execution.
#[repr(C)]
pub struct SeasonalityInitData {
    state: UnsafeCell<SeasonalityState>,
}

// Safety: DuckDB guarantees single-threaded access per query.
unsafe impl Send for SeasonalityInitData {}
unsafe impl Sync for SeasonalityInitData {}

pub struct SeasonalityVTab;

impl VTab for SeasonalityVTab {
    type InitData = SeasonalityInitData;
    type BindData = SeasonalityBindData;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let params = SeasonalityParams::from_bind_info(bind)?;

        // Add group columns first (all as VARCHAR)
        for col_name in &params.group_by {
            bind.add_result_column(col_name, LogicalTypeHandle::from(LogicalTypeId::Varchar));
        }

        // Add the seasonality output columns
        bind.add_result_column("period", LogicalTypeHandle::from(LogicalTypeId::Integer));
        bind.add_result_column("strength", LogicalTypeHandle::from(LogicalTypeId::Double));

        Ok(SeasonalityBindData { params })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(SeasonalityInitData {
            state: UnsafeCell::new(SeasonalityState {
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
        let state = unsafe { &mut *init_data.state.get() };

        // Get the connection handle from extra info
        let conn_handle_ptr = func.get_extra_info::<ConnHandle>();
        if conn_handle_ptr.is_null() {
            return Err("Internal error: connection handle not available".into());
        }
        let con = unsafe { (*conn_handle_ptr).0 };

        // Compute results on first call
        if !state.computed {
            state.computed = true;

            if params.group_by.is_empty() {
                state.rows = compute_single_seasonality(con, params)?;
            } else {
                state.rows = compute_grouped_seasonality(con, params)?;
            }
        }

        // Emit rows from the pre-computed results
        let remaining = state.rows.len() - state.position;
        if remaining == 0 {
            output.set_len(0);
            return Ok(());
        }

        let chunk_size = unsafe { libduckdb_sys::duckdb_vector_size() } as usize;
        let emit_count = remaining.min(chunk_size);
        let start = state.position;
        let end = start + emit_count;
        let num_group_cols = params.group_by.len();

        // Validate group column count
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

        // Write period (INTEGER as i32)
        let period_col = num_group_cols;
        let mut period_vector = output.flat_vector(period_col);
        let period_slice = period_vector.as_mut_slice::<i32>();
        for (row_offset, row) in state.rows[start..end].iter().enumerate() {
            period_slice[row_offset] = row.period;
        }

        // Write strength (DOUBLE)
        let strength_col = num_group_cols + 1;
        let mut strength_vector = output.flat_vector(strength_col);
        let strength_slice = strength_vector.as_mut_slice::<f64>();
        for (row_offset, row) in state.rows[start..end].iter().enumerate() {
            strength_slice[row_offset] = row.strength;
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
        ])
    }
}

/// Compute seasonality detection for a single (non-grouped) time series.
fn compute_single_seasonality(
    con: libduckdb_sys::duckdb_connection,
    params: &SeasonalityParams,
) -> Result<Vec<GroupSeasonalityRow>, Box<dyn Error>> {
    let series = unsafe {
        table_reader::read_time_series(
            con,
            &params.table_name,
            &params.timestamp_col,
            &params.value_col,
        )
    }
    .map_err(|e: String| -> Box<dyn Error> { e.into() })?;

    if series.len() < MIN_SEASONALITY_POINTS {
        return Err(format!(
            "Need at least {} data points for seasonality detection, got {}",
            MIN_SEASONALITY_POINTS,
            series.len()
        )
        .into());
    }

    let results = detect_seasonality(&series.values)
        .map_err(|e: String| -> Box<dyn Error> { e.into() })?;

    Ok(results
        .into_iter()
        .map(|r| GroupSeasonalityRow {
            group_values: Vec::new(),
            period: r.period as i32,
            strength: r.strength,
        })
        .collect())
}

/// Compute seasonality detection for each group independently.
/// Groups with insufficient data (< MIN_SEASONALITY_POINTS) are silently skipped.
fn compute_grouped_seasonality(
    con: libduckdb_sys::duckdb_connection,
    params: &SeasonalityParams,
) -> Result<Vec<GroupSeasonalityRow>, Box<dyn Error>> {
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
        // Skip groups with insufficient data
        if group.series.len() < MIN_SEASONALITY_POINTS {
            eprintln!(
                "quackstats: skipping group {:?} - insufficient data ({} points, need {})",
                group.group_key,
                group.series.len(),
                MIN_SEASONALITY_POINTS
            );
            continue;
        }

        let results = match detect_seasonality(&group.series.values) {
            Ok(r) => r,
            Err(e) => {
                eprintln!(
                    "quackstats: skipping group {:?} - seasonality detection failed: {}",
                    group.group_key, e
                );
                continue;
            }
        };

        for r in results {
            all_rows.push(GroupSeasonalityRow {
                group_values: group.group_key.clone(),
                period: r.period as i32,
                strength: r.strength,
            });
        }
    }

    Ok(all_rows)
}
