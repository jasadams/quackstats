use augurs_core::{Fit, Predict};
use augurs_ets::AutoETS;

use crate::common::types::{detect_interval, ForecastResult, TimeSeries};

/// Minimum number of data points required for ETS fitting.
const MIN_DATA_POINTS: usize = 4;

/// Run ETS forecasting on a time series.
///
/// Fits an automatic ETS model to the values, generates point forecasts
/// for `horizon` steps, and computes prediction intervals at the given
/// confidence level.
pub fn forecast_ets(
    series: &TimeSeries,
    horizon: usize,
    confidence_level: f64,
) -> Result<ForecastResult, String> {
    if series.len() < MIN_DATA_POINTS {
        return Err(format!(
            "Need at least {} data points for ETS forecasting, got {}",
            MIN_DATA_POINTS,
            series.len()
        ));
    }

    // Fit the model - use non-seasonal AutoETS (season_length=1, "ZZN")
    // Phase 2 does not support seasonality detection; that's Phase 5.
    let model = AutoETS::non_seasonal();
    let fitted = model
        .fit(&series.values)
        .map_err(|e| format!("ETS model fitting failed: {}", e))?;

    // Generate forecasts with prediction intervals
    let forecast = fitted
        .predict(horizon, confidence_level)
        .map_err(|e| format!("ETS prediction failed: {}", e))?;

    // Detect the time interval between data points
    let interval_days = detect_interval(&series.timestamps);
    let last_ts = series
        .last_timestamp()
        .ok_or_else(|| "Time series has no timestamps".to_string())?;

    // Generate future timestamps
    let timestamps: Vec<i32> = (1..=horizon as i32)
        .map(|i| last_ts + i * interval_days)
        .collect();

    // Extract intervals (fall back to point forecast if no intervals available)
    let (lower_bounds, upper_bounds) = match forecast.intervals {
        Some(intervals) => (intervals.lower, intervals.upper),
        None => {
            // No intervals available - use point forecast as both bounds
            (forecast.point.clone(), forecast.point.clone())
        }
    };

    Ok(ForecastResult {
        timestamps,
        forecasts: forecast.point,
        lower_bounds,
        upper_bounds,
    })
}
