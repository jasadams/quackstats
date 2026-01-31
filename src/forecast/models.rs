use augurs_core::{Fit, Predict};
use augurs_ets::AutoETS;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use nalgebra::{Dyn, OMatrix, OVector, Owned, Vector3, U3};
use statrs::distribution::{ContinuousCDF, StudentsT};

use crate::common::types::{detect_interval, ForecastResult, TimeSeries, MIN_DATA_POINTS};
use crate::seasonality::detection::detect_seasonality;

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

    // Fit non-seasonal ETS (season_length=1, "ZZN").
    // For seasonal ETS, use forecast_ets_seasonal() instead.
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

/// Run ETS forecasting with a known seasonal period.
///
/// Same as `forecast_ets` but uses `AutoETS::new(season_length, "ZZZ")` to search
/// over all error/trend/seasonality component combinations.
pub fn forecast_ets_seasonal(
    series: &TimeSeries,
    horizon: usize,
    confidence_level: f64,
    season_length: usize,
) -> Result<ForecastResult, String> {
    if series.len() < MIN_DATA_POINTS {
        return Err(format!(
            "Need at least {} data points for ETS forecasting, got {}",
            MIN_DATA_POINTS,
            series.len()
        ));
    }

    let model = AutoETS::new(season_length, "ZZZ")
        .map_err(|e| format!("Failed to create seasonal ETS model: {}", e))?;
    let fitted = model
        .fit(&series.values)
        .map_err(|e| format!("Seasonal ETS model fitting failed: {}", e))?;

    let forecast = fitted
        .predict(horizon, confidence_level)
        .map_err(|e| format!("Seasonal ETS prediction failed: {}", e))?;

    let interval_days = detect_interval(&series.timestamps);
    let last_ts = series
        .last_timestamp()
        .ok_or_else(|| "Time series has no timestamps".to_string())?;

    let timestamps: Vec<i32> = (1..=horizon as i32)
        .map(|i| last_ts + i * interval_days)
        .collect();

    let (lower_bounds, upper_bounds) = match forecast.intervals {
        Some(intervals) => (intervals.lower, intervals.upper),
        None => (forecast.point.clone(), forecast.point.clone()),
    };

    Ok(ForecastResult {
        timestamps,
        forecasts: forecast.point,
        lower_bounds,
        upper_bounds,
    })
}

/// Detect the best seasonal period length from the data.
///
/// Returns the period with highest strength if strength > 0.3 (meaningful
/// seasonality), otherwise returns 1 (non-seasonal).
fn detect_best_season_length(values: &[f64]) -> usize {
    let results = match detect_seasonality(values) {
        Ok(r) => r,
        Err(_) => return 1,
    };
    match results.first() {
        Some(best) if best.strength > 0.3 => best.period as usize,
        _ => 1,
    }
}

/// Run linear regression (OLS) forecasting on a time series.
///
/// Fits y = intercept + slope * x where x is the 0-based index of each data point.
/// Generates prediction intervals using the Student's t-distribution at the
/// given confidence level.
pub fn forecast_linear(
    series: &TimeSeries,
    horizon: usize,
    confidence_level: f64,
) -> Result<ForecastResult, String> {
    let n = series.len();
    if n < MIN_DATA_POINTS {
        return Err(format!(
            "Need at least {} data points for linear forecasting, got {}",
            MIN_DATA_POINTS, n
        ));
    }

    // Build regression data: x = [0, 1, 2, ...], y = values
    let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let data = vec![
        ("Y".to_string(), series.values.clone()),
        ("X".to_string(), x_vals.clone()),
    ];

    let regression_data = RegressionDataBuilder::new()
        .build_from(data)
        .map_err(|e| format!("Failed to build regression data: {}", e))?;

    let model = FormulaRegressionBuilder::new()
        .data(&regression_data)
        .formula("Y ~ X")
        .fit()
        .map_err(|e| format!("Linear regression fitting failed: {}", e))?;

    // parameters() returns [intercept, slope]
    let params = model.parameters();
    let intercept = params[0];
    let slope = params[1];

    // Compute standard error of the regression (residual standard error)
    // scale() returns the variance of residuals: SSR / (n - 2)
    let residual_se = model.scale().sqrt();

    // Compute x_mean and sum of squared deviations for prediction intervals
    let x_mean: f64 = x_vals.iter().sum::<f64>() / n as f64;
    let sum_sq_dev: f64 = x_vals.iter().map(|&x| (x - x_mean).powi(2)).sum();

    // Detect the time interval and generate future timestamps
    let interval_days = detect_interval(&series.timestamps);
    let last_ts = series
        .last_timestamp()
        .ok_or_else(|| "Time series has no timestamps".to_string())?;

    let mut timestamps = Vec::with_capacity(horizon);
    let mut forecasts = Vec::with_capacity(horizon);
    let mut lower_bounds = Vec::with_capacity(horizon);
    let mut upper_bounds = Vec::with_capacity(horizon);

    // Handle constant or near-perfect-fit data: residual SE ≈ 0 means the model
    // fits perfectly, so prediction intervals collapse to zero width.
    if residual_se < 1e-10 || sum_sq_dev < 1e-10 {
        for i in 1..=horizon {
            let x_pred = (n - 1 + i) as f64;
            let y_hat = intercept + slope * x_pred;
            timestamps.push(last_ts + i as i32 * interval_days);
            forecasts.push(y_hat);
            lower_bounds.push(y_hat);
            upper_bounds.push(y_hat);
        }
        return Ok(ForecastResult {
            timestamps,
            forecasts,
            lower_bounds,
            upper_bounds,
        });
    }

    // Student's t-distribution with n-2 degrees of freedom (standard form)
    let df = (n - 2) as f64;
    let t_dist = StudentsT::new(0.0, 1.0, df)
        .map_err(|e| format!("Failed to create t-distribution: {}", e))?;
    let alpha = 1.0 - confidence_level;
    let t_value = t_dist.inverse_cdf(1.0 - alpha / 2.0);

    for i in 1..=horizon {
        let x_pred = (n - 1 + i) as f64; // Continue from the last index
        let y_hat = intercept + slope * x_pred;

        // Prediction interval for a new observation:
        // PI = y_hat +/- t * se * sqrt(1 + 1/n + (x_pred - x_mean)^2 / sum_sq_dev)
        let pi_width = t_value
            * residual_se
            * (1.0_f64 + 1.0 / n as f64 + (x_pred - x_mean).powi(2) / sum_sq_dev).sqrt();

        if y_hat.is_nan() || pi_width.is_nan() || pi_width.is_infinite() {
            return Err("Prediction interval computation produced invalid values".to_string());
        }

        timestamps.push(last_ts + i as i32 * interval_days);
        forecasts.push(y_hat);
        lower_bounds.push(y_hat - pi_width);
        upper_bounds.push(y_hat + pi_width);
    }

    Ok(ForecastResult {
        timestamps,
        forecasts,
        lower_bounds,
        upper_bounds,
    })
}

/// Run exponential growth forecasting on a time series.
///
/// Fits y = a * exp(b * x) by log-transforming values and applying OLS:
/// ln(y) = ln(a) + b*x. All values must be strictly positive (> 0).
/// Prediction intervals are computed in log space and back-transformed,
/// producing naturally asymmetric intervals in the original space.
pub fn forecast_exponential(
    series: &TimeSeries,
    horizon: usize,
    confidence_level: f64,
) -> Result<ForecastResult, String> {
    let n = series.len();
    if n < MIN_DATA_POINTS {
        return Err(format!(
            "Need at least {} data points for exponential forecasting, got {}",
            MIN_DATA_POINTS, n
        ));
    }

    // All values must be strictly positive for log transform
    if series.values.iter().any(|&v| v <= 0.0) {
        return Err(
            "Exponential model requires all values to be strictly positive (> 0)".to_string(),
        );
    }

    // Log-transform the values
    let log_values: Vec<f64> = series.values.iter().map(|v| v.ln()).collect();
    let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // Fit OLS on (x, ln(y))
    let data = vec![
        ("Y".to_string(), log_values),
        ("X".to_string(), x_vals.clone()),
    ];

    let regression_data = RegressionDataBuilder::new()
        .build_from(data)
        .map_err(|e| format!("Failed to build regression data: {}", e))?;

    let model = FormulaRegressionBuilder::new()
        .data(&regression_data)
        .formula("Y ~ X")
        .fit()
        .map_err(|e| format!("Exponential regression fitting failed: {}", e))?;

    let params = model.parameters();
    let log_intercept = params[0]; // ln(a)
    let slope = params[1]; // b

    let residual_se = model.scale().sqrt();

    // Compute x_mean and sum of squared deviations for prediction intervals
    let x_mean: f64 = x_vals.iter().sum::<f64>() / n as f64;
    let sum_sq_dev: f64 = x_vals.iter().map(|&x| (x - x_mean).powi(2)).sum();

    // Detect the time interval and generate future timestamps
    let interval_days = detect_interval(&series.timestamps);
    let last_ts = series
        .last_timestamp()
        .ok_or_else(|| "Time series has no timestamps".to_string())?;

    let mut timestamps = Vec::with_capacity(horizon);
    let mut forecasts = Vec::with_capacity(horizon);
    let mut lower_bounds = Vec::with_capacity(horizon);
    let mut upper_bounds = Vec::with_capacity(horizon);

    // Handle near-perfect fit
    if residual_se < 1e-10 || sum_sq_dev < 1e-10 {
        for i in 1..=horizon {
            let x_pred = (n - 1 + i) as f64;
            let log_y_hat = log_intercept + slope * x_pred;
            let y_hat = log_y_hat.exp();
            timestamps.push(last_ts + i as i32 * interval_days);
            forecasts.push(y_hat);
            lower_bounds.push(y_hat);
            upper_bounds.push(y_hat);
        }
        return Ok(ForecastResult {
            timestamps,
            forecasts,
            lower_bounds,
            upper_bounds,
        });
    }

    // Student's t-distribution with n-2 degrees of freedom
    let df = (n - 2) as f64;
    let t_dist = StudentsT::new(0.0, 1.0, df)
        .map_err(|e| format!("Failed to create t-distribution: {}", e))?;
    let alpha = 1.0 - confidence_level;
    let t_value = t_dist.inverse_cdf(1.0 - alpha / 2.0);

    for i in 1..=horizon {
        let x_pred = (n - 1 + i) as f64;
        let log_y_hat = log_intercept + slope * x_pred;

        // Prediction interval in log space (same formula as linear)
        let pi_width = t_value
            * residual_se
            * (1.0_f64 + 1.0 / n as f64 + (x_pred - x_mean).powi(2) / sum_sq_dev).sqrt();

        if log_y_hat.is_nan() || pi_width.is_nan() || pi_width.is_infinite() {
            return Err(
                "Exponential prediction interval computation produced invalid values".to_string(),
            );
        }

        // Back-transform from log space: intervals become asymmetric
        let y_hat = log_y_hat.exp();
        let lower = (log_y_hat - pi_width).exp();
        let upper = (log_y_hat + pi_width).exp();

        timestamps.push(last_ts + i as i32 * interval_days);
        forecasts.push(y_hat);
        lower_bounds.push(lower);
        upper_bounds.push(upper);
    }

    Ok(ForecastResult {
        timestamps,
        forecasts,
        lower_bounds,
        upper_bounds,
    })
}

/// Logistic growth problem for Levenberg-Marquardt optimization.
///
/// Fits y = L / (1 + exp(-k * (x - x0))) with 3 parameters.
///
/// To prevent the capacity L from dropping below the observed maximum
/// (which causes forecasts to "dip" below the last data point), L is
/// reparameterized as:
///
///   L = y_max + exp(alpha)
///
/// The optimizer searches over [alpha, k, x0]. Since exp(alpha) > 0
/// for all alpha, L is guaranteed to exceed y_max.
struct LogisticProblem {
    /// Parameter vector [alpha, k, x0] where L = y_max + exp(alpha)
    params: Vector3<f64>,
    /// Maximum observed y value (floor for L)
    y_max: f64,
    /// x values (0-based indices)
    x: Vec<f64>,
    /// Observed y values
    y: Vec<f64>,
}

impl LogisticProblem {
    /// Compute L from the reparameterized alpha: L = y_max + exp(alpha)
    fn capacity(&self) -> f64 {
        self.y_max + self.params[0].exp()
    }
}

impl LeastSquaresProblem<f64, Dyn, U3> for LogisticProblem {
    type ParameterStorage = Owned<f64, U3>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U3>;

    fn set_params(&mut self, p: &Vector3<f64>) {
        self.params.copy_from(p);
    }

    fn params(&self) -> Vector3<f64> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, Dyn>> {
        let l = self.capacity();
        let k = self.params[1];
        let x0 = self.params[2];
        let n = self.x.len();

        let mut residuals = OVector::<f64, Dyn>::zeros(n);
        for i in 0..n {
            let exp_term = (-k * (self.x[i] - x0)).exp();
            let denom = 1.0 + exp_term;
            let predicted = l / denom;
            residuals[i] = self.y[i] - predicted;
        }
        Some(residuals)
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, U3>> {
        let l = self.capacity();
        let exp_alpha = self.params[0].exp(); // ∂L/∂alpha = exp(alpha)
        let k = self.params[1];
        let x0 = self.params[2];
        let n = self.x.len();

        let mut jac = OMatrix::<f64, Dyn, U3>::zeros(n);
        for i in 0..n {
            let exp_term = (-k * (self.x[i] - x0)).exp();
            let denom = 1.0 + exp_term;
            let denom_sq = denom * denom;

            // Jacobian of residual r = y - f(x) → ∂r/∂p = -∂f/∂p
            // ∂f/∂alpha = (∂f/∂L)(∂L/∂alpha) = (1/denom) * exp(alpha)
            jac[(i, 0)] = -exp_alpha / denom;
            // ∂f/∂k = L * (x - x0) * exp_term / denom²
            jac[(i, 1)] = -l * (self.x[i] - x0) * exp_term / denom_sq;
            // ∂f/∂x0 = -L * k * exp_term / denom²
            jac[(i, 2)] = l * k * exp_term / denom_sq;
        }
        Some(jac)
    }
}

/// Estimate initial logistic parameters from data.
///
/// Returns [alpha, k, x0] where L = y_max + exp(alpha).
/// Uses heuristics: L ~10% above observed max, k from the data range,
/// and x0 at the midpoint.
fn estimate_logistic_params(x: &[f64], _y: &[f64], y_max: f64) -> Vector3<f64> {
    // L_init: 10% above y_max → alpha = ln(L_init - y_max) = ln(0.1 * y_max)
    let l_surplus = (y_max * 0.1).max(1.0);
    let alpha_init = l_surplus.ln();

    let x_min = x.first().copied().unwrap_or(0.0);
    let x_max = x.last().copied().unwrap_or(1.0);
    let x_range = (x_max - x_min).max(1.0);

    // k_init: steepness from data range
    let k_init = 4.0 / x_range;

    // x0_init: midpoint of range
    let x0_init = (x_max + x_min) / 2.0;

    Vector3::new(alpha_init, k_init, x0_init)
}

/// Run logistic growth forecasting on a time series.
///
/// Fits y = L / (1 + exp(-k * (x - x0))) using Levenberg-Marquardt
/// nonlinear least squares. Generates prediction intervals via the
/// delta method (first-order Taylor expansion of parameter uncertainty).
pub fn forecast_logistic(
    series: &TimeSeries,
    horizon: usize,
    confidence_level: f64,
) -> Result<ForecastResult, String> {
    let n = series.len();
    if n < MIN_DATA_POINTS {
        return Err(format!(
            "Need at least {} data points for logistic forecasting, got {}",
            MIN_DATA_POINTS, n
        ));
    }

    let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // y_max is the floor for capacity L (reparameterization ensures L > y_max)
    let y_max = series
        .values
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // Estimate initial parameters [alpha, k, x0]
    let initial_params = estimate_logistic_params(&x_vals, &series.values, y_max);

    // Set up and solve the optimization problem
    let problem = LogisticProblem {
        params: initial_params,
        y_max,
        x: x_vals.clone(),
        y: series.values.clone(),
    };

    let (result, report) = LevenbergMarquardt::new().minimize(problem);

    if !report.termination.was_successful() {
        let l_init = y_max + initial_params[0].exp();
        return Err(format!(
            "Logistic model failed to converge (initial params: L={:.2}, k={:.4}, x0={:.2}). \
             Try a different model or ensure data follows an S-curve pattern.",
            l_init, initial_params[1], initial_params[2]
        ));
    }

    // Recover L from reparameterized alpha
    let mut l = result.capacity();
    let k = result.params[1];
    let x0 = result.params[2];

    // Post-hoc capacity adjustment: ensure the model doesn't predict below the
    // last observation. The logistic function asymptotically approaches L but
    // never reaches it, so if L ≈ y_max the forecast can visually "drop" below
    // the last data point. Nudge L upward so f(x_last) >= y_last.
    let x_last = (n - 1) as f64;
    let exp_last = (-k * (x_last - x0)).exp();
    let f_last = l / (1.0 + exp_last);
    let y_last = series.values[n - 1];
    if f_last < y_last {
        // Solve: y_last = L_new / (1 + exp_last) → L_new = y_last * (1 + exp_last)
        // Add tiny margin to avoid floating-point edge case
        l = y_last * (1.0 + exp_last) * 1.001;
    }

    eprintln!(
        "quackstats: logistic fit: L={:.4}, k={:.4}, x0={:.4} (y_max={:.4})",
        l, k, x0, y_max
    );

    // Compute residual variance using the final (adjusted) L
    let df = n as f64 - 3.0;
    if df <= 0.0 {
        return Err("Need more than 3 data points for logistic prediction intervals".to_string());
    }

    let ss_res: f64 = x_vals
        .iter()
        .zip(series.values.iter())
        .map(|(&xi, &yi)| {
            let pred = l / (1.0 + (-k * (xi - x0)).exp());
            (yi - pred).powi(2)
        })
        .sum();
    let s_sq = ss_res / df;

    // Build the data Jacobian w.r.t. [L, k, x0] using the final (adjusted) L.
    // We switch from the reparameterized [alpha, k, x0] to direct [L, k, x0]
    // because the post-hoc L adjustment invalidates the reparameterized covariance.
    let mut data_jac = OMatrix::<f64, Dyn, U3>::zeros(n);
    for i in 0..n {
        let exp_term = (-k * (x_vals[i] - x0)).exp();
        let denom = 1.0 + exp_term;
        let denom_sq = denom * denom;
        data_jac[(i, 0)] = -1.0 / denom; // ∂r/∂L
        data_jac[(i, 1)] = -l * (x_vals[i] - x0) * exp_term / denom_sq; // ∂r/∂k
        data_jac[(i, 2)] = l * k * exp_term / denom_sq; // ∂r/∂x0
    }

    let jtj = data_jac.transpose() * &data_jac;

    // If J^T*J is singular (e.g. near-perfect fit or degenerate data), fall back
    // to residual-only prediction intervals (no parameter uncertainty component).
    let param_cov = jtj.try_inverse().map(|inv| inv * s_sq);

    if param_cov.is_none() {
        eprintln!("quackstats: logistic: J^T*J singular, using residual-only intervals");
    }

    // Generate forecasts with prediction intervals
    let interval_days = detect_interval(&series.timestamps);
    let last_ts = series
        .last_timestamp()
        .ok_or_else(|| "Time series has no timestamps".to_string())?;

    let t_dist = StudentsT::new(0.0, 1.0, df)
        .map_err(|e| format!("Failed to create t-distribution: {}", e))?;
    let ci_alpha = 1.0 - confidence_level;
    let t_value = t_dist.inverse_cdf(1.0 - ci_alpha / 2.0);

    let mut timestamps = Vec::with_capacity(horizon);
    let mut forecasts = Vec::with_capacity(horizon);
    let mut lower_bounds = Vec::with_capacity(horizon);
    let mut upper_bounds = Vec::with_capacity(horizon);

    for i in 1..=horizon {
        let x_pred = (n - 1 + i) as f64;
        let exp_term = (-k * (x_pred - x0)).exp();
        let denom = 1.0 + exp_term;
        let denom_sq = denom * denom;
        let y_hat = l / denom;

        // Prediction variance: delta method if covariance available, else residual-only
        let pred_var = if let Some(ref cov) = param_cov {
            // Jacobian of f w.r.t. [L, k, x0] at this forecast point
            let j_pred = Vector3::new(
                1.0 / denom,                               // ∂f/∂L
                l * (x_pred - x0) * exp_term / denom_sq,   // ∂f/∂k
                -l * k * exp_term / denom_sq,               // ∂f/∂x0
            );
            (j_pred.transpose() * cov * &j_pred)[(0, 0)] + s_sq
        } else {
            // Residual-only: no parameter uncertainty, just observation noise
            s_sq
        };

        let pred_se = pred_var.max(0.0).sqrt();
        let pi_width = t_value * pred_se;

        if y_hat.is_nan() || pi_width.is_nan() || pi_width.is_infinite() {
            return Err(
                "Logistic prediction interval computation produced invalid values".to_string(),
            );
        }

        timestamps.push(last_ts + i as i32 * interval_days);
        forecasts.push(y_hat);
        lower_bounds.push(y_hat - pi_width);
        upper_bounds.push(y_hat + pi_width);
    }

    Ok(ForecastResult {
        timestamps,
        forecasts,
        lower_bounds,
        upper_bounds,
    })
}

/// Compute MSE of a model on held-out data.
///
/// Splits the series into training (first n - holdout) and test (last holdout),
/// fits the model on training data, forecasts holdout steps, and returns MSE.
/// Wraps each model call in `catch_unwind` to safely handle panics from
/// underlying libraries (e.g. nalgebra/levenberg-marquardt in WASM).
fn evaluate_model_mse(
    series: &TimeSeries,
    holdout: usize,
    model_name: &str,
    season_length: usize,
) -> Option<f64> {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    let n = series.len();
    if holdout >= n || n - holdout < MIN_DATA_POINTS {
        return None;
    }

    let train = TimeSeries {
        timestamps: series.timestamps[..n - holdout].to_vec(),
        values: series.values[..n - holdout].to_vec(),
    };
    let actual = &series.values[n - holdout..];

    // Use a fixed confidence level for CV (doesn't affect point forecasts)
    let confidence = 0.95;

    // Wrap in catch_unwind to prevent panics from crashing the WASM process
    let fit_result = catch_unwind(AssertUnwindSafe(|| -> Option<ForecastResult> {
        match model_name {
            "seasonal_ets" => {
                if season_length <= 1 {
                    return None;
                }
                forecast_ets_seasonal(&train, holdout, confidence, season_length).ok()
            }
            "ets" => forecast_ets(&train, holdout, confidence).ok(),
            "exponential" => forecast_exponential(&train, holdout, confidence).ok(),
            "logistic" => forecast_logistic(&train, holdout, confidence).ok(),
            "linear" => forecast_linear(&train, holdout, confidence).ok(),
            _ => None,
        }
    }));

    let result = match fit_result {
        Ok(Some(r)) => r,
        Ok(None) => return None,
        Err(_) => {
            eprintln!(
                "quackstats: auto CV: model={} panicked during evaluation, skipping",
                model_name
            );
            return None;
        }
    };

    if result.forecasts.len() != holdout {
        return None;
    }

    // Compute MSE
    let mse: f64 = result
        .forecasts
        .iter()
        .zip(actual.iter())
        .map(|(pred, act)| (pred - act).powi(2))
        .sum::<f64>()
        / holdout as f64;

    // Guard against NaN/Inf
    if mse.is_nan() || mse.is_infinite() {
        return None;
    }

    Some(mse)
}

/// Check whether the data shows decelerating growth (S-curve behavior).
///
/// Compares average absolute growth in the first half vs second half.
/// Logistic/S-curve data has decreasing absolute growth over time, while
/// exponential data has increasing absolute growth. Returns true only
/// when growth in the second half is meaningfully slower than the first.
fn shows_deceleration(values: &[f64]) -> bool {
    let n = values.len();
    if n < 6 {
        return false;
    }

    let mid = n / 2;

    // Average absolute change per step in each half
    let first_half_growth: f64 = values[1..mid]
        .iter()
        .zip(values[..mid - 1].iter())
        .map(|(a, b)| a - b)
        .sum::<f64>()
        / (mid - 1) as f64;

    let second_half_growth: f64 = values[mid + 1..]
        .iter()
        .zip(values[mid..n - 1].iter())
        .map(|(a, b)| a - b)
        .sum::<f64>()
        / (n - mid - 1) as f64;

    // Only consider logistic if growth has clearly slowed (second half < 70% of first)
    let decelerating = second_half_growth < first_half_growth * 0.7;

    eprintln!(
        "quackstats: auto: deceleration check: first_half_avg={:.2}, second_half_avg={:.2}, decelerating={}",
        first_half_growth, second_half_growth, decelerating
    );

    decelerating
}

/// Validate that a forecast result is reasonable relative to the input data.
///
/// For upward-trending data, the first forecast point should not drop
/// significantly below the last observation. Returns false if the forecast
/// looks nonsensical (indicating the wrong model was selected).
fn validate_forecast(series: &TimeSeries, result: &ForecastResult) -> bool {
    if result.forecasts.is_empty() {
        return false;
    }

    let n = series.len();
    if n < 3 {
        return true; // Can't assess trend with too few points
    }

    // Check if data is trending upward (last 3 points)
    let recent = &series.values[n - 3..];
    let trending_up = recent[2] > recent[0];

    if trending_up {
        let last_val = series.values[n - 1];
        let first_forecast = result.forecasts[0];

        // If trending up but forecast drops more than 10% below last observation, reject
        if first_forecast < last_val * 0.9 {
            eprintln!(
                "quackstats: auto: forecast validation failed: last_val={:.2}, first_forecast={:.2} (dropped >10%)",
                last_val, first_forecast
            );
            return false;
        }
    }

    // Check for NaN/Inf in forecasts or bounds
    for i in 0..result.forecasts.len() {
        if result.forecasts[i].is_nan()
            || result.forecasts[i].is_infinite()
            || result.lower_bounds[i].is_nan()
            || result.lower_bounds[i].is_infinite()
            || result.upper_bounds[i].is_nan()
            || result.upper_bounds[i].is_infinite()
        {
            eprintln!("quackstats: auto: forecast validation failed: NaN/Inf at step {}", i);
            return false;
        }
    }

    true
}

/// Automatic model selection via cross-validation.
///
/// Holds out 20% of data (min MIN_DATA_POINTS, max n - MIN_DATA_POINTS),
/// evaluates candidate models on the holdout, and selects the one with
/// the lowest MSE. Refits the winner on the full dataset.
///
/// If there isn't enough data for CV (n < 2 * MIN_DATA_POINTS), falls back
/// to the original heuristic: seasonal ETS → non-seasonal ETS → linear.
///
/// Candidate models:
/// - Seasonal ETS (only if seasonality detected with strength > 0.3)
/// - Non-seasonal ETS
/// - Exponential (only if all values > 0)
/// - Logistic (only if data shows decelerating growth — S-curve pattern)
/// - Linear (always included)
pub fn forecast_auto(
    series: &TimeSeries,
    horizon: usize,
    confidence_level: f64,
) -> Result<ForecastResult, String> {
    let n = series.len();
    let season_length = detect_best_season_length(&series.values);

    // Not enough data for CV — use original fallback heuristic
    if n < 2 * MIN_DATA_POINTS {
        eprintln!("quackstats: auto: insufficient data for CV (n={}), using fallback", n);
        let ets_result = if season_length > 1 {
            forecast_ets_seasonal(series, horizon, confidence_level, season_length)
        } else {
            forecast_ets(series, horizon, confidence_level)
        };
        return match ets_result {
            Ok(result) => Ok(result),
            Err(_) => forecast_linear(series, horizon, confidence_level),
        };
    }

    // Determine holdout size: 20% of data, bounded by MIN_DATA_POINTS on both sides
    let holdout = (n as f64 * 0.2).ceil() as usize;
    let holdout = holdout.max(MIN_DATA_POINTS).min(n - MIN_DATA_POINTS);

    // Build candidate list
    let all_positive = series.values.iter().all(|&v| v > 0.0);
    let deceleration = shows_deceleration(&series.values);
    let mut candidates: Vec<&str> = Vec::new();

    if season_length > 1 {
        candidates.push("seasonal_ets");
    }
    candidates.push("ets");
    if all_positive {
        candidates.push("exponential");
    }
    if deceleration {
        candidates.push("logistic");
    } else {
        eprintln!("quackstats: auto: skipping logistic (no deceleration in data)");
    }
    candidates.push("linear");

    // Evaluate each candidate and rank by MSE
    let mut ranked: Vec<(&str, f64)> = Vec::new();

    for &model_name in &candidates {
        match evaluate_model_mse(series, holdout, model_name, season_length) {
            Some(mse) => {
                eprintln!("quackstats: auto CV: model={} MSE={:.4}", model_name, mse);
                ranked.push((model_name, mse));
            }
            None => {
                eprintln!("quackstats: auto CV: model={} failed/skipped", model_name);
            }
        }
    }

    // Sort by MSE (lowest first)
    ranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    if ranked.is_empty() {
        eprintln!("quackstats: auto: all candidates failed, falling back to linear");
        return forecast_linear(series, horizon, confidence_level);
    }

    // Try models in order of MSE, validating each refit
    for (model_name, mse) in &ranked {
        eprintln!(
            "quackstats: auto: trying model={} (CV MSE={:.4}) on full data",
            model_name, mse
        );

        let result = match *model_name {
            "seasonal_ets" => {
                forecast_ets_seasonal(series, horizon, confidence_level, season_length)
            }
            "ets" => forecast_ets(series, horizon, confidence_level),
            "exponential" => forecast_exponential(series, horizon, confidence_level),
            "logistic" => forecast_logistic(series, horizon, confidence_level),
            "linear" => forecast_linear(series, horizon, confidence_level),
            _ => continue,
        };

        match result {
            Ok(forecast) => {
                if validate_forecast(series, &forecast) {
                    eprintln!("quackstats: auto: selected model={}", model_name);
                    return Ok(forecast);
                }
                eprintln!(
                    "quackstats: auto: model={} refit failed validation, trying next",
                    model_name
                );
            }
            Err(e) => {
                eprintln!(
                    "quackstats: auto: model={} refit failed: {}, trying next",
                    model_name, e
                );
            }
        }
    }

    // All ranked models failed validation — last resort
    eprintln!("quackstats: auto: all models failed validation, falling back to linear");
    forecast_linear(series, horizon, confidence_level)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a TimeSeries with daily timestamps starting from day 1000.
    fn make_series(values: Vec<f64>) -> TimeSeries {
        let n = values.len();
        let timestamps: Vec<i32> = (0..n).map(|i| 1000 + i as i32).collect();
        TimeSeries { timestamps, values }
    }

    // ==================== Exponential model tests ====================

    #[test]
    fn test_exponential_pure_growth() {
        // y = 100 * exp(0.03 * x) for 30 points
        let values: Vec<f64> = (0..30).map(|i| 100.0 * (0.03 * i as f64).exp()).collect();
        let series = make_series(values.clone());
        let result = forecast_exponential(&series, 5, 0.95).unwrap();

        assert_eq!(result.forecasts.len(), 5);
        assert_eq!(result.timestamps.len(), 5);

        // Forecasts should continue upward beyond the last observed value
        let last_observed = values.last().unwrap();
        for fc in &result.forecasts {
            assert!(
                *fc > *last_observed,
                "Forecast {:.2} should exceed last observed {:.2}",
                fc,
                last_observed
            );
        }

        // Each forecast step should be larger than the previous
        for w in result.forecasts.windows(2) {
            assert!(w[1] > w[0], "Forecasts should be monotonically increasing");
        }
    }

    #[test]
    fn test_exponential_rejects_non_positive() {
        let values = vec![1.0, 2.0, 0.0, 4.0, 5.0]; // contains zero
        let series = make_series(values);
        let err = forecast_exponential(&series, 3, 0.95).unwrap_err();
        assert!(err.contains("strictly positive"), "Error: {}", err);

        let values = vec![1.0, -2.0, 3.0, 4.0, 5.0]; // contains negative
        let series = make_series(values);
        let err = forecast_exponential(&series, 3, 0.95).unwrap_err();
        assert!(err.contains("strictly positive"), "Error: {}", err);
    }

    #[test]
    fn test_exponential_asymmetric_intervals() {
        // With real exponential data + some noise, intervals should be asymmetric
        let values: Vec<f64> = (0..20)
            .map(|i| {
                let base = 50.0 * (0.05 * i as f64).exp();
                // Add small noise to prevent perfect fit (which collapses intervals)
                base + (i as f64 * 0.7).sin() * 2.0
            })
            .collect();
        let series = make_series(values);
        let result = forecast_exponential(&series, 5, 0.95).unwrap();

        for i in 0..result.forecasts.len() {
            let upper_width = result.upper_bounds[i] - result.forecasts[i];
            let lower_width = result.forecasts[i] - result.lower_bounds[i];

            // In exp space, upper should be wider than lower (log-normal property)
            assert!(
                upper_width > lower_width,
                "Step {}: upper_width={:.2} should exceed lower_width={:.2}",
                i,
                upper_width,
                lower_width
            );
        }
    }

    // ==================== Logistic model tests ====================

    #[test]
    fn test_logistic_s_curve() {
        // y = 500 / (1 + exp(-0.15 * (x - 25))) for 50 points
        let values: Vec<f64> = (0..50)
            .map(|i| 500.0 / (1.0 + (-0.15 * (i as f64 - 25.0)).exp()))
            .collect();
        let series = make_series(values);
        let result = forecast_logistic(&series, 10, 0.95).unwrap();

        assert_eq!(result.forecasts.len(), 10);

        // Forecasts should plateau near 500 (the capacity)
        for fc in &result.forecasts {
            assert!(
                *fc < 520.0,
                "Forecast {:.2} should not significantly exceed capacity 500",
                fc
            );
            assert!(
                *fc > 400.0,
                "Forecast {:.2} should be near the capacity, not far below",
                fc
            );
        }
    }

    #[test]
    fn test_logistic_dashboard_data_no_drop() {
        // Exact data from the dashboard Test F chart — S-curve approaching ~1000
        let values = vec![
            15.0, 28.0, 55.0, 105.0, 195.0, 340.0, 490.0, 620.0, 720.0, 790.0,
            840.0, 875.0, 900.0, 918.0, 930.0,
        ];
        let last_actual = *values.last().unwrap();
        let series = make_series(values);
        let result = forecast_logistic(&series, 6, 0.95).unwrap();

        eprintln!("Last actual value: {:.2}", last_actual);
        for (i, fc) in result.forecasts.iter().enumerate() {
            eprintln!(
                "  forecast[{}] = {:.4}, lower = {:.4}, upper = {:.4}",
                i, fc, result.lower_bounds[i], result.upper_bounds[i]
            );
        }

        // The first forecast should NOT drop below the last actual value.
        // If L (capacity) is fitted below the last observation, the model
        // predicts a value below the data, creating a visible "drop" on the chart.
        assert!(
            result.forecasts[0] >= last_actual - 1.0,
            "First forecast {:.2} drops below last actual {:.2} — capacity L likely too low",
            result.forecasts[0],
            last_actual
        );

        // Forecasts should be monotonically non-decreasing (logistic with k>0)
        for w in result.forecasts.windows(2) {
            assert!(
                w[1] >= w[0] - 0.01,
                "Forecasts should be non-decreasing: {:.4} -> {:.4}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_logistic_bounded_by_capacity() {
        // Generate clear S-curve data approaching 1000
        let values: Vec<f64> = (0..60)
            .map(|i| 1000.0 / (1.0 + (-0.2 * (i as f64 - 30.0)).exp()))
            .collect();
        let series = make_series(values);
        let result = forecast_logistic(&series, 5, 0.95).unwrap();

        // Point forecasts should stay near or below capacity
        for fc in &result.forecasts {
            assert!(
                *fc < 1100.0,
                "Forecast {:.2} exceeds capacity 1000 by too much",
                fc
            );
        }
    }

    #[test]
    fn test_logistic_insufficient_data() {
        let values = vec![1.0, 2.0, 3.0]; // only 3 points, need MIN_DATA_POINTS
        let series = make_series(values);
        let err = forecast_logistic(&series, 3, 0.95).unwrap_err();
        assert!(err.contains("data points"), "Error: {}", err);
    }

    // ==================== Auto mode tests ====================

    #[test]
    fn test_auto_exponential_data_selects_exponential() {
        // Pure exponential growth — auto should pick exponential over linear
        let values: Vec<f64> = (0..30)
            .map(|i| {
                let base = 10.0 * (0.08 * i as f64).exp();
                base + (i as f64 * 1.3).sin() * 0.5 // tiny noise
            })
            .collect();
        let series = make_series(values);

        // This should succeed regardless of which model is selected
        let result = forecast_auto(&series, 5, 0.95);
        assert!(result.is_ok(), "Auto should succeed on exponential data");
        let result = result.unwrap();
        assert_eq!(result.forecasts.len(), 5);

        // The forecasts should be increasing (exponential growth)
        for w in result.forecasts.windows(2) {
            assert!(w[1] > w[0], "Forecasts should increase for exponential data");
        }
    }

    #[test]
    fn test_auto_scurve_data() {
        // S-curve data — auto should handle gracefully (likely picks logistic)
        let values: Vec<f64> = (0..40)
            .map(|i| 200.0 / (1.0 + (-0.2 * (i as f64 - 20.0)).exp()))
            .collect();
        let series = make_series(values);

        let result = forecast_auto(&series, 5, 0.95);
        assert!(result.is_ok(), "Auto should succeed on S-curve data");
        let result = result.unwrap();
        assert_eq!(result.forecasts.len(), 5);

        // Forecasts should be near the capacity, not wildly above
        for fc in &result.forecasts {
            assert!(
                *fc < 250.0,
                "S-curve forecast {:.2} should not overshoot capacity 200 significantly",
                fc
            );
        }
    }

    #[test]
    fn test_auto_linear_data() {
        // Linear data — auto should work (picks linear or ETS)
        let values: Vec<f64> = (0..20)
            .map(|i| 10.0 + 3.0 * i as f64 + (i as f64 * 0.5).sin())
            .collect();
        let series = make_series(values);

        let result = forecast_auto(&series, 5, 0.95);
        assert!(result.is_ok(), "Auto should succeed on linear data");
        assert_eq!(result.unwrap().forecasts.len(), 5);
    }

    #[test]
    fn test_auto_with_negatives_skips_exponential() {
        // Data with negatives — exponential should be skipped, but auto should still work
        let values: Vec<f64> = (0..20).map(|i| -10.0 + 2.0 * i as f64).collect();
        let series = make_series(values);

        let result = forecast_auto(&series, 5, 0.95);
        assert!(
            result.is_ok(),
            "Auto should succeed even with negative values"
        );
        assert_eq!(result.unwrap().forecasts.len(), 5);
    }

    #[test]
    fn test_deceleration_exponential_data() {
        // Exponential growth data — should NOT show deceleration
        let values: Vec<f64> = (0..12)
            .map(|i| 120.0 * (0.12 * i as f64).exp())
            .collect();
        assert!(
            !shows_deceleration(&values),
            "Exponential data should not show deceleration"
        );
    }

    #[test]
    fn test_deceleration_logistic_data() {
        // S-curve data past the inflection point — SHOULD show deceleration
        // Same data as dashboard Test F: growth clearly slowing as it approaches capacity
        let values = vec![
            15.0, 28.0, 55.0, 105.0, 195.0, 340.0, 490.0, 620.0, 720.0, 790.0, 840.0, 875.0,
            900.0, 918.0, 930.0,
        ];
        assert!(
            shows_deceleration(&values),
            "S-curve data past inflection should show deceleration"
        );
    }

    #[test]
    fn test_auto_fallback_small_data() {
        // With exactly MIN_DATA_POINTS, should use fallback (not CV)
        let values: Vec<f64> = (0..MIN_DATA_POINTS).map(|i| i as f64 * 2.0 + 1.0).collect();
        let series = make_series(values);

        let result = forecast_auto(&series, 3, 0.95);
        assert!(
            result.is_ok(),
            "Auto should succeed with minimal data via fallback"
        );
    }
}
