use duckdb::vtab::BindInfo;

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

        // group_by is optional; Phase 2 stores but doesn't use it
        let group_by = Vec::new();

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
