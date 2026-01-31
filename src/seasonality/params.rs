use duckdb::vtab::BindInfo;

use crate::common::params::{extract_list_strings, get_string_param};

/// Parsed and validated parameters for the detect_seasonality() table function.
pub struct SeasonalityParams {
    pub table_name: String,
    pub timestamp_col: String,
    pub value_col: String,
    pub group_by: Vec<String>,
}

impl SeasonalityParams {
    /// Extract and validate parameters from the VTab bind info.
    pub fn from_bind_info(bind: &BindInfo) -> Result<Self, Box<dyn std::error::Error>> {
        // Positional parameter: table name
        let table_name = bind.get_parameter(0).to_string();

        // Named parameters with defaults
        let timestamp_col = get_string_param(bind, "timestamp")
            .unwrap_or_else(|| "timestamp".to_string());

        let value_col = get_string_param(bind, "value")
            .unwrap_or_else(|| "value".to_string());

        // Parse group_by LIST parameter via the raw C API.
        let group_by = match bind.get_named_parameter("group_by") {
            Some(v) => extract_list_strings(&v)?,
            None => Vec::new(),
        };

        // Validate
        if table_name.is_empty() {
            return Err("table name cannot be empty".into());
        }

        Ok(SeasonalityParams {
            table_name,
            timestamp_col,
            value_col,
            group_by,
        })
    }
}
