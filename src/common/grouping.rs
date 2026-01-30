use super::types::TimeSeries;

/// A time series associated with a specific group key.
///
/// For example, with `group_by := ['region', 'product']`, one GroupedTimeSeries
/// might have `group_columns = ["region", "product"]` and
/// `group_key = ["US", "Widget"]`.
pub struct GroupedTimeSeries {
    /// The values for each group column, e.g. ["US", "Widget"].
    pub group_key: Vec<String>,
    /// The column names for each group, e.g. ["region", "product"].
    pub group_columns: Vec<String>,
    /// The time series data for this group.
    pub series: TimeSeries,
}

/// A single output row from a grouped forecast.
///
/// This flat struct makes it easy to emit rows across multiple DataChunk calls
/// without re-computing anything.
pub struct GroupForecastRow {
    /// Values for each group column (same order as group_columns).
    pub group_values: Vec<String>,
    /// Forecast timestamp (DuckDB DATE: days since epoch).
    pub timestamp: i32,
    /// Point forecast value.
    pub forecast: f64,
    /// Lower prediction interval bound.
    pub lower_bound: f64,
    /// Upper prediction interval bound.
    pub upper_bound: f64,
}
