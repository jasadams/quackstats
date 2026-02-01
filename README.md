# QuackStats

Time series forecasting and statistics as a DuckDB extension. Run forecasts, detect seasonality, and perform statistical analysis directly in SQL.

## Features

- **Multiple forecasting models** - ETS (Error-Trend-Seasonality), linear regression, exponential growth, logistic (S-curve), and automatic model selection via cross-validation
- **Prediction intervals** - Configurable confidence bounds on all forecasts (default 95%)
- **Seasonality detection** - Identify periodic patterns in your time series data
- **Multi-group support** - Forecast multiple series independently in a single query using `group_by`
- **Automatic interval detection** - Handles daily, weekly, and monthly timestamp spacing

## Installation

```sql
INSTALL quackstats FROM community;
LOAD quackstats;
```

## Usage

### Forecasting

```sql
-- Basic forecast
SELECT * FROM forecast(
    'sales',
    timestamp := 'date',
    value := 'revenue',
    horizon := 30
);

-- Grouped forecast with model selection
SELECT * FROM forecast(
    'sales',
    timestamp := 'date',
    value := 'revenue',
    horizon := 30,
    group_by := ['region', 'product'],
    model := 'auto',
    confidence_level := 0.95
);
```

**Output columns:** group columns (if specified), `forecast_timestamp`, `forecast`, `lower_bound`, `upper_bound`

**Available models:**

| Model | Description |
|-------|-------------|
| `ets` | Error-Trend-Seasonality (default) |
| `linear` | Linear regression with prediction intervals |
| `exponential` | Exponential curve fitting (y = ae^bx) |
| `logistic` | S-curve fitting for bounded growth |
| `auto` | Cross-validation across all models, picks the best |

### Seasonality Detection

```sql
SELECT * FROM detect_seasonality(
    'sales',
    timestamp := 'date',
    value := 'revenue',
    group_by := ['region']
);
```

**Output columns:** group columns (if specified), `period`, `strength`

## Building from Source

```bash
make configure
make debug
```

## Testing

```bash
make test_debug
```

## License

MIT © 2026 Alytic Pty Ltd

QuackStats is maintained by the team at [Kyomi](https://kyomi.ai) and is the forecasting engine that powers the platform.
