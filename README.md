# QuackStats

Time series forecasting and statistics as a DuckDB extension.

## Status

Phase 1 - scaffold and proof-of-concept. The `forecast()` table function is registered but returns hardcoded data. Real forecasting logic will be added in subsequent phases.

## Usage

```sql
SELECT * FROM forecast('my_table', timestamp := 'ds', value := 'revenue', horizon := 3);
```

## Building

```bash
make configure
make debug
```

## Testing

```bash
make test_debug
```

## License

Dual-licensed under MIT ([LICENSE-MIT](LICENSE-MIT)) or Apache 2.0 ([LICENSE-APACHE](LICENSE-APACHE)), at your option.
