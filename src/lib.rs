mod common;
mod forecast;
mod seasonality;

use duckdb::{Connection, Result};
use std::error::Error;
use std::ffi::CString;

use forecast::ForecastVTab;
use seasonality::SeasonalityVTab;

/// Connection handle wrapper for executing queries during VTab execution.
/// Stores a persistent duckdb_connection created at extension init time.
///
/// # Lifetime
/// The connection is created once during `quackstats_init_c_api_internal` and
/// lives for the lifetime of the database. It is intentionally never freed via
/// `duckdb_disconnect` because the database itself will clean up all connections
/// on shutdown. This avoids the need for an extension unload hook.
///
/// # Safety
/// - `Send`/`Sync`: DuckDB connections are thread-safe for independent queries.
///   Each VTab invocation executes a single read-only SELECT; concurrent VTab
///   calls on different threads each get their own query state via `duckdb_query`.
/// - `Clone`: Creates an alias to the same underlying connection pointer. This is
///   required by `register_table_function_with_extra_info` which clones the extra
///   info. No `Drop` is implemented, so aliasing is safe (no double-free risk).
pub struct ConnHandle(pub libduckdb_sys::duckdb_connection);
unsafe impl Send for ConnHandle {}
unsafe impl Sync for ConnHandle {}
impl Clone for ConnHandle {
    fn clone(&self) -> Self {
        ConnHandle(self.0)
    }
}

// We implement the entrypoint manually instead of using #[duckdb_entrypoint_c_api()]
// because we need access to the raw duckdb_database handle before it's wrapped
// in a Connection. The macro wraps it immediately, and the Connection struct
// doesn't expose a way to get it back.

/// Internal entrypoint for error handling.
///
/// # Safety
/// Called by DuckDB during extension loading.
pub unsafe fn quackstats_init_c_api_internal(
    info: libduckdb_sys::duckdb_extension_info,
    access: *const libduckdb_sys::duckdb_extension_access,
) -> std::result::Result<bool, Box<dyn Error>> {
    let have_api_struct =
        libduckdb_sys::duckdb_rs_extension_api_init(info, access, "v1.2.0").unwrap();

    if !have_api_struct {
        return Ok(false);
    }

    // Get the raw database handle BEFORE creating the Connection
    let db: libduckdb_sys::duckdb_database = *(*access).get_database.unwrap()(info);

    // Create a connection for registration (same as the macro does)
    let connection = Connection::open_from_raw(db.cast())?;

    // Create a persistent connection for use during VTab execution.
    // This connection is created at init time (when duckdb_connect works)
    // and stored as extra info for the table function.
    let mut query_con: libduckdb_sys::duckdb_connection = std::ptr::null_mut();
    let rc = libduckdb_sys::duckdb_connect(db, &mut query_con);
    if rc != libduckdb_sys::duckdb_state_DuckDBSuccess {
        return Err("Failed to create query connection for table functions".into());
    }

    // Register all table functions with the shared query connection
    register_forecast(&connection, query_con)?;
    register_seasonality(&connection, query_con)?;

    Ok(true)
}

/// # Safety
/// Entrypoint called by DuckDB when loading the extension.
#[no_mangle]
pub unsafe extern "C" fn quackstats_init_c_api(
    info: libduckdb_sys::duckdb_extension_info,
    access: *const libduckdb_sys::duckdb_extension_access,
) -> bool {
    let init_result = quackstats_init_c_api_internal(info, access);

    if let Err(x) = init_result {
        let error_c_string = CString::new(x.to_string());

        match error_c_string {
            Ok(e) => {
                (*access).set_error.unwrap()(info, e.as_ptr());
            }
            Err(_e) => {
                let error_alloc_failure = c"An error occurred but the extension failed to allocate memory for an error string";
                (*access).set_error.unwrap()(info, error_alloc_failure.as_ptr());
            }
        }
        return false;
    }

    init_result.unwrap()
}

/// Register the forecast table function with a query connection as extra info.
fn register_forecast(
    con: &Connection,
    query_con: libduckdb_sys::duckdb_connection,
) -> Result<(), Box<dyn Error>> {
    con.register_table_function_with_extra_info::<ForecastVTab, _>(
        "forecast",
        &ConnHandle(query_con),
    )?;
    Ok(())
}

/// Register the detect_seasonality table function with a query connection as extra info.
fn register_seasonality(
    con: &Connection,
    query_con: libduckdb_sys::duckdb_connection,
) -> Result<(), Box<dyn Error>> {
    con.register_table_function_with_extra_info::<SeasonalityVTab, _>(
        "detect_seasonality",
        &ConnHandle(query_con),
    )?;
    Ok(())
}
