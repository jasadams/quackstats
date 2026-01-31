#![allow(special_module_name)]

// To build the Wasm target, a staticlib crate-type is required. Since there is
// currently no way to select crate-type depending on the target, this file
// remaps the library contents as an example with crate-type = ["staticlib"].
// For WASM compilation: cargo build --target wasm32-unknown-emscripten --example quackstats
mod lib;

// Re-export modules and types so that `crate::common`, `crate::ConnHandle`, etc.
// resolve identically to the normal cdylib build (where lib.rs IS the crate root).
pub use lib::common;
pub use lib::seasonality;
pub use lib::ConnHandle;
