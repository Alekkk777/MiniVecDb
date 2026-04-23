/// Current Unix time in milliseconds.
///
/// On wasm32 the caller (microvecdb-wasm) uses `js_sys::Date::now()` directly.
/// This function is only compiled for native targets (Python binding, CLI, tests).
#[cfg(not(target_arch = "wasm32"))]
pub fn now_ms() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as f64)
        .unwrap_or(0.0)
}
