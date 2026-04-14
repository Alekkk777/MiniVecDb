/// Zero-copy helpers for exposing Rust memory to JavaScript.
///
/// ## Pattern
///
/// After calling `raw_vecs_ptr()` and `raw_vecs_len()` on a [`WasmVecDb`],
/// JavaScript can create a live view into WASM linear memory:
///
/// ```js
/// const ptr = db.raw_vecs_ptr();
/// const len = db.raw_vecs_len();
/// // Zero-copy Uint32Array view — no data is copied.
/// const view = new Uint32Array(wasm.__wbindgen_export_0.buffer, ptr, len);
/// ```
///
/// **Important**: This view is invalidated whenever the internal `Vec` reallocates
/// (i.e., after an `insert` that exceeds the pre-allocated capacity).
/// Call `with_capacity(n)` upfront to prevent reallocation during a bulk load.
pub struct ZeroCopyInfo {
    pub ptr: u32,
    pub len: u32,
}
