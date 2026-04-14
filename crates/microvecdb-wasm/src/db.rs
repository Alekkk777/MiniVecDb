use wasm_bindgen::prelude::*;
use js_sys::{Float32Array, Uint32Array, Uint8Array};

use microvecdb_core::{
    hnsw::{HnswIndex, SearchResult},
    quantize::{quantize_f32, DIMS},
    storage::VectorStore,
    DbError,
};

/// The main WASM-exposed database handle.
///
/// All insert/search operations are synchronous and run entirely in the
/// caller's WASM thread.  For use in a Web Worker, instantiate one
/// `WasmVecDb` per worker.
#[wasm_bindgen]
pub struct WasmVecDb {
    store: VectorStore,
    index: Option<HnswIndex>,
}

#[wasm_bindgen]
impl WasmVecDb {
    /// Create an empty database.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
        Self { store: VectorStore::new(), index: None }
    }

    /// Create a database pre-allocated for `capacity` vectors.
    ///
    /// Pre-allocation prevents `Vec` reallocation during bulk inserts and keeps
    /// zero-copy memory views stable.
    pub fn with_capacity(capacity: u32) -> Self {
        #[cfg(feature = "console_error_panic_hook")]
        console_error_panic_hook::set_once();
        Self { store: VectorStore::with_capacity(capacity as usize), index: None }
    }

    // ----- Insert ---------------------------------------------------------

    /// Insert one vector.
    ///
    /// `vector` must be a `Float32Array` of exactly 384 elements.
    /// Returns the internal slot index, or `u32::MAX` on error.
    pub fn insert(&mut self, doc_id: u32, vector: &Float32Array) -> u32 {
        if vector.length() != DIMS as u32 {
            return u32::MAX;
        }
        let floats = vector.to_vec();
        let bv = match quantize_f32(&floats) {
            Ok(v) => v,
            Err(_) => return u32::MAX,
        };
        match self.store.insert(doc_id, bv) {
            Ok(slot) => slot,
            Err(_) => u32::MAX,
        }
    }

    /// Batch insert.
    ///
    /// `doc_ids` must be a `Uint32Array` of N IDs.
    /// `vectors_flat` must be a `Float32Array` of N × 384 floats (row-major).
    /// Returns the number of successfully inserted vectors.
    pub fn insert_batch(&mut self, doc_ids: &Uint32Array, vectors_flat: &Float32Array) -> u32 {
        let n = doc_ids.length() as usize;
        if vectors_flat.length() as usize != n * DIMS {
            return 0;
        }
        let ids = doc_ids.to_vec();
        let all_floats = vectors_flat.to_vec();
        let mut count = 0u32;
        for i in 0..n {
            let chunk = &all_floats[i * DIMS..(i + 1) * DIMS];
            let bv = match quantize_f32(chunk) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if self.store.insert(ids[i], bv).is_ok() {
                count += 1;
            }
        }
        count
    }

    // ----- Search ---------------------------------------------------------

    /// Brute-force KNN scan (always available, no index needed).
    ///
    /// Returns an interleaved `Uint32Array`:
    /// `[doc_id_0, distance_0, doc_id_1, distance_1, ...]`
    pub fn search_scan(&self, query: &Float32Array, k: u32) -> Uint32Array {
        let results = self.scan_impl(query, k as usize);
        encode_results(&results)
    }

    /// HNSW approximate KNN search.
    ///
    /// Requires [`build_index`] to have been called first.
    /// Falls back to a brute-force scan if no index exists.
    /// `ef` is the exploration factor; higher → better recall, slower search.
    pub fn search_hnsw(&self, query: &Float32Array, k: u32, ef: u32) -> Uint32Array {
        let idx = match &self.index {
            Some(i) => i,
            None => return self.search_scan(query, k),
        };
        let floats = query.to_vec();
        let bv = match quantize_f32(&floats) {
            Ok(v) => v,
            Err(_) => return Uint32Array::new_with_length(0),
        };
        let results = idx.search(&self.store, &bv, k as usize, ef as usize);
        encode_results(&results)
    }

    // ----- Index ----------------------------------------------------------

    /// Build (or rebuild) the HNSW index from current store contents.
    ///
    /// Must be called before `search_hnsw`.  Safe to call multiple times;
    /// rebuilds from scratch each time.
    pub fn build_index(&mut self, m: u32, ef_construction: u32) {
        self.index = Some(HnswIndex::build(
            &self.store,
            m as usize,
            ef_construction as usize,
        ));
    }

    // ----- Mutation -------------------------------------------------------

    /// Soft-delete a vector by `doc_id`.
    ///
    /// Returns `true` if the vector was found and marked deleted.
    pub fn delete(&mut self, doc_id: u32) -> bool {
        let deleted = self.store.delete(doc_id);
        deleted
    }

    /// Remove all soft-deleted slots and compact the arrays.
    ///
    /// **Warning**: slot indices change. The HNSW index is cleared and must
    /// be rebuilt after compaction.
    pub fn compact(&mut self) -> u32 {
        let freed = self.store.compact();
        self.index = None; // index is now invalid
        freed as u32
    }

    // ----- Persistence ----------------------------------------------------

    /// Serialise the entire database (store + optional index) to a byte array.
    ///
    /// The result can be stored via OPFS and later passed to [`deserialize`].
    pub fn serialize(&self) -> Uint8Array {
        let mut buf = Vec::new();
        // Header byte: 0x00 = no index, 0x01 = has index
        let has_index = self.index.is_some() as u8;
        buf.push(has_index);

        let store_bytes = self.store.serialize();
        let store_len = store_bytes.len() as u32;
        buf.extend_from_slice(&store_len.to_le_bytes());
        buf.extend_from_slice(&store_bytes);

        if let Some(idx) = &self.index {
            let idx_bytes = idx.serialize();
            buf.extend_from_slice(&idx_bytes);
        }

        let out = Uint8Array::new_with_length(buf.len() as u32);
        out.copy_from(&buf);
        out
    }

    /// Deserialise a database from bytes produced by [`serialize`].
    ///
    /// Returns an error string on failure.
    pub fn deserialize(data: &Uint8Array) -> Result<WasmVecDb, JsValue> {
        let bytes = data.to_vec();
        if bytes.is_empty() {
            return Err(JsValue::from_str("empty buffer"));
        }
        let has_index = bytes[0] != 0;

        if bytes.len() < 5 {
            return Err(JsValue::from_str("buffer too short"));
        }
        let store_len = u32::from_le_bytes(bytes[1..5].try_into().unwrap()) as usize;
        if bytes.len() < 5 + store_len {
            return Err(JsValue::from_str("truncated store data"));
        }
        let store = VectorStore::deserialize(&bytes[5..5 + store_len])
            .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?;

        let index = if has_index {
            let idx_bytes = &bytes[5 + store_len..];
            Some(
                HnswIndex::deserialize(idx_bytes)
                    .map_err(|e| JsValue::from_str(&format!("{:?}", e)))?,
            )
        } else {
            None
        };

        Ok(WasmVecDb { store, index })
    }

    // ----- Zero-copy helpers ----------------------------------------------

    /// Raw pointer (byte offset) into WASM linear memory for the vector data.
    ///
    /// Use with `raw_vecs_len()` to create a zero-copy JS view:
    /// ```js
    /// const view = new Uint32Array(wasm.memory.buffer, db.raw_vecs_ptr(), db.raw_vecs_len());
    /// ```
    pub fn raw_vecs_ptr(&self) -> u32 {
        self.store.raw_vecs_ptr() as u32
    }

    /// Number of `u32` values in the raw vector data (`slot_count × 12`).
    pub fn raw_vecs_len(&self) -> u32 {
        self.store.raw_vecs_len() as u32
    }

    // ----- Stats ----------------------------------------------------------

    /// Number of slots (including soft-deleted ones).
    pub fn len(&self) -> u32 { self.store.len() as u32 }

    /// `true` if the store is empty.
    pub fn is_empty(&self) -> bool { self.store.is_empty() }

    /// `true` if an HNSW index has been built.
    pub fn has_index(&self) -> bool { self.index.is_some() }
}

// ----- Helpers -----------------------------------------------------------

fn encode_results(results: &[SearchResult]) -> Uint32Array {
    let out = Uint32Array::new_with_length(results.len() as u32 * 2);
    for (i, r) in results.iter().enumerate() {
        out.set_index(i as u32 * 2,     r.doc_id);
        out.set_index(i as u32 * 2 + 1, r.distance);
    }
    out
}

impl WasmVecDb {
    fn scan_impl(&self, query: &Float32Array, k: usize) -> Vec<SearchResult> {
        let floats = query.to_vec();
        let bv = match quantize_f32(&floats) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        self.store.scan_knn(&bv, k)
    }
}
