use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use microvecdb_core::{
    quantize::quantize_f32,
    storage::VectorStore,
    time::now_ms,
    HnswIndex,
};

const DIMS: usize = 384;

// ----- Python class ----------------------------------------------------------

/// A 1-bit quantised vector database.
///
/// Vectors must be 384-dimensional float arrays.  Internally they are quantised
/// to 384 bits (12 × u32) via sign-binarisation and compared with Hamming
/// distance, giving ~96 % RAM reduction versus raw f32.
///
/// Example::
///
///     import minivecdb
///     db = minivecdb.MiniVecDb()
///     db.insert(0, [0.1] * 384)
///     results = db.search([0.1] * 384, k=5)
#[pyclass(name = "MiniVecDb")]
pub struct PyMiniVecDb {
    store: VectorStore,
    index: Option<HnswIndex>,
}

#[pymethods]
impl PyMiniVecDb {
    /// Create an empty database.
    #[new]
    fn new() -> Self {
        Self {
            store: VectorStore::new(),
            index: None,
        }
    }

    /// Create a database pre-allocated for `capacity` vectors.
    ///
    /// Pre-allocation keeps memory addresses stable after bulk inserts, and
    /// enables tombstone recycling once the store is full.
    #[staticmethod]
    fn with_capacity(capacity: usize) -> Self {
        Self {
            store: VectorStore::with_capacity(capacity),
            index: None,
        }
    }

    /// Insert a single 384-dim float vector with the given integer `id`.
    ///
    /// The vector is quantised to 384 bits in-place.
    /// Raises ``ValueError`` if the vector is not 384-dimensional.
    /// Raises ``RuntimeError`` if the pre-allocated capacity is full with no
    /// tombstone slots available for recycling.
    fn insert(&mut self, id: u32, vector: Vec<f32>) -> PyResult<u32> {
        if vector.len() != DIMS {
            return Err(PyValueError::new_err(format!(
                "expected {DIMS}-dim vector, got {}",
                vector.len()
            )));
        }
        let bv = quantize_f32(&vector).map_err(|e| {
            PyValueError::new_err(format!("quantization error: {e:?}"))
        })?;
        let slot = self.store.insert(id, bv, now_ms()).map_err(|e| {
            PyRuntimeError::new_err(format!("insert error: {e:?}"))
        })?;
        // Any new insert invalidates the HNSW index.
        self.index = None;
        Ok(slot)
    }

    /// Soft-delete the vector with the given ``id``.
    ///
    /// Returns ``True`` if the vector was found and tombstoned.
    fn delete(&mut self, id: u32) -> bool {
        let deleted = self.store.delete(id);
        if deleted {
            self.index = None;
        }
        deleted
    }

    /// Number of slots (including tombstones).
    fn __len__(&self) -> usize {
        self.store.len()
    }

    /// ``True`` if the store has no slots at all.
    fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Build (or rebuild) the HNSW approximate nearest-neighbour index.
    ///
    /// Call this after a bulk ``insert`` sequence for faster subsequent searches.
    ///
    /// :param m: Max connections per node per layer (default 16).
    /// :param ef_construction: Exploration factor during build (default 200).
    #[pyo3(signature = (m=16, ef_construction=200))]
    fn build_index(&mut self, m: usize, ef_construction: usize) {
        self.index = Some(HnswIndex::build(&self.store, m, ef_construction));
    }

    /// ``True`` if an HNSW index has been built.
    fn has_index(&self) -> bool {
        self.index.is_some()
    }

    /// Search for the ``k`` nearest neighbours of ``query``.
    ///
    /// Uses the HNSW index when available, falls back to brute-force scan
    /// otherwise.
    ///
    /// :param query: 384-dim float list/array.
    /// :param k: Number of results.
    /// :param ef: HNSW exploration factor (ignored for brute-force). Default 50.
    /// :returns: List of ``(doc_id, score)`` tuples, score ∈ [0, 1], descending.
    #[pyo3(signature = (query, k, ef=50))]
    fn search(
        &self,
        query: Vec<f32>,
        k: usize,
        ef: usize,
    ) -> PyResult<Vec<(u32, f32)>> {
        if query.len() != DIMS {
            return Err(PyValueError::new_err(format!(
                "expected {DIMS}-dim query, got {}",
                query.len()
            )));
        }
        let bq = quantize_f32(&query).map_err(|e| {
            PyValueError::new_err(format!("quantization error: {e:?}"))
        })?;

        let results = if let Some(idx) = &self.index {
            idx.search(&self.store, &bq, k, ef)
        } else {
            self.store.scan_knn(&bq, k)
        };

        Ok(results
            .into_iter()
            .map(|r| (r.doc_id, r.score))
            .collect())
    }

    /// Brute-force KNN scan (always available, no index required).
    ///
    /// :returns: List of ``(doc_id, score)`` tuples, score ∈ [0, 1], descending.
    fn search_scan(&self, query: Vec<f32>, k: usize) -> PyResult<Vec<(u32, f32)>> {
        if query.len() != DIMS {
            return Err(PyValueError::new_err(format!(
                "expected {DIMS}-dim query, got {}",
                query.len()
            )));
        }
        let bq = quantize_f32(&query).map_err(|e| {
            PyValueError::new_err(format!("quantization error: {e:?}"))
        })?;
        Ok(self
            .store
            .scan_knn(&bq, k)
            .into_iter()
            .map(|r| (r.doc_id, r.score))
            .collect())
    }

    /// Tombstone all vectors older than ``ttl_ms`` milliseconds.
    ///
    /// Uses the system clock (``std::time::SystemTime``) internally.
    ///
    /// :returns: Number of vectors tombstoned.
    fn run_gc(&mut self, ttl_ms: f64) -> usize {
        let n = self.store.run_gc(now_ms(), ttl_ms);
        if n > 0 {
            self.index = None;
        }
        n
    }

    /// Remove all tombstone slots and compact the backing arrays.
    ///
    /// **Warning**: slot indices change after compaction — the HNSW index is
    /// cleared and must be rebuilt.
    ///
    /// :returns: Number of slots freed.
    fn compact(&mut self) -> usize {
        let freed = self.store.compact();
        if freed > 0 {
            self.index = None;
        }
        freed
    }

    /// Serialise the database to ``bytes`` for persistence.
    ///
    /// The result can be written to disk and later restored with
    /// :meth:`deserialize`.
    fn serialize<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.store.serialize())
    }

    /// Restore a database from bytes produced by :meth:`serialize`.
    ///
    /// Raises ``ValueError`` on malformed input.
    #[staticmethod]
    fn deserialize(data: &[u8]) -> PyResult<Self> {
        let store = VectorStore::deserialize(data).map_err(|e| {
            PyValueError::new_err(format!("deserialize error: {e:?}"))
        })?;
        Ok(Self { store, index: None })
    }
}

// ----- Module ----------------------------------------------------------------

/// MiniVecDb — 1-bit quantised vector database compiled from Rust.
///
/// Loaded as `minivecdb._minivecdb` so the top-level `minivecdb` package
/// can be a regular Python package with `__init__.py` and `langchain.py`
/// alongside this native extension.
#[pymodule]
#[pyo3(name = "_minivecdb")]
fn minivecdb_ext(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMiniVecDb>()?;
    Ok(())
}
