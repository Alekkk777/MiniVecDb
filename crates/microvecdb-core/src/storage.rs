use alloc::vec::Vec;

use crate::{
    distance::{hamming_distance, similarity_score},
    error::DbError,
    quantize::{BinaryVec, WORDS},
    hnsw::SearchResult,
};

// ----- Bit-packed metadata -----------------------------------------------

/// Packs a `doc_id` and an `is_deleted` flag into a single `u32`.
///
/// Layout:
/// - Bit  31   : `1` = slot is soft-deleted, `0` = active.
/// - Bits 0-30 : document ID (max ≈ 2.1 billion).
#[derive(Clone, Copy, Default)]
#[repr(transparent)]
pub struct PackedMeta(pub u32);

const DELETED_BIT: u32 = 1 << 31;
const ID_MASK: u32 = !DELETED_BIT; // 0x7FFF_FFFF

impl PackedMeta {
    /// Maximum allowed doc ID (bit 31 is reserved).
    pub const MAX_DOC_ID: u32 = ID_MASK;

    #[inline]
    pub fn new(doc_id: u32, deleted: bool) -> Self {
        debug_assert!(doc_id <= Self::MAX_DOC_ID);
        Self((doc_id & ID_MASK) | if deleted { DELETED_BIT } else { 0 })
    }

    #[inline]
    pub fn doc_id(self) -> u32 { self.0 & ID_MASK }

    #[inline]
    pub fn is_deleted(self) -> bool { (self.0 & DELETED_BIT) != 0 }

    #[inline]
    pub fn mark_deleted(&mut self) { self.0 |= DELETED_BIT; }
}

// ----- VectorStore -------------------------------------------------------

/// Structure-of-Arrays vector store.
///
/// Two flat, cache-friendly arrays are kept in sync:
/// - `binary_vecs`: the HOT array accessed in every search iteration.
/// - `meta`: the COLD array read only when a result is promoted.
///
/// During a brute-force KNN scan the CPU streams through `binary_vecs`
/// uninterrupted (no metadata bytes to evict from L1 cache).
pub struct VectorStore {
    /// Flat SoA array: slot `i` occupies `binary_vecs[i]`.
    binary_vecs: Vec<BinaryVec>,
    /// Parallel metadata array: same length as `binary_vecs`.
    meta: Vec<PackedMeta>,
}

impl VectorStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self { binary_vecs: Vec::new(), meta: Vec::new() }
    }

    /// Pre-allocate space for `capacity` vectors to avoid reallocation.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            binary_vecs: Vec::with_capacity(capacity),
            meta: Vec::with_capacity(capacity),
        }
    }

    /// Number of slots (including soft-deleted ones).
    #[inline]
    pub fn len(&self) -> usize { self.binary_vecs.len() }

    /// `true` if no vectors have been inserted yet.
    #[inline]
    pub fn is_empty(&self) -> bool { self.binary_vecs.is_empty() }

    /// Insert a quantized vector and return its internal slot index.
    ///
    /// # Errors
    /// - [`DbError::DocIdTooLarge`] if `doc_id` has bit 31 set.
    pub fn insert(&mut self, doc_id: u32, vec: BinaryVec) -> Result<u32, DbError> {
        if doc_id > PackedMeta::MAX_DOC_ID {
            return Err(DbError::DocIdTooLarge);
        }
        let slot = self.binary_vecs.len() as u32;
        self.binary_vecs.push(vec);
        self.meta.push(PackedMeta::new(doc_id, false));
        Ok(slot)
    }

    /// Soft-delete the slot with the given `doc_id`.
    ///
    /// Returns `true` if the slot was found and marked deleted.
    /// Deleted slots are excluded from all scans but remain in memory until
    /// [`compact`](Self::compact) is called.
    pub fn delete(&mut self, doc_id: u32) -> bool {
        for m in &mut self.meta {
            if m.doc_id() == doc_id && !m.is_deleted() {
                m.mark_deleted();
                return true;
            }
        }
        false
    }

    /// Return a reference to the quantized vector at `slot`.
    #[inline]
    pub fn get_vec(&self, slot: u32) -> Option<&BinaryVec> {
        self.binary_vecs.get(slot as usize)
    }

    /// Return the metadata at `slot`.
    #[inline]
    pub fn get_meta(&self, slot: u32) -> Option<PackedMeta> {
        self.meta.get(slot as usize).copied()
    }

    /// Return `true` if `slot` is active (not deleted).
    #[inline]
    pub fn is_active(&self, slot: u32) -> bool {
        self.meta
            .get(slot as usize)
            .map(|m| !m.is_deleted())
            .unwrap_or(false)
    }

    /// Brute-force KNN scan.
    ///
    /// Iterates **all** active slots, computes Hamming distance to `query`,
    /// and returns the top-`k` results ordered by ascending distance.
    pub fn scan_knn(&self, query: &BinaryVec, k: usize) -> Vec<SearchResult> {
        if k == 0 || self.binary_vecs.is_empty() {
            return Vec::new();
        }

        // We maintain a bounded max-heap of size k (furthest result at the top).
        // When the heap exceeds k we drop the furthest.
        let mut heap: Vec<SearchResult> = Vec::with_capacity(k + 1);

        for (slot, (vec, meta)) in
            self.binary_vecs.iter().zip(self.meta.iter()).enumerate()
        {
            if meta.is_deleted() {
                continue;
            }
            let dist = hamming_distance(query, vec);
            let result = SearchResult {
                doc_id: meta.doc_id(),
                slot: slot as u32,
                distance: dist,
                score: similarity_score(dist),
            };

            if heap.len() < k {
                heap.push(result);
                // Bubble up to maintain max-heap by distance
                sift_up(&mut heap);
            } else if let Some(worst) = heap.first() {
                if dist < worst.distance {
                    heap[0] = result;
                    sift_down(&mut heap);
                }
            }
        }

        // Sort ascending (closest first)
        heap.sort_unstable_by_key(|r| r.distance);
        heap
    }

    // ----- Serialization -------------------------------------------------

    /// Serialise the store to a byte vector for OPFS persistence.
    ///
    /// Format (little-endian):
    /// ```text
    /// [0..4]            magic   = b"MVDB"
    /// [4..8]            version = 1u32
    /// [8..12]           count   = N (number of slots)
    /// [12..12+N*4]      meta    = N × PackedMeta (u32 each)
    /// [12+N*4 ..]       vecs    = N × BinaryVec  (12 × u32 = 48 bytes each)
    /// ```
    pub fn serialize(&self) -> Vec<u8> {
        let n = self.binary_vecs.len();
        let mut buf = Vec::with_capacity(12 + n * 4 + n * WORDS * 4);

        buf.extend_from_slice(b"MVDB");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(n as u32).to_le_bytes());

        for m in &self.meta {
            buf.extend_from_slice(&m.0.to_le_bytes());
        }
        for v in &self.binary_vecs {
            for &word in v.iter() {
                buf.extend_from_slice(&word.to_le_bytes());
            }
        }
        buf
    }

    /// Deserialise a store from bytes produced by [`serialize`](Self::serialize).
    ///
    /// # Errors
    /// Returns [`DbError::CorruptedData`] if the magic, version, or lengths
    /// are invalid.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, DbError> {
        if bytes.len() < 12 {
            return Err(DbError::CorruptedData);
        }
        if &bytes[0..4] != b"MVDB" {
            return Err(DbError::CorruptedData);
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 1 {
            return Err(DbError::CorruptedData);
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

        let meta_start = 12;
        let meta_end = meta_start + n * 4;
        let vecs_end = meta_end + n * WORDS * 4;

        if bytes.len() < vecs_end {
            return Err(DbError::CorruptedData);
        }

        let mut meta = Vec::with_capacity(n);
        for chunk in bytes[meta_start..meta_end].chunks_exact(4) {
            meta.push(PackedMeta(u32::from_le_bytes(chunk.try_into().unwrap())));
        }

        let mut binary_vecs = Vec::with_capacity(n);
        for chunk in bytes[meta_end..vecs_end].chunks_exact(WORDS * 4) {
            let mut bv = [0u32; WORDS];
            for (i, word_bytes) in chunk.chunks_exact(4).enumerate() {
                bv[i] = u32::from_le_bytes(word_bytes.try_into().unwrap());
            }
            binary_vecs.push(bv);
        }

        Ok(Self { binary_vecs, meta })
    }

    /// Remove all soft-deleted slots, compacting the arrays in place.
    ///
    /// Returns the number of slots freed.
    /// **Warning**: slot indices change after compaction — rebuild any HNSW
    /// index afterward.
    pub fn compact(&mut self) -> usize {
        let before = self.binary_vecs.len();
        let mut i = 0;
        while i < self.binary_vecs.len() {
            if self.meta[i].is_deleted() {
                self.binary_vecs.swap_remove(i);
                self.meta.swap_remove(i);
            } else {
                i += 1;
            }
        }
        before - self.binary_vecs.len()
    }

    // ----- Zero-copy WASM helpers ----------------------------------------

    /// Raw pointer to the start of the binary vector data.
    ///
    /// In WASM, JS can create a zero-copy view via:
    /// ```js
    /// new Uint32Array(wasm.memory.buffer, ptr, len)
    /// ```
    /// **Invalidated** if the Vec reallocates (i.e., after an insert that
    /// exceeds capacity). Use `with_capacity` to prevent reallocation.
    #[inline]
    pub fn raw_vecs_ptr(&self) -> *const u32 {
        self.binary_vecs.as_ptr() as *const u32
    }

    /// Number of `u32` values in the raw vector data (`slots × WORDS`).
    #[inline]
    pub fn raw_vecs_len(&self) -> usize {
        self.binary_vecs.len() * WORDS
    }
}

impl Default for VectorStore {
    fn default() -> Self { Self::new() }
}

// ----- Binary-heap helpers (no-std, descending by distance) ---------------

fn sift_up(heap: &mut Vec<SearchResult>) {
    let mut i = heap.len() - 1;
    while i > 0 {
        let parent = (i - 1) / 2;
        if heap[i].distance > heap[parent].distance {
            heap.swap(i, parent);
            i = parent;
        } else {
            break;
        }
    }
}

fn sift_down(heap: &mut Vec<SearchResult>) {
    let len = heap.len();
    let mut i = 0;
    loop {
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        let mut largest = i;
        if left < len && heap[left].distance > heap[largest].distance {
            largest = left;
        }
        if right < len && heap[right].distance > heap[largest].distance {
            largest = right;
        }
        if largest == i {
            break;
        }
        heap.swap(i, largest);
        i = largest;
    }
}

// ----- Tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantize::quantize_f32;

    fn make_vec(fill: f32) -> BinaryVec {
        quantize_f32(&[fill; 384]).unwrap()
    }

    #[test]
    fn insert_and_retrieve() {
        let mut store = VectorStore::new();
        let v = make_vec(1.0);
        let slot = store.insert(42, v).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(store.get_vec(0), Some(&v));
        assert_eq!(store.get_meta(0).unwrap().doc_id(), 42);
        assert!(!store.get_meta(0).unwrap().is_deleted());
    }

    #[test]
    fn soft_delete_excludes_from_scan() {
        let mut store = VectorStore::new();
        let v = make_vec(1.0);
        store.insert(1, v).unwrap();
        store.insert(2, v).unwrap();
        store.delete(1);
        let results = store.scan_knn(&v, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 2);
    }

    #[test]
    fn scan_returns_closest_first() {
        let mut store = VectorStore::new();
        // doc 1: all-ones (closest to query)
        let close = quantize_f32(&[1.0_f32; 384]).unwrap();
        // doc 2: all-zeros (furthest from all-ones query)
        let far = quantize_f32(&[-1.0_f32; 384]).unwrap();
        store.insert(1, close).unwrap();
        store.insert(2, far).unwrap();
        let query = close; // query is all-ones
        let results = store.scan_knn(&query, 2);
        assert_eq!(results[0].doc_id, 1); // closest first
        assert_eq!(results[0].distance, 0);
        assert_eq!(results[1].doc_id, 2);
        assert_eq!(results[1].distance, 384);
    }

    #[test]
    fn doc_id_too_large_is_rejected() {
        let mut store = VectorStore::new();
        let v = make_vec(1.0);
        assert!(store.insert(PackedMeta::MAX_DOC_ID + 1, v).is_err());
    }

    #[test]
    fn compact_removes_deleted_slots() {
        let mut store = VectorStore::new();
        for i in 0..10u32 {
            store.insert(i, make_vec(1.0)).unwrap();
        }
        store.delete(3);
        store.delete(7);
        let freed = store.compact();
        assert_eq!(freed, 2);
        assert_eq!(store.len(), 8);
    }

    #[test]
    fn serialization_round_trip() {
        let mut store = VectorStore::new();
        for i in 0..5u32 {
            let v = make_vec(if i % 2 == 0 { 1.0 } else { -1.0 });
            store.insert(i * 100, v).unwrap();
        }
        let bytes = store.serialize();
        let restored = VectorStore::deserialize(&bytes).unwrap();
        assert_eq!(store.len(), restored.len());
        for slot in 0..store.len() as u32 {
            assert_eq!(store.get_vec(slot), restored.get_vec(slot));
            assert_eq!(
                store.get_meta(slot).unwrap().doc_id(),
                restored.get_meta(slot).unwrap().doc_id()
            );
        }
    }
}
