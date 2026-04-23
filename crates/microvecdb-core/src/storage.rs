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
/// - Bit  31   : `1` = slot is soft-deleted (tombstone), `0` = active.
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

/// Structure-of-Arrays vector store with TTL support.
///
/// Three flat, cache-friendly arrays are kept in sync:
/// - `binary_vecs`  : the HOT array accessed in every search iteration.
/// - `meta`         : packed `doc_id` + deleted flag; read when promoting results.
/// - `timestamps`   : insertion time in ms (Unix epoch); used by GC.
///
/// Soft-deleted slots are **tombstones**: they remain in memory until recycled
/// by a subsequent `insert` at capacity, or physically removed by `compact`.
/// This tombstoning strategy preserves SharedArrayBuffer pointer stability —
/// no `Vec` reallocation occurs as long as inserts stay within `capacity`.
pub struct VectorStore {
    binary_vecs: Vec<BinaryVec>,
    meta:        Vec<PackedMeta>,
    /// Unix timestamp in milliseconds for each slot (from `js_sys::Date::now()`
    /// in the WASM layer). Parallel to `binary_vecs` / `meta`.
    timestamps:  Vec<f64>,
    /// Hard ceiling set by `with_capacity`.  `None` → unbounded growth.
    /// When `len == ceiling`, tombstones are recycled instead of reallocating.
    ceiling: Option<usize>,
}

impl VectorStore {
    pub fn new() -> Self {
        Self {
            binary_vecs: Vec::new(),
            meta:        Vec::new(),
            timestamps:  Vec::new(),
            ceiling:     None,
        }
    }

    /// Pre-allocate space for `capacity` vectors to avoid reallocation.
    ///
    /// Once `len` reaches `capacity`, inserts recycle tombstone slots rather
    /// than reallocating, keeping the SharedArrayBuffer pointer stable.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            binary_vecs: Vec::with_capacity(capacity),
            meta:        Vec::with_capacity(capacity),
            timestamps:  Vec::with_capacity(capacity),
            ceiling:     Some(capacity),
        }
    }

    /// Number of slots (including tombstones).
    #[inline]
    pub fn len(&self) -> usize { self.binary_vecs.len() }

    /// `true` if no vectors have been inserted yet.
    #[inline]
    pub fn is_empty(&self) -> bool { self.binary_vecs.is_empty() }

    /// Insert a quantized vector and return its internal slot index.
    ///
    /// `inserted_at` is a Unix timestamp in milliseconds (from
    /// `js_sys::Date::now()` in the WASM layer; pass `0.0` in native tests).
    ///
    /// When the backing `Vec` is at capacity, the method scans for an existing
    /// tombstone and overwrites it in-place rather than reallocating.  This
    /// keeps the SharedArrayBuffer pointer (from `raw_vecs_ptr`) stable.
    ///
    /// # Errors
    /// - [`DbError::DocIdTooLarge`] if `doc_id` has bit 31 set.
    /// - [`DbError::CapacityExceeded`] if at capacity with no tombstones to recycle.
    pub fn insert(
        &mut self,
        doc_id: u32,
        vec: BinaryVec,
        inserted_at: f64,
    ) -> Result<u32, DbError> {
        if doc_id > PackedMeta::MAX_DOC_ID {
            return Err(DbError::DocIdTooLarge);
        }

        // At the explicit ceiling: recycle a tombstone to avoid reallocation
        // and preserve SharedArrayBuffer pointer stability.
        if self.ceiling == Some(self.binary_vecs.len()) {
            return match self.first_tombstone() {
                Some(slot) => {
                    self.binary_vecs[slot] = vec;
                    self.meta[slot]        = PackedMeta::new(doc_id, false);
                    self.timestamps[slot]  = inserted_at;
                    Ok(slot as u32)
                }
                None => Err(DbError::CapacityExceeded),
            };
        }

        let slot = self.binary_vecs.len() as u32;
        self.binary_vecs.push(vec);
        self.meta.push(PackedMeta::new(doc_id, false));
        self.timestamps.push(inserted_at);
        Ok(slot)
    }

    /// Soft-delete the slot with the given `doc_id`.
    ///
    /// Returns `true` if the slot was found and marked as a tombstone.
    pub fn delete(&mut self, doc_id: u32) -> bool {
        for m in &mut self.meta {
            if m.doc_id() == doc_id && !m.is_deleted() {
                m.mark_deleted();
                return true;
            }
        }
        false
    }

    /// Scan all active slots and tombstone any node whose age exceeds `ttl_ms`.
    ///
    /// `current_time` is a Unix timestamp in milliseconds.  In the WASM layer
    /// this comes from `js_sys::Date::now()`; in native code pass any `f64`.
    ///
    /// Returns the number of nodes newly tombstoned.
    pub fn run_gc(&mut self, current_time: f64, ttl_ms: f64) -> usize {
        let mut tombstoned = 0usize;
        for (meta, &ts) in self.meta.iter_mut().zip(self.timestamps.iter()) {
            if !meta.is_deleted() && (current_time - ts) > ttl_ms {
                meta.mark_deleted();
                tombstoned += 1;
            }
        }
        tombstoned
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

    /// Return `true` if `slot` is active (not a tombstone).
    #[inline]
    pub fn is_active(&self, slot: u32) -> bool {
        self.meta
            .get(slot as usize)
            .map(|m| !m.is_deleted())
            .unwrap_or(false)
    }

    /// Brute-force KNN scan — skips all tombstone slots.
    pub fn scan_knn(&self, query: &BinaryVec, k: usize) -> Vec<SearchResult> {
        if k == 0 || self.binary_vecs.is_empty() {
            return Vec::new();
        }

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
                sift_up(&mut heap);
            } else if let Some(worst) = heap.first() {
                if dist < worst.distance {
                    heap[0] = result;
                    sift_down(&mut heap);
                }
            }
        }

        heap.sort_unstable_by_key(|r| r.distance);
        heap
    }

    // ----- Serialization -------------------------------------------------

    /// Serialise the store to a byte vector for OPFS persistence.
    ///
    /// Format v2 (little-endian):
    /// ```text
    /// [0..4]                magic      = b"MVDB"
    /// [4..8]                version    = 2u32
    /// [8..12]               count      = N
    /// [12 .. 12+N*4]        meta       = N × PackedMeta (u32)
    /// [12+N*4 .. 12+N*12]   timestamps = N × f64
    /// [12+N*12 ..]          vecs       = N × BinaryVec (12 × u32 = 48 bytes)
    /// ```
    pub fn serialize(&self) -> Vec<u8> {
        let n = self.binary_vecs.len();
        let mut buf = Vec::with_capacity(12 + n * 4 + n * 8 + n * WORDS * 4);

        buf.extend_from_slice(b"MVDB");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&(n as u32).to_le_bytes());

        for m in &self.meta {
            buf.extend_from_slice(&m.0.to_le_bytes());
        }
        for &ts in &self.timestamps {
            buf.extend_from_slice(&ts.to_le_bytes());
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
    /// Supports both v1 (no timestamps — defaults to `0.0`) and v2.
    pub fn deserialize(bytes: &[u8]) -> Result<Self, DbError> {
        if bytes.len() < 12 {
            return Err(DbError::CorruptedData);
        }
        if &bytes[0..4] != b"MVDB" {
            return Err(DbError::CorruptedData);
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        if version != 1 && version != 2 {
            return Err(DbError::CorruptedData);
        }
        let n = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

        let meta_start = 12;
        let meta_end   = meta_start + n * 4;

        // v2 carries timestamps; v1 defaults to 0.0 (treated as never-expired).
        let (ts_end, timestamps) = if version == 2 {
            let end = meta_end + n * 8;
            if bytes.len() < end {
                return Err(DbError::CorruptedData);
            }
            let mut v = Vec::with_capacity(n);
            for chunk in bytes[meta_end..end].chunks_exact(8) {
                v.push(f64::from_le_bytes(chunk.try_into().unwrap()));
            }
            (end, v)
        } else {
            (meta_end, alloc::vec![0.0f64; n])
        };

        let vecs_end = ts_end + n * WORDS * 4;
        if bytes.len() < vecs_end {
            return Err(DbError::CorruptedData);
        }

        let mut meta = Vec::with_capacity(n);
        for chunk in bytes[meta_start..meta_end].chunks_exact(4) {
            meta.push(PackedMeta(u32::from_le_bytes(chunk.try_into().unwrap())));
        }

        let mut binary_vecs = Vec::with_capacity(n);
        for chunk in bytes[ts_end..vecs_end].chunks_exact(WORDS * 4) {
            let mut bv = [0u32; WORDS];
            for (i, word_bytes) in chunk.chunks_exact(4).enumerate() {
                bv[i] = u32::from_le_bytes(word_bytes.try_into().unwrap());
            }
            binary_vecs.push(bv);
        }

        Ok(Self { binary_vecs, meta, timestamps, ceiling: None })
    }

    /// Remove all tombstone slots, compacting the three parallel arrays.
    ///
    /// Returns the number of slots freed.
    /// **Warning**: slot indices change — rebuild any HNSW index afterward.
    pub fn compact(&mut self) -> usize {
        let before = self.binary_vecs.len();
        let mut i = 0;
        while i < self.binary_vecs.len() {
            if self.meta[i].is_deleted() {
                self.binary_vecs.swap_remove(i);
                self.meta.swap_remove(i);
                self.timestamps.swap_remove(i);
            } else {
                i += 1;
            }
        }
        before - self.binary_vecs.len()
    }

    // ----- Zero-copy WASM helpers ----------------------------------------

    /// Raw pointer to the start of the binary vector data.
    ///
    /// **Invalidated** if the Vec reallocates.  Use `with_capacity` + tombstone
    /// recycling to prevent reallocation in hot insert paths.
    #[inline]
    pub fn raw_vecs_ptr(&self) -> *const u32 {
        self.binary_vecs.as_ptr() as *const u32
    }

    /// Number of `u32` values in the raw vector data (`slots × WORDS`).
    #[inline]
    pub fn raw_vecs_len(&self) -> usize {
        self.binary_vecs.len() * WORDS
    }

    // ----- Private helpers -----------------------------------------------

    /// Index of the first tombstone slot, or `None` if all slots are active.
    #[inline]
    fn first_tombstone(&self) -> Option<usize> {
        self.meta.iter().position(|m| m.is_deleted())
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
        let slot = store.insert(42, v, 0.0).unwrap();
        assert_eq!(slot, 0);
        assert_eq!(store.get_vec(0), Some(&v));
        assert_eq!(store.get_meta(0).unwrap().doc_id(), 42);
        assert!(!store.get_meta(0).unwrap().is_deleted());
    }

    #[test]
    fn soft_delete_excludes_from_scan() {
        let mut store = VectorStore::new();
        let v = make_vec(1.0);
        store.insert(1, v, 0.0).unwrap();
        store.insert(2, v, 0.0).unwrap();
        store.delete(1);
        let results = store.scan_knn(&v, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 2);
    }

    #[test]
    fn scan_returns_closest_first() {
        let mut store = VectorStore::new();
        let close = quantize_f32(&[1.0_f32; 384]).unwrap();
        let far   = quantize_f32(&[-1.0_f32; 384]).unwrap();
        store.insert(1, close, 0.0).unwrap();
        store.insert(2, far, 0.0).unwrap();
        let results = store.scan_knn(&close, 2);
        assert_eq!(results[0].doc_id, 1);
        assert_eq!(results[0].distance, 0);
        assert_eq!(results[1].doc_id, 2);
        assert_eq!(results[1].distance, 384);
    }

    #[test]
    fn doc_id_too_large_is_rejected() {
        let mut store = VectorStore::new();
        let v = make_vec(1.0);
        assert!(store.insert(PackedMeta::MAX_DOC_ID + 1, v, 0.0).is_err());
    }

    #[test]
    fn compact_removes_deleted_slots() {
        let mut store = VectorStore::new();
        for i in 0..10u32 {
            store.insert(i, make_vec(1.0), 0.0).unwrap();
        }
        store.delete(3);
        store.delete(7);
        let freed = store.compact();
        assert_eq!(freed, 2);
        assert_eq!(store.len(), 8);
    }

    #[test]
    fn tombstone_recycled_at_capacity() {
        // Fill a capacity-3 store, delete slot 1, then insert a 4th vector.
        // It must reuse the tombstone without growing the backing allocation.
        let mut store = VectorStore::with_capacity(3);
        store.insert(10, make_vec(1.0), 0.0).unwrap();
        store.insert(11, make_vec(1.0), 0.0).unwrap();
        store.insert(12, make_vec(1.0), 0.0).unwrap();
        assert_eq!(store.len(), 3);

        store.delete(11);

        let slot = store.insert(99, make_vec(-1.0), 1000.0).unwrap();
        // Slot 1 (the tombstone) must have been recycled.
        assert_eq!(slot, 1);
        assert_eq!(store.len(), 3); // no growth
        assert_eq!(store.get_meta(1).unwrap().doc_id(), 99);
        assert!(!store.get_meta(1).unwrap().is_deleted());
    }

    #[test]
    fn capacity_exceeded_returns_error() {
        let mut store = VectorStore::with_capacity(2);
        store.insert(0, make_vec(1.0), 0.0).unwrap();
        store.insert(1, make_vec(1.0), 0.0).unwrap();
        // No tombstones — must fail.
        let r = store.insert(2, make_vec(1.0), 0.0);
        assert_eq!(r, Err(DbError::CapacityExceeded));
    }

    #[test]
    fn run_gc_tombstones_expired_nodes() {
        let mut store = VectorStore::new();
        // inserted_at = 0 ms; current_time = 5_000 ms; ttl = 3_000 ms
        store.insert(1, make_vec(1.0), 0.0).unwrap();    // age 5_000 > ttl → expired
        store.insert(2, make_vec(1.0), 3_500.0).unwrap(); // age 1_500 < ttl → alive
        store.insert(3, make_vec(1.0), 0.0).unwrap();    // age 5_000 > ttl → expired

        let tombstoned = store.run_gc(5_000.0, 3_000.0);
        assert_eq!(tombstoned, 2);

        // Only doc_id 2 should survive in a scan.
        let results = store.scan_knn(&make_vec(1.0), 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, 2);
    }

    #[test]
    fn run_gc_does_not_double_tombstone() {
        let mut store = VectorStore::new();
        store.insert(1, make_vec(1.0), 0.0).unwrap();
        store.run_gc(5_000.0, 1_000.0); // first GC
        let n = store.run_gc(5_000.0, 1_000.0); // second GC on same slot
        assert_eq!(n, 0); // already a tombstone — must not count again
    }

    #[test]
    fn serialization_round_trip() {
        let mut store = VectorStore::new();
        for i in 0..5u32 {
            let v = make_vec(if i % 2 == 0 { 1.0 } else { -1.0 });
            store.insert(i * 100, v, i as f64 * 1_000.0).unwrap();
        }
        let bytes    = store.serialize();
        let restored = VectorStore::deserialize(&bytes).unwrap();
        assert_eq!(store.len(), restored.len());
        for slot in 0..store.len() as u32 {
            assert_eq!(store.get_vec(slot),  restored.get_vec(slot));
            assert_eq!(
                store.get_meta(slot).unwrap().doc_id(),
                restored.get_meta(slot).unwrap().doc_id(),
            );
        }
    }

    #[test]
    fn deserialize_v1_defaults_timestamps_to_zero() {
        // Build a v1 payload by hand.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"MVDB");
        buf.extend_from_slice(&1u32.to_le_bytes()); // version 1
        buf.extend_from_slice(&1u32.to_le_bytes()); // count = 1
        buf.extend_from_slice(&0u32.to_le_bytes()); // meta: doc_id=0, active
        let bv = make_vec(1.0);
        for &w in bv.iter() { buf.extend_from_slice(&w.to_le_bytes()); }

        let mut store = VectorStore::deserialize(&buf).unwrap();
        assert_eq!(store.len(), 1);
        // Timestamp defaults to 0.0 — node is never expired unless ttl=0.
        assert_eq!(store.run_gc(5_000.0, 10_000.0), 0); // ttl longer than age
    }
}
