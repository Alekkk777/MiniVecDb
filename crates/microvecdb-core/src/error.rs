/// All error variants produced by the database engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DbError {
    /// The vector slice has the wrong number of elements (expected 384).
    InvalidDimension { got: usize },
    /// The document ID uses bit 31, which is reserved for the deletion flag.
    DocIdTooLarge,
    /// HNSW `search` was called before `build_index`.
    IndexNotBuilt,
    /// The byte buffer passed to `deserialize` is malformed.
    CorruptedData,
    /// The store is at maximum capacity (2^31 − 1 slots).
    CapacityExceeded,
}
