#![no_std]

extern crate alloc;

mod alloc_setup;

pub mod distance;
pub mod error;
pub mod hnsw;
pub mod quantize;
pub mod storage;

pub use distance::hamming_distance;
pub use error::DbError;
pub use hnsw::{HnswIndex, SearchResult};
pub use quantize::{quantize_f32, BinaryVec, DIMS, WORDS};
pub use storage::VectorStore;
