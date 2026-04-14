use alloc::vec::Vec;
use crate::distance::similarity_score;

/// A single result returned by HNSW or brute-force KNN search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    /// The document ID supplied by the caller during `insert`.
    pub doc_id: u32,
    /// Internal slot index inside [`VectorStore`].
    pub slot: u32,
    /// Hamming distance to the query vector (0 = identical, 384 = opposite).
    pub distance: u32,
    /// Normalised similarity: `1 - distance / 384`. Range: `[0.0, 1.0]`.
    pub score: f32,
}

/// A candidate node during HNSW graph traversal.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Candidate {
    pub distance: u32,
    pub slot: u32,
}

impl Candidate {
    pub(super) fn new(slot: u32, distance: u32) -> Self {
        Self { distance, slot }
    }

    pub(super) fn to_result(self, doc_id: u32) -> SearchResult {
        SearchResult {
            doc_id,
            slot: self.slot,
            distance: self.distance,
            score: similarity_score(self.distance),
        }
    }
}

// Manual Ord implementations so we can use Vec-as-heap tricks.
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Reverse: smaller distance = higher priority
        other.distance.cmp(&self.distance)
    }
}

// ----- Tiny no-std priority queues ----------------------------------------

/// Min-heap (smallest distance at top) for the candidate frontier.
pub(super) struct MinHeap(pub Vec<Candidate>);

impl MinHeap {
    pub fn new() -> Self { Self(Vec::new()) }
    pub fn with_capacity(n: usize) -> Self { Self(Vec::with_capacity(n)) }

    pub fn push(&mut self, c: Candidate) {
        self.0.push(c);
        sift_up_min(&mut self.0);
    }

    pub fn pop(&mut self) -> Option<Candidate> {
        if self.0.is_empty() { return None; }
        let last = self.0.len() - 1;
        self.0.swap(0, last);
        let top = self.0.pop();
        if !self.0.is_empty() { sift_down_min(&mut self.0); }
        top
    }

    pub fn peek(&self) -> Option<&Candidate> { self.0.first() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}

/// Max-heap (largest distance at top) for the results set (bounded to k).
pub(super) struct MaxHeap(pub Vec<Candidate>);

impl MaxHeap {
    pub fn new() -> Self { Self(Vec::new()) }
    pub fn with_capacity(n: usize) -> Self { Self(Vec::with_capacity(n)) }

    pub fn push(&mut self, c: Candidate) {
        self.0.push(c);
        sift_up_max(&mut self.0);
    }

    pub fn pop_worst(&mut self) -> Option<Candidate> {
        if self.0.is_empty() { return None; }
        let last = self.0.len() - 1;
        self.0.swap(0, last);
        let top = self.0.pop();
        if !self.0.is_empty() { sift_down_max(&mut self.0); }
        top
    }

    pub fn peek_worst(&self) -> Option<&Candidate> { self.0.first() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
}

// ----- Bitset visited set -------------------------------------------------

/// O(1) insert/lookup visited set backed by a flat bit-array.
/// Supports a "generation" trick: we keep a per-slot generation counter and
/// compare it to the current generation rather than clearing the array on each
/// query (clearing a Vec<u64> of size N/64 is O(N/64) — expensive for large N).
pub(super) struct VisitedSet {
    gen: u32,
    pub(super) gen_map: Vec<u32>, // gen_map[slot] = generation when last visited
}

impl VisitedSet {
    pub fn new(capacity: usize) -> Self {
        // gen starts at 1, gen_map filled with 0 → is_visited() correctly
        // returns false for every slot until visit() is called.
        Self { gen: 1, gen_map: alloc::vec![0u32; capacity] }
    }

    /// Ensure the internal map can hold at least `n` slots.
    pub(super) fn ensure_capacity(&mut self, n: usize) {
        if self.gen_map.len() < n {
            self.gen_map.resize(n, 0);
        }
    }

    /// Start a new query — "clear" all visited marks in O(1).
    pub fn reset(&mut self) {
        self.gen = self.gen.wrapping_add(1);
        // On overflow, zero the map to avoid false positives.
        if self.gen == 0 {
            self.gen_map.iter_mut().for_each(|v| *v = 0);
            self.gen = 1;
        }
    }

    /// Mark `slot` as visited.
    #[inline]
    pub fn visit(&mut self, slot: u32) {
        let idx = slot as usize;
        if idx >= self.gen_map.len() {
            self.gen_map.resize(idx + 1, 0);
        }
        self.gen_map[idx] = self.gen;
    }

    /// Return `true` if `slot` was already visited in the current generation.
    #[inline]
    pub fn is_visited(&self, slot: u32) -> bool {
        self.gen_map
            .get(slot as usize)
            .map(|&g| g == self.gen)
            .unwrap_or(false)
    }
}

// ----- Heap helpers (manual impls, no std::collections) -------------------

fn sift_up_min(v: &mut Vec<Candidate>) {
    let mut i = v.len() - 1;
    while i > 0 {
        let p = (i - 1) / 2;
        if v[i].distance < v[p].distance { v.swap(i, p); i = p; } else { break; }
    }
}

fn sift_down_min(v: &mut Vec<Candidate>) {
    let len = v.len();
    let mut i = 0;
    loop {
        let l = 2 * i + 1; let r = 2 * i + 2;
        let mut s = i;
        if l < len && v[l].distance < v[s].distance { s = l; }
        if r < len && v[r].distance < v[s].distance { s = r; }
        if s == i { break; }
        v.swap(i, s); i = s;
    }
}

fn sift_up_max(v: &mut Vec<Candidate>) {
    let mut i = v.len() - 1;
    while i > 0 {
        let p = (i - 1) / 2;
        if v[i].distance > v[p].distance { v.swap(i, p); i = p; } else { break; }
    }
}

fn sift_down_max(v: &mut Vec<Candidate>) {
    let len = v.len();
    let mut i = 0;
    loop {
        let l = 2 * i + 1; let r = 2 * i + 2;
        let mut s = i;
        if l < len && v[l].distance > v[s].distance { s = l; }
        if r < len && v[r].distance > v[s].distance { s = r; }
        if s == i { break; }
        v.swap(i, s); i = s;
    }
}
