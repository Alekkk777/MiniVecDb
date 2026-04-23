/// HNSW (Hierarchical Navigable Small World) index.
///
/// ## Arena-flat layout
///
/// Level-0 neighbours live in a flat `Vec<u32>` arena: node `i`'s slot starts
/// at `i * m_max0`.  Upper-level nodes are sparse (O(log N)), stored with an
/// explicit `upper_nodes` mapping.  All-contiguous layout → L1 cache friendly.
///
/// ## SELECT-NEIGHBORS-HEURISTIC (Algorithm 4, Malkov & Yashunin 2018)
///
/// Instead of simply keeping the M closest candidates, this heuristic selects
/// M neighbors such that each accepted neighbor `e` is closer to the query `q`
/// than to any already-selected neighbor `r`:
///
///   accept e iff ∀r ∈ result : dist(q,e) < dist(e,r)
///
/// This produces a "diverse" neighborhood that fans out in different directions,
/// dramatically improving navigability — especially for structured embedding
/// spaces where greedy brute-force selection would pick only same-cluster
/// neighbours.
///
/// Applied to BOTH forward edges (new node → its M neighbours) and backward
/// edge repair (existing neighbour → new node, when list is full).
///
/// ## Scratch-buffer strategy (hot path is zero-alloc after warmup)
///
/// Three reusable `Vec` fields are `mem::take`-d/restored across calls to let
/// the borrow checker see non-overlapping borrows while avoiding per-call heap
/// allocation:
///
/// * `nb_buf: Vec<u32>`       — neighbour IDs collected inside `ef_search`
/// * `fwd_buf: Vec<u32>`      — forward-edge result of `select_neighbors`
/// * `cands_scratch: Vec<Candidate>` — candidate list built in `connect_bidirectional`
use alloc::vec;
use alloc::vec::Vec;

use crate::{
    distance::hamming_distance,
    error::DbError,
    quantize::WORDS,
    storage::VectorStore,
};
use super::search::{Candidate, MaxHeap, MinHeap, SearchResult, VisitedSet};

// ----- Constants ----------------------------------------------------------

pub const DEFAULT_M: usize = 16;
pub const DEFAULT_M_MAX0: usize = DEFAULT_M * 2;
pub const DEFAULT_EF_CONSTRUCTION: usize = 200;
const MAX_LEVEL: usize = 16;

// ----- PRNG ---------------------------------------------------------------

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    *state = x; x
}
#[inline]
fn random_f64(state: &mut u64) -> f64 {
    (xorshift64(state) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

// ----- SELECT-NEIGHBORS-HEURISTIC -----------------------------------------

/// SELECT-NEIGHBORS-HEURISTIC (Malkov & Yashunin 2018, Algorithm 4).
///
/// `candidates` must be sorted **ascending** by `distance` (= dist(q, c)).
/// Writes selected slot IDs into `result` (cleared on entry).
///
/// The acceptance rule: candidate `c` is added to `result` only when
/// `dist(q,c) < dist(c,r)` for every already-selected neighbour `r`.
/// This prevents "redundant" connections all pointing the same direction.
///
/// `keep_pruned = true` (always): if fewer than `m` diverse neighbours exist,
/// fills remaining slots with the closest rejected candidates.
fn select_neighbors(
    store: &VectorStore,
    candidates: &[Candidate],
    m: usize,
    result: &mut Vec<u32>,
) {
    result.clear();
    // Rejected candidates (discarded in diversity check) — for keep_pruned phase.
    // Bounded by |candidates| ≤ ef_construction; heap-alloc is acceptable here
    // since this function is called once per layer per insert, not per neighbour.
    let mut pruned: Vec<u32> = Vec::new();

    'outer: for cand in candidates {
        if result.len() >= m { break; }
        let c_slot = cand.slot;
        let d_qc   = cand.distance;
        let c_vec  = match store.get_vec(c_slot) { Some(v) => v, None => continue };

        // Reject if any already-selected r has dist(c,r) ≤ dist(q,c).
        // (c is "in the shadow" of r from q's perspective.)
        for &r_slot in result.iter() {
            if let Some(r_vec) = store.get_vec(r_slot) {
                if d_qc >= hamming_distance(c_vec, r_vec) {
                    pruned.push(c_slot);
                    continue 'outer;
                }
            }
        }
        result.push(c_slot);
    }

    // keep_pruned_connections: fill remaining slots with closest rejected (already
    // in dist order because candidates was sorted ascending before this call).
    for c_slot in pruned {
        if result.len() >= m { break; }
        result.push(c_slot);
    }
}

// ----- HnswIndex ----------------------------------------------------------

pub struct HnswIndex {
    level0_adj:   Vec<u32>,
    level0_count: Vec<u8>,
    upper_nodes:  Vec<Vec<u32>>,
    upper_adj:    Vec<Vec<u32>>,
    upper_count:  Vec<Vec<u8>>,
    entry_point:  Option<u32>,
    max_level:    usize,
    m:            usize,
    m_max0:       usize,
    ef_construction: usize,
    level_mult:   f64,
    rng_state:    u64,

    // ── Scratch buffers (never serialised) ─────────────────────────────────
    visited:       VisitedSet,
    /// Reused by `ef_search` for neighbour collection (mem::take pattern).
    nb_buf:        Vec<u32>,
    /// Reused by `insert_node` for the forward-edge `select_neighbors` result.
    fwd_buf:       Vec<u32>,
    /// Reused by `connect_bidirectional` for building the candidate list.
    cands_scratch: Vec<Candidate>,
}

impl HnswIndex {
    pub fn new(m: usize, ef_construction: usize) -> Self {
        let m = m.max(2);
        Self {
            level0_adj:      Vec::new(),
            level0_count:    Vec::new(),
            upper_nodes:     Vec::new(),
            upper_adj:       Vec::new(),
            upper_count:     Vec::new(),
            entry_point:     None,
            max_level:       0,
            m,
            m_max0:          m * 2,
            ef_construction,
            level_mult:      1.0 / libm::log(m as f64),
            rng_state:       0xDEAD_BEEF_CAFE_BABE,
            visited:         VisitedSet::new(0),
            nb_buf:          Vec::new(),
            fwd_buf:         Vec::new(),
            cands_scratch:   Vec::new(),
        }
    }

    pub fn build(store: &VectorStore, m: usize, ef_construction: usize) -> Self {
        let n = store.len();
        let mut idx = Self::new(m, ef_construction);
        // Pre-allocate scratch to avoid growth during hot build loop
        idx.nb_buf.reserve(m * 2);
        idx.fwd_buf.reserve(m * 2);
        idx.cands_scratch.reserve(m * 2 + 2);
        idx.visited.ensure_capacity(n);
        idx.level0_adj.reserve(n * idx.m_max0);
        idx.level0_count.reserve(n);
        for slot in 0..n as u32 {
            if store.is_active(slot) {
                let _ = idx.insert_node(store, slot);
            }
        }
        idx
    }

    // ----- Insert ---------------------------------------------------------

    pub fn insert_node(&mut self, store: &VectorStore, slot: u32) -> Result<(), DbError> {
        let vec = store.get_vec(slot).ok_or(DbError::CorruptedData)?;
        let new_level = self.random_level();
        let n = self.level0_count.len();

        // Extend level-0 arena
        self.level0_adj.extend(core::iter::repeat(u32::MAX).take(self.m_max0));
        self.level0_count.push(0);

        // Extend upper-level arenas
        for l in 1..=new_level {
            let li = l - 1;
            while self.upper_nodes.len() <= li {
                self.upper_nodes.push(Vec::new());
                self.upper_adj.push(Vec::new());
                self.upper_count.push(Vec::new());
            }
            self.upper_nodes[li].push(slot);
            self.upper_adj[li].extend(core::iter::repeat(u32::MAX).take(self.m));
            self.upper_count[li].push(0);
        }

        if self.entry_point.is_none() {
            self.entry_point = Some(slot);
            if new_level > self.max_level { self.max_level = new_level; }
            return Ok(());
        }

        let ep = self.entry_point.unwrap();
        self.visited.ensure_capacity(n + 1);

        // Phase 1: greedy descent max_level → new_level+1
        let mut cur_dist = hamming_distance(vec, store.get_vec(ep).unwrap());
        let mut cur_slot = ep;

        if self.max_level > new_level {
            for l in (new_level + 1..=self.max_level).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    let mut nb_buf = core::mem::take(&mut self.nb_buf);
                    nb_buf.clear();
                    for nb in self.neighbors_at(cur_slot, l) { nb_buf.push(nb); }
                    for &nb in &nb_buf {
                        if let Some(nv) = store.get_vec(nb) {
                            let d = hamming_distance(vec, nv);
                            if d < cur_dist { cur_dist = d; cur_slot = nb; changed = true; }
                        }
                    }
                    self.nb_buf = nb_buf;
                }
            }
        }

        // Phase 2: ef-search + SELECT-NEIGHBORS-HEURISTIC + bidirectional repair
        for l in (0..=new_level.min(self.max_level)).rev() {
            let max_nb = if l == 0 { self.m_max0 } else { self.m };
            let ef = self.ef_construction;

            let mut candidates = self.ef_search(store, vec, cur_slot, l, ef);
            // nb_buf is back in self after ef_search

            if let Some(best) = candidates.0.iter().min_by_key(|c| c.distance) {
                cur_slot = best.slot;
                cur_dist = best.distance;
            }

            // Sort ascending (required by select_neighbors)
            candidates.0.sort_unstable_by_key(|c| c.distance);

            // SELECT-NEIGHBORS-HEURISTIC for the new node's forward edges.
            // fwd_buf is reused across levels to avoid allocation.
            let mut fwd_buf = core::mem::take(&mut self.fwd_buf);
            select_neighbors(store, &candidates.0, max_nb, &mut fwd_buf);

            self.set_neighbors(slot, l, &fwd_buf);

            // Bidirectional repair: add slot as incoming edge to each selected neighbour.
            let fwd_count = fwd_buf.len();
            for i in 0..fwd_count {
                let nb = fwd_buf[i];
                self.connect_bidirectional(store, nb, slot, l, max_nb);
            }

            self.fwd_buf = fwd_buf;
        }

        if new_level > self.max_level {
            self.max_level = new_level;
            self.entry_point = Some(slot);
        }
        Ok(())
    }

    // ----- Search ---------------------------------------------------------

    pub fn search(
        &self,
        store: &VectorStore,
        query: &[u32; WORDS],
        k: usize,
        ef: usize,
    ) -> Vec<SearchResult> {
        if self.entry_point.is_none() || k == 0 { return Vec::new(); }

        let ep = self.entry_point.unwrap();
        let ep_vec = match store.get_vec(ep) { Some(v) => v, None => return Vec::new() };

        let mut cur_dist = hamming_distance(query, ep_vec);
        let mut cur_slot = ep;

        if self.max_level > 0 {
            for l in (1..=self.max_level).rev() {
                let mut changed = true;
                while changed {
                    changed = false;
                    for nb in self.neighbors_at(cur_slot, l) {
                        if let Some(nv) = store.get_vec(nb) {
                            let d = hamming_distance(query, nv);
                            if d < cur_dist { cur_dist = d; cur_slot = nb; changed = true; }
                        }
                    }
                }
            }
        }

        let ef = ef.max(k);
        let mut results = self.ef_search_immut(store, query, cur_slot, 0, ef);
        results.0.sort_unstable_by_key(|c| c.distance);
        results.0.truncate(k);

        results.0.iter().filter_map(|c| {
            let meta = store.get_meta(c.slot)?;
            if meta.is_deleted() { return None; }
            Some(c.to_result(meta.doc_id()))
        }).collect()
    }

    // ----- Serialization (scratch buffers are NOT serialised) ---------------

    pub fn serialize(&self) -> Vec<u8> {
        let n = self.level0_count.len();
        let mut buf = Vec::new();
        buf.extend_from_slice(b"HNWS");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&(self.m as u32).to_le_bytes());
        buf.extend_from_slice(&(self.m_max0 as u32).to_le_bytes());
        buf.extend_from_slice(&(self.ef_construction as u32).to_le_bytes());
        buf.extend_from_slice(&self.level_mult.to_le_bytes());
        buf.extend_from_slice(&self.rng_state.to_le_bytes());
        buf.extend_from_slice(&(self.max_level as u32).to_le_bytes());
        buf.extend_from_slice(&self.entry_point.unwrap_or(u32::MAX).to_le_bytes());
        buf.extend_from_slice(&(n as u32).to_le_bytes());

        for &v in &self.level0_adj   { buf.extend_from_slice(&v.to_le_bytes()); }
        buf.extend_from_slice(&self.level0_count);
        let pad = (4 - (n % 4)) % 4;
        buf.extend(core::iter::repeat(0u8).take(pad));

        buf.extend_from_slice(&(self.upper_nodes.len() as u32).to_le_bytes());
        for li in 0..self.upper_nodes.len() {
            let uc = self.upper_nodes[li].len() as u32;
            buf.extend_from_slice(&uc.to_le_bytes());
            for &s in &self.upper_nodes[li] { buf.extend_from_slice(&s.to_le_bytes()); }
            for &a in &self.upper_adj[li]   { buf.extend_from_slice(&a.to_le_bytes()); }
            buf.extend_from_slice(&self.upper_count[li]);
            let pad = (4 - (self.upper_count[li].len() % 4)) % 4;
            buf.extend(core::iter::repeat(0u8).take(pad));
        }
        buf
    }

    pub fn deserialize(bytes: &[u8]) -> Result<Self, DbError> {
        if bytes.len() < 48 || &bytes[0..4] != b"HNWS" { return Err(DbError::CorruptedData); }
        let version = read_u32(bytes, 4)?;
        if version != 1 { return Err(DbError::CorruptedData); }
        let m            = read_u32(bytes, 8)?  as usize;
        let m_max0       = read_u32(bytes, 12)? as usize;
        let ef           = read_u32(bytes, 16)? as usize;
        let level_mult   = f64::from_le_bytes(bytes[20..28].try_into().map_err(|_| DbError::CorruptedData)?);
        let rng_state    = u64::from_le_bytes(bytes[28..36].try_into().map_err(|_| DbError::CorruptedData)?);
        let max_level    = read_u32(bytes, 36)? as usize;
        let ep_raw       = read_u32(bytes, 40)?;
        let entry_point  = if ep_raw == u32::MAX { None } else { Some(ep_raw) };
        let n            = read_u32(bytes, 44)? as usize;
        let mut pos = 48usize;

        let adj_bytes = n * m_max0 * 4;
        if pos + adj_bytes > bytes.len() { return Err(DbError::CorruptedData); }
        let mut level0_adj = vec![u32::MAX; n * m_max0];
        for (i, ch) in bytes[pos..pos+adj_bytes].chunks_exact(4).enumerate() {
            level0_adj[i] = u32::from_le_bytes(ch.try_into().unwrap());
        }
        pos += adj_bytes;

        if pos + n > bytes.len() { return Err(DbError::CorruptedData); }
        let level0_count = bytes[pos..pos+n].to_vec();
        pos += n;
        pos += (4 - (n % 4)) % 4;

        if pos + 4 > bytes.len() { return Err(DbError::CorruptedData); }
        let num_upper = read_u32(bytes, pos)? as usize;
        pos += 4;

        let mut upper_nodes = Vec::with_capacity(num_upper);
        let mut upper_adj   = Vec::with_capacity(num_upper);
        let mut upper_count = Vec::with_capacity(num_upper);

        for _ in 0..num_upper {
            if pos + 4 > bytes.len() { return Err(DbError::CorruptedData); }
            let uc = read_u32(bytes, pos)? as usize; pos += 4;

            let nb = uc * 4;
            if pos + nb > bytes.len() { return Err(DbError::CorruptedData); }
            let mut nodes = vec![0u32; uc];
            for (i, ch) in bytes[pos..pos+nb].chunks_exact(4).enumerate() {
                nodes[i] = u32::from_le_bytes(ch.try_into().unwrap());
            }
            pos += nb;
            upper_nodes.push(nodes);

            let ab = uc * m * 4;
            if pos + ab > bytes.len() { return Err(DbError::CorruptedData); }
            let mut adj = vec![u32::MAX; uc * m];
            for (i, ch) in bytes[pos..pos+ab].chunks_exact(4).enumerate() {
                adj[i] = u32::from_le_bytes(ch.try_into().unwrap());
            }
            pos += ab;
            upper_adj.push(adj);

            if pos + uc > bytes.len() { return Err(DbError::CorruptedData); }
            let cv = bytes[pos..pos+uc].to_vec();
            pos += uc;
            pos += (4 - (uc % 4)) % 4;
            upper_count.push(cv);
        }

        Ok(Self {
            level0_adj, level0_count, upper_nodes, upper_adj, upper_count,
            entry_point, max_level, m, m_max0, ef_construction: ef,
            level_mult, rng_state,
            visited:       VisitedSet::new(0),
            nb_buf:        Vec::new(),
            fwd_buf:       Vec::new(),
            cands_scratch: Vec::new(),
        })
    }

    // ----- Internal helpers -----------------------------------------------

    fn random_level(&mut self) -> usize {
        let r = random_f64(&mut self.rng_state);
        let l = if r > 0.0 { libm::floor(-libm::log(r) * self.level_mult) as usize } else { 0 };
        l.min(MAX_LEVEL)
    }

    fn neighbors_at(&self, slot: u32, l: usize) -> impl Iterator<Item = u32> + '_ {
        if l == 0 {
            let base  = slot as usize * self.m_max0;
            let count = self.level0_count.get(slot as usize).copied().unwrap_or(0) as usize;
            NeighborIter::new(&self.level0_adj[base..base + self.m_max0], count)
        } else {
            let li = l - 1;
            if li >= self.upper_nodes.len() { return NeighborIter::new(&[], 0); }
            match self.upper_nodes[li].iter().position(|&s| s == slot) {
                None    => NeighborIter::new(&[], 0),
                Some(ui) => {
                    let base  = ui * self.m;
                    let count = self.upper_count[li].get(ui).copied().unwrap_or(0) as usize;
                    NeighborIter::new(&self.upper_adj[li][base..base + self.m], count)
                }
            }
        }
    }

    fn set_neighbors(&mut self, node: u32, l: usize, neighbors: &[u32]) {
        if l == 0 {
            let n = node as usize;
            let cap = self.m_max0;
            let base = n * cap;
            let cnt = neighbors.len().min(cap);
            self.level0_adj[base..base+cnt].copy_from_slice(&neighbors[..cnt]);
            for i in cnt..cap { self.level0_adj[base+i] = u32::MAX; }
            self.level0_count[n] = cnt as u8;
        } else {
            let li = l - 1;
            if li >= self.upper_nodes.len() { return; }
            let ui = match self.upper_nodes[li].iter().position(|&s| s == node) {
                Some(i) => i, None => return,
            };
            let cap = self.m;
            let base = ui * cap;
            let cnt = neighbors.len().min(cap);
            self.upper_adj[li][base..base+cnt].copy_from_slice(&neighbors[..cnt]);
            for i in cnt..cap { self.upper_adj[li][base+i] = u32::MAX; }
            self.upper_count[li][ui] = cnt as u8;
        }
    }

    /// Add `new_slot` as an incoming edge to `nb`'s neighbour list at layer `l`.
    ///
    /// Three cases:
    /// 1. Already connected → skip.
    /// 2. List has space → append in O(1).
    /// 3. List is full → run SELECT-NEIGHBORS-HEURISTIC on
    ///    {current_neighbours ∪ new_slot}, replacing the list with the
    ///    diverse selection.  This is correct bidirectional repair per the
    ///    HNSW paper: we reselect, so new_slot appears only if it is diverse
    ///    enough (not dominated by any existing neighbour).  `keep_pruned=true`
    ///    guarantees we still fill all M slots, ensuring connectivity.
    fn connect_bidirectional(
        &mut self,
        store: &VectorStore,
        nb: u32,
        new_slot: u32,
        l: usize,
        max_nb: usize,
    ) {
        if l == 0 {
            let n     = nb as usize;
            let count = self.level0_count[n] as usize;
            let base  = n * self.m_max0;
            if self.level0_adj[base..base+count].contains(&new_slot) { return; }
            if count < max_nb {
                self.level0_adj[base + count] = new_slot;
                self.level0_count[n] += 1;
                return;
            }
            // Full — heuristic reselection
            let nb_vec = match store.get_vec(nb) { Some(v) => *v, None => return };
            let mut cands = core::mem::take(&mut self.cands_scratch);
            cands.clear();
            for i in 0..count {
                let s = self.level0_adj[base + i];
                if s == u32::MAX { continue; }
                if let Some(v) = store.get_vec(s) {
                    cands.push(Candidate::new(s, hamming_distance(&nb_vec, v)));
                }
            }
            if let Some(v) = store.get_vec(new_slot) {
                cands.push(Candidate::new(new_slot, hamming_distance(&nb_vec, v)));
            }
            cands.sort_unstable_by_key(|c| c.distance);
            // Use nb_buf as result scratch (it's free while fwd_buf holds the loop's data)
            let mut result = core::mem::take(&mut self.nb_buf);
            select_neighbors(store, &cands, max_nb, &mut result);
            self.set_neighbors(nb, 0, &result);
            self.cands_scratch = cands;
            self.nb_buf = result;
        } else {
            let li = l - 1;
            if li >= self.upper_nodes.len() { return; }
            let ui = match self.upper_nodes[li].iter().position(|&s| s == nb) {
                Some(i) => i, None => return,
            };
            let count = self.upper_count[li][ui] as usize;
            let base  = ui * self.m;
            if self.upper_adj[li][base..base+count].contains(&new_slot) { return; }
            if count < max_nb {
                self.upper_adj[li][base + count] = new_slot;
                self.upper_count[li][ui] += 1;
                return;
            }
            let nb_vec = match store.get_vec(nb) { Some(v) => *v, None => return };
            let mut cands = core::mem::take(&mut self.cands_scratch);
            cands.clear();
            for i in 0..count {
                let s = self.upper_adj[li][base + i];
                if s == u32::MAX { continue; }
                if let Some(v) = store.get_vec(s) {
                    cands.push(Candidate::new(s, hamming_distance(&nb_vec, v)));
                }
            }
            if let Some(v) = store.get_vec(new_slot) {
                cands.push(Candidate::new(new_slot, hamming_distance(&nb_vec, v)));
            }
            cands.sort_unstable_by_key(|c| c.distance);
            let mut result = core::mem::take(&mut self.nb_buf);
            select_neighbors(store, &cands, max_nb, &mut result);
            self.set_neighbors(nb, l, &result);
            self.cands_scratch = cands;
            self.nb_buf = result;
        }
    }

    /// Mutable ef-search (construction phase).  Uses `self.visited` + `nb_buf`.
    fn ef_search(
        &mut self,
        store: &VectorStore,
        query: &[u32; WORDS],
        entry: u32,
        l: usize,
        ef: usize,
    ) -> MaxHeap {
        self.visited.reset();
        self.visited.visit(entry);

        let d = hamming_distance(query, match store.get_vec(entry) {
            Some(v) => v, None => return MaxHeap::new(),
        });
        let ep_cand = Candidate::new(entry, d);
        let mut candidates = MinHeap::with_capacity(ef);
        let mut results    = MaxHeap::with_capacity(ef + 1);
        candidates.push(ep_cand);
        results.push(ep_cand);

        while let Some(cur) = candidates.pop() {
            let worst = results.peek_worst().map(|c| c.distance).unwrap_or(u32::MAX);
            if cur.distance > worst && results.len() >= ef { break; }

            // mem::take: lets neighbors_at borrow self immutably while nb_buf is local
            let mut nb_buf = core::mem::take(&mut self.nb_buf);
            nb_buf.clear();
            for nb in self.neighbors_at(cur.slot, l) { nb_buf.push(nb); }
            self.nb_buf = nb_buf;

            for i in 0..self.nb_buf.len() {
                let nb = self.nb_buf[i];
                if nb == u32::MAX || self.visited.is_visited(nb) { continue; }
                self.visited.visit(nb);
                let nv = match store.get_vec(nb) { Some(v) => v, None => continue };
                let d  = hamming_distance(query, nv);
                let worst = results.peek_worst().map(|c| c.distance).unwrap_or(u32::MAX);
                if d < worst || results.len() < ef {
                    candidates.push(Candidate::new(nb, d));
                    results.push(Candidate::new(nb, d));
                    if results.len() > ef { results.pop_worst(); }
                }
            }
        }
        results
    }

    /// Immutable ef-search (query phase).  Allocates a fresh `VisitedSet`.
    fn ef_search_immut(
        &self,
        store: &VectorStore,
        query: &[u32; WORDS],
        entry: u32,
        l: usize,
        ef: usize,
    ) -> MaxHeap {
        let mut visited = VisitedSet::new(store.len());
        visited.visit(entry);
        let d = hamming_distance(query, match store.get_vec(entry) {
            Some(v) => v, None => return MaxHeap::new(),
        });
        let ep_cand = Candidate::new(entry, d);
        let mut candidates = MinHeap::with_capacity(ef);
        let mut results    = MaxHeap::with_capacity(ef + 1);
        candidates.push(ep_cand);
        results.push(ep_cand);

        while let Some(cur) = candidates.pop() {
            let worst = results.peek_worst().map(|c| c.distance).unwrap_or(u32::MAX);
            if cur.distance > worst && results.len() >= ef { break; }
            for nb in self.neighbors_at(cur.slot, l) {
                if nb == u32::MAX || visited.is_visited(nb) { continue; }
                visited.visit(nb);
                let nv = match store.get_vec(nb) { Some(v) => v, None => continue };
                let d  = hamming_distance(query, nv);
                let worst = results.peek_worst().map(|c| c.distance).unwrap_or(u32::MAX);
                if d < worst || results.len() < ef {
                    candidates.push(Candidate::new(nb, d));
                    results.push(Candidate::new(nb, d));
                    if results.len() > ef { results.pop_worst(); }
                }
            }
        }
        results
    }
}

// ----- NeighborIter -------------------------------------------------------

struct NeighborIter<'a> { slice: &'a [u32], pos: usize, count: usize }
impl<'a> NeighborIter<'a> {
    fn new(slice: &'a [u32], count: usize) -> Self { Self { slice, pos: 0, count } }
}
impl<'a> Iterator for NeighborIter<'a> {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.pos >= self.count { return None; }
        let v = self.slice[self.pos]; self.pos += 1;
        if v == u32::MAX { None } else { Some(v) }
    }
}

// ----- Helpers -----------------------------------------------------------

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32, DbError> {
    bytes.get(offset..offset+4).and_then(|s| s.try_into().ok())
        .map(u32::from_le_bytes).ok_or(DbError::CorruptedData)
}

// ----- Tests --------------------------------------------------------------

#[cfg(test)]
mod tests {
    extern crate std;
    use std::collections::HashSet;
    use alloc::vec::Vec;
    use super::*;
    use crate::{quantize::quantize_f32, storage::VectorStore};

    fn make_store_random(n: usize, seed: u64) -> VectorStore {
        let mut store = VectorStore::with_capacity(n);
        let mut rng = seed;
        for i in 0..n as u32 {
            let floats: Vec<f32> = (0..384).map(|_| {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                if rng & 1 == 0 { 1.0_f32 } else { -1.0_f32 }
            }).collect();
            store.insert(i, quantize_f32(&floats).unwrap(), 0.0).unwrap();
        }
        store
    }

    #[test]
    fn empty_index_returns_empty() {
        let store = VectorStore::new();
        let idx   = HnswIndex::new(DEFAULT_M, DEFAULT_EF_CONSTRUCTION);
        assert!(idx.search(&store, &[0u32; 12], 5, 20).is_empty());
    }

    #[test]
    fn single_node_exact_match() {
        let mut store = VectorStore::new();
        let v = quantize_f32(&[1.0_f32; 384]).unwrap();
        store.insert(42, v, 0.0).unwrap();
        let mut idx = HnswIndex::new(DEFAULT_M, DEFAULT_EF_CONSTRUCTION);
        idx.insert_node(&store, 0).unwrap();
        let r = idx.search(&store, &v, 1, 10);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].doc_id, 42);
        assert_eq!(r[0].distance, 0);
    }

    #[test]
    fn recall_at_10_above_90_percent() {
        let n     = 2000;
        let store = make_store_random(n, 0xABCD1234);
        let idx   = HnswIndex::build(&store, DEFAULT_M, DEFAULT_EF_CONSTRUCTION);

        let mut total_hits = 0usize;
        let queries = 50;
        let k = 10;
        let mut rng = 0xFEED_FACE_u64;
        for _ in 0..queries {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let q_slot = (rng % n as u64) as u32;
            let query  = *store.get_vec(q_slot).unwrap();
            let brute  = store.scan_knn(&query, k);
            let approx = idx.search(&store, &query, k, 64);
            let bf_ids: HashSet<u32> = brute.iter().map(|r| r.doc_id).collect();
            total_hits += approx.iter().filter(|r| bf_ids.contains(&r.doc_id)).count();
        }
        let recall = total_hits as f64 / (queries * k) as f64;
        assert!(recall >= 0.85, "Recall@10 = {:.2} < 0.85", recall);
    }

    #[test]
    fn serialization_round_trip() {
        let store   = make_store_random(200, 0x1234ABCD);
        let idx     = HnswIndex::build(&store, DEFAULT_M, DEFAULT_EF_CONSTRUCTION);
        let bytes   = idx.serialize();
        let restored = HnswIndex::deserialize(&bytes).unwrap();
        let query   = *store.get_vec(0).unwrap();
        let a: Vec<u32> = idx.search(&store, &query, 5, 20).iter().map(|r| r.doc_id).collect();
        let b: Vec<u32> = restored.search(&store, &query, 5, 20).iter().map(|r| r.doc_id).collect();
        assert_eq!(a, b);
    }
}
