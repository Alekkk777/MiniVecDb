// Define the global allocator only when compiling for wasm32.
// For native builds (tests, CLI), the standard library provides one automatically.
#[cfg(all(target_arch = "wasm32", feature = "allocator"))]
use lol_alloc::{AssumeSingleThreaded, FreeListAllocator};

#[cfg(all(target_arch = "wasm32", feature = "allocator"))]
#[global_allocator]
// SAFETY: The WASM module runs single-threaded by design. Each Web Worker
// gets its own WASM instance and linear memory — there is no shared memory
// between instances, so no concurrent allocator access is possible.
static ALLOCATOR: AssumeSingleThreaded<FreeListAllocator> =
    unsafe { AssumeSingleThreaded::new(FreeListAllocator::new()) };
