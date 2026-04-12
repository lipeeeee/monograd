# optimization
- gemm/matmul ops should be optimized pre-written kernels, with diferences on tiling based on nvidia/amd (32 for nvidia 16 for amd). also different dtpes might require different kernels.
    - there is a lib called clblast that optimizes for every single gpu iteration. should use that for maximum perf gain
    - also pre-compiled kernels? maybe it doesn't result in any perf gain but still interesting
- new op: V(vector)CONST, makes expand a noop, even if it has no cost(does it have cost or is EXPAND free/boundary?)
- .to() creates COPY OP before realize, it should be a NOOP, maybe not.

- compgt complt
    - then decompose rest
- OptOps — kernel-level loop optimization: Extracts last 10-15% optimization
- Kernel compile caching
- minimize ram/vram access because it's slow
- We aren't caching LOAD ops, not sure if in the future its worth it to make them cachable
- choosing kernel opts
    - global_size: round_up_multipleof2(len(data)) if len(data) <= gpu_compute_units * gpu_max_threads else gpu_compute_units * gpu_max_threads
    - local_size: gpu warp constraints. Maybe change for amd vs nvidia / 256 is good default
- assert codegen uses 'gid' when writting indexes and that we are generating 'gid' in every kernel
- Missing Memory Pool: device.py directly allocates a new OpenCL buffer for every operation. During a training loop, this will cause severe VRAM fragmentation and massive allocation overhead. A memory pool that reuses identically sized allocations is required.
- The Buffer class supports offsets and bases (self._base), but this logic is not fully integrated with the execution engine. If a KernelTask operates on a memory view, the codegen must inject the offset into the physical address calculation.
- _collect_inputs iterates through the sources of every operation in the group. For large fused groups, this scales poorly. Replacing the nested loops with a set-based traversal will reduce scheduler overhead. (does this matter tho? (does this matter tho? profile it later)
- opencl kernel profiling/benchmarking into saving optimal params
- flatten? (zero-copy preferably)
- graph viz

# After codegen
- monospec
- stress test scheduler and understand it fully with tests
- test contiguous & padding behaviours & _reduceop(axis_tuple) and axis_scalar & new scheduler
- there is alot of repeted code in codegen, look into it
- make DEBUG >= 1 print(source) into 1 line
- document contextvars/envvars
- re-read schedule.py
- support np dtypes in to_dtype
- !look for cache optims!


# Debug levels
- DEBUG >= 1 — High level execution flow
Kernel source after generation, schedule summary (how many tasks, what kinds), graph roots when .numpy() is called.
- DEBUG >= 2 — Per-task details
KernelTask pretty-print, BufferRef strides for each input, which rules fired in the rewriter.
- DEBUG >= 3 — Internal state
val_map contents, render_op_chain intermediate expressions, UPat match captures.
- DEBUG >= 4 — Very noisy internals
Every BufferRef created, every UOp cache hit/miss, every rule attempted (not just matched).

# important
- GlobalCounters for mem_used;global_ops;global_mem;kernel_count;
- environmentalize/globalize data/ folder for downloding and caching (is it a good name?)


# unimportant
- tensor fns such as zeros(), arrange(), ones(), randn()
- mnist.py should be cleaned
- If user doesnt have pyopencl installed, we extract it via ctypes(not very supported )
