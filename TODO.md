# next
- github actions run pytest
- setup test env with pytest
- re-read schedule.py


# optimization
- gemm/matmul ops should be optimized pre-written kernels, with diferences on tiling based on nvidia/amd (32 for nvidia 16 for amd). also different dtpes might require different kernels.
- there is a lib called clblast that optimizes for every single gpu iteration. should use that for maximum perf gain
- also pre-compiled kernels? maybe it doesn't result in any perf gain but still interesting
- constant folding
- MUL+ADD = MULACC
- new op: V(vector)CONST, makes expand a noop, even if it has no cost(does it have cost or is EXPAND free/boundary?)
- .to() creates COPY OP before realize, it should be a NOOP


# important
- GlobalCounters for mem_used;global_ops;global_mem;kernel_count;
- environmentalize/globalize data/ folder for downloding and caching (is it a good name?)
- UPAT to lower number of gpu instructions, see tinygrad
- If user doesnt have pyopencl installed, we extract it via ctypes(not very supported )


# unimportant
- mnist.py should be cleaned
