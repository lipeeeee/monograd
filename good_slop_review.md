# Monograd engineering review

Date: 2026-07-11

## Scope and method

This review is based on the implementation, not the README. I read the source under `monograd/`, the supporting examples and project metadata, and traced the path from Tensor construction through UOps, rewriting, scheduling, layout reconstruction, OpenCL code generation, and buffers. I deliberately did not read the test suites or `slop_review.md`.

I also ran small, independent probes using the installed Python 3.12, NumPy 2.4.2, and PyOpenCL 2026.1.2. These probes constructed graphs and inspected schedules/generated kernels; they did not use the repository's tests. Findings marked **confirmed by probe** were reproduced directly.

I am assuming, as requested, that a future high-level runtime will cache compiled kernels and tune legal kernel parameters. I do not treat the absence of that runtime, a complete backward pass, or every neural-network operation as a failure by itself. I do critique current interfaces where they would make that runtime hard or unsafe to build.

## Short verdict

Monograd is a good compiler and GPU-systems learning project, but it is not yet a sound foundation for a usable tensor framework. Its best work is in the scheduler/code-generation direction: it attempts real fusion, masked views, reduction specialization, and tiled GEMM instead of hiding everything behind NumPy or a vendor BLAS call.

The largest issue is not missing features. It is that the representation of layout, storage identity, and execution state does not preserve tensor semantics once views, fusion, and materialization interact. Fixing those invariants is more important than adding operations, a memory pool, or more tuning parameters.

The system currently looks like this:

```text
Tensor API
    -> immutable/interned UOp DAG
    -> algebraic rewrite
    -> scheduler and fusion
    -> BufferRef reconstructs views and indexing
    -> KernelTask
    -> OpenCL source generation
    -> future compiled-kernel runtime
    -> Buffer/OpenCL execution

UOp ---------------- weak side table ----------------> Buffer
```

The main fracture is between fusion and `BufferRef`: layout belongs to a viewed value or to an edge in the expression graph, while code generation often identifies a value only by its base UOp.

## Rating

### As a first serious project for learning systems programming and low-level deep learning: **8/10**

This is an unusually strong starting project. It forces contact with several real topics at once:

- IR and graph design
- canonicalization and graph rewriting
- fusion and scheduling
- shape, stride, broadcast, and padding semantics
- native-resource ownership
- host/device transfers
- OpenCL launch geometry
- reductions, tiling, local memory, and register blocking
- dtype and numerical-semantics questions
- the boundary between a compiler artifact and a runtime invocation

The bugs are serious for a framework, but many are exactly the valuable kind of systems bugs: identity versus equality, ownership versus aliasing, logical views versus physical storage, numerical equivalence versus algebraic equivalence, and compilation versus execution state. Discovering and fixing these will teach more than adding twenty high-level layers.

Why it is not 9 or 10:

- The current abstractions do not yet enforce their invariants, so several optimizations silently miscompile valid-looking graphs.
- Python and PyOpenCL hide some lower-level concerns such as ABI layout, explicit native error handling, command-buffer ownership, and synchronization. They are excellent for iteration, but eventually some C++, Rust, C, or direct OpenCL API work would deepen the systems side.
- The supported operation surface is ahead of the compiler's actual coverage, making it hard to tell what is intended to work.
- There is not yet a disciplined numerical contract or backend contract.

### As the current basis of a usable general tensor framework: **3.5/10**

That lower score is not a contradiction. A project can be excellent for learning while still being far from reliable software. Today, view/fusion correctness, device transfers, dtype legality, buffer ownership, realization state, and autograd architecture all need foundational work.

### As a portfolio artifact today: **7/10**, potentially **9/10** after a focused correctness pass

The interesting part to an employer is not the number of implemented layers. It is whether the repository demonstrates that the author can define invariants, find counterexamples, measure behavior, and repair abstractions. A short design document plus a handful of compelling correctness/performance case studies would raise its value substantially.

## What is done well

### 1. The project attacks the right low-level problems

It would have been much easier to wrap NumPy, call CLBlast, and present a large API. Instead, Monograd contains its own graph representation, rewrite engine, scheduler, view indexing, reductions, and a tiled matmul kernel. That scope is much more relevant to systems and ML-compiler work.

### 2. The scheduler is compact and understandable

`run_scheduler` in `monograd/engine/schedule.py` is small enough to reason about. Counting consumers, treating reductions/BLAS/copies as boundaries, materializing fan-out, and pulling elementwise producers into a kernel are sensible first heuristics.

In particular, fusing an elementwise producer into a reduction can avoid a full intermediate VRAM round trip. That is the right performance instinct: optimize memory traffic, not just arithmetic count.

### 3. Indexing logic is centralized

`BufferRef` attempts to make shape, strides, expansion, permutation, and padding one compiler concern instead of duplicating index calculations in every kernel. This is directionally correct. A first-class layout abstraction should evolve from this code rather than replacing the whole compiler.

### 4. Padding as a masked load is a good decision

Representing padding as an address calculation plus a validity predicate can avoid a physical padded allocation and copy. The generated ternary load is a useful technique for fused convolution/reduction-style operations. It needs a more general layout representation, but the optimization idea is good.

### 5. Reduction layout is treated as meaningful

The code distinguishes full reduction, strided outer reduction, and a workgroup-parallel inner reduction. Even though the algorithms need more work, recognizing that reduction axes and physical strides determine the kernel strategy is exactly right.

### 6. The GEMM kernel shows real hardware awareness

`_codegen_blas` includes:

- cooperative local-memory loads
- two-dimensional tiling
- register blocking with work per thread
- FMA accumulation
- boundary guards
- a `+1` padding dimension on the A tile to reduce bank conflicts
- shape-specialized launch geometry

This is substantially more educational than calling a library. It also creates a good place to learn occupancy, local-memory pressure, vectorized loads, accumulator types, and device-specific tuning.

### 7. Generated kernels are readable

Readable generated code is a major advantage while developing a compiler. The current sources are easy to inspect and connect back to graph operations. Keep that property even if the internal lowering becomes more structured.

### 8. The project has good seeds for canonicalization

Immutable DTypes, UOp interning, a small pattern matcher, and fixpoint rewriting are worthwhile compiler experiments. The implementation needs stricter identity and numerical rules, but the direction is appropriate.

## Critical correctness findings

### 1. Multiple views of the same base are conflated

`BufferRef.from_uop` walks through movement operations and records the underlying base UOp plus reconstructed strides. In code generation, `_resolve_base` strips movement operations, while `val_map` is keyed by `id(base_uop)`.

That means two distinct views of one allocation cannot be represented simultaneously.

Example:

```python
y = x + x.T
```

**Confirmed by probe:** the task correctly collected two inputs with strides `(2, 1)` and `(1, 2)`, but the generated expression was equivalent to:

```c
in1[transposed_index] + in1[transposed_index]
```

`in0` was declared but never read. The second entry overwrote the first entry in `val_map` because both `BufferRef`s had the same base UOp identity.

This is not a local dictionary bug; it exposes the missing concept. A compiler value must be identified by both its storage and its view/layout. Possible designs:

- A first-class immutable `Layout(shape, strides, offset, mask)` attached to each expression edge.
- An indexed-expression IR where loads are explicit `(buffer, index, predicate, fallback)` values before arithmetic fusion.
- As a temporary safety rule, materialize when a kernel would consume incompatible views of the same base.

The first or second option is the durable fix.

### 2. Movement of an internal fused expression is dropped

Example:

```python
a = x + 1
b = a.T * 2
```

The scheduler traverses movement nodes and pulls both `ADD` and `MUL` into one kernel. `render_op_chain` then resolves the source of `MUL` through `PERMUTE` back to the scalar temporary produced by `ADD`. There is no index remapping of that temporary.

**Confirmed by probe:** the generated kernel computed:

```c
v0 = in0[gid] + 1;
v1 = v0 * 2;
```

The transpose vanished.

Movement over an in-register fused value is not the same operation as movement over a storage buffer. To fuse it correctly, code generation must evaluate the producer at the consumer's remapped index. Until an indexed IR supports that, nontrivial movement over computed values should be a kernel boundary.

### 3. Contiguity is not tracked correctly

`Tensor.is_contiguous` walks through `RESHAPE`, `EXPAND`, and `PERMUTE` and normally returns true unless it encounters padding. `BufferRef.from_uop` then handles every reshape by discarding prior strides and assigning fresh row-major strides.

This breaks cases such as:

```python
x.expand((2, 3)).reshape((6,))
x.T.reshape(...)
```

**Confirmed by probe:** a physical `(1, 3)` allocation expanded to `(2, 3)` and reshaped to `(6,)` became a `BufferRef` with strides `(1,)`. It can address elements 3 through 5 even though the base contains only three elements.

A reshape may be a view only if the old strides can be legally collapsed and split into the new dimensions. Broadcast dimensions with stride zero and arbitrary permutations often require materialization. `is_contiguous` should be a property of a concrete layout, not a walk over op categories.

### 4. Double-cast elimination is unsound

The optimizer implements:

```text
CAST(CAST(x, T1), T2) -> CAST(x, T2)
```

without conditions.

**Confirmed by probe:** `float32 -> int32 -> float32` was rewritten to a cast directly from the original float32 load. Values such as `1.9` therefore keep their fractional part rather than passing through integer truncation.

This rule is only legal when the intermediate conversion is proven lossless for the source/target relationship. The safe initial choice is to remove it. Later it can use explicit cast-kind and range facts.

### 5. Other algebraic rewrites need a numerical policy

- `x * 0 -> 0` is not IEEE-equivalent for NaN and infinity.
- `x + 0 -> x` can change signed-zero behavior.
- FMA contraction changes rounding compared with separate multiply and add.
- Python constant folding does not exactly model float16/float32 rounding or fixed-width integer behavior.

These can be useful optimizations, but Monograd should state whether it promises strict, contract-enabled, or fast-math semantics. A good model is an optimization context such as `STRICT`, `ALLOW_CONTRACT`, and `FAST_MATH`.

### 6. Dtype priority is not a sufficient promotion system

`most_upper_dtype` selects the dtype with maximum numeric priority. This produces results such as:

- `int64 + float16 -> float16`
- `int32 + uint32 -> uint32`

Python scalars are also created with the existing tensor's dtype before promotion. Thus an integer tensor combined with `0.5` can turn the scalar into integer zero rather than promoting the expression.

Each operation needs type inference that distinguishes:

- input/storage dtype
- compute dtype
- accumulator dtype
- output dtype
- supported combinations on the selected backend

This becomes especially important for half/BF16 inputs with float32 accumulation, quantized inference, and integer reductions.

### 7. The Tensor API accepts operations that OpenCL cannot render

Examples include float modulo and bitwise operations, integer `pow`, and integer matmul through an FMA-based kernel. `Ops.MAX` is listed as a binary ALU operation and used by the optimizer but has no `CL_OP` renderer. `RELU` emits `max(value, 0)`, which can create mismatched OpenCL overload types.

Validate operations before scheduling rather than waiting for string rendering or the OpenCL compiler to reject them.

### 8. Intermediate Tensor objects are incompletely initialized

Most operations use `Tensor.__new__` and set only `uop` and `requires_grad`.

**Confirmed by probe:** reading either `.grad` or `.name` on the result of `tensor + 1` raises `AttributeError`.

Use one internal Tensor constructor/factory that always establishes all object invariants. Avoid having leaf construction and operation-result construction initialize different object shapes.

## Runtime, device, and memory findings

### 1. Compiled artifact and invocation are conflated

The planned runtime cache needs two different concepts:

```text
KernelSpec / CompiledArtifact
    normalized source or IR
    device and driver fingerprint
    build options
    argument ABI
    launch constraints
    tunable parameters

KernelInvocation
    concrete input buffers/views
    output allocation
    scalar values
    dynamic local-memory sizes
    dependency events
```

Current `CompiledKernel.args` contains `BufferRef`s tied to specific UOps, so it is not naturally reusable across equivalent graph instances. The generated ABI also includes `n` and sometimes dynamic `__local` scratch storage, but `CompiledKernel` does not describe those arguments explicitly.

Kernel names are not safe cache keys because they omit input layouts, masks, reduction axes, pad values, device identity, build options, and tuning parameters. Caching by normalized source plus device/build fingerprint is a safer first implementation.

### 2. CPU execution and device transfer semantics are inconsistent

CPU is the default Tensor device, but computational code generation only emits OpenCL C. `COPY` is also rendered as an OpenCL elementwise kernel.

Cross-device copies should be runtime commands:

- CPU to GPU: host-to-device enqueue
- GPU to CPU: device-to-host enqueue
- GPU to GPU in one context: device copy, unless layout conversion requires a kernel
- CPU to CPU: host copy

An OpenCL kernel cannot directly consume a NumPy allocation as though it were a device buffer. Either implement a CPU backend or make GPU the only executable backend for now and explicitly treat CPU tensors as host data.

Matmul should also verify that both operands are on compatible devices/contexts, as binary elementwise operations already do.

### 3. One process-global OpenCL context is too restrictive

`OpenCLContext` caches one context and one queue using `create_some_context()`. This prevents clean handling of multiple GPUs, noninteractive selection, per-device capability queries, transfer queues, and context-specific cache keys.

A real `OpenCLDevice` object should own:

- platform/device handles
- context
- one or more queues
- capability data
- allocator
- compiler/program cache
- synchronization policy

`Device.GPU` is a device class, not a device identity.

### 4. Buffer views have ownership bugs

The base/view direction is useful, but current invariants are unsafe:

- A view reports allocated if its base is allocated, even when the view has no `_buf` attribute.
- A view allocation stores the exact same PyOpenCL wrapper in both base and view.
- Both destructors can call `release()` on the same wrapper.
- Bounds are not checked.
- Base/view dtype compatibility is not checked.
- Offsets are applied to host copy operations but not represented in `BufferRef` or kernel addressing.

Only one owner should release the native object. A view should contain an owner reference, byte offset, length, dtype, and layout. It should never independently own the same native handle.

### 5. Realization state is invisible to the scheduler

Buffers are associated with UOps through a weak side table, but the scheduler recognizes boundaries only by op kind. If an arithmetic UOp has already been realized, a later schedule can still fuse and recompute it. Rewriting can also replace a realized UOp with a new UOp that has no buffer association.

Storage and realization belong to runtime values, not purely semantic expression nodes. The scheduler needs an explicit input/materialization boundary supplied by the runtime or lowering pass.

### 6. UOp equality and hashing disagree

`UOp` receives structural equality from the dataclass but uses object identity as its hash. Two uncached, structurally equal LOAD nodes can compare equal while having different hashes, violating Python's hash contract.

Given side tables and materialization, identity equality and identity hashing are the safest current model. If structural expressions are desired, make all fields structurally hashable and keep storage outside them.

## Scheduler and code-generation findings

### 1. Full reduction is correct but structurally under-parallelized

The full reduction always launches one workgroup of 256 threads and produces one output. This is logically suitable for a scalar result, but large tensors use only one compute unit.

The future tuner cannot fix this merely by increasing global size: multiple workgroups would create multiple partial outputs. A second-stage reduction, atomic strategy, or device-specific cooperative mechanism is required. This is an algorithm choice, not a launch-parameter choice.

The tuner must also enforce power-of-two assumptions in the tree reduction, maximum workgroup size, per-dimension limits, and local-memory capacity.

### 2. BLAS is present but not integrated

The scheduler creates a `BLAS` task and `_codegen_blas` contains a substantial tiled kernel, but `codegen()` never dispatches `TaskKind.BLAS`.

**Confirmed by probe:** scheduling a 2x2 matmul produced a BLAS task, and calling `codegen()` raised the generic unhandled-task `RuntimeError`.

This is acceptable as unfinished work, but the broader structural lesson is that op support is distributed across enum membership, shape/device inference, scheduling, codegen, dtype legality, and runtime launch. A capability table or exhaustive lowering registry would make incomplete coverage explicit.

### 3. Internal operation metadata is too ad hoc

`UOp.arg` means unrelated things for different operations: device, shape, reduction axes, padding masks, and values. Shape and device are recovered through large op-specific conditionals.

Small typed argument records would improve validation and make cache-key construction much safer, for example:

- `LoadArg(shape, device)`
- `ReduceArg(axes, keepdims_shape)`
- `PadArg(shape, valid_ranges, value)`
- `CopyArg(destination_device)`

This keeps the compact UOp design while reducing tuple-position bugs.

### 4. Fan-out materialization is a reasonable starting heuristic

Materializing every multi-consumer ALU result avoids duplicated computation and keeps scheduling understandable. It is not always optimal: duplicating a cheap add can cost less than a VRAM write/read. Keep the current policy until correctness is stable, then add a simple cost model based on estimated arithmetic and bytes moved.

## Autograd and `nn`

The absence of backward is expected in an unfinished project. The concern is that the current data model does not yet reveal where backward should live.

Tensor operations retain only source UOps. `Tensor.parents` returns UOps, while `requires_grad`, `.grad`, names, and parameter identity exist only on Tensor wrappers. UOp interning can also cause multiple Tensor wrappers to share an expression node.

Choose one architecture deliberately:

1. Symbolically differentiate UOps, with explicit parameter/leaf UOps and a gradient graph.
2. Maintain a Tensor-level tape containing parent Tensor references and backward functions.
3. Produce a separate backward IR during forward construction.

For a tinygrad-like compiler, symbolic UOp differentiation is conceptually attractive, but leaf identity and realized-buffer ownership must be explicit.

The current `nn` and MNIST code should be treated as sketches rather than supported API: they reference nonexistent `monograd.ops`, `.data`, and `.backward()`. Quarantining them under an experimental label would make actual capability clearer.

## Recommended implementation order

1. Define `Layout(shape, strides, offset, mask)` and write down its invariants.
2. Fix view identity and stop keying loads solely by base UOp.
3. Make reshape legality layout-driven; insert contiguous materialization when required.
4. Prevent fusion across movement of computed values until indexed lowering supports it.
5. Separate semantic UOps, runtime values, compiled artifacts, and invocations.
6. Define backend transfer behavior and decide whether CPU execution is supported.
7. Repair buffer ownership, bounds, offsets, and synchronization lifetime.
8. Add dtype inference, operation legality, and accumulator types.
9. Remove unsound rewrites and introduce an explicit numerical policy.
10. Add exhaustive lowering/capability checks so unsupported operations fail early.
11. Implement multi-stage reductions.
12. Integrate GEMM, then tune it against queried device limits and measured profiles.
13. Choose and document the autograd architecture.
14. Only then prioritize pooling, more layers, broad autotuning, and API expansion.

## How to turn Monograd into a strong portfolio project

The highest-value additions are evidence, not surface area:

- A design note explaining UOp identity, Layout, RuntimeValue, KernelSpec, and KernelInvocation.
- A correctness case study for `x + x.T`, fused transpose, expand/reshape, padding, and offsets.
- A reduction case study comparing one-workgroup and two-stage strategies.
- A GEMM study using roofline reasoning, occupancy, local-memory use, register pressure, and achieved bandwidth/FLOPs.
- A profiler trace showing transfer, compilation, allocation, and kernel time separately.
- A numerical-semantics note covering dtype promotion, accumulation, FMA contraction, NaN, infinity, and signed zero.
- One non-Python systems component: for example a small C++/Rust OpenCL runtime, binary cache, allocator, or profiler integration.

Do not hide failed experiments. A brief explanation of why an optimization was rejected is strong evidence of engineering judgment.

## Relevant jobs and job titles

Monograd maps to a much wider market than “kernel engineer” or “framework engineer.” Titles vary heavily between companies, so search by both title and responsibility.

### Compiler and program transformation

- Machine Learning Compiler Engineer
- Deep Learning Compiler Engineer
- AI Compiler Engineer
- GPU Compiler Engineer
- Compiler Backend Engineer
- Compiler Performance Engineer
- MLIR/LLVM Compiler Engineer
- Tensor Compiler Engineer
- Graph Compiler Engineer
- JIT Compiler Engineer
- Programming Languages / Compiler Software Engineer

Typical work includes IR design, graph lowering, fusion, layout propagation, vectorization, code generation, register allocation, and target-specific optimization. Monograd's UOps, rewrites, scheduler, and code generator are directly relevant.

Current examples include Apple's GPU Compiler Engineer and Machine Learning Compiler Engineer roles, and NVIDIA's DL Compiler and backend compiler roles:

- [Apple GPU Compiler Engineer, Graphics, Game and ML](https://jobs.apple.com/en-ie/details/200649008-0836/gpu-compiler-engineer-graphics-game-and-ml)
- [Apple Machine Learning Compiler Engineer](https://jobs.apple.com/en-us/details/200595326/machine-learning-compiler-engineer)
- [NVIDIA Senior Software Engineer, DL Compilers](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Senior-Software-Engineer--DL-Compilers_JR2015756)
- [NVIDIA Compiler Engineer, Backend — New College Grad](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Compiler-Engineer--Backend--New-College-Grad-2026_JR2017290)

### Accelerator runtimes, drivers, and systems software

- GPU Compute Runtime Engineer
- Accelerator Runtime Engineer
- Language Runtime Engineer
- GPU Driver Engineer
- User-Mode Driver Engineer
- Heterogeneous Compute Systems Engineer
- Device Runtime Engineer
- Firmware / Runtime Integration Engineer
- Systems Software Engineer, GPU Compute

These roles work on queues, memory allocation, virtual memory, synchronization, command submission, driver/runtime interfaces, device bring-up, and profiling. The Buffer, device, transfer, and future runtime parts of Monograd are the relevant foundation.

An example is AMD's GPU language-runtime role, which explicitly spans runtime components, driver interfaces, hardware counters, and ASIC bring-up:

- [AMD Lead Runtime Engineer](https://careers.amd.com/careers-home/jobs/78150?lang=en-us)

### AI performance and performance modeling

- Deep Learning Performance Engineer
- ML Systems Performance Engineer
- GPU Performance Engineer
- Compiler Performance Engineer
- Performance Software Engineer
- Performance Architect
- Workload Characterization Engineer
- AI Benchmarking Engineer
- Systems Performance Analyst/Engineer

These jobs profile end-to-end workloads, construct roofline or architectural models, find bottlenecks across frameworks/runtimes/kernels, and maintain regression methodology. This is a particularly good match if the enjoyable part of Monograd is asking “why is this slow?” rather than implementing a particular layer.

Examples:

- [NVIDIA Deep Learning Performance Architect](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Deep-Learning-Performance-Architect_JR2019478)
- [AMD Triton Compiler/GPU Kernel Performance Engineer](https://careers.amd.com/careers-home/jobs/79527?lang=en-us)
- [AMD Compiler Performance Engineer](https://careers.amd.com/careers-home/jobs/86163?lang=en-us)

### Inference and training systems

- ML Systems Engineer
- AI Systems Engineer
- LLM Inference Engineer
- Inference Runtime Engineer
- Model Serving Engineer
- Distributed Inference Engineer
- Distributed Training Engineer
- Training Performance Engineer
- AI Infrastructure Engineer
- GPU Networking / Communication Systems Engineer

These roles may barely touch compiler code. They focus on batching, scheduling, KV-cache management, memory pressure, distributed execution, collectives, RDMA, fault tolerance, latency, and throughput. The graph scheduling, memory planning, runtime, and performance skills from Monograd transfer well.

Examples:

- [AMD LLM Inference Framework Engineer](https://careers.amd.com/jobs/76255?lang=en-us)
- [AMD GPU Software Performance Engineer — Post-Training](https://careers.amd.com/careers-home/jobs/80742?lang=en-us)
- [AMD High-Performance GPU Communication / Inference Infrastructure](https://careers.amd.com/careers-home/jobs/84428?lang=en-us)

### Numerical libraries and high-performance computing

- GPU Math Libraries Engineer
- Numerical Software Engineer
- Linear Algebra Library Engineer
- HPC Software Engineer
- Parallel Computing Engineer
- Scientific Computing Engineer
- BLAS/DNN Library Engineer
- Collective Communications Library Engineer

Work can include GEMM, convolutions, FFTs, sparse operations, numerical accuracy, MPI/NCCL/RCCL, CPU vectorization, and GPU kernels. Monograd's reduction and GEMM work is directly relevant, but these roles usually expect deeper numerical analysis and benchmarking.

### Hardware/software co-design and architecture

- Hardware-Software Co-Design Engineer
- ML Accelerator Software Engineer
- AI Performance Architect
- GPU/AI System Architect
- Workload Modeling Engineer
- Architecture Performance Engineer
- Pre-Silicon Performance Engineer
- Accelerator Modeling Engineer

These engineers use real workloads and performance models to influence caches, memory systems, instructions, dataflow, synchronization, and accelerator features. They frequently bridge framework, compiler, runtime, firmware, and silicon teams.

AMD currently describes such a role as working across Linux DRM/GEM, ROCm runtime, IREE, PyTorch, and Triton while feeding workload evidence into GPU architecture:

- [AMD Hardware-Software Co-Design](https://careers.amd.com/careers-home/jobs/80662?lang=en-us)

### On-device and embedded ML

- On-Device ML Systems Engineer
- Edge AI Runtime Engineer
- Embedded ML Engineer
- Mobile ML Performance Engineer
- Neural Engine / NPU Software Engineer
- Model Optimization Engineer
- ML Deployment Engineer

The work includes heterogeneous CPU/GPU/NPU execution, power and memory limits, quantization, graph partitioning, model compression, and runtime integration.

Apple's current on-device compiler role explicitly covers heterogeneous execution, runtime performance, model compression, benchmarking, and debugging tooling:

- [Apple On-Device ML Compiler Engineer](https://jobs.apple.com/en-ca/details/200631247/on-device-ml-compiler-engineer-model-compilation-graphics-games-and-machine-learning)

### Validation, tooling, and developer infrastructure

- Compiler Verification Engineer
- GPU Software Validation Engineer
- ML Systems Quality Engineer
- Performance Regression Engineer
- Developer Tools Engineer
- GPU Profiler Engineer
- Benchmark Infrastructure Engineer
- Hardware Lab / Systems Validation Engineer
- Release Infrastructure Engineer for AI systems

These are serious systems roles, not secondary testing jobs. They build differential testing, fuzzing, performance gates, distributed hardware labs, tracing, bisection, stress testing, and release qualification. Someone who enjoys finding failures like Monograd's view/fusion cases could be very good at this work.

Examples:

- [NVIDIA Deep Learning Compiler Verification and Infra Intern](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Deep-Learning-Compiler-Verification-and-Infra-Development-Intern---2026_JR2007481)
- [AMD AI/ML System-Level Validation](https://careers.amd.com/careers-home/jobs/86351?lang=en-us)

### Research-oriented roles

- ML Systems Research Engineer
- Compiler Research Engineer
- Research Engineer, Efficient AI
- Applied Scientist, Compilers
- Research Scientist, Computer Architecture
- Research Engineer, Hardware-Aware ML

These roles study new scheduling, compilation, numerics, accelerator, and distributed-system techniques. Research scientist positions often prefer a graduate degree/publications, while research engineer roles may value strong systems implementation and open-source contributions just as much.

NVIDIA currently advertises applied compiler research spanning low-level optimization and learning-guided compiler decisions:

- [NVIDIA Senior AI Compiler Engineer — Applied Research](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/AI-Research-Engineer---Applied-Scientist-Compilers_JR2016804-1)

## Which paths fit Monograd best?

Based on the code as it exists, the closest matches are:

1. **ML compiler / tensor compiler engineering** — strongest direct match.
2. **GPU runtime or accelerator systems software** — especially after repairing device, buffer, and invocation design.
3. **Deep learning performance engineering** — especially after adding profiling and quantitative case studies.
4. **Inference runtime / ML systems engineering** — if the scheduler expands toward memory planning, batching, and execution.
5. **Hardware-software co-design** — if you enjoy architecture and performance modeling.
6. **Compiler or GPU validation infrastructure** — the ability to generate minimal semantic counterexamples is highly valuable.

Kernel engineering is only one branch among these.

## Skills to add next for employability

### Core systems

- Modern C++ or Rust, including ownership, RAII, threading, and FFI
- Linux processes, virtual memory, files, dynamic libraries, and debugging
- Native build systems and toolchains
- Concurrency, atomics, memory ordering, and synchronization
- Profilers, tracing, hardware counters, and reproducible benchmarking

### Compiler

- LLVM fundamentals
- MLIR dialects, passes, rewrite patterns, and lowering
- Dataflow analysis, dominance, SSA, alias analysis, and liveness
- Register allocation and instruction scheduling at a conceptual level
- One open-source compiler contribution, even a small diagnostic or correctness fix

### Architecture and parallel computing

- Cache/memory hierarchy, SIMD/SIMT, occupancy, divergence, coalescing
- CUDA or HIP in addition to OpenCL
- CPU SIMD and multithreading so GPU tradeoffs have a comparison point
- Roofline analysis and arithmetic intensity
- Floating-point behavior and numerical stability

### ML systems

- PyTorch internals and custom operators
- ONNX and model graph semantics
- Triton, TVM, IREE, XLA, or TorchInductor
- vLLM/SGLang concepts: paged KV cache, continuous batching, prefill/decode
- Distributed fundamentals: collectives, NCCL/RCCL, MPI, RDMA, topology

Do not try to learn all of these simultaneously. A good sequence is:

1. Repair Monograd's layout and runtime-value model.
2. Reimplement a small runtime component in C++ or Rust.
3. Profile and document reduction/GEMM behavior.
4. Learn MLIR/LLVM by reproducing one Monograd lowering concept there.
5. Contribute one focused fix to a related open-source system.

## Final assessment

For someone newly entering systems programming and low-level deep learning, Monograd is a very good choice of project and already shows stronger instincts than a conventional “build a neural network from scratch” exercise. It crosses meaningful abstraction boundaries and contains enough original machinery to expose real compiler/runtime failures.

The next leap is not more code. It is making the semantics explicit:

- What exactly identifies a value?
- What exactly identifies storage?
- When is a layout a view, and when is it a copy?
- Which rewrites preserve the chosen numerical contract?
- What belongs to a reusable compiled artifact versus one execution?
- Who owns every native resource, and when may it be released?

Answering those questions in code would turn Monograd from an interesting tinygrad-inspired prototype into a genuinely strong systems portfolio project.
