# Monograd — Codebase Review

## Overview

Monograd is a deep learning framework mid-refactor between two generations of code. The new compiler-style core (`uop/`, `engine/`, `tensor.py`, `device.py`, `dtype.py`) is architecturally sound and clearly tinygrad-inspired. The old `nn/` layer (`Conv2d`, `MaxPool2d`, optimizers, `mnist.py`) is completely broken and references a deleted API.

The compiler pipeline you chose — `Tensor → UOp DAG → rewrite_graph → run_scheduler → codegen` — is the right architecture. It gives you the most leverage over what actually runs on hardware and is exactly how modern frameworks approach this. The bones are good.


---

## Directory Structure

```
monograd/
├── .github/workflows/ci.yml         # CI pipeline (CPU + GPU test jobs)
├── examples/
│   └── mnist.py                     # CNN MNIST training example (OLD API — broken)
├── extra/
│   ├── pytorch_matmul_cuda_tflops.py # PyTorch GEMM benchmark
│   ├── reduce_kernel_full.cl         # Reference OpenCL reduce kernel (template)
│   └── reduce_kernel_strided.cl      # Reference OpenCL strided reduce kernel (template)
├── monograd/
│   ├── __init__.py
│   ├── device.py                     # Device enum + Buffer (CPU/GPU allocation)
│   ├── dtype.py                      # DType system (singleton metaclass)
│   ├── tensor.py                     # Tensor class (wraps UOp, broadcasts, ops)
│   ├── utils.py                      # DEBUG env vars, toposort, im2col (orphaned)
│   ├── engine/
│   │   ├── schedule.py               # Scheduler: UOp DAG → KernelTask list
│   │   ├── optimize.py               # Graph rewriter: UPat pattern matching
│   │   └── codegen.py                # OpenCL C code generator
│   ├── mixin/
│   │   ├── __init__.py               # OpMixin = MathMixin + MovementMixin
│   │   ├── math.py                   # Arithmetic ops (add, mul, sub, div, relu, etc.)
│   │   └── movement.py               # Shape ops (reshape, expand, permute, pad, transpose)
│   ├── nn/
│   │   ├── __init__.py               # Module, Linear, Conv2d, MaxPool2d (OLD API — broken)
│   │   ├── optim.py                  # SGD + Adam stub (OLD API — broken)
│   │   └── datasets.py               # MNIST downloader
│   └── uop/
│       ├── __init__.py               # Ops enum + GroupOp classification
│       └── ops.py                    # UOp dataclass + UOpMetaClass (singleton cache)
├── paper.md
├── TODO.md
├── README.md
├── pyproject.toml
└── flags.txt
```

---

## Architecture Assessment

### What is correct

| Decision | Notes |
|---|---|
| UOp singleton cache via `WeakValueDictionary` | Structural sharing for free. Same sub-expression built twice returns same object — correct. |
| Movement ops as stride transforms in `BufferRef` | Zero-copy reshape/permute/expand. The analytical stride computation is the right call. |
| Greedy backward fusion in scheduler (`_pull`) | Avoids intermediate buffers for ALU chains before a reduce. Right approach. |
| Fan-out materialization | Prevents recomputation when a node has multiple consumers. Correct. |
| Fixed-point UPat rewriter | Clean, declarative rule system. Good structure. |
| Tiled GEMM kernel | TSM=128/TSN=128/TSK=16 with register tiling is solid for small nets. |
| `arg` encoding per op type | Keeps `UOp` a single frozen dataclass without subclassing. Tinygrad does the same. |
| `DTypeMetaClass` singleton | Type identity comparisons via `is` — correct. |
| Frozen `@dataclass(slots=True)` for `UOp` | Memory-efficient, fast attribute access, correct for an IR node. |

### What is wrong structurally

**The `arg` encoding is undocumented at the call site.** Every op type interprets `arg` differently (device, shape tuple, value, axes, pad mask) with no type annotation or docstring enforcing the contract. This is fine if you're the only author and you know the convention, but it is a latent bug magnet. At minimum add a docstring to `UOp` enumerating the `arg` format per op.

**`BufferRef` strides are computed by replaying the movement chain but SHRINK is not handled.** If a SHRINK UOp ever appears in the chain (and it will if you implement `__getitem__`), `BufferRef.from_uop` silently ignores it or falls through to the wrong case.

**The scheduler's `_pull` recursion has no depth limit.** For very deep ALU chains this will hit Python's default recursion limit. Tinygrad converts this to an explicit stack.

---

## Critical Bugs

### 1. No execution path (fatal)

`codegen()` returns `CompiledKernel` objects but nothing ever enqueues them. `device.py` creates an OpenCL context and command queue and uses them for `copyin`/`copyout` (host↔device transfers) but never for kernel execution. There is no `.realize()`, `.numpy()`, or `.item()`. The entire new engine is unreachable from user code.

The fix is a `realize()` method on `Tensor` that runs: `run_scheduler → rewrite_graph per task → codegen → cl.enqueue_nd_range_kernel → copyout`. This single method unlocks the entire system.

### 2. No backward pass (fatal)

Zero backward implementation. No `backward()`, no gradient accumulation, no gradient computation for any op. The README advertises "Reverse-mode autograd differentiation" — this is currently false.

The UOp DAG is well-suited for this. You need to:
- Add `requires_grad` tracking to `Tensor`
- After `realize()`, walk the UOp DAG in reverse topological order
- For each op, apply the analytic gradient rule to accumulate into `.grad`

### 3. `UOp.__hash__` / `__eq__` contract violation for LOAD ops

`@dataclass(eq=True)` gives field-based `__eq__`. `__hash__ = lambda self: id(self)` gives identity-based hash. For cached ops this is harmless (same args → same object). For LOAD ops (excluded from cache), two LOADs with identical fields compare `==` but have different hashes. This violates Python's fundamental requirement that `a == b` implies `hash(a) == hash(b)`. Placing LOAD ops in sets or as dict keys produces silently incorrect behavior.

Fix: either use `__hash__ = object.__hash__` everywhere (identity semantics) and only compare structurally via a separate method, or define `__eq__` to be identity-based as well for consistency.

### 4. `nn/` layer is fully broken

- `monograd/nn/__init__.py` line 1: `import monograd.ops as Ops` — this module does not exist. The entire file fails at import.
- All optimizers in `optim.py` use `p.data` and `p.grad.data` — `Tensor` has no `.data` property.
- `Adam` is literally `class Adam(Optimizer): pass`.
- `examples/mnist.py` uses `loss.backward()`, `logits.data`, `tensor.data` — none exist on the new `Tensor`.

### 5. BLAS codegen bypasses `BufferRef` (silent wrong results)

`_codegen_blas` computes A/B fetch indices with hardcoded row-major arithmetic (`global_rowA * K + global_colA`) instead of using `BufferRef.load_expr`. If either input matrix was permuted or transposed before the matmul, the kernel reads wrong memory locations and produces wrong results silently. This is a real issue since `matmul` is commonly preceded by a transpose (e.g., `(x @ w.T)`).

### 6. Float-type-specific C literals used for all dtypes

Several `CL_OP` entries use `f`-suffixed float32 literals:
- `RECIP`: `(1.0f / a)` — truncates to float32 precision for float64 inputs
- `EXP`: multiplies by `1.4426950408889634f` — same problem
- `LOG`: multiplies by `0.6931471805599453f` — same problem
- `RELU`: `max(a, 0)` — integer `0` can cause type ambiguity with `half` in some OpenCL implementations

Fix: use `cl_const(val, dtype)` (which you already have) to emit correctly-typed literals.

---

## Design Problems

### 7. `CONTIGUOUS` wasted arg field

`_unop` stores `self.device` in `arg` for all unary ops including `CONTIGUOUS`. But `UOp.device` for `CONTIGUOUS` reads from `src[0].device`, not `arg`. The stored device in `arg` is never read. Minor, but architecturally inconsistent.

### 8. `SHRINK` is declared but entirely dead

`Ops.SHRINK` is in the enum and `GroupOp.Movement`, recognized by the scheduler as invisible. But there is no `shrink()` method on `Tensor`/`MovementMixin`, no `BufferRef.from_uop` handling for SHRINK, and no codegen case. It exists only in the enum. If you want slicing (`tensor[2:5]`), this is where it goes — but right now it's a phantom.

### 9. `_ternop` always raises `NotImplementedError`

`Tensor._ternop()` raises `NotImplementedError("need 3-way broadcast function")`. Yet `Ops.WHERE` is in `CL_OP` and `GroupOp.Ternary`, and `MULACC` is generated by the optimizer. There is no user-facing path to create a `WHERE` UOp. This means no conditional masking, no `where()` op.

### 10. `REDUCE_FULL` single-workgroup performance bottleneck

`_codegen_reduce_full` always launches exactly one workgroup of 256 threads regardless of tensor size. For a tensor with 10 million elements, 256 threads process ~39,000 elements each serially before the local tree reduction. This is catastrophically slow compared to a multi-workgroup parallel reduction with partial sums. The reference kernels in `extra/` document this limitation explicitly.

### 11. No kernel caching

Every `codegen()` call recompiles from scratch. `cl.Program.build()` is expensive (100ms+ per kernel on some drivers). For a training loop this means paying full compilation cost every forward pass. `CompiledKernel` even has a `# NOTE: Should this be cached???` comment acknowledging this.

Fix: cache compiled `cl.Program` objects keyed by source string hash. The UOp singleton cache already ensures identical computations produce structurally identical graphs — so the same operation always produces the same source, making this a straightforward `dict` cache.

### 12. No memory pool

`Buffer.allocate()` makes a fresh `cl.Buffer` allocation every time. The TODO correctly identifies this causes VRAM fragmentation and overhead. For training loops that allocate/free the same shapes repeatedly, a simple pool keyed by `(size, device)` reclaims most of this overhead.

### 13. `dtypes.bool` uses C type `"char"`

`dtypes.bool = DType(0, 1, "char", '?')`. The C type name emitted is `"char"` (OpenCL `char`), which is a signed 8-bit integer, not a boolean. OpenCL has `bool` as a keyword — using `char` for boolean semantics works in practice but is semantically wrong and could produce unexpected results with relational ops that return `int`.

---

## Code Quality Issues

### 14. Dual dead debug system

`utils.py` contains both:
- `ContextVar("DEBUG", 0)` — new system, used throughout the codebase
- `DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("true", "1", "t")` + `dbg()` function — old system, completely unused

Delete the old system.

### 15. `im2col` utilities are orphaned

`get_im2col_indices`, `im2col_indices`, `col2im_indices` in `utils.py` are NumPy-based convolution helpers from the pre-refactor autograd backend. They are unreachable from the new UOp system. Either move them to `extra/` (for reference) or delete them.

### 16. `Module.parameters()` uses `inspect.getmembers`

`inspect.getmembers(self)` triggers all property getters and is O(n) in all attributes. Standard approach is traversing `__dict__` recursively. This is slow for any non-trivial model.

### 17. Inconsistent indentation

`nn/` and `examples/` use 4-space indentation (old style). Core engine files (`tensor.py`, `engine/`, `uop/`) use 2-space. Pick one and enforce it.

### 18. `get_broadcasted_shape` cache grows unbounded

`@functools.cache` on `get_broadcasted_shape` with tuple arguments works but is unbounded. For dynamic networks with many unique shapes this is a slow memory leak. `@functools.lru_cache(maxsize=256)` is a safer choice.

### 19. `GEMM` op declared but never generated

`Ops.GEMM` exists in the enum and `GroupOp.BLAS` but nothing ever creates a GEMM UOp. `MATMUL` is used instead. This appears to be a planned future op. Fine, but worth noting as dead enumeration.

---

## Priority Roadmap

For a "usable framework for small nets" this is the order that matters:

```
BLOCKING (nothing works without these)
  1. realize() → launch compiled kernels → copyout → numpy()
  2. Backward pass implementation
  3. Fix nn/ layer to use new Tensor API
  4. Implement Adam

HIGH (correctness issues)
  5. Fix BLAS codegen to use BufferRef for transposed inputs
  6. Fix LOAD UOp hash/eq contract
  7. Fix float-type-specific C literals (use cl_const everywhere)

MEDIUM (performance)
  8. Kernel caching (compile once per source hash, reuse)
  9. Multi-workgroup REDUCE_FULL
  10. Memory pool for Buffer.allocate()

LOW (cleanup)
  11. Delete dead code (im2col utilities, old debug system)
  12. Implement SHRINK for tensor slicing
  13. Implement WHERE / _ternop for conditional ops
  14. Fix dtypes.bool C type from "char" to "bool"
  15. Document arg encoding per op in UOp docstring
  16. Convert scheduler _pull recursion to explicit stack
```

---

---

## Proposed README.md

---

<div align="center">
  <h1>monograd</h1>

  Something between [PyTorch](https://github.com/pytorch/pytorch) and [tinygrad](https://github.com/tinygrad/tinygrad)
</div>

---

### Monograd supports
- **Tensor library** and API with NumPy interop
- Reverse-mode **autograd** differentiation over a lazy UOp DAG
- **OpenCL GPU kernels** with JIT compilation and kernel caching
- **nn / optim / datasets** for training real networks

The architectural foundation of monograd strictly follows a modern compiler-based design: tensors are lazy graph nodes. No computation runs until `.realize()` is called — at which point the UOp graph is optimized, scheduled into kernels, compiled to OpenCL C, cached, and executed on the target device.

It is a lightweight deep learning framework built from scratch. Designed to be readable and hackable, small enough to understand entirely, capable enough to train real networks.

---

### How it works

```
Tensor ops  →  UOp DAG  →  rewrite_graph  →  run_scheduler  →  codegen  →  OpenCL kernel
                                (algebraic         (fusion +         (tiled GEMM,
                               simplification)     scheduling)       reduce, elementwise)
```

Operations on `Tensor` build a lazy computation graph. Calling `.realize()` triggers the full compiler pipeline: the graph is algebraically simplified, fused into kernels, compiled to OpenCL C (with the result cached), and executed on device. Results are read back via `.numpy()`.

---

### Making a simple neural net in monograd

```python
from monograd.tensor import Tensor
from monograd.nn import Module, Linear
from monograd.nn import optim

class MyNet(Module):
    def __init__(self):
        self.l1 = Linear(784, 128)
        self.l2 = Linear(128, 10)

    def forward(self, x):
        x = self.l1(x).relu()
        x = self.l2(x)
        return x

model = MyNet()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for x_batch, y_batch in dataloader:
    optimizer.zero_grad()
    loss = model(x_batch).cross_entropy(y_batch)
    loss.backward()
    optimizer.step()
```

---

### NumPy interop

```python
import numpy as np
from monograd.tensor import Tensor

# From NumPy
x = Tensor(np.random.randn(32, 784).astype(np.float32))

# Back to NumPy (triggers realize)
result = (x @ x.T).numpy()
```

---

### monograd vs PyTorch

```python
from monograd.tensor import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = Tensor([[2.0, 0.0, -2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad.numpy())  # dz/dx
print(y.grad.numpy())  # dz/dy
```

Same thing in PyTorch:

```python
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = torch.tensor([[2.0, 0.0, -2.0]], requires_grad=True)
z = y.matmul(x).sum()
z.backward()

print(x.grad)
print(y.grad)
```

---

### Installing monograd

```bash
pip install monograd
```

Requires `numpy`. GPU support requires `pyopencl` and an OpenCL-capable device (most GPUs and CPUs).

```bash
pip install monograd[gpu]  # includes pyopencl
```

---

### Available Optimizers
- **SGD** — Stochastic Gradient Descent with momentum
- **Adam** — Adaptive Moment Estimation

### Available Network Layers
- **Linear**
- **Conv2d**
- **MaxPool2d**

### Available Ops
- Arithmetic: `add`, `sub`, `mul`, `div`
- Unary: `relu`, `log`, `exp`, `sqrt`, `sin`, `recip`, `neg`
- Reduce: `sum`, `max`
- Movement: `reshape`, `expand`, `permute`, `transpose`, `pad`
- Linear algebra: `matmul`
- Type: `cast`

---

### Device selection

```python
from monograd.tensor import Tensor
from monograd.device import Device

# CPU (default)
x = Tensor([1.0, 2.0, 3.0], device=Device.CPU)

# GPU via OpenCL
x = Tensor([1.0, 2.0, 3.0], device=Device.GPU)
```

---

### Running tests

```bash
chmod +x test-all.sh
./test-all.sh
```

GPU tests require an OpenCL device and run automatically if one is detected. Set `TEST_GPU=1` to force GPU tests.

### Run MNIST

```bash
PYTHONPATH=. python3 examples/mnist.py
```

---

### Roadmap

- [x] Core tensor ops (add, mul, relu, reshape, permute, matmul, reduce)
- [x] UOp lazy DAG with structural sharing (singleton cache)
- [x] Algebraic graph optimizer (UPat fixed-point rewriter)
- [x] Kernel scheduler with elementwise fusion
- [x] OpenCL C code generator (elementwise, reduce, tiled GEMM)
- [x] CPU and GPU buffer management
- [ ] `.realize()` execution path + kernel launcher
- [ ] `.numpy()` / NumPy retrieval
- [ ] Backward pass (reverse-mode autograd over UOp DAG)
- [ ] JIT kernel caching (compile-once per source hash)
- [ ] Adam optimizer
- [ ] Conv2d + MaxPool2d on new API
- [ ] Memory pool for GPU buffer reuse
- [ ] Multi-workgroup parallel reduction
- [ ] Tensor slicing (`SHRINK` op)
- [ ] Conditional ops (`WHERE`)
