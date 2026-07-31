from dataclasses import dataclass
from typing import Callable
from math import prod
from monograd.engine.schedule import BufferRef, KernelTask, TaskKind
from monograd.dtype import DType, ConstType, DTypeLike, dtypes, to_dtype
from monograd.uop.ops import UOp, identity_element
from monograd.uop import GroupOp, Ops
from monograd.utils import DEBUG

# NOTE: We shouldnt launch kernels with large global sizes
# query gpu to figure out limits and adapt (also shouldnt launch not multiple of 2 global sizes)
# hardcoding _local_size_tmp = 256 is safe, but sub-optimal.
# If a user runs tensor.sum() on a tensor with exactly 64 elements, we will spawn 256 threads.
  # 192 of them will do zero work but will still consume hardware registers and participate in the barrier() syncs.
  # The local_size should eventually be calculated dynamically based on min(next_power_of_2(n), MAX_WORK_GROUP_SIZE).
_local_size_tmp = 256

C_KERNEL_TEMPLATE = """{pragmas}
__kernel void {name}({args}__global {out_dtype}* out{extra_args}, const int n) {{
{body}
}}"""

@dataclass
class CompiledKernel: # NOTE: Should this be cached???!!!!!!
  source: str
  name: str
  global_size: tuple[int, ...]        # OpenCL global work size
  local_size: tuple[int, ...] | None  # None = let driver decide
  args: list[BufferRef]               # inputs in argument order
  output_shape: tuple[int, ...]
  output_dtype: DType 

  def __repr__(self):
    return f"CompiledKernel(global_size={self.global_size}, local_size={self.local_size}, output_shape={self.output_shape}, source=\n{self.source})"

CL_OP: dict[Ops, Callable] = {
  # binary
  Ops.ADD:    lambda a, b:    f"({a} + {b})",
  Ops.MUL:    lambda a, b:    f"({a} * {b})",
  Ops.MAX:    lambda a, b:    f"max({a}, {b})",
  Ops.MOD:    lambda a, b:    f"({a} % {b})",
  Ops.POW:    lambda a, b:    f"pow({a}, {b})",
  Ops.XOR:    lambda a, b:    f"({a} ^ {b})",
  Ops.OR:     lambda a, b:    f"({a} | {b})",
  Ops.AND:    lambda a, b:    f"({a} & {b})",
  # unary
  Ops.RECIP:  lambda a:       f"(1.0f / {a})",
  Ops.SIN:    lambda a:       f"sin({a})",
  Ops.SQRT:   lambda a:       f"sqrt({a})",
  Ops.EXP:    lambda a:       f"exp2({a} * 1.4426950408889634f)",   # exp(x) = exp2(x / ln(2)) = exp2(x * log2(e))
  Ops.LOG:    lambda a:       f"(log2({a}) * 0.6931471805599453f)", # log(x) = log2(x) * ln(2)
  # Ops.CAST:   None,         # handled in *cl_const()* 
  # ternary
  Ops.MULACC: lambda a, b, c: f"fma({a}, {b}, {c})",
  Ops.WHERE:  lambda a, b, c: f"({a} != 0 ? {b} : {c})", # a != 0 allows a to be valid in all/most cases
  # reduce
  Ops.REDUCEMAX:    lambda a, b:    f"max({a}, {b})",
  Ops.SUM:          lambda a, b:    f"{a} + {b}", # this is computed in parallel
}


# *** rendering helpers ****
def cl_type(dtype:DType) -> str: return dtype.name # NOTE: Cache??
def cl_const(val:ConstType, dtype:DType) -> str: # NOTE: Cache??
  if val == -float('inf'): return "-INFINITY"
  if val == float('inf'): return "INFINITY"
  if dtype is dtypes.bool: return "1" if val else "0"
  if dtypes.is_float(dtype): # handle weirdness of c99 llvm when casting floats
    if dtype is dtypes.half: return f"{float(val)}h"
    if dtype is dtypes.float: return f"{float(val)}f" # opencl does (half)1.0f
    if dtype is dtypes.double: return f"{float(val)}"
  assert not dtypes.is_float(dtype), f"unhandled float dtype: {dtype}"
  return f"({cl_type(dtype)})({int(val)})"
def render_pragmas(task:KernelTask) -> str:
  lines:list[str] = []
  all_dtypes:set[DType] = {task.output_dtype} | {ref.uop.dtype for ref in task.inputs} | {op.dtype for op in task.ops}
  seen:set[int] = set()
  stack = [src for op in task.ops for src in op.src]
  while stack:
    uop = stack.pop()
    if id(uop) in seen: continue
    seen.add(id(uop)); all_dtypes.add(uop.dtype); stack.extend(uop.src)
  if dtypes.float16 in all_dtypes: lines.append("#pragma OPENCL EXTENSION cl_khr_fp16 : enable")
  if dtypes.float64 in all_dtypes: lines.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable")
  return "\n".join(lines) + ("\n" if len(lines) > 0 else "")
def kernel_name(task:KernelTask) -> str:
  kind = task.kind.name.lower()
  ops = "_".join(u.op.name.lower() for u in task.ops)
  shape = "x".join(str(s) for s in task.output_shape)
  dtype = task.output_dtype.name.lower() + str(task.output_dtype.bitsize)
  return f"{dtype}_{kind}_{ops}_{shape}"
def _resolve_base(uop:UOp) -> UOp:
  cur = uop
  while cur.op in GroupOp.Movement: cur = cur.src[0]
  return cur
def render_op(uop:UOp, src_exprs:list[str]) -> str:
  if uop.op is Ops.CONST: return cl_const(uop.arg[0], uop.dtype) 
  if uop.op is Ops.CAST:
    assert len(src_exprs) == 1, "cast received more than 1 arg when rendering"
    return f"({cl_type(uop.dtype)}){src_exprs[0]}"
  if uop.op is Ops.RELU:
    assert len(src_exprs) == 1, "relu received more than 1 arg when rendering"
    return f"max({src_exprs[0]}, {cl_const(0, uop.dtype)})"
  assert uop.op in CL_OP, f"unhandled op in render_op: {uop.op}"
  return CL_OP[uop.op](*src_exprs)
def render_kernel(task:KernelTask, body:str, global_size:tuple, local_size:tuple|None=None, extra_args:str="", name_suffix:str="") -> CompiledKernel:
  name = kernel_name(task) + name_suffix
  args = ", ".join(f"__global const {cl_type(b.uop.dtype)}* in{i}" for i, b in enumerate(task.inputs))
  if args: args += ", "
  source = C_KERNEL_TEMPLATE.format(
      pragmas=render_pragmas(task),
      name=name,
      args=args,
      out_dtype=cl_type(task.output_dtype),
      body=body,
      extra_args=extra_args) # extra_args mainly used for __local stuff
  if DEBUG >= 1: print(source)
  return CompiledKernel(source, name, global_size, local_size, task.inputs, task.output_shape, task.output_dtype)


# **** actual codegen ****
def render_op_chain(uops:list[UOp], val_map:dict[int, str]) -> list[str]: # elem-wise kernel body
  lines:list[str] = []
  for i, op in enumerate(uops):
    var = f"v{i}"
    src_exprs:list[str] = []
    for src in op.src:
      _base = _resolve_base(src)
      if _base.op is Ops.CONST: src_exprs.append(cl_const(_base.arg[0], _base.dtype))
      elif id(_base) in val_map: src_exprs.append(val_map[id(_base)])
      else: raise RuntimeError(f"src {_base.op} not in val_map - toposort broken?")
    lines.append(f"{cl_type(op.dtype)} {var} = {render_op(op, src_exprs)};")
    if DEBUG >= 5: print(f"{op.op.name}({op.src}) -> {lines[-1].strip()}")
    val_map[id(op)] = var
  return lines
def codegen(task:KernelTask) -> CompiledKernel:
  if task.kind is TaskKind.ELEMENTWISE: return _codegen_elementwise(task)
  if task.kind is TaskKind.REDUCE_FULL: return _codegen_reduce_full(task, _local_size_tmp)
  if task.kind is TaskKind.REDUCE_STRIDED:
    reduce_axes = task.ops[-1].arg[0]
    if isinstance(reduce_axes, int): reduce_axes = (reduce_axes,)
    # Check if any of the reduced axes have a stride of 1 in the primary input
    primary_strides = task.inputs[0].strides
    if any(primary_strides[a] == 1 for a in reduce_axes): return _codegen_reduce_strided_inner(task, _local_size_tmp)
    else: return _codegen_reduce_strided_outer(task)
  if task.kind is TaskKind.COPY: return _codegen_copy(task)
  if task.kind is TaskKind.BLAS: return _codegen_matmul(task)
  raise RuntimeError(f"how come i didnt get treated? {task};kind={task.kind}")
def _codegen_elementwise(task:KernelTask) -> CompiledKernel:
  n:int = prod(task.output_shape)
  val_map:dict[int, str] = {id(b.uop): b.load_expr(f"in{i}", 'gid') for i, b in enumerate(task.inputs)}
  math_lines:list[str] = render_op_chain(task.ops, val_map)
  source:str = f"""  int gid = get_global_id(0);
  if (gid >= n) return;
  {'\n  '.join(math_lines)}
  out[gid] = {val_map[id(task.ops[-1])]};"""
  return render_kernel(task, source, global_size=(n,))
def _codegen_reduce_full(task:KernelTask, local_size:int) -> CompiledKernel:
  uop:UOp = task.output_uop
  dtype:str = cl_type(task.output_dtype)
  val_map = {id(b.uop): b.load_expr(f"in{i}", 'i') for i, b in enumerate(task.inputs)}
  math_lines = render_op_chain(task.ops[:-1], val_map)
  reduce_val = val_map[id(_resolve_base(task.ops[-1].src[0]))] # we apply reduction via this var; which is the last from math ops
  math_lines.append(f"acc = {render_op(uop, ['acc', reduce_val])};") # NOTE: this is the actual reduction line
  source:str = f"""  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int g_stride = get_global_size(0);
  int l_size = get_local_size(0);
  {dtype} acc = {cl_const(identity_element(uop.op, task.output_dtype), task.output_dtype)};
  for (int i = gid; i < n; i += g_stride) {{
    {'\n    '.join(math_lines)}
  }}
  scratch[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = l_size / 2; s > 0; s >>= 1) {{
    if (lid < s) scratch[lid] = {render_op(uop, ['scratch[lid]', 'scratch[lid + s]'])};
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
  if (lid == 0) {{
    int group_id = get_group_id(0);
    out[group_id] = scratch[0];
  }}"""
  return render_kernel(task, source, global_size=(local_size,), local_size=(local_size,), extra_args=f", __local {dtype}* scratch")
def _codegen_reduce_strided_outer(task:KernelTask) -> CompiledKernel:
  n:int = prod(task.output_shape)
  uop:UOp = task.output_uop
  dtype:str = cl_type(task.output_dtype)
  axes:tuple[int, ...] = task.ops[-1].arg[0]
  if isinstance(axes, int): axes = (axes,)
  reduce_volume:int = prod(task.inputs[0].shape[a] for a in axes) # flattned reduce size
  val_map = {id(b.uop): b.reduce_load_expr_multi(axes, f"in{i}", gid='gid', k='k') for i, b in enumerate(task.inputs)}
  math_lines = render_op_chain(task.ops[:-1], val_map)
  reduce_val = val_map[id(_resolve_base(task.ops[-1].src[0]))] # we apply reduction via this var; which is the last from math ops
  math_lines.append(f"acc = {render_op(uop, ['acc', reduce_val])};") # NOTE: this is the actual reduction line
  source:str = f"""  int gid = get_global_id(0);
  if (gid >= n) return;
  {dtype} acc = {cl_const(identity_element(uop.op, task.output_dtype), task.output_dtype)};
  for (int k = 0; k < {reduce_volume}; k++) {{ // k < reduce_volume
    {'\n    '.join(math_lines)}
  }}
  out[gid] = acc;"""
  return render_kernel(task, source, global_size=(n,))
def _codegen_reduce_strided_inner(task:KernelTask, local_size:int) -> CompiledKernel:
  # (1024, 256).reduce(axis=1)  ---->  n_batch = 1024, and we launch 1024 threads, each processes 256 elems
  n_batch:int = prod(task.output_shape)
  uop:UOp = task.output_uop
  dtype:str = cl_type(task.output_dtype)
  axes:tuple[int, ...] = task.ops[-1].arg[0]
  if isinstance(axes, int): axes = (axes,)
  reduce_volume:int = prod(task.inputs[0].shape[a] for a in axes) # flattned reduce size
  # Trick BufferRef: Pass 'batch_idx' as the output coordinate variable instead of 'gid'
  # val_map = {id(b.uop): b.reduce_load_expr(axis, f"in{i}", 'batch_idx') for i, b in enumerate(task.inputs)}
  val_map = {id(b.uop): b.reduce_load_expr_multi(axes, f"in{i}", gid='batch_idx', k='k') for i, b in enumerate(task.inputs)}
  math_lines = render_op_chain(task.ops[:-1], val_map)
  reduce_val = val_map[id(_resolve_base(task.ops[-1].src[0]))]
  math_lines.append(f"acc = {render_op(uop, ['acc', reduce_val])};")
  source:str = f"""  int lid = get_local_id(0);
  int l_size = get_local_size(0);
  int batch_idx = get_global_id(1); 
  if (batch_idx >= {n_batch}) return;
  {dtype} acc = {cl_const(identity_element(uop.op, task.output_dtype), task.output_dtype)};
  for (int k = lid; k < {reduce_volume}; k += l_size) {{ 
    {'\n    '.join(math_lines)}
  }}
  scratch[lid] = acc;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = l_size / 2; s > 0; s >>= 1) {{
    if (lid < s) scratch[lid] = {render_op(uop, ['scratch[lid]', 'scratch[lid + s]'])};
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
  if (lid == 0) {{
    out[batch_idx] = scratch[0];
  }}"""
  return render_kernel(task, source, global_size=(local_size, n_batch), local_size=(local_size, 1), extra_args=f", __local {dtype}* scratch")
def _codegen_copy(task:KernelTask) -> CompiledKernel:
  n:int = prod(task.output_shape)
  source:str = f"""  int gid = get_global_id(0);
  if (gid >= n) return;
  out[gid] = {task.inputs[0].load_expr('in0', 'gid')};"""
  return render_kernel(task, source, global_size=(n,))
def _unpack_matmul(task:KernelTask) -> tuple[UOp, UOp, int, int, int]:
  assert task.output_uop.op is Ops.MATMUL, f"MATMUL codegen received {task.output_uop.op}"
  assert all(op.op in GroupOp.ALU for op in task.ops[:-1]), f"invalid MATMUL prologue: {task.ops[:-1]}"
  left, right = task.output_uop.src
  assert len(left.shape) == len(right.shape) == 2, "MATMUL codegen currently supports only 2D inputs"
  assert task.output_dtype is not dtypes.bool, "boolean MATMUL is unsupported"
  assert left.dtype is task.output_dtype and right.dtype is task.output_dtype, "MATMUL operands must match its output dtype"
  m, k = left.shape
  b_k, n = right.shape
  assert k == b_k, f"MATMUL shape mismatch: {left.shape} x {right.shape}"
  assert task.output_shape == (m, n), f"MATMUL output mismatch: expected {(m, n)}, got {task.output_shape}"
  return left, right, m, n, k
def _matmul_renderer(task:KernelTask) -> Callable[[UOp, str], str]:
  group_ids = {id(op) for op in task.ops}
  input_names = {id(ref): f"in{i}" for i, ref in enumerate(task.inputs)}
  def render(uop:UOp, index:str) -> str:
    if uop.op is Ops.CONST: return cl_const(uop.arg[0], uop.dtype)
    if uop.op in GroupOp.Movement:
      ref = BufferRef.from_uop(uop)
      if id(ref.uop) not in group_ids and ref.uop.op is not Ops.CONST:
        assert id(ref) in input_names, f"missing MATMUL input for {uop}"
        value = ref.load_expr(input_names[id(ref)], index)
      else:
        mapped_index, mask = ref.index_expr(index)
        value = render(ref.uop, mapped_index)
        if mask != "true":
          assert ref.padding_op is not None, f"masked MATMUL input has no padding op: {ref}"
          value = f"({mask} ? {value} : {cl_const(ref.padding_op.arg[2], uop.dtype)})"
      return f"({cl_type(uop.dtype)})({value})"
    if id(uop) not in group_ids:
      ref = BufferRef.from_uop(uop)
      assert id(ref) in input_names, f"missing MATMUL input for {uop}"
      return f"({cl_type(uop.dtype)})({ref.load_expr(input_names[id(ref)], index)})"
    assert uop.op in GroupOp.ALU, f"unsupported fused MATMUL op: {uop.op}"
    return f"({cl_type(uop.dtype)})({render_op(uop, [render(src, index) for src in uop.src])})"
  return render
def _codegen_matmul(task:KernelTask) -> CompiledKernel:
  if task.output_dtype in {dtypes.float16, dtypes.float32}: return _codegen_matmul_tiled(task)
  return _codegen_matmul_naive(task)
def _codegen_matmul_naive(task:KernelTask) -> CompiledKernel:
  left, right, m, cols, depth = _unpack_matmul(task)
  render = _matmul_renderer(task)
  dtype, zero = cl_type(task.output_dtype), cl_const(0, task.output_dtype)
  load_a = render(left, f"(row * {depth} + k)")
  load_b = render(right, f"(k * {cols} + col)")
  update = CL_OP[Ops.MULACC](load_a, load_b, "acc") if dtypes.is_float(task.output_dtype) else f"(acc + ({load_a} * {load_b}))"
  source = f"""  int gid = get_global_id(0);
  if (gid >= n) return;
  int row = gid / {cols};
  int col = gid % {cols};
  {dtype} acc = {zero};
  for (int k = 0; k < {depth}; k++) acc = {update};
  out[gid] = acc;"""
  return render_kernel(task, source, global_size=(m * cols,), name_suffix="_naive")
def _codegen_matmul_tiled(task:KernelTask, tile_m:int=128, tile_n:int=128, tile_k:int=16,
                          work_m:int=8, work_n:int=8) -> CompiledKernel:
  assert dtypes.is_float(task.output_dtype), "tiled MATMUL requires a floating-point dtype"
  assert min(tile_m, tile_n, tile_k, work_m, work_n) > 0, "MATMUL tuning parameters must be positive"
  assert tile_m % work_m == 0, f"tile_m ({tile_m}) must be divisible by work_m ({work_m})"
  assert tile_n % work_n == 0, f"tile_n ({tile_n}) must be divisible by work_n ({work_n})"
  left, right, m, n, k = _unpack_matmul(task)
  render = _matmul_renderer(task)
  threads_m, threads_n = tile_m // work_m, tile_n // work_n
  thread_count = threads_m * threads_n
  assert tile_k * tile_m % thread_count == 0, "A tile cannot be evenly loaded by the workgroup"
  assert tile_k * tile_n % thread_count == 0, "B tile cannot be evenly loaded by the workgroup"
  assert thread_count <= 1024, f"workgroup size {thread_count} exceeds the OpenCL limit"
  loads_a, loads_b = tile_k * tile_m // thread_count, tile_k * tile_n // thread_count
  dtype, zero = cl_type(task.output_dtype), cl_const(0, task.output_dtype)
  load_a = render(left, f"(global_row_a * {k} + global_col_a)")
  load_b = render(right, f"(global_row_b * {n} + global_col_b)")
  fma = CL_OP[Ops.MULACC]("a_value", "b_values[wn]", "acc[wm][wn]")
  # The extra A column breaks bank-aligned row strides in local memory.
  source = f"""  const int thread_n = get_local_id(0);
  const int thread_m = get_local_id(1);
  const int offset_n = {tile_n} * get_group_id(0);
  const int offset_m = {tile_m} * get_group_id(1);
  const int thread_id = thread_m * {threads_n} + thread_n;
  __local {dtype} tile_a[{tile_m}][{tile_k + 1}];
  __local {dtype} tile_b[{tile_k}][{tile_n}];
  {dtype} a_value;
  {dtype} b_values[{work_n}];
  {dtype} acc[{work_m}][{work_n}];
  #pragma unroll
  for (int wm = 0; wm < {work_m}; wm++) {{
    #pragma unroll
    for (int wn = 0; wn < {work_n}; wn++) acc[wm][wn] = {zero};
  }}
  const int tile_count = ({k} + {tile_k - 1}) / {tile_k};
  for (int tile = 0; tile < tile_count; tile++) {{
    const int tile_offset = {tile_k} * tile;
    #pragma unroll
    for (int load = 0; load < {loads_a}; load++) {{
      const int load_id = load * {thread_count} + thread_id;
      const int row_a = load_id / {tile_k};
      const int col_a = load_id % {tile_k};
      const int global_row_a = offset_m + row_a;
      const int global_col_a = tile_offset + col_a;
      tile_a[row_a][col_a] = global_row_a < {m} && global_col_a < {k} ? {load_a} : {zero};
    }}
    #pragma unroll
    for (int load = 0; load < {loads_b}; load++) {{
      const int load_id = load * {thread_count} + thread_id;
      const int row_b = load_id / {tile_n};
      const int col_b = load_id % {tile_n};
      const int global_row_b = tile_offset + row_b;
      const int global_col_b = offset_n + col_b;
      tile_b[row_b][col_b] = global_row_b < {k} && global_col_b < {n} ? {load_b} : {zero};
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int kk = 0; kk < {tile_k}; kk++) {{
      #pragma unroll
      for (int wn = 0; wn < {work_n}; wn++) b_values[wn] = tile_b[kk][thread_n + wn * {threads_n}];
      #pragma unroll
      for (int wm = 0; wm < {work_m}; wm++) {{
        a_value = tile_a[thread_m + wm * {threads_m}][kk];
        #pragma unroll
        for (int wn = 0; wn < {work_n}; wn++) acc[wm][wn] = {fma};
      }}
    }}
    barrier(CLK_LOCAL_MEM_FENCE);
  }}
  #pragma unroll
  for (int wm = 0; wm < {work_m}; wm++) {{
    const int row = offset_m + thread_m + wm * {threads_m};
    #pragma unroll
    for (int wn = 0; wn < {work_n}; wn++) {{
      const int col = offset_n + thread_n + wn * {threads_n};
      if (row < {m} && col < {n}) out[row * {n} + col] = acc[wm][wn];
    }}
  }}"""
  local_size = (threads_n, threads_m, 1)
  global_size = (((n + tile_n - 1) // tile_n) * threads_n, ((m + tile_m - 1) // tile_m) * threads_m, 1)
  suffix = f"_tiled_{tile_m}x{tile_n}x{tile_k}_{work_m}x{work_n}"
  return render_kernel(task, source, global_size, local_size, name_suffix=suffix)
