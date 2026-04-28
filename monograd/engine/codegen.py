from dataclasses import dataclass
from typing import Callable
from math import prod
from monograd.engine.schedule import BufferRef, KernelTask, TaskKind
from monograd.dtype import DType, ConstType, DTypeLike, dtypes, to_dtype
from monograd.uop.ops import UOp, identity_element
from monograd.uop import GroupOp, Ops
from monograd.utils import DEBUG

# 1 entrypoint: codegen() -> CompiledKernel
# for every codegen: we should have a main template that supports math ops
#   do we store templates outside functions?; doesnt really have much utility
# there is alot of repeated code. 

# WARN: We shouldnt launch kernels with large global sizes
# query gpu to figure out limits and adapt (also shouldnt launch not multiple of 2 global sizes)
_local_size_tmp = 256 # NOTE: local_size should be computed

C_KERNEL_TEMPLATE = """{pragmas}
__kernel void {name}({args}, __global {out_dtype}* out{extra_args}, const int n) {{
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
  Ops.MOD:    lambda a, b:    f"({a} % {b})",
  Ops.POW:    lambda a, b:    f"pow({a}, {b})",
  Ops.XOR:    lambda a, b:    f"({a} ^ {b})",
  Ops.OR:     lambda a, b:    f"({a} | {b})",
  Ops.AND:    lambda a, b:    f"({a} & {b})",
  # unary
  Ops.RECIP:  lambda a:       f"(1.0f / {a})",
  Ops.RELU:   lambda a:       f"max({a}, 0)",
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
  all_dtypes:set[DType] = {task.output_dtype} | {ref.uop.dtype for ref in task.inputs}
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
  assert uop.op in CL_OP, f"unhandled op in render_op: {uop.op}"
  return CL_OP[uop.op](*src_exprs)
def render_kernel(task:KernelTask, body:str, global_size:tuple, local_size:tuple|None=None, extra_args:str="") -> CompiledKernel:
  name = kernel_name(task)
  args = ", ".join(f"__global const {cl_type(b.uop.dtype)}* in{i}" for i, b in enumerate(task.inputs))
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
  if task.kind is TaskKind.REDUCE_STRIDED: return _codegen_reduce_strided(task)
  if task.kind is TaskKind.COPY: return _codegen_copy(task)
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
  n:int = prod(task.inputs[0].uop.shape)
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
def _codegen_reduce_strided(task:KernelTask) -> CompiledKernel:
  n:int = prod(task.output_shape)
  uop:UOp = task.output_uop
  dtype:str = cl_type(task.output_dtype)
  axis:int = task.ops[-1].arg[0][0]
  assert isinstance(axis, int), f"_codegen_reduce_strided received invalid axis: {axis}, only 1 axis supported on strided"
  val_map = {id(b.uop): b.reduce_load_expr(axis, f"in{i}", 'gid') for i, b in enumerate(task.inputs)}
  math_lines = render_op_chain(task.ops[:-1], val_map)
  reduce_val = val_map[id(_resolve_base(task.ops[-1].src[0]))] # we apply reduction via this var; which is the last from math ops
  math_lines.append(f"acc = {render_op(uop, ['acc', reduce_val])};") # NOTE: this is the actual reduction line
  source:str = f"""  int gid = get_global_id(0);
  if (gid >= n) return;
  {dtype} acc = {cl_const(identity_element(uop.op, task.output_dtype), task.output_dtype)};
  for (int k = 0; k < {task.inputs[0].shape[axis]}; k++) {{ // k < reduce_size
    {'\n    '.join(math_lines)}
  }}
  out[gid] = acc;"""
  return render_kernel(task, source, global_size=(n,))
def _codegen_copy(task:KernelTask) -> CompiledKernel:
  n:int = prod(task.output_shape)
  source:str = """  int gid = get_global_id(0);
  if (gid >= n) return;
  out[gid] = {task.inputs[0].load_expr('in0', 'gid')};
  """
  return render_kernel(task, source, global_size=(n,))
