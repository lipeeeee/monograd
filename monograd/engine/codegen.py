from dataclasses import dataclass
from typing import Callable
from math import prod
from monograd.engine.schedule import BufferRef, KernelTask, TaskKind
from monograd.dtype import DType, ConstType, DTypeLike, dtypes, to_dtype
from monograd.uop.ops import UOp, identity_element
from monograd.uop import GroupOp, Ops
from monograd.utils import DEBUG


# WARN: We shouldnt launch kernels with large global sizes
# query gpu to figure out limits and adapt (also shouldnt launch not multiple of 2 global sizes)
_local_size_tmp = 256 # NOTE: local_size should be computed

@dataclass
class CompiledKernel:
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
  # Ops.CAST:   None,  # handled specially
  # ternary
  Ops.MULACC: lambda a, b, c: f"fma({a}, {b}, {c})",
  Ops.WHERE:  lambda a, b, c: f"({a} ? {b} : {c})",
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
    lines.append(f"  {cl_type(op.dtype)} {var} = {render_op(op, src_exprs)};")
    if DEBUG >= 5: print(f"{op.op.name}({op.src}) -> {lines[-1].strip()}")
    val_map[id(op)] = var
  return lines
def codegen(task:KernelTask) -> CompiledKernel:
  if task.kind is TaskKind.ELEMENTWISE: return _codegen_elementwise(task)
  if task.kind is TaskKind.REDUCE: return _codegen_reduce(task)
  if task.kind is TaskKind.COPY: return _codegen_copy(task)
  raise RuntimeError(f"how come i didnt get treated? {task};kind={task.kind}")
def _codegen_elementwise(task:KernelTask) -> CompiledKernel:
  name:str = kernel_name(task)
  n:int = prod(task.output_shape)
  val_map:dict[int, str] = {}
  args:list[str] = []
  for i, buff in enumerate(task.inputs): # NOTE: do not generate variable lines for each input and hope compiler catches them and makes them 1 memory read only
    args.append(f"__global const {cl_type(buff.uop.dtype)}* in{i}")
    val_map[id(buff.uop)] = buff.load_expr(f"in{i}", "gid") # f"in{i}[{buff.index_expr('gid')}]"
  args.extend([f"__global {cl_type(task.output_dtype)}* out", "const int n"])
  body:str = "\n".join(render_op_chain(task.ops, val_map))
  source:str = f"""{render_pragmas(task)}
__kernel void {name}({', '.join(args)}) {{
  int gid = get_global_id(0);
  if (gid >= n) return;
{body}
  out[gid] = v{len(task.ops) - 1};
}}"""
  if DEBUG >= 1: print(source)
  return CompiledKernel(source, name, global_size=(n,), local_size=None, args=task.inputs, 
                        output_shape=task.output_shape, output_dtype=task.output_dtype)
def _codegen_reduce(task:KernelTask) -> CompiledKernel:
  axes:tuple[int, ...]  = task.ops[0].arg[0]
  input_uop:UOp         = task.ops[0].src[0]
  if len(axes) == len(input_uop.shape): return _codegen_reduce_full(task, local_size=_local_size_tmp)
  else: return _codegen_reduce_strided(task)
def _codegen_reduce_full(task:KernelTask, local_size:int) -> CompiledKernel:
  n:int = prod(task.output_shape)
  uop:UOp = task.output_uop
  dtype:str = cl_type(task.output_dtype)
  name:str = kernel_name(task)
  source:str = f"""{render_pragmas(task)}
__kernel void {name}(__global const {dtype}* in, __global {dtype}* out, __local {dtype}* scratch, const int n) {{
  int lid = get_local_id(0);
  int gid = get_global_id(0);
  int g_stride = get_global_size(0);
  int l_size = get_local_size(0);
  {dtype} acc = {cl_const(identity_element(uop.op, task.output_dtype), task.output_dtype)};
  for (int i = gid; i < n; i += g_stride) {{
    acc = {render_op(uop, ['acc', 'in[i]'])};
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
  }}
}}"""
  if DEBUG >= 1: print(source)
  return CompiledKernel(source, name, global_size=(n,), local_size=(local_size,), args=task.inputs, 
                        output_shape=task.output_shape, output_dtype=task.output_dtype)
def _codegen_reduce_strided(task:KernelTask) -> CompiledKernel:
  n:int = prod(task.output_shape)
  uop:UOp = task.output_uop
  name:str = kernel_name(task)
  dtype:str = cl_type(task.output_dtype)
  input_buf:BufferRef = task.inputs[0]
  axis:int = task.ops[0].arg[0][0] 
  assert isinstance(axis, int)
  source:str = f"""{render_pragmas(task)}
__kernel void {name}(__global const {dtype}* in, __global {dtype}* out, const int n) {{
  int gid = get_global_id(0);
  if (gid >= n) return;
  {dtype} acc = {cl_const(identity_element(uop.op, task.output_dtype), task.output_dtype)};
  for (int k = 0; k < {input_buf.shape[axis]}; k++) {{ // k < reduce_size
    {dtype} val = {input_buf.reduce_load_expr(axis, 'in', 'gid')};
    acc = {render_op(uop, ['acc', 'val'])};
  }}
  out[gid] = acc;
}}"""
  if DEBUG >= 1: print(source)
  return CompiledKernel(source, name, global_size=(n,), local_size=None, args=task.inputs, 
                        output_shape=task.output_shape, output_dtype=task.output_dtype)
def _codegen_copy(task:KernelTask) -> CompiledKernel:
  n:int = prod(task.output_shape)
  name:str = kernel_name(task)
  dtype:str = cl_type(task.output_dtype)
  source:str = f"""{render_pragmas(task)}
__kernel void {name}(__global const {dtype}* in, __global {dtype}* out, const int n) {{
  int gid = get_global_id(0);
  if (gid >= n) return;
  out[gid] = {task.inputs[0].load_expr('in', 'gid')};
}}"""
  if DEBUG >= 1: print(source)
  return CompiledKernel(source, name, global_size=(n,), local_size=None, args=task.inputs, 
                        output_shape=task.output_shape, output_dtype=task.output_dtype)
