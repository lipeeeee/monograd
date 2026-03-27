from dataclasses import dataclass
from typing import Callable
from monograd.engine.schedule import BufferRef, KernelTask, TaskKind
from monograd.dtype import DType, ConstType, dtypes
from monograd.uop.ops import UOp
from monograd.uop import GroupOp, Ops
from monograd.utils import DEBUG

# 1. cl_ or c_
# 2. we never see ops.neg because its decomposed, its ok to remove right?
# 3. try to remove first assert in `render_op`

@dataclass
class CompiledKernel:
  source: str
  name: str
  global_size: tuple[int, ...]        # OpenCL global work size
  local_size: tuple[int, ...] | None  # None = let driver decide
  args: list[BufferRef]               # inputs in argument order
  output_shape: tuple[int, ...]
  output_dtype: DType 

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
  Ops.RELU:   lambda a:       f"max({a}, 0)",
  Ops.SIN:    lambda a:       f"sin({a})",
  Ops.SQRT:   lambda a:       f"sqrt({a})",
  Ops.EXP:    lambda a:       f"exp2({a} * 1.4426950408889634f)",   # exp(x) = exp2(x / ln(2)) = exp2(x * log2(e))
  Ops.LOG:    lambda a:       f"(log2({a}) * 0.6931471805599453f)", # log(x) = log2(x) * ln(2)
  # Ops.CAST:   None,  # handled specially
  # ternary
  Ops.MULACC: lambda a, b, c: f"fma({a}, {b}, {c})",
  Ops.WHERE:  lambda a, b, c: f"({a} ? {b} : {c})",
}


# *** rendering helpers ****
def cl_type(dtype:DType) -> str: return dtype.name
def cl_const(val:ConstType, dtype:DType) -> str:
  if dtype is dtypes.bool: return "1" if val else "0"
  if dtypes.is_float(dtype): # handle weirdness of c99 llvm when casting floats
    if dtype is dtypes.half: return f"{float(val)}h"
    if dtype is dtypes.float: return f"{float(val)}f"
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
  return f"{kind}_{ops}_{shape}"
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
def render_op_chain(uops:list[UOp], val_map:dict[int, str]) -> list[str]: # kernel body
  lines:list[str] = []
  for i, op in enumerate(uops):
    var = f"v{i}"
    src_exprs:list[str] = []
    for src in op.src:
      _base = _resolve_base(src) # NOTE: this sucks because we are looping through move ops, can improve?
      if _base.op is Ops.CONST: src_exprs.append(cl_const(_base.arg[0], _base.dtype))
      elif id(_base) in val_map: src_exprs.append(val_map[id(_base)])
      else: raise RuntimeError(f"src {_base.op} not in val_map - toposort broken?")
    lines.append(f" {cl_type(op.dtype)} {var} = {render_op(op, src_exprs)};")
    val_map[id(op)] = var
  return lines
def codegen(task:KernelTask) -> CompiledKernel:
  if task.kind is TaskKind.ELEMENTWISE: return _codegen_elementwise(task)
  raise RuntimeError(f"how come i didnt get treated? {task};kind={task.kind}")
def _codegen_elementwise(task:KernelTask) -> CompiledKernel:
  inputs:list[str] = []
  val_map:dict[int, str] = {}
  for i, buff in enumerate(task.inputs):
    # NOTE: do not generate variable lines for each input and hope compiler catches them and makes them 1 memory read only
    var = f"in{i}"
    inputs.append(f"__global const {cl_type(buff.uop.dtype)} in{i}*") # NOTE: can i generate the signature's inputs here?
    val_map[id(buff.uop)] = f"{var}[{buff.index_expr('gid')}]"
  signature:str = f"__kernel void {kernel_name(task)}({', '.join(inputs)})"+" {"
  kernel_body:str = "\n".join(render_op_chain(task.ops, val_map))
  if DEBUG >= 1: # kernel printing
    print(task.ops)
    print(val_map)
    print(signature)
    print(kernel_body + "\n}")
