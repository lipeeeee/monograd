# Comprehensive coverage for monograd/schedule.py:
#  _row_major_strides, is_scalar/is_fusable/is_invisible/is_boundary,
#  BufferRef.from_uop, BufferRef.index_expr,
#  _collect_inputs, run_scheduler, KernelTask properties
import unittest
import numpy as np
from math import prod
from monograd.device import Device
from monograd.dtype import dtypes
from monograd.uop import GroupOp, Ops
from monograd.uop.ops import UOp
from monograd.engine.schedule import (
  _row_major_strides, is_scalar, is_fusable, is_invisible, is_boundary,
  BufferRef, KernelTask, TaskKind, run_scheduler, _collect_inputs, pprint_schedule,
)

# **** helpers ****
F32 = dtypes.float32
CPU = Device.CPU
GPU = Device.GPU

def load(shape, dtype=F32, device=CPU) -> UOp:
  u = UOp(Ops.LOAD, dtype, (), (shape, device))
  u.assign_buffer(int(np.prod(shape)))
  return u

def const(val, dtype=F32, device=CPU) -> UOp:
  return UOp(Ops.CONST, dtype, (), (val, device))

def reshape(src, shape) -> UOp:
  return UOp(Ops.RESHAPE, src.dtype, (src,), shape)

def expand(src, shape) -> UOp:
  return UOp(Ops.EXPAND, src.dtype, (src,), shape)

def permute(src, order) -> UOp:
  return UOp(Ops.PERMUTE, src.dtype, (src,), order)

def add(a, b) -> UOp:    return UOp(Ops.ADD,   a.dtype, (a, b), CPU)
def mul(a, b) -> UOp:    return UOp(Ops.MUL,   a.dtype, (a, b), CPU)
def relu(a) -> UOp:      return UOp(Ops.RELU,  a.dtype, (a,),   CPU)
def exp(a) -> UOp:       return UOp(Ops.EXP,   a.dtype, (a,),   CPU)
def log(a) -> UOp:       return UOp(Ops.LOG,   a.dtype, (a,),   CPU)
def sqrt(a) -> UOp:      return UOp(Ops.SQRT,  a.dtype, (a,),   CPU)
def cast(a, dtype) -> UOp: return UOp(Ops.CAST, dtype, (a,),    CPU)

# **** decomposed ops ****
# NEG, SUB, DIV are no longer primitive ops — they decompose into:
#   neg(x)    = mul(x, CONST(-1))
#   sub(a, b) = add(a, neg(b))
#   div(a, b) = mul(a, recip(b))
def recip(a) -> UOp: return UOp(Ops.RECIP, a.dtype, (a,), CPU)
def neg(a) -> UOp: return mul(a, UOp(Ops.CONST, a.dtype, (), (-1.0, CPU)))
def sub(a, b) -> UOp: return add(a, neg(b))
def div(a, b) -> UOp: return mul(a, recip(b))

def sumop(src, axes, out_shape) -> UOp:
  return UOp(Ops.SUM, src.dtype, (src,), (axes, out_shape))

def matmul(a, b) -> UOp:
  return UOp(Ops.MATMUL, a.dtype, (a, b))

def broadcast(src, target_shape) -> UOp:
  """helper: reshape to add leading 1-dims then expand"""
  ndim_diff = len(target_shape) - len(src.shape)
  padded = (1,) * ndim_diff + src.shape
  r = reshape(src, padded) if padded != src.shape else src
  return expand(r, target_shape) if r.shape != target_shape else r

def eval_index(expr: str, gid: int) -> int:
  """evaluate a C index expression as Python integer division"""
  return eval(expr.replace("/", "//"), {"gid": gid})

def eval_reduce_index(expr: str, gid: int, k: int) -> int:
  """evaluate a C reduce index expression as Python integer division"""
  return eval(expr.replace("/", "//"), {"gid": gid, "k": k})

def tasks_kinds(root: UOp) -> list[TaskKind]:
  return [t.kind for t in run_scheduler(root)]

def tasks_ops(root: UOp) -> list[list[Ops]]:
  return [[u.op for u in t.ops] for t in run_scheduler(root)]

def all_ops_in_graph(root: UOp) -> list[Ops]:
  """collect all op types reachable from root"""
  visited, ops = set(), []
  def walk(u):
    if id(u) in visited: return
    visited.add(id(u))
    ops.append(u.op)
    for s in u.src: walk(s)
  walk(root)
  return ops


# **** _row_major_strides ****
class TestRowMajorStrides(unittest.TestCase):
  def test_empty_shape(self):
    self.assertEqual(_row_major_strides(()), ())

  def test_scalar_shape(self):
    self.assertEqual(_row_major_strides((1,)), (1,))

  def test_1d(self):
    self.assertEqual(_row_major_strides((4,)), (1,))
    self.assertEqual(_row_major_strides((8,)), (1,))

  def test_2d(self):
    self.assertEqual(_row_major_strides((2, 3)), (3, 1))
    self.assertEqual(_row_major_strides((4, 8)), (8, 1))

  def test_3d(self):
    self.assertEqual(_row_major_strides((2, 3, 4)), (12, 4, 1))

  def test_4d(self):
    self.assertEqual(_row_major_strides((2, 3, 4, 5)), (60, 20, 5, 1))

  def test_last_stride_always_one(self):
    for shape in [(4,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
      self.assertEqual(_row_major_strides(shape)[-1], 1)

  def test_first_stride_is_product_of_rest(self):
    for shape in [(2, 3), (2, 3, 4), (3, 4, 5)]:
      strides = _row_major_strides(shape)
      self.assertEqual(strides[0], prod(shape[1:]))

  def test_contiguous_property(self):
    """element at [i,j] for shape (R,C) is at flat index i*C + j"""
    shape = (3, 4)
    strides = _row_major_strides(shape)
    for i in range(shape[0]):
      for j in range(shape[1]):
        flat = i * strides[0] + j * strides[1]
        self.assertEqual(flat, i * shape[1] + j)


# **** uop classification helpers ****
class TestUOpClassifiers(unittest.TestCase):
  def setUp(self):
    self.a = load((4,))
    self.b = load((4,))

  def test_is_fusable_binary(self):
    for op in GroupOp.Binary:
      u = UOp(op, F32, (self.a, self.b), CPU)
      self.assertTrue(is_fusable(u), f"{op} should be fusable")

  def test_is_fusable_unary(self):
    for op in GroupOp.Unary:
      u = UOp(op, F32, (self.a,), CPU)
      self.assertTrue(is_fusable(u), f"{op} should be fusable")

  def test_is_fusable_false_for_reduce(self):
    s = sumop(self.a, (0,), (1,))
    self.assertFalse(is_fusable(s))

  def test_is_fusable_false_for_load(self):
    self.assertFalse(is_fusable(self.a))

  def test_is_fusable_false_for_movement(self):
    r = reshape(self.a, (2, 2))
    self.assertFalse(is_fusable(r))

  def test_is_fusable_recip(self):
    u = recip(self.a)
    self.assertTrue(is_fusable(u))

  def test_is_invisible_movement_ops(self):
    r = reshape(self.a, (2, 2))
    e = expand(load((1, 4)), (3, 4))
    p = permute(load((3, 4)), (1, 0))
    self.assertTrue(is_invisible(r))
    self.assertTrue(is_invisible(e))
    self.assertTrue(is_invisible(p))

  def test_is_invisible_false_for_others(self):
    self.assertFalse(is_invisible(self.a))
    self.assertFalse(is_invisible(add(self.a, self.b)))
    self.assertFalse(is_invisible(sumop(self.a, (0,), (1,))))

  def test_is_boundary_reduce(self):
    s = sumop(self.a, (0,), (1,))
    self.assertTrue(is_boundary(s))

  def test_is_boundary_blas(self):
    x = load((4, 8))
    w = load((8, 16))
    mm = matmul(x, w)
    self.assertTrue(is_boundary(mm))

  def test_is_boundary_copy(self):
    c = UOp(Ops.COPY, F32, (self.a,), GPU)
    self.assertTrue(is_boundary(c))

  def test_is_boundary_false_for_fusable(self):
    self.assertFalse(is_boundary(add(self.a, self.b)))
    self.assertFalse(is_boundary(relu(self.a)))

  def test_is_scalar_const(self):
    c = const(2.0)
    self.assertTrue(is_scalar(c))

  def test_is_scalar_all_zero_strides(self):
    self.assertTrue(is_scalar(load((4,)), strides=(0,)))
    self.assertTrue(is_scalar(load((4,)), strides=(0, 0)))

  def test_is_scalar_false_normal(self):
    self.assertFalse(is_scalar(load((4,)), strides=(1,)))
    self.assertFalse(is_scalar(load((4,)), strides=(4, 1)))


# **** BufferRef.from_uop — stride computation ****
class TestBufferRefFromUop(unittest.TestCase):
  def test_bare_load_contiguous_1d(self):
    ref = BufferRef.from_uop(load((8,)))
    self.assertEqual(ref.shape,   (8,))
    self.assertEqual(ref.strides, (1,))

  def test_bare_load_contiguous_2d(self):
    ref = BufferRef.from_uop(load((3, 4)))
    self.assertEqual(ref.shape,   (3, 4))
    self.assertEqual(ref.strides, (4, 1))

  def test_bare_load_contiguous_3d(self):
    ref = BufferRef.from_uop(load((2, 3, 4)))
    self.assertEqual(ref.shape,   (2, 3, 4))
    self.assertEqual(ref.strides, (12, 4, 1))

  def test_base_uop_is_load(self):
    """ref.uop must be the base LOAD, not the movement op"""
    a = load((4,))
    r = reshape(a, (2, 2))
    ref = BufferRef.from_uop(r)
    self.assertIs(ref.uop, a)

  def test_reshape_resets_to_row_major(self):
    a   = load((12,))
    r   = reshape(a, (3, 4))
    ref = BufferRef.from_uop(r)
    self.assertEqual(ref.shape,   (3, 4))
    self.assertEqual(ref.strides, (4, 1))

  def test_expand_broadcast_dim0(self):
    """(1,4) → expand(3,4): dim0 stride becomes 0"""
    a   = load((4,))
    r   = reshape(a, (1, 4))
    e   = expand(r, (3, 4))
    ref = BufferRef.from_uop(e)
    self.assertEqual(ref.strides, (0, 1))
    self.assertEqual(ref.shape,   (3, 4))

  def test_expand_broadcast_dim1(self):
    """(3,1) → expand(3,4): dim1 stride becomes 0"""
    a   = load((3,))
    r   = reshape(a, (3, 1))
    e   = expand(r, (3, 4))
    ref = BufferRef.from_uop(e)
    self.assertEqual(ref.strides, (1, 0))
    self.assertEqual(ref.shape,   (3, 4))

  def test_expand_broadcast_both_dims(self):
    """(1,1) → expand(3,4): all strides 0"""
    a   = load((1,))
    r   = reshape(a, (1, 1))
    e   = expand(r, (3, 4))
    ref = BufferRef.from_uop(e)
    self.assertTrue(all(s == 0 for s in ref.strides))

  def test_expand_broadcast_scalar_to_3d(self):
    a   = load((1,))
    r   = reshape(a, (1, 1, 1))
    e   = expand(r, (2, 3, 4))
    ref = BufferRef.from_uop(e)
    self.assertEqual(ref.strides, (0, 0, 0))

  def test_expand_non_broadcast_dim_unchanged(self):
    """dims that were already correct size must keep their stride"""
    a   = load((4,))
    r   = reshape(a, (1, 4))
    e   = expand(r, (3, 4))
    ref = BufferRef.from_uop(e)
    self.assertEqual(ref.strides[1], 1)

  def test_permute_2d_transpose(self):
    a   = load((3, 4))
    p   = permute(a, (1, 0))
    ref = BufferRef.from_uop(p)
    self.assertEqual(ref.shape,   (4, 3))
    self.assertEqual(ref.strides, (1, 4))

  def test_permute_3d(self):
    a   = load((2, 3, 4))
    p   = permute(a, (2, 0, 1))
    ref = BufferRef.from_uop(p)
    self.assertEqual(ref.shape,   (4, 2, 3))
    self.assertEqual(ref.strides, (1, 12, 4))

  def test_reshape_then_expand(self):
    """(6,) → reshape(2,3) → expand — reshape resets then expand zeros dims"""
    a   = load((6,))
    r   = reshape(a, (1, 6))
    e   = expand(r, (4, 6))
    ref = BufferRef.from_uop(e)
    self.assertEqual(ref.strides, (0, 1))

  def test_alu_output_is_contiguous(self):
    """output of a previous task — treated as a fresh contiguous buffer"""
    a   = load((3, 4))
    b   = load((3, 4))
    m   = mul(a, b)
    ref = BufferRef.from_uop(m)
    self.assertEqual(ref.shape,   (3, 4))
    self.assertEqual(ref.strides, (4, 1))
    self.assertIs(ref.uop, m)

  def test_cache_returns_same_object(self):
    a   = load((4,))
    r   = reshape(a, (2, 2))
    ref1 = BufferRef.from_uop(r)
    ref2 = BufferRef.from_uop(r)
    self.assertIs(ref1, ref2)

  def test_different_movement_chains_different_refs(self):
    """EXPAND and bare LOAD over same base must give different strides"""
    a    = load((4,))
    r    = reshape(a, (1, 4))
    e    = expand(r, (3, 4))
    ref_load   = BufferRef.from_uop(a)
    ref_expand = BufferRef.from_uop(e)
    self.assertNotEqual(ref_load.strides, ref_expand.strides)


# **** BufferRef.index_expr ****
class TestIndexExpr(unittest.TestCase):
  def test_const_returns_zero(self):
    c   = const(2.0)
    ref = BufferRef.from_uop(c)
    self.assertEqual(ref.index_expr("gid"), "0")

  def test_fully_broadcast_returns_zero(self):
    a   = load((1,))
    r   = reshape(a, (1, 1))
    e   = expand(r, (3, 4))
    ref = BufferRef.from_uop(e)
    self.assertEqual(ref.index_expr("gid"), "0")

  def test_1d_contiguous_sequential(self):
    ref  = BufferRef.from_uop(load((6,)))
    expr = ref.index_expr("gid")
    for i in range(6):
      self.assertEqual(eval_index(expr, i), i)

  def test_2d_contiguous_sequential(self):
    """flat gid should map to same position in row-major layout"""
    ref  = BufferRef.from_uop(load((2, 3)))
    expr = ref.index_expr("gid")
    for i in range(6):
      self.assertEqual(eval_index(expr, i), i)

  def test_broadcast_dim0_repeats_rows(self):
    """(1,3)→(2,3): row 0 and row 1 read same elements"""
    a    = load((3,))
    r    = reshape(a, (1, 3))
    e    = expand(r, (2, 3))
    ref  = BufferRef.from_uop(e)
    expr = ref.index_expr("gid")
    for i in range(6):
      self.assertEqual(eval_index(expr, i), i % 3)

  def test_broadcast_dim1_repeats_cols(self):
    """(3,1)→(3,2): each row reads same single element"""
    a    = load((3,))
    r    = reshape(a, (3, 1))
    e    = expand(r, (3, 2))
    ref  = BufferRef.from_uop(e)
    expr = ref.index_expr("gid")
    expected = [0, 0, 1, 1, 2, 2]
    for i in range(6):
      self.assertEqual(eval_index(expr, i), expected[i])

  def test_transposed_access_pattern(self):
    """permute (2,3)→(3,2): verify index maps correctly"""
    a    = load((2, 3))
    p    = permute(a, (1, 0))
    ref  = BufferRef.from_uop(p)
    self.assertEqual(ref.strides, (1, 3))

  def test_3d_contiguous_sequential(self):
    ref  = BufferRef.from_uop(load((2, 3, 4)))
    expr = ref.index_expr("gid")
    for i in range(24):
      self.assertEqual(eval_index(expr, i), i)


# **** BufferRef.reduce_index_expr ****
class TestReduceIndexExpr(unittest.TestCase):
  def test_reduce_axis_0_2d(self):
    """Shape (3, 4) reduce axis 0. Output shape (4,). gid goes 0-3, k goes 0-2."""
    ref = BufferRef.from_uop(load((3, 4)))
    expr = ref.reduce_index_expr(reduce_axis=0)
    # Expected: k * 4 + gid
    for gid in range(4):
      for k in range(3):
        self.assertEqual(eval_reduce_index(expr, gid, k), k * 4 + gid)

  def test_reduce_axis_1_2d(self):
    """Shape (3, 4) reduce axis 1. Output shape (3,). gid goes 0-2, k goes 0-3."""
    ref = BufferRef.from_uop(load((3, 4)))
    expr = ref.reduce_index_expr(reduce_axis=1)
    # Expected: gid * 4 + k
    for gid in range(3):
      for k in range(4):
        self.assertEqual(eval_reduce_index(expr, gid, k), gid * 4 + k)

  def test_reduce_middle_axis_3d(self):
    """Shape (2, 3, 4) reduce axis 1. Output (2, 4) -> 8 threads."""
    ref = BufferRef.from_uop(load((2, 3, 4)))
    expr = ref.reduce_index_expr(reduce_axis=1)
    # Input strides: (12, 4, 1)
    # Formula: (gid // 4) * 12 + k * 4 + (gid % 4)
    for gid in range(8):
      for k in range(3):
        expected = (gid // 4) * 12 + k * 4 + (gid % 4)
        self.assertEqual(eval_reduce_index(expr, gid, k), expected)

  def test_transposed_reduce(self):
    """Permute (3, 4) to (4, 3), reduce axis 0."""
    a = load((3, 4))
    p = permute(a, (1, 0))
    ref = BufferRef.from_uop(p)
    expr = ref.reduce_index_expr(reduce_axis=0)
    # Original strides: (4, 1). New shape: (4, 3). New strides: (1, 4).
    # Expected: k * 1 + gid * 4
    for gid in range(3):
      for k in range(4):
        self.assertEqual(eval_reduce_index(expr, gid, k), k * 1 + gid * 4)

  def test_broadcast_reduce(self):
    """Expand (1, 4) to (3, 4), reduce axis 0."""
    a = load((4,))
    r = reshape(a, (1, 4))
    e = expand(r, (3, 4))
    ref = BufferRef.from_uop(e)
    expr = ref.reduce_index_expr(reduce_axis=0)
    # Shape: (3, 4). Strides: (0, 1).
    # Since stride 0 is 0, 'k * 0' is skipped in the expression string.
    for gid in range(4):
      for k in range(3):
        self.assertEqual(eval_reduce_index(expr, gid, k), gid)


# **** _collect_inputs ****
class TestCollectInputs(unittest.TestCase):
  def test_single_binary_op_two_inputs(self):
    a, b = load((4,)), load((5,))
    refs = _collect_inputs([mul(a, b)])
    self.assertEqual(len(refs), 2)

  def test_const_not_included(self):
    a = load((4,))
    c = const(2.0)
    refs = _collect_inputs([mul(a, c)])
    uop_ops = [r.uop.op for r in refs]
    self.assertNotIn(Ops.CONST, uop_ops)

  def test_shared_load_deduplicated(self):
    """same LOAD used as both operands — must appear only once"""
    a    = load((4,))
    refs = _collect_inputs([mul(a, a)])
    self.assertEqual(len(refs), 1)

  def test_internal_op_not_in_inputs(self):
    """intermediate op inside the group must not be an input"""
    a, b, c = load((4,)), load((4,)), load((4,))
    m       = mul(a, b)
    out     = add(m, c)
    refs    = _collect_inputs([m, out])
    input_uops = [r.uop for r in refs]
    self.assertNotIn(m, input_uops)

  def test_broadcast_input_has_zero_stride(self):
    a  = load((4,))
    r  = reshape(a, (1, 4))
    e  = expand(r, (3, 4))
    b  = load((3, 4))
    refs = _collect_inputs([add(b, e)])
    broadcast_ref = next(ref for ref in refs if 0 in ref.strides)
    self.assertEqual(broadcast_ref.strides[0], 0)

  def test_prev_task_output_as_input(self):
    """output of a boundary op (reduce) used as input to next group"""
    a = load((4,))
    s = sumop(a, (0,), (1,))
    out = relu(s)
    refs = _collect_inputs([out])
    input_uops = [r.uop for r in refs]
    self.assertIn(s, input_uops)

  def test_three_distinct_loads(self):
    a, b, c = load((4,)), load((5,)), load((6,))
    m       = mul(a, b)
    out     = add(m, c)
    refs    = _collect_inputs([m, out])
    self.assertEqual(len(refs), 3)

  def test_movement_ops_resolved_to_base(self):
    """BufferRef.uop must be the base LOAD, not the EXPAND"""
    a   = load((4,))
    r   = reshape(a, (1, 4))
    e   = expand(r, (3, 4))
    b   = load((3, 4))
    refs = _collect_inputs([add(b, e)])
    base_uops = [ref.uop for ref in refs]
    self.assertIn(a, base_uops)
    self.assertNotIn(e, base_uops)
    self.assertNotIn(r, base_uops)


# **** KernelTask properties ****
class TestKernelTaskProperties(unittest.TestCase):
  def test_elementwise_output_dtype(self):
    a, b  = load((4,)), load((4,))
    tasks = run_scheduler(add(a, b))
    self.assertIs(tasks[0].output_dtype, F32)

  def test_elementwise_output_shape(self):
    a, b  = load((3, 4)), load((3, 4))
    tasks = run_scheduler(add(a, b))
    self.assertEqual(tasks[0].output_shape, (3, 4))

  def test_elementwise_output_device(self):
    a, b  = load((4,)), load((4,))
    tasks = run_scheduler(add(a, b))
    self.assertEqual(tasks[0].output_device, CPU)

  def test_reduce_output_shape(self):
    a     = load((4,))
    s     = sumop(a, (0,), (1,))
    tasks = run_scheduler(s)
    self.assertEqual(tasks[-1].output_shape, (1,))

  def test_reduce_output_shape_axis1(self):
    a     = load((3, 4))
    s     = sumop(a, (1,), (3, 1))
    tasks = run_scheduler(s)
    self.assertEqual(tasks[-1].output_shape, (3, 1))

  def test_matmul_output_shape(self):
    x, w  = load((4, 8)), load((8, 16))
    tasks = run_scheduler(matmul(x, w))
    self.assertEqual(tasks[0].output_shape, (4, 16))

  def test_output_dtype_is_last_ops_dtype(self):
    """output_dtype must reflect the last op, not the first"""
    a     = load((4,), dtype=dtypes.float16)
    c     = cast(a, dtypes.float32)
    tasks = run_scheduler(relu(c))
    self.assertIs(tasks[0].output_dtype, dtypes.float32)


# **** run_scheduler — task count and kinds ****
class TestSchedulerTaskCount(unittest.TestCase):
  def test_single_binary_one_task(self):
    a, b = load((4,)), load((4,))
    self.assertEqual(len(run_scheduler(add(a, b))), 1)

  def test_single_unary_one_task(self):
    a = load((4,))
    self.assertEqual(len(run_scheduler(relu(a))), 1)

  def test_long_elementwise_chain_one_task(self):
    """mul→add→relu→exp — all fusable, no boundary ops → single kernel"""
    a, b = load((4,)), load((4,))
    # neg is now MUL(x, CONST(-1)), still fusable
    out  = neg(exp(relu(add(mul(a, b), b))))
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 1)
    self.assertEqual(tasks[0].kind, TaskKind.ELEMENTWISE)

  def test_recip_fuses_with_elementwise(self):
    a, b = load((4,)), load((4,))
    out  = mul(a, recip(b))   # decomposed div
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 1)

  def test_reduce_splits_into_two_tasks(self):
    a     = load((4,))
    tasks = run_scheduler(sumop(a, (0,), (1,)))
    self.assertEqual(len(tasks), 1)
    self.assertEqual(tasks[0].kind, TaskKind.REDUCE)

  def test_elementwise_then_reduce(self):
    a     = load((4,))
    out   = sumop(relu(a), (0,), (1,))
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 2)
    self.assertEqual(tasks[0].kind, TaskKind.ELEMENTWISE)
    self.assertEqual(tasks[1].kind, TaskKind.REDUCE)

  def test_reduce_then_elementwise(self):
    a   = load((4,))
    s   = sumop(a, (0,), (1,))
    out = relu(s)
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 2)
    self.assertEqual(tasks[0].kind, TaskKind.REDUCE)
    self.assertEqual(tasks[1].kind, TaskKind.ELEMENTWISE)

  def test_elementwise_reduce_elementwise(self):
    a   = load((4,))
    out = relu(sumop(mul(a, a), (0,), (1,)))
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 3)
    kinds = [t.kind for t in tasks]
    self.assertEqual(kinds, [
      TaskKind.ELEMENTWISE,
      TaskKind.REDUCE,
      TaskKind.ELEMENTWISE,
    ])

  def test_two_consecutive_reduces(self):
    a   = load((4,))
    s1  = sumop(a, (0,), (1,))
    s2  = sumop(s1, (0,), (1,))
    tasks = run_scheduler(s2)
    self.assertEqual(len(tasks), 2)
    self.assertTrue(all(t.kind == TaskKind.REDUCE for t in tasks))

  def test_matmul_own_task(self):
    x, w  = load((4, 8)), load((8, 16))
    tasks = run_scheduler(matmul(x, w))
    self.assertEqual(len(tasks), 1)
    self.assertEqual(tasks[0].kind, TaskKind.BLAS)

  def test_matmul_then_elementwise(self):
    x, w  = load((4, 8)), load((8, 16))
    out   = relu(matmul(x, w))
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 2)
    self.assertEqual(tasks[0].kind, TaskKind.BLAS)
    self.assertEqual(tasks[1].kind, TaskKind.ELEMENTWISE)

  def test_elementwise_then_matmul(self):
    x = load((4, 8))
    w = load((8, 16))
    xr  = relu(x)
    out = matmul(xr, w)
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 2)
    self.assertEqual(tasks[0].kind, TaskKind.ELEMENTWISE)
    self.assertEqual(tasks[1].kind, TaskKind.BLAS)

  def test_copy_own_task(self):
    a   = load((4,))
    cpy = UOp(Ops.COPY, F32, (a,), GPU)
    tasks = run_scheduler(cpy)
    self.assertEqual(len(tasks), 1)
    self.assertEqual(tasks[0].kind, TaskKind.COPY)

  def test_elementwise_copy_elementwise(self):
    a   = load((4,), device=CPU)
    r   = relu(a)
    cpy = UOp(Ops.COPY, F32, (r,), GPU)
    e   = UOp(Ops.EXP, F32, (cpy,), GPU)
    tasks = run_scheduler(e)
    self.assertEqual(len(tasks), 3)
    self.assertEqual(tasks[0].kind, TaskKind.ELEMENTWISE)
    self.assertEqual(tasks[1].kind, TaskKind.COPY)
    self.assertEqual(tasks[2].kind, TaskKind.ELEMENTWISE)


# **** run_scheduler — ops inside tasks ****
class TestSchedulerOps(unittest.TestCase):
  def test_ops_in_execution_order(self):
    """toposort guarantees dependencies before consumers"""
    a, b = load((4,)), load((4,))
    m    = mul(a, b)
    r    = relu(m)
    e    = exp(r)
    tasks = run_scheduler(e)
    op_names = [u.op for u in tasks[0].ops]
    self.assertLess(op_names.index(Ops.MUL),  op_names.index(Ops.RELU))
    self.assertLess(op_names.index(Ops.RELU), op_names.index(Ops.EXP))

  def test_movement_ops_never_in_tasks(self):
    """RESHAPE, EXPAND, PERMUTE must never appear in any task's ops"""
    a  = load((6,))
    r  = reshape(a, (2, 3))
    e  = expand(reshape(load((3,)), (1, 3)), (2, 3))
    out = add(r, e)
    for task in run_scheduler(out):
      for op in task.ops:
        self.assertNotIn(op.op, GroupOp.Movement,
          f"{op.op} should not be in task ops")

  def test_load_const_never_in_tasks(self):
    """LOAD and CONST must never appear in any task's ops"""
    a = load((4,))
    c = const(2.0)
    for task in run_scheduler(mul(a, c)):
      for op in task.ops:
        self.assertNotIn(op.op, {Ops.LOAD, Ops.CONST})

  def test_boundary_op_in_own_task(self):
    a = load((4,))
    s = sumop(relu(a), (0,), (1,))
    tasks = run_scheduler(s)
    self.assertIn(Ops.SUM,  [u.op for u in tasks[1].ops])
    self.assertNotIn(Ops.SUM, [u.op for u in tasks[0].ops])

  def test_cast_fuses_with_elementwise(self):
    """CAST is in GroupOp.Unary so must fuse"""
    a    = load((4,), dtype=dtypes.float16)
    c    = cast(a, dtypes.float32)
    out  = relu(c)
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 1)
    ops = [u.op for u in tasks[0].ops]
    self.assertIn(Ops.CAST, ops)
    self.assertIn(Ops.RELU, ops)

  def test_recip_fuses_in_chain(self):
    """RECIP must appear in the fused ops list alongside other elementwise ops"""
    a, b = load((4,)), load((4,))
    out  = relu(div(a, b))   # = relu(mul(a, recip(b)))
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 1)
    ops = [u.op for u in tasks[0].ops]
    self.assertIn(Ops.RECIP, ops)
    self.assertIn(Ops.MUL,   ops)
    self.assertIn(Ops.RELU,  ops)


# **** run_scheduler — inputs correctness ****
class TestSchedulerInputs(unittest.TestCase):
  def test_two_loads_two_inputs(self):
    a, b  = load((4,)), load((5,))
    tasks = run_scheduler(add(a, b))
    self.assertEqual(len(tasks[0].inputs), 2)

  def test_shared_load_deduplicated(self):
    a     = load((4,))
    tasks = run_scheduler(add(mul(a, a), a))
    input_uops = [r.uop for r in tasks[0].inputs]
    self.assertEqual(input_uops.count(a), 1)

  def test_const_not_in_inputs(self):
    a     = load((4,))
    tasks = run_scheduler(mul(a, const(2.0)))
    for ref in tasks[0].inputs:
      self.assertIsNot(ref.uop.op, Ops.CONST)

  def test_broadcast_input_has_zero_stride(self):
    a  = load((4,))
    r  = reshape(a, (1, 4))
    e  = expand(r, (3, 4))
    b  = load((3, 4))
    tasks = run_scheduler(add(b, e))
    broadcast_refs = [ref for ref in tasks[0].inputs if 0 in ref.strides]
    self.assertEqual(len(broadcast_refs), 1)
    self.assertEqual(broadcast_refs[0].strides, (0, 1))

  def test_intermediate_not_in_inputs(self):
    a, b, c = load((4,)), load((4,)), load((4,))
    m       = mul(a, b)
    out     = add(m, c)
    tasks   = run_scheduler(out)
    input_uops = [r.uop for r in tasks[0].inputs]
    self.assertNotIn(m, input_uops)

  def test_reduce_task_input_is_elementwise_output(self):
    a   = load((4,))
    m   = mul(a, a)
    s   = sumop(m, (0,), (1,))
    tasks = run_scheduler(s)
    reduce_inputs = [r.uop for r in tasks[-1].inputs]
    self.assertIn(m, reduce_inputs)

  def test_matmul_task_has_two_inputs(self):
    x, w  = load((4, 8)), load((8, 16))
    tasks = run_scheduler(matmul(x, w))
    self.assertEqual(len(tasks[0].inputs), 2)

  def test_inputs_have_correct_strides_for_contiguous(self):
    a, b  = load((3, 4)), load((3, 4))
    tasks = run_scheduler(add(a, b))
    for ref in tasks[0].inputs:
      self.assertEqual(ref.strides, (4, 1))


# **** linear layer, softmax, layernorm ****
class TestRealWorldPatterns(unittest.TestCase):
  """
  These tests mirror actual neural network operations.
  They verify that the scheduler produces the expected number of tasks
  and that fusion happens where it should.
  """

  def test_linear_layer_matmul_plus_bias(self):
    """y = x @ W + bias — MATMUL task + ELEMENTWISE task"""
    x    = load((4, 8))
    w    = load((8, 16))
    bias = load((16,))
    br   = broadcast(bias, (4, 16))
    out  = add(matmul(x, w), br)
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 2)
    self.assertEqual(tasks[0].kind, TaskKind.BLAS)
    self.assertEqual(tasks[1].kind, TaskKind.ELEMENTWISE)

  def test_linear_layer_with_relu(self):
    """relu(x @ W + bias) — relu fuses with add in elementwise task"""
    x    = load((4, 8))
    w    = load((8, 16))
    bias = load((16,))
    br   = broadcast(bias, (4, 16))
    out  = relu(add(matmul(x, w), br))
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 2)
    elem_ops = [u.op for u in tasks[1].ops]
    self.assertIn(Ops.ADD,  elem_ops)
    self.assertIn(Ops.RELU, elem_ops)

  def test_softmax_four_tasks(self):
    """
    softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))

    sub decomposes to ADD(x, MUL(max, -1)) — still elementwise, still fuses.
    div decomposes to MUL(exp, RECIP(sum)) — still elementwise, still fuses.

    Expected tasks (unchanged from pre-decomposition):
      0: REDUCE  — reducemax(x, axis=-1)
      1: ELEMENTWISE — (x - max) then exp  (MUL+ADD+EXP fused)
      2: REDUCE  — sum(exp, axis=-1)
      3: ELEMENTWISE — div result  (RECIP+MUL fused)
    """
    x      = load((2, 4))
    x_max  = UOp(Ops.REDUCEMAX, F32, (x,), ((1,), (2, 1)))
    br_max = broadcast(x_max, (2, 4))
    x_sub  = sub(x, br_max)    # = add(x, mul(br_max, const(-1)))
    x_exp  = exp(x_sub)
    x_sum  = sumop(x_exp, (1,), (2, 1))
    br_sum = broadcast(x_sum, (2, 4))
    out    = div(x_exp, br_sum)  # = mul(x_exp, recip(br_sum))
    tasks  = run_scheduler(out)
    self.assertEqual(len(tasks), 4)
    self.assertEqual(tasks[0].kind, TaskKind.REDUCE)      # max
    self.assertEqual(tasks[1].kind, TaskKind.ELEMENTWISE) # neg+add(sub)+exp fused
    self.assertEqual(tasks[2].kind, TaskKind.REDUCE)      # sum
    self.assertEqual(tasks[3].kind, TaskKind.ELEMENTWISE) # recip+mul(div) fused

  def test_softmax_sub_and_exp_fuse(self):
    """
    sub and exp between the two reduces must be in the same task.
    sub is now decomposed to ADD(x, MUL(max, -1)) — ADD and EXP must both be present.
    """
    x      = load((2, 4))
    x_max  = UOp(Ops.REDUCEMAX, F32, (x,), ((1,), (2, 1)))
    br_max = broadcast(x_max, (2, 4))
    x_sub  = sub(x, br_max)
    x_exp  = exp(x_sub)
    x_sum  = sumop(x_exp, (1,), (2, 1))
    br_sum = broadcast(x_sum, (2, 4))
    out    = div(x_exp, br_sum)
    tasks  = run_scheduler(out)
    elem_task = tasks[1]
    ops = [u.op for u in elem_task.ops]
    # sub is decomposed: ADD(x, MUL(br_max, CONST(-1)))
    self.assertIn(Ops.ADD, ops)   # from decomposed sub
    self.assertIn(Ops.MUL, ops)   # from neg part of sub
    self.assertIn(Ops.EXP, ops)   # exp still present

  def test_softmax_div_recip_fuse(self):
    """
    div between sum and output must be in the same elementwise task.
    div is now decomposed to MUL(exp, RECIP(sum)) — RECIP and MUL must both be present.
    """
    x      = load((2, 4))
    x_max  = UOp(Ops.REDUCEMAX, F32, (x,), ((1,), (2, 1)))
    br_max = broadcast(x_max, (2, 4))
    x_sub  = sub(x, br_max)
    x_exp  = exp(x_sub)
    x_sum  = sumop(x_exp, (1,), (2, 1))
    br_sum = broadcast(x_sum, (2, 4))
    out    = div(x_exp, br_sum)
    tasks  = run_scheduler(out)
    elem_task = tasks[3]
    ops = [u.op for u in elem_task.ops]
    self.assertIn(Ops.RECIP, ops)  # from decomposed div
    self.assertIn(Ops.MUL,   ops)  # outer mul of decomposed div

  def test_mse_loss(self):
    """
    loss = mean((pred - target)^2) = sum((pred-target)^2) / N
    sub and div are decomposed but task count is unchanged.

    Expected: ELEMENTWISE → REDUCE → ELEMENTWISE (3 tasks)
    """
    pred   = load((4,))
    target = load((4,))
    diff   = sub(pred, target)   # = add(pred, mul(target, -1))
    sq     = mul(diff, diff)
    s      = sumop(sq, (0,), (1,))
    n      = const(4.0)
    out    = div(s, n)            # = mul(s, recip(const(4.0)))
    tasks  = run_scheduler(out)
    self.assertEqual(len(tasks), 3)
    self.assertEqual(tasks[0].kind, TaskKind.ELEMENTWISE)
    self.assertEqual(tasks[1].kind, TaskKind.REDUCE)
    self.assertEqual(tasks[2].kind, TaskKind.ELEMENTWISE)

  def test_mse_sub_and_square_fuse(self):
    """
    sub is decomposed to ADD+MUL; squaring is MUL.
    All three ops must fuse into the pre-reduce elementwise task.
    """
    pred   = load((4,))
    target = load((4,))
    diff   = sub(pred, target)   # = add(pred, mul(target, -1))
    sq     = mul(diff, diff)
    s      = sumop(sq, (0,), (1,))
    tasks  = run_scheduler(s)
    elem_ops = [u.op for u in tasks[0].ops]
    self.assertIn(Ops.ADD, elem_ops)  # from decomposed sub
    self.assertIn(Ops.MUL, elem_ops)  # from neg and/or squaring

  def test_two_layer_mlp(self):
    """h = relu(x @ W1 + b1); y = h @ W2 + b2 — BLAS→ELEM→BLAS→ELEM"""
    x  = load((8, 16))
    w1 = load((16, 32))
    b1 = load((32,))
    w2 = load((32, 10))
    b2 = load((10,))
    h1  = matmul(x, w1)
    h1b = add(h1, broadcast(b1, (8, 32)))
    h1r = relu(h1b)
    h2  = matmul(h1r, w2)
    out = add(h2, broadcast(b2, (8, 10)))
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 4)
    kinds = [t.kind for t in tasks]
    self.assertEqual(kinds, [
      TaskKind.BLAS,
      TaskKind.ELEMENTWISE,
      TaskKind.BLAS,
      TaskKind.ELEMENTWISE,
    ])

  def test_layer_norm(self):
    """
    y = (x - mean(x)) / sqrt(var(x) + eps)
    sub and div decompose but reduce structure is unchanged.
    Must have at least 2 reduce tasks in correct order.
    """
    x    = load((4, 8))
    s1   = sumop(x, (1,), (4, 1))
    n    = const(8.0)
    mean = div(s1, n)               # = mul(s1, recip(const(8.0)))
    br_mean = broadcast(mean, (4, 8))
    diff    = sub(x, br_mean)       # = add(x, mul(br_mean, -1))
    sq   = mul(diff, diff)
    s2   = sumop(sq, (1,), (4, 1))
    var  = div(s2, n)
    eps  = const(1e-5)
    out  = div(diff, sqrt(add(var, eps)))
    tasks = run_scheduler(out)
    reduce_count = sum(1 for t in tasks if t.kind == TaskKind.REDUCE)
    self.assertGreaterEqual(reduce_count, 2)
    reduce_indices = [i for i, t in enumerate(tasks) if t.kind == TaskKind.REDUCE]
    self.assertLess(reduce_indices[0], reduce_indices[1])


# **** edge cases ****
class TestSchedulerEdgeCases(unittest.TestCase):
  def test_single_load_no_tasks(self):
    """bare LOAD with no ops — nothing to schedule"""
    tasks = run_scheduler(load((4,)))
    self.assertEqual(len(tasks), 0)

  def test_single_const_no_tasks(self):
    tasks = run_scheduler(const(1.0))
    self.assertEqual(len(tasks), 0)

  def test_movement_only_no_tasks(self):
    """RESHAPE with no consumers — nothing to schedule"""
    a = load((6,))
    tasks = run_scheduler(reshape(a, (2, 3)))
    self.assertEqual(len(tasks), 0)

  def test_multiple_reduces_same_input(self):
    """two reduces over same input — each gets own task"""
    a  = load((4,))
    s1 = sumop(a, (0,), (1,))
    tasks = run_scheduler(s1)
    self.assertEqual(len(tasks), 1)
    self.assertEqual(tasks[0].kind, TaskKind.REDUCE)

  def test_long_chain_stays_one_task(self):
    """10 chained unary ops must fuse into a single kernel"""
    a = load((4,))
    out = a
    for _ in range(10):
      out = relu(out)
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 1)
    self.assertEqual(len(tasks[0].ops), 10)

  def test_neg_fuses_in_chain(self):
    """neg (decomposed to MUL) must fuse — no extra task created"""
    a = load((4,))
    out = neg(relu(a))   # = mul(relu(a), const(-1))
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 1)

  def test_broadcast_scalar_to_2d(self):
    """scalar constant broadcast to full 2D shape — no extra tasks"""
    a = load((3, 4))
    c = const(2.0)
    tasks = run_scheduler(mul(a, c))
    self.assertEqual(len(tasks), 1)
    for ref in tasks[0].inputs:
      self.assertIsNot(ref.uop.op, Ops.CONST)

  def test_cast_in_chain_no_task_break(self):
    """CAST mid-chain must not break fusion"""
    a    = load((4,), dtype=dtypes.float16)
    c    = cast(a, dtypes.float32)
    out  = relu(exp(c))
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 1)

  def test_diamond_pattern_shared_input(self):
    """
    a → relu → add → out
    a → exp  ↗
    Single elementwise task, a deduplicated in inputs.
    """
    a   = load((4,))
    r   = relu(a)
    e   = exp(a)
    out = add(r, e)
    tasks = run_scheduler(out)
    self.assertEqual(len(tasks), 1)
    input_uops = [ref.uop for ref in tasks[0].inputs]
    self.assertEqual(input_uops.count(a), 1)

  def test_output_shape_correct_after_broadcast_fusion(self):
    """fused kernel output shape must be the broadcast shape"""
    a    = load((2, 3))
    b    = broadcast(load((3,)), (2, 3))
    out  = add(a, b)
    tasks = run_scheduler(out)
    self.assertEqual(tasks[0].output_shape, (2, 3))

  def test_scheduler_deterministic(self):
    """running scheduler twice on same graph must give same result"""
    a, b = load((4,)), load((4,))
    out  = relu(add(a, b))
    tasks1 = run_scheduler(out)
    tasks2 = run_scheduler(out)
    self.assertEqual(len(tasks1), len(tasks2))
    for t1, t2 in zip(tasks1, tasks2):
      self.assertEqual(t1.kind, t2.kind)
      self.assertEqual(len(t1.ops), len(t2.ops))
      self.assertEqual(t1.output_shape, t2.output_shape)


if __name__ == "__main__":
  unittest.main(verbosity=2)
