# Coverage for monograd/tensor.py, mixin/math.py, mixin/movement.py:
# Tensor construction, _binop, _unop, _reduceop, cast, to, matmul,
# MathMixin operators, MovementMixin reshape/expand/permute/transpose
import unittest
import numpy as np
from monograd.device import Device
from monograd.dtype import dtypes
from monograd.tensor import Tensor
from monograd.uop import Ops, GroupOp
from monograd.uop.ops import UOp


# **** helpers ****
def uop_chain(uop: UOp) -> list[Ops]:
  """collect all op types from root down to leaves"""
  ops, visited = [], set()
  def walk(u):
    if id(u) in visited: return
    visited.add(id(u))
    ops.append(u.op)
    for s in u.src: walk(s)
  walk(uop)
  return ops

def has_op(t: Tensor, op: Ops) -> bool:
  return op in uop_chain(t.uop)


# **** tensor construction ****
class TestTensorConstruction(unittest.TestCase):
  def test_from_list_shape(self):
    t = Tensor([1, 2, 3, 4])
    self.assertEqual(t.shape, (4,))

  def test_from_list_2d_shape(self):
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    self.assertEqual(t.shape, (2, 3))

  def test_from_numpy_shape(self):
    arr = np.zeros((3, 4), dtype=np.float32)
    t   = Tensor(arr)
    self.assertEqual(t.shape, (3, 4))

  def test_from_numpy_size_not_len(self):
    """regression — must use buf.size not len(buf) for multi-dim arrays"""
    arr = np.ones((3, 4), dtype=np.float32)
    t   = Tensor(arr)
    # buffer must hold 12 elements, not 3
    self.assertEqual(t.uop.buffer.size, 12)

  def test_from_list_default_dtype_float32(self):
    t = Tensor([1, 2, 3])
    self.assertIs(t.dtype, dtypes.float32)

  def test_from_list_explicit_dtype(self):
    t = Tensor([1, 2, 3], dtype="int32")
    self.assertIs(t.dtype, dtypes.int32)

  def test_from_int_scalar_creates_const(self):
    t = Tensor(2)
    self.assertIs(t.uop.op, Ops.CONST)

  def test_from_float_scalar_creates_const(self):
    t = Tensor(3.14)
    self.assertIs(t.uop.op, Ops.CONST)

  def test_from_bool_scalar_creates_const(self):
    t = Tensor(True)
    self.assertIs(t.uop.op, Ops.CONST)

  def test_from_uop_passthrough(self):
    uop = UOp(Ops.LOAD, dtypes.float32, (), ((4,), Device.CPU))
    uop.assign_buffer(4)
    t = Tensor(uop)
    self.assertIs(t.uop, uop)

  def test_from_uop_dtype_mismatch_raises(self):
    uop = UOp(Ops.LOAD, dtypes.float32, (), ((4,), Device.CPU))
    uop.assign_buffer(4)
    with self.assertRaises(AssertionError):
      Tensor(uop, dtype="int32")

  def test_requires_grad_default_true(self):
    t = Tensor([1, 2, 3])
    self.assertTrue(t.requires_grad)

  def test_requires_grad_false(self):
    t = Tensor([1, 2, 3], requires_grad=False)
    self.assertFalse(t.requires_grad)

  def test_device_default_cpu(self):
    t = Tensor([1, 2, 3])
    self.assertEqual(t.device, Device.CPU)

  def test_name_stored(self):
    t = Tensor([1, 2, 3], name="weights")
    self.assertEqual(t.name, "weights")

  def test_grad_initially_none(self):
    t = Tensor([1, 2, 3])
    self.assertIsNone(t.grad)

  def test_ndim(self):
    self.assertEqual(Tensor([1, 2, 3]).ndim, 1)
    self.assertEqual(Tensor([[1, 2], [3, 4]]).ndim, 2)

  def test_parents_are_uop_src(self):
    t = Tensor([1, 2, 3])
    self.assertEqual(t.parents, t.uop.src)


# **** _binop - shape, dtype, device ****
class TestTensorBinop(unittest.TestCase):
  def setUp(self):
    self.a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    self.b = Tensor([[1.0, 1.0], [1.0, 1.0]])

  def test_output_shape_same_shape(self):
    c = self.a + self.b
    self.assertEqual(c.shape, (2, 2))

  def test_output_dtype_same_dtype(self):
    c = self.a + self.b
    self.assertIs(c.dtype, dtypes.float32)

  def test_dtype_promotion_int_plus_float(self):
    a = Tensor([1, 2, 3], dtype="int32")
    b = Tensor([1.0, 2.0, 3.0], dtype="float32")
    c = a + b
    self.assertIs(c.dtype, dtypes.float32)

  def test_dtype_promotion_inserts_cast(self):
    a = Tensor([1, 2, 3], dtype="int32")
    b = Tensor([1.0, 2.0, 3.0], dtype="float32")
    c = a + b
    self.assertTrue(has_op(c, Ops.CAST), "CAST should be inserted for promotion")

  def test_no_cast_when_same_dtype(self):
    a = Tensor([1.0, 2.0], dtype="float32")
    b = Tensor([1.0, 2.0], dtype="float32")
    c = a + b
    self.assertFalse(has_op(c, Ops.CAST), "no CAST needed for same dtype")

  def test_broadcast_1d_to_2d(self):
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2,3)
    b = Tensor([1.0, 2.0, 3.0])                     # (3,)
    c = a + b
    self.assertEqual(c.shape, (2, 3))

  def test_broadcast_inserts_reshape_and_expand(self):
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2,2)
    b = Tensor([1.0, 2.0])                # (2,)
    c = a + b
    self.assertTrue(has_op(c, Ops.RESHAPE))
    self.assertTrue(has_op(c, Ops.EXPAND))

  def test_requires_grad_propagates(self):
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([1.0, 2.0], requires_grad=False)
    c = a + b
    self.assertTrue(c.requires_grad)

  def test_requires_grad_false_when_both_false(self):
    a = Tensor([1.0, 2.0], requires_grad=False)
    b = Tensor([1.0, 2.0], requires_grad=False)
    c = a + b
    self.assertFalse(c.requires_grad)

  def test_device_mismatch_raises(self):
    a = Tensor([1.0, 2.0], device=Device.CPU)
    b_uop = UOp(Ops.LOAD, dtypes.float32, (), ((2,), Device.GPU))
    b_uop.assign_buffer(2)
    b = Tensor(b_uop)
    with self.assertRaises(AssertionError):
      _ = a + b

  def test_incompatible_shapes_raise(self):
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])  # (2,2)
    b = Tensor([1.0, 2.0, 3.0])           # (3,) — incompatible
    with self.assertRaises(AssertionError):
      _ = a + b

  def test_reverse_flag_swaps_operands(self):
    """reverse=True used by __radd__ etc — lhs/rhs should swap"""
    a = Tensor([1.0, 2.0])
    b = Tensor(2)
    normal  = a._binop(Ops.SUB, b, reverse=False)
    reverse = a._binop(Ops.SUB, b, reverse=True)
    # the src order of the UOp should differ
    self.assertNotEqual(normal.uop.src, reverse.uop.src)


# **** _unop ****
class TestTensorUnop(unittest.TestCase):
  def setUp(self):
    self.a = Tensor([1.0, 2.0, 3.0])

  def test_unop_preserves_shape(self):
    for op in GroupOp.Unary - {Ops.CAST}:
      r = self.a._unop(op)
      self.assertEqual(r.shape, self.a.shape, f"{op} should preserve shape")

  def test_unop_preserves_dtype(self):
    for op in GroupOp.Unary - {Ops.CAST}:
      r = self.a._unop(op)
      self.assertIs(r.dtype, self.a.dtype, f"{op} should preserve dtype")

  def test_unop_correct_op_in_uop(self):
    for op in GroupOp.Unary - {Ops.CAST}:
      r = self.a._unop(op)
      self.assertIs(r.uop.op, op, f"uop.op should be {op}")

  def test_unop_src_is_self(self):
    r = self.a._unop(Ops.NEG)
    self.assertIn(self.a.uop, r.uop.src)


# **** _reduceop ****
class TestTensorReduceop(unittest.TestCase):
  def setUp(self):
    self.a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # (2,3)

  def test_reduce_all_axes(self):
    r = self.a._reduceop(Ops.SUM, axis=None)
    self.assertEqual(r.shape, (1,))

  def test_reduce_axis0(self):
    r = self.a._reduceop(Ops.SUM, axis=0)
    self.assertEqual(r.shape, (3,))

  def test_reduce_axis1(self):
    r = self.a._reduceop(Ops.SUM, axis=1)
    self.assertEqual(r.shape, (2,))

  def test_keepdim_true_axis0(self):
    r = self.a._reduceop(Ops.SUM, axis=0, keepdim=True)
    self.assertEqual(r.shape, (1, 3))

  def test_keepdim_true_axis1(self):
    r = self.a._reduceop(Ops.SUM, axis=1, keepdim=True)
    self.assertEqual(r.shape, (2, 1))

  def test_negative_axis(self):
    r = self.a._reduceop(Ops.SUM, axis=-1)
    self.assertEqual(r.shape, (2,))

  def test_negative_axis_keepdim(self):
    r = self.a._reduceop(Ops.SUM, axis=-1, keepdim=True)
    self.assertEqual(r.shape, (2, 1))

  def test_tuple_axis(self):
    r = self.a._reduceop(Ops.SUM, axis=(0, 1))
    self.assertEqual(r.shape, (1,))

  def test_preserves_dtype(self):
    r = self.a._reduceop(Ops.SUM, axis=0)
    self.assertIs(r.dtype, self.a.dtype)

  def test_invalid_axis_raises(self):
    with self.assertRaises((ValueError, TypeError)):
      self.a._reduceop(Ops.SUM, axis="bad")  # type: ignore

  def test_0d_scalar_reduce(self):
    """0D tensor reduce should produce axis=()"""
    scalar_uop = UOp(Ops.CONST, dtypes.float32, (), (1.0, Device.CPU))
    t = Tensor(scalar_uop)
    # should not raise
    r = t._reduceop(Ops.SUM)
    self.assertIsNotNone(r)


# **** cast ****
class TestTensorCast(unittest.TestCase):
  def test_cast_same_dtype_noop(self):
    t = Tensor([1.0, 2.0])
    self.assertIs(t.cast(dtypes.float32), t)

  def test_cast_different_dtype(self):
    t    = Tensor([1.0, 2.0])
    cast = t.cast(dtypes.float16)
    self.assertIs(cast.dtype, dtypes.float16)

  def test_cast_creates_cast_uop(self):
    t    = Tensor([1.0, 2.0])
    cast = t.cast(dtypes.int32)
    self.assertIs(cast.uop.op, Ops.CAST)

  def test_cast_preserves_shape(self):
    t    = Tensor([[1.0, 2.0], [3.0, 4.0]])
    cast = t.cast(dtypes.int32)
    self.assertEqual(cast.shape, t.shape)

  def test_cast_string_dtype(self):
    t    = Tensor([1.0, 2.0])
    cast = t.cast("int32")
    self.assertIs(cast.dtype, dtypes.int32)

  def test_cast_preserves_requires_grad(self):
    t    = Tensor([1.0, 2.0], requires_grad=True)
    cast = t.cast(dtypes.float16)
    self.assertTrue(cast.requires_grad)


# **** to(device) ****
class TestTensorTo(unittest.TestCase):
  def test_to_same_device_noop(self):
    t = Tensor([1.0, 2.0], device=Device.CPU)
    self.assertIs(t.to(Device.CPU), t)

  def test_to_same_device_string_noop(self):
    t = Tensor([1.0, 2.0], device=Device.CPU)
    self.assertIs(t.to("cpu"), t)

  def test_to_different_device_creates_copy_uop(self):
    t    = Tensor([1.0, 2.0], device=Device.CPU)
    copy = t.to(Device.GPU)
    self.assertIs(copy.uop.op, Ops.COPY)

  def test_to_different_device_preserves_shape(self):
    t    = Tensor([[1.0, 2.0], [3.0, 4.0]])
    copy = t.to(Device.GPU)
    self.assertEqual(copy.shape, t.shape)

  def test_to_different_device_preserves_dtype(self):
    t    = Tensor([1.0, 2.0])
    copy = t.to(Device.GPU)
    self.assertIs(copy.dtype, t.dtype)

  def test_to_string_device(self):
    t    = Tensor([1.0, 2.0])
    copy = t.to("gpu")
    self.assertIs(copy.uop.op, Ops.COPY)


# **** matmul/gemm ****
class TestTensorMatmul(unittest.TestCase):
  def test_matmul_output_shape(self):
    a  = Tensor(np.zeros((4, 8), dtype=np.float32))
    b  = Tensor(np.zeros((8, 16), dtype=np.float32))
    mm = a @ b
    self.assertEqual(mm.shape, (4, 16))

  def test_matmul_op_is_matmul(self):
    a  = Tensor(np.zeros((4, 8), dtype=np.float32))
    b  = Tensor(np.zeros((8, 16), dtype=np.float32))
    mm = a @ b
    self.assertIs(mm.uop.op, Ops.MATMUL)

  def test_matmul_shape_mismatch_raises(self):
    a = Tensor(np.zeros((4, 8), dtype=np.float32))
    b = Tensor(np.zeros((9, 16), dtype=np.float32))
    with self.assertRaises(AssertionError):
      _ = a @ b

  def test_matmul_dtype_promotion(self):
    a  = Tensor(np.zeros((4, 8), dtype=np.float32))
    b  = Tensor(np.zeros((8, 16), dtype=np.float16))
    mm = a @ b
    self.assertIs(mm.dtype, dtypes.float32)

  def test_matmul_promotes_inserts_cast(self):
    a  = Tensor(np.zeros((4, 8), dtype=np.float16))
    b  = Tensor(np.zeros((8, 16), dtype=np.float32))
    mm = a @ b
    self.assertTrue(has_op(mm, Ops.CAST))

  def test_matmul_requires_grad_propagates(self):
    a  = Tensor(np.zeros((4, 8), dtype=np.float32), requires_grad=True)
    b  = Tensor(np.zeros((8, 16), dtype=np.float32), requires_grad=False)
    mm = a @ b
    self.assertTrue(mm.requires_grad)

  def test_matmul_non_2d_raises(self):
    a = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    b = Tensor(np.zeros((4, 5), dtype=np.float32))
    with self.assertRaises(AssertionError):
      _ = a @ b


# **** mathmixin ****
class TestMathMixin(unittest.TestCase):
  def setUp(self):
    self.a = Tensor([1.0, 2.0, 3.0])
    self.b = Tensor([4.0, 5.0, 6.0])

  def _assert_op(self, t: Tensor, op: Ops):
    self.assertIs(t.uop.op, op, f"expected {op}, got {t.uop.op}")

  # binary operators
  def test_add(self):        self._assert_op(self.a + self.b, Ops.ADD)
  def test_sub(self):        self._assert_op(self.a - self.b, Ops.SUB)
  def test_mul(self):        self._assert_op(self.a * self.b, Ops.MUL)
  def test_truediv(self):    self._assert_op(self.a / self.b, Ops.DIV)
  def test_pow(self):        self._assert_op(self.a ** self.b, Ops.POW)

  # reflected operators — scalar on left
  def test_radd_scalar(self):
    r = 1.0 + self.a
    self._assert_op(r, Ops.ADD)

  def test_rsub_scalar(self):
    r = 1.0 - self.a
    self._assert_op(r, Ops.SUB)

  def test_rmul_scalar(self):
    r = 2.0 * self.a
    self._assert_op(r, Ops.MUL)

  def test_rtruediv_scalar(self):
    r = 1.0 / self.a
    self._assert_op(r, Ops.DIV)

  # unary operators
  def test_neg(self):        self._assert_op(-self.a, Ops.NEG)
  def test_log(self):        self._assert_op(self.a.log(), Ops.LOG)
  def test_exp(self):        self._assert_op(self.a.exp(), Ops.EXP)
  def test_sqrt(self):       self._assert_op(self.a.sqrt(), Ops.SQRT)
  def test_relu(self):       self._assert_op(self.a.relu(), Ops.RELU)
  def test_sin(self):        self._assert_op(self.a.sin(), Ops.SIN)

  # reduce
  def test_sum_no_axis(self):
    r = self.a.sum()
    self.assertIs(r.uop.op if r.uop.op is Ops.SUM else r.uop.src[0].op, Ops.SUM)

  def test_matmul_operator(self):
    a = Tensor(np.zeros((3, 4), dtype=np.float32))
    b = Tensor(np.zeros((4, 5), dtype=np.float32))
    self._assert_op(a @ b, Ops.MATMUL)

  def test_ufix_wraps_scalar_in_const_like(self):
    """adding a Python int should produce a CONST node in the graph"""
    r = self.a + 1
    self.assertTrue(has_op(r, Ops.CONST))

  def test_ufix_passes_tensor_through(self):
    """adding a Tensor should not produce an extra CONST"""
    r = self.a + self.b
    # both sources of ADD should be LOAD nodes not CONST
    add_uop = r.uop
    src_ops = [s.op for s in add_uop.src]
    self.assertNotIn(Ops.CONST, src_ops)


# **** movementmixin - reshape ****
class TestReshape(unittest.TestCase):
  def setUp(self):
    self.t = Tensor(np.zeros((2, 3), dtype=np.float32))  # 6 elements

  def test_reshape_same_shape_noop(self):
    r = self.t.reshape((2, 3))
    self.assertIs(r, self.t)

  def test_reshape_to_1d(self):
    r = self.t.reshape((6,))
    self.assertEqual(r.shape, (6,))

  def test_reshape_to_different_2d(self):
    r = self.t.reshape((3, 2))
    self.assertEqual(r.shape, (3, 2))

  def test_reshape_with_minus1(self):
    r = self.t.reshape((-1,))
    self.assertEqual(r.shape, (6,))

  def test_reshape_with_minus1_partial(self):
    r = self.t.reshape((-1, 3))
    self.assertEqual(r.shape, (2, 3))

  def test_reshape_size_mismatch_raises(self):
    with self.assertRaises(AssertionError):
      self.t.reshape((2, 4))

  def test_reshape_multiple_minus1_raises(self):
    with self.assertRaises(AssertionError):
      self.t.reshape((-1, -1))

  def test_reshape_inserts_reshape_uop(self):
    r = self.t.reshape((6,))
    self.assertIs(r.uop.op, Ops.RESHAPE)

  def test_reshape_preserves_dtype(self):
    r = self.t.reshape((6,))
    self.assertIs(r.dtype, self.t.dtype)


# **** movementmixin - expand ****
class TestExpand(unittest.TestCase):
  def test_expand_1d_to_2d(self):
    t = Tensor(np.zeros((1, 4), dtype=np.float32))
    e = t.expand((3, 4))
    self.assertEqual(e.shape, (3, 4))

  def test_expand_same_shape_noop(self):
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    e = t.expand((3, 4))
    self.assertIs(e, t)

  def test_expand_inserts_expand_uop(self):
    t = Tensor(np.zeros((1, 4), dtype=np.float32))
    e = t.expand((3, 4))
    self.assertIs(e.uop.op, Ops.EXPAND)

  def test_expand_preserves_dtype(self):
    t = Tensor(np.zeros((1, 4), dtype=np.float32))
    e = t.expand((3, 4))
    self.assertIs(e.dtype, t.dtype)

  def test_expand_non_broadcast_dim_raises(self):
    """can only expand dims that are size 1"""
    t = Tensor(np.zeros((2, 4), dtype=np.float32))
    with self.assertRaises(AssertionError):
      t.expand((3, 4))


# **** movementmixin - permute ****
class TestPermute(unittest.TestCase):
  def test_permute_2d_transpose(self):
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    p = t.permute((1, 0))
    self.assertEqual(p.shape, (4, 3))

  def test_permute_3d(self):
    t = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    p = t.permute((2, 0, 1))
    self.assertEqual(p.shape, (4, 2, 3))

  def test_permute_inserts_permute_uop(self):
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    p = t.permute((1, 0))
    self.assertIs(p.uop.op, Ops.PERMUTE)

  def test_permute_arg_stored(self):
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    p = t.permute((1, 0))
    self.assertEqual(p.uop.arg, (1, 0))

  def test_permute_wrong_length_raises(self):
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    with self.assertRaises(AssertionError):
      t.permute((0,))

  def test_permute_invalid_axes_raises(self):
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    with self.assertRaises(AssertionError):
      t.permute((0, 2))  # axis 2 doesn't exist for 2D

  def test_permute_identity_noop(self):
    """permuting with identity order should behave as noop shape-wise"""
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    p = t.permute((0, 1))
    self.assertEqual(p.shape, (3, 4))


# **** movementmixin - transpose ****
class TestTranspose(unittest.TestCase):
  def test_transpose_default_swaps_last_two(self):
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    r = t.transpose()
    self.assertEqual(r.shape, (4, 3))

  def test_transpose_explicit_axes(self):
    t = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    r = t.transpose(0, 2)
    self.assertEqual(r.shape, (4, 3, 2))

  def test_transpose_is_permute(self):
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    r = t.transpose()
    self.assertIs(r.uop.op, Ops.PERMUTE)

  def test_T_property_reverses_all_axes(self):
    t = Tensor(np.zeros((2, 3, 4), dtype=np.float32))
    r = t.T
    self.assertEqual(r.shape, (4, 3, 2))

  def test_T_calls_transpose(self):
    t = Tensor(np.zeros((3, 4), dtype=np.float32))
    self.assertEqual(t.T.shape, t.transpose().shape)

  def test_T_2d(self):
    t = Tensor(np.zeros((5, 7), dtype=np.float32))
    self.assertEqual(t.T.shape, (7, 5))


# **** get_broadcasted_shape ****
class TestGetBroadcastedShape(unittest.TestCase):
  def test_same_shape(self):
    a = Tensor(np.zeros((3, 4), dtype=np.float32))
    b = Tensor(np.zeros((3, 4), dtype=np.float32))
    c = a + b
    self.assertEqual(c.shape, (3, 4))

  def test_1d_broadcasts_with_2d(self):
    a = Tensor(np.zeros((2, 3), dtype=np.float32))
    b = Tensor(np.zeros((3,), dtype=np.float32))
    c = a + b
    self.assertEqual(c.shape, (2, 3))

  def test_scalar_broadcasts_with_nd(self):
    a = Tensor(np.zeros((2, 3), dtype=np.float32))
    b = Tensor(2.0)
    c = a + b
    self.assertEqual(c.shape, (2, 3))

  def test_incompatible_shapes_raise(self):
    a = Tensor(np.zeros((2, 3), dtype=np.float32))
    b = Tensor(np.zeros((2, 4), dtype=np.float32))
    with self.assertRaises(AssertionError):
      _ = a + b


if __name__ == "__main__":
  unittest.main(verbosity=2)
