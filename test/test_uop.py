# Coverage for monograd/uop/ops.py: 
# UOpMetaClass, UOp (shape, device, buffer, assign_buffer, cast, has_buffer_assigned, __hash__, __repr__)
import gc
import unittest
import numpy as np
from monograd.device import Buffer, Device
from monograd.dtype import dtypes
from monograd.uop import GroupOp, Ops
from monograd.uop.ops import UOp, UOpMetaClass, _uop_buffers


# **** helpers ****
F32 = dtypes.float32
I32 = dtypes.int32
CPU = Device.CPU
GPU = Device.GPU

def make_load(shape:tuple, dtype=F32, device=CPU) -> UOp:
  """LOAD UOp with an allocated buffer."""
  u = UOp(Ops.LOAD, dtype, (), (shape, device))
  u.assign_buffer(int(np.prod(shape)))
  return u

def make_const(val, dtype=F32, device=CPU) -> UOp:
  return UOp(Ops.CONST, dtype, (), (val, device))


# **** duplication & caching ****
class TestUOpMetaClass(unittest.TestCase):
  def test_identical_args_same_object(self):
    a = make_const(1.0)
    b = make_const(1.0)
    self.assertIs(a, b)

  def test_different_val_different_object(self):
    a = make_const(1.0)
    b = make_const(2.0)
    self.assertIsNot(a, b)

  def test_different_dtype_different_object(self):
    a = UOp(Ops.CONST, dtypes.float32, (), (1.0, CPU))
    b = UOp(Ops.CONST, dtypes.int32,   (), (1.0, CPU))
    self.assertIsNot(a, b)

  def test_cache_is_weak(self):
    """GC must be able to evict UOps — cache must not hold strong refs"""
    _ = UOp(Ops.CONST, dtypes.float32, (), (999.0, CPU))
    before = len(UOpMetaClass.ucache)
    del _
    gc.collect()
    after = len(UOpMetaClass.ucache)
    self.assertLessEqual(after, before)


# **** shape ****
class TestUOpShape(unittest.TestCase):
  def test_load_shape_from_arg(self):
    self.assertEqual(make_load((3, 4)).shape, (3, 4))

  def test_load_1d(self):
    self.assertEqual(make_load((8,)).shape, (8,))

  def test_const_shape_always_1(self):
    self.assertEqual(make_const(2.0).shape, (1,))

  def test_copy_inherits_src_shape(self):
    src = make_load((2, 3))
    cpy = UOp(Ops.COPY, F32, (src,), GPU)
    self.assertEqual(cpy.shape, (2, 3))

  def test_reshape_shape_from_arg(self):
    src = make_load((6,))
    r   = UOp(Ops.RESHAPE, F32, (src,), (2, 3))
    self.assertEqual(r.shape, (2, 3))

  def test_expand_shape_from_arg(self):
    src = make_load((1, 3))
    e   = UOp(Ops.EXPAND, F32, (src,), (4, 3))
    self.assertEqual(e.shape, (4, 3))

  def test_permute_reorders_dims(self):
    src = make_load((3, 4))
    p   = UOp(Ops.PERMUTE, F32, (src,), (1, 0))
    self.assertEqual(p.shape, (4, 3))

  def test_permute_3d(self):
    src = make_load((2, 3, 4))
    p   = UOp(Ops.PERMUTE, F32, (src,), (2, 0, 1))
    self.assertEqual(p.shape, (4, 2, 3))

  def test_unary_inherits_src_shape(self):
    src = make_load((4, 8))
    for op in GroupOp.Unary:
      u = UOp(op, F32, (src,), CPU)
      self.assertEqual(u.shape, (4, 8), f"Unary {op} should inherit shape")

  def test_binary_inherits_src0_shape(self):
    a = make_load((4, 8))
    b = make_load((4, 8))
    for op in GroupOp.Binary:
      u = UOp(op, F32, (a, b), CPU)
      self.assertEqual(u.shape, (4, 8), f"Binary {op} should inherit shape")

  def test_reduce_shape_from_arg(self):
    src    = make_load((4, 8))
    reduce = UOp(Ops.SUM, F32, (src,), ((1,), (4, 1)))
    self.assertEqual(reduce.shape, (4, 1))

  def test_matmul_shape(self):
    a  = make_load((4, 8))
    b  = make_load((8, 16))
    mm = UOp(Ops.MATMUL, F32, (a, b))
    self.assertEqual(mm.shape, (4, 16))

  def test_matmul_batched_shape(self):
    a  = make_load((2, 4, 8))
    b  = make_load((8, 16))
    mm = UOp(Ops.MATMUL, F32, (a, b))
    self.assertEqual(mm.shape, (2, 4, 16))

  def test_unknown_op_raises(self):
    with self.assertRaises(NotImplementedError):
      _ = UOp(Ops.SINK, F32, ()).shape


# **** device ****
class TestUOpDevice(unittest.TestCase):
  def test_load_cpu(self):
    self.assertEqual(make_load((4,), device=CPU).device, CPU)

  def test_load_gpu(self):
    self.assertEqual(make_load((4,), device=GPU).device, GPU)

  def test_const_device_from_arg(self):
    self.assertEqual(UOp(Ops.CONST, F32, (), (1.0, CPU)).device, CPU)
    self.assertEqual(UOp(Ops.CONST, F32, (), (2.0, GPU)).device, GPU)

  def test_copy_device_from_arg(self):
    src = make_load((4,), device=CPU)
    cpy = UOp(Ops.COPY, F32, (src,), GPU)
    self.assertEqual(cpy.device, GPU)

  def test_cast_device_from_arg(self):
    src  = make_load((4,), device=CPU)
    cast = UOp(Ops.CAST, dtypes.float16, (src,), CPU)
    self.assertEqual(cast.device, CPU)

  def test_movement_inherits_from_src(self):
    src = make_load((4,), device=CPU)
    r   = UOp(Ops.RESHAPE, F32, (src,), (2, 2))
    e   = UOp(Ops.EXPAND,  F32, (src,), (4,))
    self.assertEqual(r.device, CPU)
    self.assertEqual(e.device, CPU)

  def test_unary_device_from_arg(self):
    src = make_load((4,), device=CPU)
    neg = UOp(Ops.NEG, F32, (src,), CPU)
    self.assertEqual(neg.device, CPU)

  def test_binary_device_from_arg(self):
    a   = make_load((4,), device=CPU)
    b   = make_load((4,), device=CPU)
    add = UOp(Ops.ADD, F32, (a, b), CPU)
    self.assertEqual(add.device, CPU)

  def test_reduce_inherits_from_src(self):
    src    = make_load((4,), device=CPU)
    reduce = UOp(Ops.SUM, F32, (src,), ((0,), (1,)))
    self.assertEqual(reduce.device, CPU)


# **** .assign_buffer / .buffer / .has_buffer_assigned ****
class TestUOpBuffer(unittest.TestCase):
  def tearDown(self):
    # clear buffer cache after each test
    _uop_buffers.clear()
    UOpMetaClass.ucache.clear()

  def test_assign_buffer_allocates(self):
    u   = UOp(Ops.LOAD, F32, (), ((4,), CPU))
    buf = u.assign_buffer(4)
    self.assertIsInstance(buf, Buffer)
    self.assertTrue(buf.is_allocated())

  def test_assign_buffer_idempotent(self):
    u    = UOp(Ops.LOAD, F32, (), ((4,), CPU))
    buf1 = u.assign_buffer(4)
    buf2 = u.assign_buffer(4)
    self.assertIs(buf1, buf2)

  def test_buffer_property_returns_assigned(self):
    u   = UOp(Ops.LOAD, F32, (), ((4,), CPU))
    buf = u.assign_buffer(4)
    self.assertIs(u.buffer, buf)

  def test_buffer_stored_in_weak_dict(self):
    u = UOp(Ops.LOAD, F32, (), ((4,), CPU))
    u.assign_buffer(4)
    self.assertIn(u, _uop_buffers)

  def test_assign_buffer_with_initial_value(self):
    u   = UOp(Ops.LOAD, F32, (), ((4,), CPU))
    arr = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf = u.assign_buffer(4, arr)
    out = memoryview(np.empty(4, dtype=np.float32))
    buf.copyout(out)
    np.testing.assert_array_equal(np.frombuffer(out, dtype=np.float32), arr)

  def test_buffer_property_raises_without_assignment(self):
    u = UOp(Ops.LOAD, F32, (), ((999,), CPU))
    with self.assertRaises(KeyError):
      _ = u.buffer


# **** cast ****
class TestUOpCast(unittest.TestCase):
  def test_cast_same_dtype_is_noop(self):
    src = make_load((4,))
    self.assertIs(src.cast(F32), src)

  def test_cast_different_dtype_creates_cast_uop(self):
    src  = make_load((4,))
    cast = src.cast(dtypes.float16)
    self.assertIs(cast.op, Ops.CAST)
    self.assertIs(cast.dtype, dtypes.float16)
    self.assertEqual(cast.src, (src,))

  def test_cast_correct_device(self):
    src  = make_load((4,), device=CPU)
    cast = src.cast(dtypes.int32)
    self.assertEqual(cast.device, CPU)

  def test_cast_correct_shape(self):
    src  = make_load((3, 4))
    cast = src.cast(dtypes.float16)
    self.assertEqual(cast.shape, (3, 4))

  def test_cast_chain(self):
    src = make_load((4,))
    c1  = src.cast(dtypes.float16)
    c2  = c1.cast(dtypes.float32)
    self.assertIs(c2.op, Ops.CAST)
    self.assertIs(c2.dtype, dtypes.float32)
    self.assertEqual(c2.src, (c1,))

  def test_cast_to_int(self):
    src  = make_load((4,))
    cast = src.cast(dtypes.int32)
    self.assertIs(cast.dtype, dtypes.int32)


# **** hashing & id ****
class TestUOpHashAndIdentity(unittest.TestCase):
  def test_hash_equals_id(self):
    u = make_load((4,))
    self.assertEqual(hash(u), id(u))

  def test_stable_hash(self):
    u = make_load((4,))
    self.assertEqual(hash(u), hash(u))

  def test_usable_as_dict_key(self):
    u = make_load((4,))
    d = {u: "value"}
    self.assertEqual(d[u], "value")

  def test_usable_in_set(self):
    u = make_load((4,))
    s = {u}
    self.assertIn(u, s)


# **** __repr__ ****
class TestUOpRepr(unittest.TestCase):
  def test_repr_contains_op_name(self):
    self.assertIn("CONST", repr(make_const(1.0)))
    self.assertIn("LOAD",  repr(make_load((4,))))

  def test_repr_contains_dtype(self):
    self.assertIn("float", repr(make_const(1.0)))

  def test_repr_is_string(self):
    self.assertIsInstance(repr(make_load((4,))), str)


# **** group op membership ****
class TestGroupOpMembership(unittest.TestCase):
  def test_unary_ops(self):
    for op in (Ops.NEG, Ops.RELU, Ops.LOG, Ops.EXP, Ops.SQRT, Ops.CAST):
      self.assertIn(op, GroupOp.Unary, f"{op} should be in Unary")

  def test_binary_ops(self):
    for op in (Ops.ADD, Ops.SUB, Ops.MUL, Ops.DIV, Ops.MAX,
               Ops.POW, Ops.MOD, Ops.OR, Ops.XOR, Ops.AND):
      self.assertIn(op, GroupOp.Binary, f"{op} should be in Binary")

  def test_movement_ops(self):
    for op in (Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE):
      self.assertIn(op, GroupOp.Movement, f"{op} should be in Movement")

  def test_reduce_ops(self):
    for op in (Ops.SUM, Ops.REDUCEMAX):
      self.assertIn(op, GroupOp.Reduce, f"{op} should be in Reduce")

  def test_blas_ops(self):
    self.assertIn(Ops.MATMUL, GroupOp.BLAS)

  def test_alu_is_union(self):
    self.assertEqual(GroupOp.ALU, GroupOp.Unary | GroupOp.Binary | GroupOp.Ternary)

  def test_unary_and_binary_disjoint(self):
    self.assertTrue(GroupOp.Unary.isdisjoint(GroupOp.Binary))

  def test_movement_not_in_alu(self):
    self.assertTrue(GroupOp.Movement.isdisjoint(GroupOp.ALU))

  def test_all_contains_every_op(self):
    for op in Ops:
      self.assertIn(op, GroupOp.All, f"{op} missing from GroupOp.All")


if __name__ == "__main__":
  unittest.main(verbosity=2)
