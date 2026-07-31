import unittest
from monograd.device import Device
from monograd.dtype import dtypes
from monograd.engine.codegen import codegen, _codegen_matmul_tiled
from monograd.engine.schedule import TaskKind, run_scheduler
from monograd.uop import Ops
from monograd.uop.ops import UOp

GPU = Device.GPU

def load(shape:tuple[int, ...], dtype=dtypes.float32) -> UOp:
  return UOp(Ops.LOAD, dtype, (), (shape, GPU))
def matmul(a:UOp, b:UOp) -> UOp: return UOp(Ops.MATMUL, a.dtype, (a, b), GPU)
def relu(x:UOp) -> UOp: return UOp(Ops.RELU, x.dtype, (x,), GPU)
def add(a:UOp, b:UOp) -> UOp: return UOp(Ops.ADD, a.dtype, (a, b), GPU)
def mul(a:UOp, b:UOp) -> UOp: return UOp(Ops.MUL, a.dtype, (a, b), GPU)
def reshape(x:UOp, shape:tuple[int, ...]) -> UOp: return UOp(Ops.RESHAPE, x.dtype, (x,), shape)
def expand(x:UOp, shape:tuple[int, ...]) -> UOp: return UOp(Ops.EXPAND, x.dtype, (x,), shape)
def permute(x:UOp, order:tuple[int, ...]) -> UOp: return UOp(Ops.PERMUTE, x.dtype, (x,), order)
def const(value, dtype=dtypes.float32) -> UOp: return UOp(Ops.CONST, dtype, (), (value, GPU))
def cast(x:UOp, dtype) -> UOp: return UOp(Ops.CAST, dtype, (x,), GPU)

class TestMatmulCodegen(unittest.TestCase):
  def test_float32_selects_tiled_kernel(self):
    task = run_scheduler(matmul(load((129, 17)), load((17, 130))))[-1]
    kernel = codegen(task)
    self.assertEqual(task.kind, TaskKind.BLAS)
    self.assertEqual(kernel.local_size, (16, 16, 1))
    self.assertEqual(kernel.global_size, (32, 32, 1))
    self.assertTrue(kernel.name.endswith("_tiled_128x128x16_8x8"))
    self.assertIn("__local float tile_a[128][17]", kernel.source)
    self.assertIn("global_row_a < 129 && global_col_a < 17", kernel.source)
    self.assertIn("global_row_b < 17 && global_col_b < 130", kernel.source)

  def test_tuning_parameters_control_geometry_and_identity(self):
    task = run_scheduler(matmul(load((129, 17)), load((17, 130))))[-1]
    kernel = _codegen_matmul_tiled(task, tile_m=64, tile_n=32, tile_k=8, work_m=4, work_n=4)
    self.assertEqual(kernel.local_size, (8, 16, 1))
    self.assertEqual(kernel.global_size, (40, 48, 1))
    self.assertTrue(kernel.name.endswith("_tiled_64x32x8_4x4"))
    self.assertIn("__local float tile_a[64][9]", kernel.source)
    self.assertIn("__local float tile_b[8][32]", kernel.source)

  def test_integer_matmul_uses_naive_kernel(self):
    task = run_scheduler(matmul(load((3, 4), dtypes.int32), load((4, 5), dtypes.int32)))[-1]
    kernel = codegen(task)
    self.assertEqual(kernel.global_size, (15,))
    self.assertIsNone(kernel.local_size)
    self.assertTrue(kernel.name.endswith("_naive"))
    self.assertIn("for (int k = 0; k < 4; k++)", kernel.source)
    self.assertNotIn("fma(", kernel.source)

  def test_boolean_matmul_is_rejected(self):
    task = run_scheduler(matmul(load((1, 4), dtypes.bool), load((4, 1), dtypes.bool)))[0]
    with self.assertRaisesRegex(AssertionError, "boolean MATMUL is unsupported"): codegen(task)

  def test_elementwise_matmul_input_is_fused(self):
    x, w = load((4, 8)), load((8, 16))
    activated = relu(x)
    tasks = run_scheduler(matmul(activated, w))
    self.assertEqual([task.kind for task in tasks], [TaskKind.BLAS])
    self.assertEqual([u.op for u in tasks[0].ops], [Ops.RELU, Ops.MATMUL])
    self.assertEqual([ref.uop for ref in tasks[0].inputs], [x, w])
    self.assertIn(", 0.0f)", codegen(tasks[0]).source)

  def test_multi_input_producer_is_fused(self):
    a, b, w = load((4, 8)), load((4, 8)), load((8, 16))
    lhs = add(a, b)
    tasks = run_scheduler(matmul(lhs, w))
    self.assertEqual([task.kind for task in tasks], [TaskKind.BLAS])
    self.assertEqual([u.op for u in tasks[0].ops], [Ops.ADD, Ops.MATMUL])
    self.assertEqual([ref.uop for ref in tasks[0].inputs], [a, b, w])
    source = codegen(tasks[0]).source
    self.assertIn("in0", source)
    self.assertIn("in1", source)
    self.assertIn("in2", source)

  def test_movement_over_computed_input_is_fused(self):
    a, b, c, w = load((2, 3)), load((2, 3)), load((3, 2)), load((2, 4))
    summed = add(a, b)
    transformed = mul(permute(summed, (1, 0)), c)
    tasks = run_scheduler(matmul(transformed, w))
    self.assertEqual([task.kind for task in tasks], [TaskKind.BLAS])
    self.assertEqual([u.op for u in tasks[0].ops], [Ops.ADD, Ops.MUL, Ops.MATMUL])
    self.assertEqual([ref.uop for ref in tasks[0].inputs], [a, b, c, w])
    self.assertIn("% 2", codegen(tasks[0]).source)

  def test_broadcast_over_computed_input_is_fused(self):
    x, w = load((3,)), load((3, 4))
    left = expand(reshape(relu(x), (1, 3)), (2, 3))
    task = run_scheduler(matmul(left, w))[0]
    self.assertEqual([u.op for u in task.ops], [Ops.RELU, Ops.MATMUL])
    self.assertEqual([ref.uop for ref in task.inputs], [x, w])

  def test_padding_over_computed_input_is_fused(self):
    x, w = load((2, 3)), load((3, 4))
    padded = UOp(Ops.PAD, x.dtype, (relu(x),), ((3, 3), ((1, 3), (0, 3)), -2.0))
    task = run_scheduler(matmul(padded, w))[0]
    source = codegen(task).source
    self.assertEqual([u.op for u in task.ops], [Ops.RELU, Ops.MATMUL])
    self.assertIn("?", source)
    self.assertIn("-2.0f", source)

  def test_both_operands_fuse(self):
    a, b, x, y = load((4, 8)), load((4, 8)), load((8, 16)), load((8, 16))
    task = run_scheduler(matmul(relu(add(a, b)), mul(x, y)))[0]
    self.assertEqual([u.op for u in task.ops], [Ops.ADD, Ops.RELU, Ops.MUL, Ops.MATMUL])
    self.assertEqual([ref.uop for ref in task.inputs], [a, b, x, y])
    source = codegen(task).source
    self.assertIn("max(", source)
    self.assertIn("in2", source)
    self.assertIn("in3", source)

  def test_constant_stays_inline(self):
    a, w = load((4, 8)), load((8, 16))
    scalar = expand(const(2.0), (4, 8))
    task = run_scheduler(matmul(relu(mul(a, scalar)), w))[0]
    self.assertEqual([ref.uop for ref in task.inputs], [a, w])
    self.assertIn("2.0f", codegen(task).source)

  def test_constant_operands_need_no_kernel_arguments(self):
    left, right = expand(const(2.0), (2, 3)), expand(const(3.0), (3, 4))
    task = run_scheduler(matmul(left, right))[0]
    kernel = codegen(task)
    self.assertEqual(task.inputs, [])
    self.assertIn(f"{kernel.name}(__global float* out", kernel.source)

  def test_binary_max_fuses(self):
    a, b, w = load((4, 8)), load((4, 8)), load((8, 16))
    maximum = UOp(Ops.MAX, a.dtype, (a, b), GPU)
    task = run_scheduler(matmul(maximum, w))[0]
    self.assertIn("max(", codegen(task).source)

  def test_cast_fuses_and_enables_input_dtype(self):
    a, w = load((4, 8), dtypes.float16), load((8, 16))
    task = run_scheduler(matmul(cast(a, dtypes.float32), w))[0]
    kernel = codegen(task)
    self.assertEqual([u.op for u in task.ops], [Ops.CAST, Ops.MATMUL])
    self.assertIn("__global const half* in0", kernel.source)
    self.assertIn("(float)", kernel.source)
    self.assertIn("cl_khr_fp16", kernel.source)

  def test_fused_intermediate_keeps_declared_dtype(self):
    a, b = load((1, 1), dtypes.int8), load((1, 1), dtypes.int8)
    summed = UOp(Ops.ADD, dtypes.int8, (a, b), GPU)
    left = cast(summed, dtypes.float32)
    w = load((1, 1))
    task = run_scheduler(matmul(left, w))[0]
    source = codegen(task).source
    self.assertEqual([u.op for u in task.ops], [Ops.ADD, Ops.CAST, Ops.MATMUL])
    self.assertIn("(char)", source)

  def test_padded_narrow_input_keeps_declared_dtype(self):
    x, w = load((1, 2), dtypes.int8), load((2, 1), dtypes.int8)
    padded = UOp(Ops.PAD, x.dtype, (x,), ((2, 2), ((1, 2), (0, 2)), -2))
    task = run_scheduler(matmul(relu(padded), w))[0]
    source = codegen(task).source
    self.assertIn("max(", source)
    self.assertIn("(char)", source)

  def test_alias_reuses_one_kernel_argument(self):
    x = load((4, 4))
    task = run_scheduler(matmul(x, x))[0]
    kernel = codegen(task)
    self.assertEqual(len(task.inputs), 1)
    self.assertIn("* in0", kernel.source)
    self.assertNotIn("* in1", kernel.source)

  def test_shared_movement_schedules_producer_before_matmul(self):
    for reverse in (False, True):
      a, b, w = load((2, 3)), load((2, 3)), load((2, 2))
      summed = add(a, b)
      view = permute(summed, (1, 0))
      side, product = relu(view), matmul(view, w)
      tasks = run_scheduler(add(product, side) if reverse else add(side, product))
      self.assertEqual([task.kind for task in tasks], [TaskKind.ELEMENTWISE, TaskKind.BLAS, TaskKind.ELEMENTWISE])
      self.assertIs(tasks[0].output_uop, summed)
      self.assertIs(tasks[1].output_uop, product)

if __name__ == "__main__":
  unittest.main(verbosity=2)
