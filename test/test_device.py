# Coverage for monograd/device.py: Device, to_device, Buffer (CPU + GPU)
# Run GPU tests with: TEST_GPU=1 pytest tests/test_device.py -v
import unittest
import numpy as np
from monograd.device import Buffer, Device, to_device, OpenCLContext
from monograd.dtype import dtypes
from monograd.utils import TEST_GPU


# **** device + to_device ****
class TestDevice(unittest.TestCase):
  def test_cpu_and_gpu_are_distinct(self):
    self.assertNotEqual(Device.CPU, Device.GPU)

  def test_str_lowercase(self):
    self.assertEqual(str(Device.CPU), "cpu")
    self.assertEqual(str(Device.GPU), "gpu")

  def test_to_device_passthrough(self):
    self.assertIs(to_device(Device.CPU), Device.CPU)
    self.assertIs(to_device(Device.GPU), Device.GPU)

  def test_to_device_from_string_lowercase(self):
    self.assertIs(to_device("cpu"), Device.CPU)
    self.assertIs(to_device("gpu"), Device.GPU)

  def test_to_device_from_string_uppercase(self):
    self.assertIs(to_device("CPU"), Device.CPU)
    self.assertIs(to_device("GPU"), Device.GPU)

  def test_to_device_from_string_mixed(self):
    self.assertIs(to_device("Cpu"), Device.CPU)
    self.assertIs(to_device("Gpu"), Device.GPU)


# **** Shared Buffer tests cpu+gpu ****
class BufferTests(unittest.TestCase):
  """
  Mixin containing all device-agnostic Buffer tests.
  Subclasses set self.device = Device.CPU or Device.GPU.
  """
  device:Device = Device.CPU

  # **** allocation ****
  def test_not_allocated_before_allocate(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    self.assertFalse(buf.is_allocated())

  def test_allocated_after_allocate(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    buf.allocate()
    self.assertTrue(buf.is_allocated())

  def test_double_allocate_raises(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    buf.allocate()
    with self.assertRaises(AssertionError):
      buf.allocate()

  def test_ensure_allocated_idempotent(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    b1 = buf.ensure_allocated()
    b2 = buf.ensure_allocated()
    self.assertIs(b1, b2)

  def test_allocate_returns_self(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    ret = buf.allocate()
    self.assertIs(ret, buf)


  # **** nbytes ****
  def test_nbytes_float32(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    self.assertEqual(buf.nbytes, 4 * 4)   # 4 elements * 4 bytes

  def test_nbytes_float16(self):
    buf = Buffer(self.device, 8, dtypes.float16)
    self.assertEqual(buf.nbytes, 8 * 2)

  def test_nbytes_int8(self):
    buf = Buffer(self.device, 16, dtypes.int8)
    self.assertEqual(buf.nbytes, 16 * 1)

  def test_nbytes_float64(self):
    buf = Buffer(self.device, 2, dtypes.float64)
    self.assertEqual(buf.nbytes, 2 * 8)


  # **** copyin copyout roundtrip ****
  def test_copyin_copyout_float32(self):
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf  = Buffer(self.device, 4, dtypes.float32)
    buf.allocate()
    buf.copyin(memoryview(data))
    out = memoryview(np.empty(4, dtype=np.float32))
    buf.copyout(out)
    np.testing.assert_array_equal(np.frombuffer(out, dtype=np.float32), data)

  def test_copyin_copyout_int32(self):
    data = np.array([10, 20, 30, 40], dtype=np.int32)
    buf  = Buffer(self.device, 4, dtypes.int32)
    buf.allocate()
    buf.copyin(memoryview(data))
    out = memoryview(np.empty(4, dtype=np.int32))
    buf.copyout(out)
    np.testing.assert_array_equal(np.frombuffer(out, dtype=np.int32), data)

  def test_copyin_copyout_float64(self):
    data = np.array([1.1, 2.2, 3.3], dtype=np.float64)
    buf  = Buffer(self.device, 3, dtypes.float64)
    buf.allocate()
    buf.copyin(memoryview(data))
    out = memoryview(np.empty(3, dtype=np.float64))
    buf.copyout(out)
    np.testing.assert_array_almost_equal(np.frombuffer(out, dtype=np.float64), data)

  def test_copyin_copyout_bool(self):
    data = np.array([True, False, True, False], dtype=np.bool_)
    buf  = Buffer(self.device, 4, dtypes.bool)
    buf.allocate()
    buf.copyin(memoryview(data))
    out = memoryview(np.empty(4, dtype=np.bool_))
    buf.copyout(out)
    np.testing.assert_array_equal(np.frombuffer(out, dtype=np.bool_), data)

  def test_copyin_copyout_int8(self):
    data = np.array([-128, -1, 0, 1, 127], dtype=np.int8)
    buf  = Buffer(self.device, 5, dtypes.int8)
    buf.allocate()
    buf.copyin(memoryview(data))
    out = memoryview(np.empty(5, dtype=np.int8))
    buf.copyout(out)
    np.testing.assert_array_equal(np.frombuffer(out, dtype=np.int8), data)

  def test_copyout_returns_memoryview(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    buf.allocate()
    buf.copyin(memoryview(np.zeros(4, dtype=np.float32)))
    out = memoryview(np.empty(4, dtype=np.float32))
    ret = buf.copyout(out)
    self.assertIsInstance(ret, memoryview)

  def test_copyin_without_allocation_raises(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    with self.assertRaises(AssertionError):
      buf.copyin(memoryview(np.zeros(4, dtype=np.float32)))

  def test_copyout_without_allocation_raises(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    with self.assertRaises(AssertionError):
      buf.copyout(memoryview(np.empty(4, dtype=np.float32)))


  # **** allocate with intial value ****
  def test_allocate_with_initial_value(self):
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    buf  = Buffer(self.device, 4, dtypes.float32)
    buf.allocate(initial_value=data)
    out = memoryview(np.empty(4, dtype=np.float32))
    buf.copyout(out)
    np.testing.assert_array_equal(np.frombuffer(out, dtype=np.float32), data)

  def test_allocate_with_initial_value_2d(self):
    """2D array initial value — size must equal total elements"""
    data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    buf  = Buffer(self.device, 4, dtypes.float32)
    buf.allocate(initial_value=data)
    out = memoryview(np.empty(4, dtype=np.float32))
    buf.copyout(out)
    np.testing.assert_array_equal(
      np.frombuffer(out, dtype=np.float32),
      data.flatten()
    )


  # **** as_buffer ****
  def test_as_buffer_returns_correct_data(self):
    data = np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)
    buf  = Buffer(self.device, 4, dtypes.float32)
    buf.allocate(initial_value=data)
    result = np.frombuffer(buf.as_buffer(), dtype=np.float32)
    np.testing.assert_array_equal(result, data)

  def test_as_buffer_returns_memoryview(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    buf.allocate(initial_value=np.zeros(4, dtype=np.float32))
    self.assertIsInstance(buf.as_buffer(), memoryview)


  # **** base/view ****
  def test_base_of_base_is_self(self):
    buf = Buffer(self.device, 8, dtypes.float32)
    self.assertIs(buf.base, buf)

  def test_view_base_points_to_parent(self):
    base = Buffer(self.device, 8, dtypes.float32)
    view = Buffer(self.device, 4, dtypes.float32, base=base, offset=2)
    self.assertIs(view.base, base)

  def test_base_offset_must_be_zero(self):
    with self.assertRaises(AssertionError):
      Buffer(self.device, 4, dtypes.float32, base=None, offset=2)

  def test_view_base_cannot_have_base(self):
    base  = Buffer(self.device, 8, dtypes.float32)
    view  = Buffer(self.device, 4, dtypes.float32, base=base, offset=0)
    with self.assertRaises(AssertionError):
      Buffer(self.device, 2, dtypes.float32, base=view, offset=0)

  def test_view_device_must_match_base(self):
    base = Buffer(Device.CPU, 8, dtypes.float32)
    with self.assertRaises(AssertionError):
      Buffer(Device.GPU, 4, dtypes.float32, base=base, offset=0)


  # **** __repr__ ****
  def test_repr_is_string(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    self.assertIsInstance(repr(buf), str)

  def test_repr_contains_device(self):
    buf = Buffer(self.device, 4, dtypes.float32)
    self.assertIn(str(self.device), repr(buf))


# **** cpu concrete buffer tests ****
class TestBufferCPU(BufferTests, unittest.TestCase):
  device = Device.CPU

  def test_cpu_buf_is_numpy_array(self):
    """internal _buf on CPU must be a numpy array"""
    buf = Buffer(Device.CPU, 4, dtypes.float32)
    buf.allocate()
    self.assertTrue(buf.is_allocated())
    self.assertIsInstance(buf._buf, np.ndarray)

  def test_cpu_empty_buf_is_uint8(self):
    """uninitialized CPU buffer is raw uint8 bytes"""
    buf = Buffer(Device.CPU, 4, dtypes.float32)
    buf.allocate()
    self.assertTrue(buf.is_allocated())
    self.assertEqual(buf._buf.dtype, np.uint8) # type: ignore

  def test_cpu_initialized_buf_dtype(self):
    """initialized CPU buffer dtype matches DType"""
    data = np.array([1.0, 2.0], dtype=np.float32)
    buf  = Buffer(Device.CPU, 2, dtypes.float32)
    buf.allocate(initial_value=data)
    self.assertTrue(buf.is_allocated())
    self.assertEqual(buf._buf.dtype, np.float32) # type: ignore


# **** gpu concrete buffer tests ****
@unittest.skipUnless(TEST_GPU, "set TEST_GPU=1 to run GPU tests")
class TestBufferGPU(BufferTests, unittest.TestCase):
  device = Device.GPU

  def test_gpu_buf_is_cl_buffer(self):
    import pyopencl as cl
    buf = Buffer(Device.GPU, 4, dtypes.float32)
    buf.allocate()
    self.assertIsInstance(buf._buf, cl.Buffer)

  def test_gpu_has_opencl_context(self):
    """GPU buffer must release cl.Buffer on deletion"""
    import pyopencl as cl
    buf = Buffer(Device.GPU, 4, dtypes.float32)
    self.assertTrue(hasattr(buf, "CL_CTX"))
    self.assertTrue(hasattr(buf, "CL_QUEUE"))
    
  def test_gpu_opencl_context_singleton(self):
    """OpenCLContext must return same ctx and queue every call"""
    ctx1 = OpenCLContext.cl_ctx()
    ctx2 = OpenCLContext.cl_ctx()
    self.assertIs(ctx1, ctx2)
    q1 = OpenCLContext.cl_queue()
    q2 = OpenCLContext.cl_queue()
    self.assertIs(q1, q2)


if __name__ == "__main__":
  unittest.main(verbosity=2)
