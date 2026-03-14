# Coverage for monograd/dtype.py: DTypeMetaClass, DType, dtypes, to_np_dtype, to_dtype, most_upper_dtype
import unittest
import numpy as np
from dataclasses import FrozenInstanceError
from monograd.dtype import DType, dtypes, to_np_dtype, to_dtype, most_upper_dtype

# **** singleton cache ****
class TestDTypeMetaClass(unittest.TestCase):
  def test_same_args_return_identical_object(self):
    """DTypeMetaClass must deduplicate: identical args → same object"""
    a = DType.new(5, 32, "int", 'i')
    b = DType.new(5, 32, "int", 'i')
    self.assertIs(a, b)

  def test_different_args_return_different_objects(self):
    a = DType.new(5, 32, "int",  'i')
    b = DType.new(5, 32, "int2", 'i')
    self.assertIsNot(a, b)

  def test_all_builtin_dtypes_are_singletons(self):
    """every dtypes.X re-created with same args must be the exact same object"""
    for dt in dtypes.all:
      recreated = DType.new(dt.priority, dt.bitsize, dt.name, dt.fmt)
      self.assertIs(dt, recreated, f"{dt.name} is not a singleton")


# **** dtypes propreties ****
class TestDTypeProperties(unittest.TestCase):
  def test_frozen_immutable(self):
    with self.assertRaises((FrozenInstanceError, TypeError)):
      dtypes.float32.priority = 99  # type: ignore

  def test_itemsize_bytes(self):
    self.assertEqual(dtypes.float32.itemsize, 4)
    self.assertEqual(dtypes.float64.itemsize, 8)
    self.assertEqual(dtypes.float16.itemsize, 2)
    self.assertEqual(dtypes.int32.itemsize,   4)
    self.assertEqual(dtypes.int64.itemsize,   8)
    self.assertEqual(dtypes.int8.itemsize,    1)
    self.assertEqual(dtypes.uint8.itemsize,   1)
    self.assertEqual(dtypes.bool.itemsize,    1)

  def test_itemsize_formula_all(self):
    import math
    for dt in dtypes.all:
      if dt is dtypes.void: continue
      self.assertEqual(dt.itemsize, math.ceil(dt.bitsize / 8), f"wrong itemsize for {dt.name}")

  def test_np_dtype_matches_fmt(self):
    for dt in dtypes.all:
      if dt.fmt is None: continue
      self.assertEqual(dt.np_dtype, np.dtype(dt.fmt), f"np_dtype mismatch for {dt.name}")

  def test_priority_floats_beat_same_bitsize_ints(self):
    self.assertGreater(dtypes.float16.priority, dtypes.int16.priority)
    self.assertGreater(dtypes.float32.priority, dtypes.int32.priority)
    self.assertGreater(dtypes.float64.priority, dtypes.int64.priority)

  def test_priority_wider_beats_narrower(self):
    self.assertGreater(dtypes.float32.priority, dtypes.float16.priority)
    self.assertGreater(dtypes.float64.priority, dtypes.float32.priority)
    self.assertGreater(dtypes.int32.priority,   dtypes.int16.priority)
    self.assertGreater(dtypes.int64.priority,   dtypes.int32.priority)

  def test_void_lowest_priority(self):
    for dt in dtypes.all:
      if dt is dtypes.void: continue
      self.assertGreater(dt.priority, dtypes.void.priority)

  def test_name_nonempty_string(self):
    for dt in dtypes.all:
      self.assertIsInstance(dt.name, str)
      self.assertGreater(len(dt.name), 0)

  def test_void_bitsize_zero(self):
    self.assertEqual(dtypes.void.bitsize, 0)

  def test_nonvoid_bitsize_positive(self):
    for dt in dtypes.all:
      if dt is dtypes.void: continue
      self.assertGreater(dt.bitsize, 0)


# **** dtype classification ****
class TestDtypesClassification(unittest.TestCase):
  def test_is_float_true_for_floats(self):
    for dt in dtypes.floats:
      self.assertTrue(dtypes.is_float(dt), f"{dt.name} should be float")

  def test_is_float_false_for_ints(self):
    for dt in dtypes.ints:
      self.assertFalse(dtypes.is_float(dt), f"{dt.name} should not be float")

  def test_is_float_false_for_bool(self):
    self.assertFalse(dtypes.is_float(dtypes.bool))

  def test_is_int_true_for_ints(self):
    for dt in dtypes.ints:
      self.assertTrue(dtypes.is_int(dt), f"{dt.name} should be int")

  def test_is_int_false_for_floats(self):
    for dt in dtypes.floats:
      self.assertFalse(dtypes.is_int(dt), f"{dt.name} should not be int")

  def test_is_int_false_for_bool(self):
    self.assertFalse(dtypes.is_int(dtypes.bool))

  def test_is_unsigned_true_for_uints(self):
    for dt in dtypes.uints:
      self.assertTrue(dtypes.is_unsigned(dt), f"{dt.name} should be unsigned")

  def test_is_unsigned_false_for_sints(self):
    for dt in dtypes.sints:
      self.assertFalse(dtypes.is_unsigned(dt), f"{dt.name} should not be unsigned")

  def test_is_unsigned_false_for_floats(self):
    for dt in dtypes.floats:
      self.assertFalse(dtypes.is_unsigned(dt))

  def test_is_bool(self):
    self.assertTrue(dtypes.is_bool(dtypes.bool))
    for dt in dtypes.ints + dtypes.floats:
      self.assertFalse(dtypes.is_bool(dt))

  def test_sints_and_uints_disjoint(self):
    self.assertTrue(set(dtypes.sints).isdisjoint(set(dtypes.uints)))

  def test_ints_is_union_of_sints_and_uints(self):
    self.assertEqual(set(dtypes.ints), set(dtypes.sints) | set(dtypes.uints))


# **** min/max dtype ****
class TestDtypesMinMax(unittest.TestCase):
  def test_float_min_is_neg_inf(self):
    for dt in dtypes.floats:
      self.assertEqual(dtypes.min(dt), -float("inf"), f"{dt.name} min should be -inf")

  def test_float_max_is_pos_inf(self):
    for dt in dtypes.floats:
      self.assertEqual(dtypes.max(dt), float("inf"), f"{dt.name} max should be +inf")

  def test_bool_min_max(self):
    self.assertEqual(dtypes.min(dtypes.bool), False)
    self.assertEqual(dtypes.max(dtypes.bool), True)

  def test_signed_int_min(self):
    self.assertEqual(dtypes.min(dtypes.int8),  -128)
    self.assertEqual(dtypes.min(dtypes.int16), -32768)
    self.assertEqual(dtypes.min(dtypes.int32), -(2**31))
    self.assertEqual(dtypes.min(dtypes.int64), -(2**63))

  def test_unsigned_int_min_zero(self):
    for dt in dtypes.uints:
      self.assertEqual(dtypes.min(dt), 0, f"{dt.name} min should be 0")

  def test_signed_int_max(self):
    self.assertEqual(dtypes.max(dtypes.int8),  127)
    self.assertEqual(dtypes.max(dtypes.int16), 32767)
    self.assertEqual(dtypes.max(dtypes.int32), 2**31 - 1)
    self.assertEqual(dtypes.max(dtypes.int64), 2**63 - 1)

  def test_unsigned_int_max(self):
    self.assertEqual(dtypes.max(dtypes.uint8),  255)
    self.assertEqual(dtypes.max(dtypes.uint16), 65535)
    self.assertEqual(dtypes.max(dtypes.uint32), 2**32 - 1)
    self.assertEqual(dtypes.max(dtypes.uint64), 2**64 - 1)

  def test_void_min_raises(self):
    with self.assertRaises(RuntimeError):
      dtypes.min(dtypes.void)

  def test_void_max_raises(self):
    with self.assertRaises(RuntimeError):
      dtypes.max(dtypes.void)


# **** from_py dtypes ****
class TestDtypesFromPy(unittest.TestCase):
  def test_float_literal(self):
    self.assertIs(dtypes.from_py(1.0),   dtypes.default_float)
    self.assertIs(dtypes.from_py(-3.14), dtypes.default_float)

  def test_int_literal(self):
    self.assertIs(dtypes.from_py(0),    dtypes.default_int)
    self.assertIs(dtypes.from_py(-100), dtypes.default_int)

  def test_bool_literal(self):
    # bool must be caught before int since bool subclasses int
    self.assertIs(dtypes.from_py(True),  dtypes.bool)
    self.assertIs(dtypes.from_py(False), dtypes.bool)

  def test_list_raises(self):
    with self.assertRaises(RuntimeError):
      dtypes.from_py([1, 2, 3])

  def test_tuple_raises(self):
    with self.assertRaises(RuntimeError):
      dtypes.from_py((1, 2))

  def test_string_raises(self):
    with self.assertRaises(RuntimeError):
      dtypes.from_py("hello")

  def test_none_raises(self):
    with self.assertRaises(RuntimeError):
      dtypes.from_py(None)

  def test_default_float_is_float32(self):
    self.assertIs(dtypes.default_float, dtypes.float32)

  def test_default_int_is_int32(self):
    self.assertIs(dtypes.default_int, dtypes.int32)


# **** dtypes aliases ****
class TestDtypesAliases(unittest.TestCase):
  def test_float_aliases(self):
    self.assertIs(dtypes.half,   dtypes.float16)
    self.assertIs(dtypes.float,  dtypes.float32)
    self.assertIs(dtypes.double, dtypes.float64)

  def test_signed_int_aliases(self):
    self.assertIs(dtypes.char,  dtypes.int8)
    self.assertIs(dtypes.short, dtypes.int16)
    self.assertIs(dtypes.int,   dtypes.int32)
    self.assertIs(dtypes.long,  dtypes.int64)

  def test_unsigned_int_aliases(self):
    self.assertIs(dtypes.uchar,  dtypes.uint8)
    self.assertIs(dtypes.ushort, dtypes.uint16)
    self.assertIs(dtypes.uint,   dtypes.uint32)
    self.assertIs(dtypes.ulong,  dtypes.uint64)


# **** to_np_dtype ****
class TestToNpDtype(unittest.TestCase):
  def test_float32(self): self.assertEqual(to_np_dtype(dtypes.float32), np.dtype('float32'))
  def test_float64(self): self.assertEqual(to_np_dtype(dtypes.float64), np.dtype('float64'))
  def test_float16(self): self.assertEqual(to_np_dtype(dtypes.float16), np.dtype('float16'))
  def test_int8(self):    self.assertEqual(to_np_dtype(dtypes.int8),    np.dtype('int8'))
  def test_int16(self):   self.assertEqual(to_np_dtype(dtypes.int16),   np.dtype('int16'))
  def test_int32(self):   self.assertEqual(to_np_dtype(dtypes.int32),   np.dtype('int32'))
  def test_int64(self):   self.assertEqual(to_np_dtype(dtypes.int64),   np.dtype('int64'))
  def test_uint8(self):   self.assertEqual(to_np_dtype(dtypes.uint8),   np.dtype('uint8'))
  def test_uint32(self):  self.assertEqual(to_np_dtype(dtypes.uint32),  np.dtype('uint32'))
  def test_bool(self):    self.assertEqual(to_np_dtype(dtypes.bool),    np.dtype('bool'))

  def test_returns_np_dtype_instance(self):
    for dt in dtypes.all:
      if dt.fmt is None: continue
      self.assertIsInstance(to_np_dtype(dt), np.dtype)

  def test_cached_same_object(self):
    a = to_np_dtype(dtypes.float32)
    b = to_np_dtype(dtypes.float32)
    self.assertIs(a, b)


# **** to_dtype ****
class TestToDtype(unittest.TestCase):
  def test_dtype_passthrough(self):
    self.assertIs(to_dtype(dtypes.float32), dtypes.float32)
    self.assertIs(to_dtype(dtypes.int32),   dtypes.int32)
    self.assertIs(to_dtype(dtypes.bool),    dtypes.bool)

  def test_string_lookup_lowercase(self):
    self.assertIs(to_dtype("float32"), dtypes.float32)
    self.assertIs(to_dtype("float16"), dtypes.float16)
    self.assertIs(to_dtype("float64"), dtypes.float64)
    self.assertIs(to_dtype("int8"),    dtypes.int8)
    self.assertIs(to_dtype("int32"),   dtypes.int32)
    self.assertIs(to_dtype("int64"),   dtypes.int64)
    self.assertIs(to_dtype("uint8"),   dtypes.uint8)
    self.assertIs(to_dtype("uint32"),  dtypes.uint32)
    self.assertIs(to_dtype("bool"),    dtypes.bool)

  def test_string_lookup_uppercase(self):
    self.assertIs(to_dtype("FLOAT32"), dtypes.float32)
    self.assertIs(to_dtype("INT32"),   dtypes.int32)
    self.assertIs(to_dtype("BOOL"),    dtypes.bool)

  def test_string_lookup_mixed_case(self):
    self.assertIs(to_dtype("Float32"), dtypes.float32)
    self.assertIs(to_dtype("Int32"),   dtypes.int32)

  def test_alias_strings(self):
    self.assertIs(to_dtype("float"),  dtypes.float32)
    self.assertIs(to_dtype("double"), dtypes.float64)
    self.assertIs(to_dtype("half"),   dtypes.float16)
    self.assertIs(to_dtype("int"),    dtypes.int32)
    self.assertIs(to_dtype("long"),   dtypes.int64)


# **** most_upper_dtype ****
class TestMostUpperDtype(unittest.TestCase):
  def test_same_dtype_returns_itself(self):
    self.assertIs(most_upper_dtype(dtypes.float32, dtypes.float32), dtypes.float32)
    self.assertIs(most_upper_dtype(dtypes.int32,   dtypes.int32),   dtypes.int32)

  def test_float_beats_int(self):
    self.assertIs(most_upper_dtype(dtypes.int32,  dtypes.float32), dtypes.float32)
    self.assertIs(most_upper_dtype(dtypes.int64,  dtypes.float16), dtypes.float16)
    self.assertIs(most_upper_dtype(dtypes.uint32, dtypes.float32), dtypes.float32)

  def test_wider_float_wins(self):
    self.assertIs(most_upper_dtype(dtypes.float16, dtypes.float32), dtypes.float32)
    self.assertIs(most_upper_dtype(dtypes.float32, dtypes.float64), dtypes.float64)
    self.assertIs(most_upper_dtype(dtypes.float16, dtypes.float64), dtypes.float64)

  def test_wider_int_wins(self):
    self.assertIs(most_upper_dtype(dtypes.int8,  dtypes.int32),  dtypes.int32)
    self.assertIs(most_upper_dtype(dtypes.int32, dtypes.int64),  dtypes.int64)
    self.assertIs(most_upper_dtype(dtypes.uint8, dtypes.uint64), dtypes.uint64)

  def test_bool_always_loses(self):
    for dt in dtypes.floats + dtypes.ints:
      self.assertIs(most_upper_dtype(dtypes.bool, dt), dt, f"bool should lose to {dt.name}")

  def test_three_args(self):
    self.assertIs(most_upper_dtype(dtypes.int8,  dtypes.int32,  dtypes.float32), dtypes.float32)
    self.assertIs(most_upper_dtype(dtypes.bool,  dtypes.int16,  dtypes.float64), dtypes.float64)

  def test_commutative(self):
    pairs = [
      (dtypes.int32,   dtypes.float32),
      (dtypes.float16, dtypes.float64),
      (dtypes.bool,    dtypes.int8),
    ]
    for a, b in pairs:
      self.assertIs(most_upper_dtype(a, b), most_upper_dtype(b, a),
        f"not commutative for {a.name}, {b.name}")

  def test_single_arg(self):
    self.assertIs(most_upper_dtype(dtypes.float32), dtypes.float32)

  def test_all_floats_returns_float64(self):
    self.assertIs(most_upper_dtype(*dtypes.floats), dtypes.float64)

  def test_all_ints_returns_uint64(self):
    self.assertIs(most_upper_dtype(*dtypes.ints), dtypes.uint64)

  def test_cached(self):
    a = most_upper_dtype(dtypes.float32, dtypes.int32)
    b = most_upper_dtype(dtypes.float32, dtypes.int32)
    self.assertIs(a, b)


if __name__ == "__main__":
  unittest.main(verbosity=2)
