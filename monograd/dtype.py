from __future__ import annotations
from typing import Final, ClassVar, Callable, Literal
from dataclasses import dataclass

class DTypeMetaClass(type):
  dcache: dict[tuple, DType] = {}
  def __call__(self, *args, **kwds):
    if (ret:=DTypeMetaClass.dcache.get(args, None)) is not None: return ret
    DTypeMetaClass.dcache[args] = ret = super().__call__(args)
    return ret

@dataclass(frozen=True)
class DType(metaclass=DTypeMetaClass):
  priority: int
  bitsize: int
  name: str
  fmt: str

  @staticmethod
  def new(priority:int, bitsize:int, name:str, fmt:str): return DType(priority, bitsize, name, fmt)

class dtypes:
  @staticmethod
  def is_int(x: DType) -> bool: return False
  @staticmethod
  def is_float(x: DType) -> bool: return False
  @staticmethod
  def is_unsigned(x: DType) -> bool: return False
  @staticmethod
  def is_bool(x: DType) -> bool: return False
  @staticmethod
  def from_py(x) -> DType:
    if x.__class__ is float: return dtypes.default_float
    if x.__class__ is int: return dtypes.default_int
    if x.__class__ is bool: return dtypes.bool
    if x.__class__ is list or x.__class__ is tuple: raise RuntimeError(f"TODO: infer list & tuples")
    raise RuntimeError(f"Could not infer dtype of {x} with type {type(x)}")

  void: Final[DType] = DType.new(-1, 0, "void", None)
  bool: Final[DType] = DType.new(0, 1, "bool", '?')
  int8: Final[DType] = DType.new(1, 8, "signed char", 'b')
  uint8: Final[DType] = DType.new(2, 8, "unsigned char", 'B')
  int16: Final[DType] = DType.new(3, 16, "short", 'h')
  uint16: Final[DType] = DType.new(4, 16, "unsigned short", 'H')
  int32: Final[DType] = DType.new(5, 32, "int", 'i')
  uint32: Final[DType] = DType.new(6, 32, "unsigned int", 'I')
  int64: Final[DType] = DType.new(7, 64, "long", 'q')
  uint64: Final[DType] = DType.new(8, 64, "unsigned long", 'Q')
  float16: Final[DType] = DType.new(11, 16, "half", 'e')
  float32: Final[DType] = DType.new(13, 32, "float", 'f')
  float64: Final[DType] = DType.new(14, 64, "double", 'd')

  # aliases
  half = float16; float = float32; double = float64
  uchar = uint8; ushort = uint16; uint = uint32; ulong = uint64
  char = int8; short = int16; int = int32; long = int64

  default_float: ClassVar[DType] = float32
  default_int: ClassVar[DType] = int32
