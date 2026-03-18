# allow semicolons to put multiple ops on one line
# flake8: noqa: E702
from enum import auto, IntEnum, Enum

class Ops(IntEnum):
  LOAD = auto(); CONST = auto();
  COPY = auto(); SINK = auto();

  # Unary Ops
  RELU = auto(); LOG = auto(); EXP = auto()
  SIN = auto(); SQRT = auto(); CAST = auto()
  RECIP = auto() # recip is a fast gpu operation (1.0f/val)
    
  # Binary Ops
  ADD = auto(); MUL = auto();
  MAX = auto(); POW = auto(); MOD = auto()
  OR = auto(); XOR = auto(); AND = auto()

  # BLAS
  MATMUL = auto(); GEMM = auto()

  # Ternary Ops
  WHERE = auto();
    
  # Movement/Shape Ops
  RESHAPE = auto(); EXPAND = auto(); PERMUTE = auto()
  PAD = auto(); SHRINK = auto();

  # Reduce ops
  SUM = auto(); REDUCEMAX = auto()

  def __str__(self): return Enum.__str__(self)
  def __repr__(self): return str(self)

class GroupOp:
  Unary = {Ops.SIN, Ops.SQRT, Ops.LOG, Ops.RELU, Ops.CAST, Ops.EXP, Ops.RECIP}
  Binary = {Ops.ADD, Ops.MUL, Ops.MOD, Ops.MAX, Ops.XOR, Ops.OR, Ops.AND, Ops.POW}
  Ternary = {Ops.WHERE}
  ALU = set.union(Unary, Binary, Ternary)

  Input = {Ops.LOAD, Ops.CONST}
  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE, Ops.PAD, Ops.SHRINK}
  Reduce = {Ops.SUM, Ops.REDUCEMAX}
  BLAS = {Ops.MATMUL, Ops.GEMM}

  All = set(Ops)
