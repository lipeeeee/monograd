# allow semicolons to put multiple ops on one line
# flake8: noqa: E702
from enum import auto, IntEnum, Enum

class Ops(IntEnum):
  LOAD = auto(); CONST = auto();
  COPY = auto(); SINK = auto();

  # Unary Ops
  NEG = auto(); RELU = auto(); LOG = auto(); EXP = auto()
  SIN = auto(); SQRT = auto(); CAST = auto()
    
  # Binary Ops
  ADD = auto(); SUB = auto(); MUL = auto();
  MAX = auto(); POW = auto(); MOD = auto()
  OR = auto(); XOR = auto(); AND = auto()
  DIV = auto()

  # BLAS
  MATMUL = auto(); GEMM = auto()

  # Ternary Ops
  WHERE = auto(); MULACC = auto()
    
  # Movement/Shape Ops
  RESHAPE = auto(); EXPAND = auto(); PERMUTE = auto()

  # Reduce ops
  SUM = auto();

  def __str__(self): return Enum.__str__(self)
  def __repr__(self): return str(self)

class GroupOp:
  Unary = {Ops.SIN, Ops.SQRT, Ops.NEG, Ops.LOG, Ops.RELU, Ops.CAST, Ops.EXP}
  Binary = {Ops.ADD, Ops.MUL, Ops.MATMUL, Ops.MAX, Ops.MOD, Ops.XOR, Ops.OR, Ops.AND, Ops.SUB, Ops.POW, Ops.DIV}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)

  Movement = {Ops.RESHAPE, Ops.EXPAND, Ops.PERMUTE}
  Reduce = {Ops.SUM}
  BLAS = {Ops.MATMUL, Ops.GEMM}

  All = set(Ops)
