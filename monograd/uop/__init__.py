# allow semicolons to put multiple ops on one line
# flake8: noqa: E702
from enum import auto, IntEnum, Enum

class Ops(IntEnum):
  # No shape
  LOAD = auto(); STORE = auto(); CONST = auto()
  COPY = auto()

  # Unary Ops
  NEG = auto(); RELU = auto(); LOG = auto(); EXP = auto()
  SIN = auto(); SQRT = auto()
    
  # Binary Ops
  ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto()
  MAX = auto(); POW = auto(); MOD = auto()
  OR = auto(); XOR = auto(); AND = auto()

  # Ternary Ops
  WHERE = auto(); MULACC = auto()
    
  # Movement/Shape Ops
  RESHAPE = auto()
  PERMUTE = auto()

  def __str__(self): return Enum.__str__(self)
  def __repr__(x): return str(x)

class GroupOp:
  Unary = {Ops.SIN, Ops.SQRT, Ops.NEG}
  Binary = {Ops.ADD, Ops.MUL, Ops.MAX, Ops.MOD, Ops.XOR, Ops.OR, Ops.AND, Ops.SUB, Ops.POW}
  Ternary = {Ops.WHERE, Ops.MULACC}
  ALU = set.union(Unary, Binary, Ternary)

  # Buffer types
  Buffer = {Ops.LOAD, Ops.STORE, Ops.CONST}

  # BinaryOps where f(f(a,b),c) = f(a,f(b,c))
  # Associative = {Ops.ADD, Ops.MUL, Ops.AND, Ops.OR, Ops.MAX}

  # BinaryOps that satisfy f(x,x)=x see https://en.wikipedia.org/wiki/Idempotence
  # Idempotent = {Ops.OR, Ops.AND, Ops.MAX}

  All = set(Ops)
