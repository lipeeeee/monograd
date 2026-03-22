# optimize DAG graph
#   future:
#     - max(x, y) and min(x, y) can be pre-computed in graph using dtypes if one x or y is a CONST op
from __future__ import annotations
import operator
from typing import Any, Callable
from monograd.dtype import DType, dtypes
from monograd.uop import Ops
from monograd.uop.ops import UOp, absorbing_element, identity_element

class UPat:
  def __init__(self, op:Ops|set[Ops]|None= None, dtype:DType|None=None, src:list[UPat]|None=None, 
               arg:Any=None, name:str|None=None, commutative=False):
    self.op = {op} if isinstance(op, Ops) else op
    self.dtype, self.src, self.arg, self.name, self.commutative = dtype, src, arg, name, commutative
  def match(self, uop:UOp) -> dict|None:
    # Returns dict of named captures if matched, None if not.
    if self.op is not None and uop.op not in self.op: return None
    if self.dtype is not None and uop.dtype != self.dtype: return None
    if self.arg is not None and uop.arg != self.arg: return None
    captures: dict[str, UOp] = {}
    if self.src is not None:
      if len(self.src) != len(uop.src): return None
      # try normal order first
      result = self._match_src(self.src, uop.src)
      # if commutative and normal order failed, try reversed
      if result is None and self.commutative and len(self.src) == 2:
        result = self._match_src(self.src, (uop.src[1], uop.src[0]))
      if result is None: return None
      captures.update(result)
    else: captures = {}
    # record this node if named
    if self.name is not None: captures[self.name] = uop
    return captures
  def _match_src(self, patterns, srcs) -> dict[str, UOp] | None:
    captures = {}
    for pat, child in zip(patterns, srcs):
      result = pat.match(child)
      if result is None: return None
      captures.update(result)
    return captures

# **** rewrite engine ****
Rule = tuple[UPat, Callable[..., UOp | None]]
def rewrite_graph(root:UOp, rules:list[Rule]) -> UOp:
  while True:
    new_root = _rewrite_pass(root, rules)
    if new_root is root: break
    root = new_root
  return root
def _rewrite_pass(root:UOp, rules:list[Rule]) -> UOp:
  memo:dict[int, UOp] = {}
  def rewrite(u: UOp) -> UOp:
    if id(u) in memo: return memo[id(u)]
    new_src = tuple(rewrite(s) for s in u.src)
    node = UOp(u.op, u.dtype, new_src, u.arg) if new_src != u.src else u
    for pat, action in rules:
      captures = pat.match(node)
      if captures is not None:
        result = action(**captures)
        if result is not None:
          memo[id(u)] = result
          return result
    memo[id(u)] = node
    return node
  return rewrite(root)


# **** patterns/rules ****
# ADD(MUL(x,y),z) -> MULACC
def _fuse_mulacc(mul: UOp, a: UOp, b: UOp, c: UOp) -> UOp|None:
  if not dtypes.is_float(mul.dtype): return None # there is no `fma` operation for integer mulacc in opencl
  return UOp(Ops.MULACC, mul.dtype, (a, b, c), mul.arg)
FUSE_MULACC = (
  UPat(Ops.ADD,
    src=[UPat(Ops.MUL, src=[UPat(name="a"), UPat(name="b")], name="mul"),
         UPat(name="c")],
    commutative=True),
  _fuse_mulacc
)

# CONST *op* CONST -> CONST
_FOLD_OPS: dict[Ops, Callable] = {
  Ops.ADD: operator.add,
  Ops.MUL: operator.mul,
  Ops.MAX: max,
}
def _fold_consts(x:UOp, y:UOp, z:UOp) -> UOp:
  assert isinstance(x.arg, (int, float)) and isinstance(y.arg, (int, float)), f"why isn't CONST.arg not a number?"
  return UOp(Ops.CONST, z.dtype, (), _FOLD_OPS[z.op](x.arg, y.arg))
FOLD_CONSTS = (
  UPat(set(_FOLD_OPS.keys()), src=[UPat(Ops.CONST, name="x"), UPat(Ops.CONST, name="y")], name="z"),
  _fold_consts
)

# identity: x + identity(ADD) -> x
def _elim_identity(x:UOp, y:UOp, z:UOp) -> UOp|None:
  try:
    if y.arg == identity_element(z.op, z.dtype): return x
  except KeyError: return None
  return None
ELIM_IDENTITY = (
  UPat({Ops.ADD, Ops.MUL, Ops.MAX},
    src=[UPat(name="x"), UPat(Ops.CONST, name="y")],
    commutative=True, name="z"),
  _elim_identity
)

# absorbing: x * absorbing(MUL) -> absorbing(MUL)
def _elim_absorbing(x:UOp, y:UOp, z:UOp) -> UOp | None:
  try:
    if y.arg == absorbing_element(z.op, z.dtype): 
      if y.dtype == z.dtype: return y
      return UOp(Ops.CONST, z.dtype, (), y.arg) # NOTE: this makes cast a noop when the type mismatches!!! actually genius
  except KeyError: return None
  return None
ELIM_ABSORBING = (
  UPat({Ops.MUL, Ops.AND, Ops.MAX},
    src=[UPat(name="x"), UPat(Ops.CONST, name="y")],
    commutative=True, name="z"),
  _elim_absorbing
)

# CAST(CAST(x, _), T2) -> CAST(x, T2)
def _elim_double_cast(x:UOp, z:UOp) -> UOp:
  return UOp(Ops.CAST, z.dtype, (x.src[0],), z.arg)
ELIM_DOUBLE_CAST = (
  UPat(Ops.CAST, src=[UPat(Ops.CAST, name="x")], name="z"),
  _elim_double_cast
)

# POW(x, CONST(2)) -> MUL(x, x)
# pow() is expensive
def _strength_reduce_pow2(x:UOp, y:UOp, z:UOp) -> UOp|None:
  if y.arg == 2.0: return UOp(Ops.MUL, z.dtype, (x, x), z.arg)
  return None
STRENGTH_REDUCE_POW2 = (
  UPat(Ops.POW, src=[UPat(name="x"), UPat(Ops.CONST, name="y")], name="z"),
  _strength_reduce_pow2
)

# MUL(MUL(x, CONST(-1)), CONST(-1)) -> x
# double negation
def _elim_double_neg(inner_x:UOp, c1:UOp, c2:UOp) -> UOp|None:
  if c1.arg == -1.0 and c2.arg == -1.0: return inner_x
  return None
ELIM_DOUBLE_NEG = (
  UPat(Ops.MUL,
    src=[UPat(Ops.MUL, src=[UPat(name="inner_x"), UPat(Ops.CONST, name="c1")], commutative=True), 
         UPat(Ops.CONST, name="c2")],
    commutative=True),
  _elim_double_neg
)

# fold const recips
def _fold_recip_const(x:UOp, z:UOp) -> UOp|None:
  if x.arg != 0: return UOp(Ops.CONST, z.dtype, (), 1.0 / x.arg)
  return None
FOLD_RECIP_CONST = (
  UPat(Ops.RECIP, src=[UPat(Ops.CONST, name="x")], name="z"),
  _fold_recip_const
)

DEFAULT_RULES: list[Rule] = [
  ELIM_DOUBLE_CAST,       # CAST(CAST(x, dt1), dt2) -> CAST(x, dt2)
  FOLD_RECIP_CONST,       # RECIP(CONST) -> CONST(arg=1.0/old_arg)
  FOLD_CONSTS,            # CONST *op* CONST -> CONST
  ELIM_IDENTITY,          # x * CONST(_identity_) -> x
  ELIM_ABSORBING,         # x * CONST(_absorb_) -> CONST(_absorb_)
  ELIM_DOUBLE_NEG,        # -(-x) -> x
  STRENGTH_REDUCE_POW2,   # x**2 -> x*x
  FUSE_MULACC,            # ADD(MUL(a,b),c) -> MULACC
]
