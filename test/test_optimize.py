# Coverage for monograd/optimize.py: UPat.match, rewrite_graph, _rewrite_pass, and every DEFAULT_RULES rule.
import unittest
import numpy as np
from monograd.device import Device
from monograd.dtype import dtypes
from monograd.uop import Ops 
from monograd.uop.ops import UOp
from monograd.engine.optimize import (
  UPat, Rule,
  rewrite_graph,
  FUSE_MULACC, FOLD_CONSTS, ELIM_IDENTITY, ELIM_ABSORBING,
  ELIM_DOUBLE_CAST, STRENGTH_REDUCE_POW2, ELIM_DOUBLE_NEG,
  FOLD_RECIP_CONST, DEFAULT_RULES,
)


# **** helpers ****
F32 = dtypes.float32
F16 = dtypes.float16
I32 = dtypes.int32
CPU = Device.CPU

def load(shape, dtype=F32) -> UOp:
  u = UOp(Ops.LOAD, dtype, (), (shape, CPU))
  u.assign_buffer(int(np.prod(shape)))
  return u

def const(val, dtype=F32) -> UOp:
  return UOp(Ops.CONST, dtype, (), (val, CPU))

def add(a, b, dtype=F32)  -> UOp: return UOp(Ops.ADD,  dtype, (a, b), CPU)
def mul(a, b, dtype=F32)  -> UOp: return UOp(Ops.MUL,  dtype, (a, b), CPU)
def cast(a, dtype)        -> UOp: return UOp(Ops.CAST, dtype, (a,),   CPU)
def recip(a, dtype=F32)   -> UOp: return UOp(Ops.RECIP, dtype, (a,),  CPU)
def maxop(a, b, dtype=F32)-> UOp: return UOp(Ops.MAX,  dtype, (a, b), CPU)
def andop(a, b, dtype=F32)-> UOp: return UOp(Ops.AND,  dtype, (a, b), CPU)
def powop(a, b, dtype=F32)-> UOp: return UOp(Ops.POW,  dtype, (a, b), CPU)
def neg(a, dtype=F32)     -> UOp: return mul(a, const(-1.0, dtype), dtype)

def rewrite1(root: UOp, rule: Rule) -> UOp:
  """Apply a single rule to fixpoint."""
  return rewrite_graph(root, [rule])

def all_ops(root: UOp) -> list[Ops]:
  """Collect all op types reachable from root."""
  visited, ops = set(), []
  def walk(u):
    if id(u) in visited: return
    visited.add(id(u))
    ops.append(u.op)
    for s in u.src: walk(s)
  walk(root)
  return ops



# **** UPat.match; unit tests for the pattern matcher itself ****
class TestUPatMatch(unittest.TestCase):

  # op matching
  def test_op_none_matches_any(self):
    u = add(load((4,)), load((4,)))
    self.assertIsNotNone(UPat().match(u))

  def test_op_single_matches(self):
    u = add(load((4,)), load((4,)))
    self.assertIsNotNone(UPat(Ops.ADD).match(u))

  def test_op_single_rejects(self):
    u = add(load((4,)), load((4,)))
    self.assertIsNone(UPat(Ops.MUL).match(u))

  def test_op_set_matches_any_in_set(self):
    u = add(load((4,)), load((4,)))
    self.assertIsNotNone(UPat({Ops.ADD, Ops.MUL}).match(u))

  def test_op_set_rejects_not_in_set(self):
    u = add(load((4,)), load((4,)))
    self.assertIsNone(UPat({Ops.MUL, Ops.MAX}).match(u))

  # dtype matching
  def test_dtype_none_matches_any(self):
    u = const(1.0, F32)
    self.assertIsNotNone(UPat(Ops.CONST, dtype=None).match(u))

  def test_dtype_matches(self):
    u = const(1.0, F32)
    self.assertIsNotNone(UPat(Ops.CONST, dtype=F32).match(u))

  def test_dtype_rejects(self):
    u = const(1.0, F32)
    self.assertIsNone(UPat(Ops.CONST, dtype=F16).match(u))

  # arg matching
  def test_arg_none_matches_any(self):
    u = const(2.0)
    self.assertIsNotNone(UPat(Ops.CONST, arg=None).match(u))

  def test_arg_matches_exact(self):
    arg = (2.0, CPU)
    u   = const(2.0)
    self.assertIsNotNone(UPat(Ops.CONST, arg=arg).match(u))

  def test_arg_rejects_wrong(self):
    u = const(2.0)
    self.assertIsNone(UPat(Ops.CONST, arg=(3.0, CPU)).match(u))

  # src matching
  def test_src_none_matches_any_src(self):
    u = add(load((4,)), load((4,)))
    self.assertIsNotNone(UPat(Ops.ADD, src=None).match(u))

  def test_src_correct_length_matches(self):
    a, b = load((4,)), load((4,))
    u    = add(a, b)
    self.assertIsNotNone(UPat(Ops.ADD, src=[UPat(), UPat()]).match(u))

  def test_src_wrong_length_rejects(self):
    a, b = load((4,)), load((4,))
    u    = add(a, b)
    self.assertIsNone(UPat(Ops.ADD, src=[UPat()]).match(u))

  def test_src_nested_match(self):
    a    = load((4,))
    c    = const(2.0)
    m    = mul(a, c)
    pat  = UPat(Ops.MUL, src=[UPat(Ops.LOAD), UPat(Ops.CONST)])
    self.assertIsNotNone(pat.match(m))

  def test_src_nested_reject(self):
    a, b = load((4,)), load((4,))
    m    = mul(a, b)
    pat  = UPat(Ops.MUL, src=[UPat(Ops.LOAD), UPat(Ops.CONST)])
    self.assertIsNone(pat.match(m))

  # name captures 
  def test_name_captured_in_result(self):
    u       = const(2.0)
    captures = UPat(Ops.CONST, name="x").match(u)
    self.assertIsNotNone(captures)
    self.assertIn("x", captures) # type: ignore
    self.assertIs(captures["x"], u) # type: ignore

  def test_nested_names_all_captured(self):
    a = load((4,))
    c = const(2.0)
    m = mul(a, c)
    pat = UPat(Ops.MUL,
      src=[UPat(Ops.LOAD, name="x"), UPat(Ops.CONST, name="y")],
      name="z")
    captures = pat.match(m)
    self.assertIsNotNone(captures)
    self.assertIs(captures["x"], a) # type: ignore
    self.assertIs(captures["y"], c) # type: ignore
    self.assertIs(captures["z"], m) # type: ignore

  def test_no_name_returns_empty_captures(self):
    u        = const(2.0)
    captures = UPat(Ops.CONST).match(u)
    self.assertIsNotNone(captures)
    self.assertEqual(captures, {})

  # commutativity 
  def test_commutative_matches_normal_order(self):
    a   = load((4,))
    c   = const(2.0)
    m   = mul(a, c) # LOAD, CONST - normal order
    pat = UPat(Ops.MUL,
      src=[UPat(Ops.LOAD, name="x"), UPat(Ops.CONST, name="y")],
      commutative=True)
    self.assertIsNotNone(pat.match(m))

  def test_commutative_matches_reversed_order(self):
    a   = load((4,))
    c   = const(2.0)
    m   = mul(c, a) # CONST, LOAD - reversed
    pat = UPat(Ops.MUL,
      src=[UPat(Ops.LOAD, name="x"), UPat(Ops.CONST, name="y")],
      commutative=True)
    captures = pat.match(m)
    self.assertIsNotNone(captures)
    # x must be LOAD, y must be CONST - regardless of src order
    self.assertIs(captures["x"], a) # type: ignore
    self.assertIs(captures["y"], c) # type: ignore

  def test_non_commutative_rejects_reversed_order(self):
    a   = load((4,))
    c   = const(2.0)
    m   = mul(c, a) # CONST on left
    pat = UPat(Ops.MUL,
      src=[UPat(Ops.LOAD, name="x"), UPat(Ops.CONST, name="y")],
      commutative=False)
    self.assertIsNone(pat.match(m))



# **** rewrite_graph; engine behaviour ****
class TestRewriteEngine(unittest.TestCase):

  def test_no_match_returns_same_object(self):
    """If no rule fires, must return the exact same root object."""
    a    = load((4,))
    rule = (UPat(Ops.CAST), lambda z: None) # never matches ADD
    root = add(a, a)
    self.assertIs(rewrite_graph(root, [rule]), root)

  def test_single_rule_fires(self):
    """A matching rule must transform the graph."""
    a    = load((4,))
    c    = const(0.0)
    root = add(a, c)
    result = rewrite1(root, ELIM_IDENTITY)
    self.assertIs(result, a)

  def test_fixpoint_cascading(self):
    """
    FOLD_RECIP_CONST produces a CONST that FOLD_CONSTS can then fold.
    Two passes needed — fixpoint must catch it.
    """
    # RECIP(CONST(2)) * CONST(4)
    # pass1: RECIP(CONST(2)) → CONST(0.5)   [FOLD_RECIP_CONST]
    # pass2: CONST(0.5) * CONST(4) → CONST(2.0)  [FOLD_CONSTS]
    c2   = const(2.0)
    c4   = const(4.0)
    root = mul(recip(c2), c4)
    result = rewrite_graph(root, [FOLD_RECIP_CONST, FOLD_CONSTS])
    self.assertIs(result.op, Ops.CONST)
    self.assertAlmostEqual(result.arg[0], 2.0)

  def test_shared_node_rewritten_once(self):
    """
    Diamond: a feeds both arms of an ADD.
    The rewrite of `a` must only happen once — memo prevents double rewrite.
    """
    c    = const(0.0)
    a    = load((4,))
    # add(a + 0, a + 0) — a+0 is the same UOp object both times
    arm  = add(a, c)
    root = add(arm, arm)
    result = rewrite1(root, ELIM_IDENTITY)
    # both arms should reduce to `a`, root becomes add(a, a)
    self.assertIs(result.src[0], a)
    self.assertIs(result.src[1], a)

  def test_rule_order_matters(self):
    """First matching rule wins — later rules don't fire for same node."""
    a    = load((4,))
    c    = const(1.0)
    root = mul(a, c)
    # ELIM_IDENTITY fires first, removes the mul
    result = rewrite_graph(root, [ELIM_IDENTITY, FUSE_MULACC])
    self.assertIs(result, a)  # ELIM_IDENTITY consumed the node



# **** Individual rules ****
class TestFoldConsts(unittest.TestCase):
  def test_add_consts(self):
    root = add(const(2.0), const(3.0))
    r    = rewrite1(root, FOLD_CONSTS)
    self.assertIs(r.op, Ops.CONST)
    self.assertAlmostEqual(r.arg[0], 5.0)

  def test_mul_consts(self):
    root = mul(const(3.0), const(4.0))
    r    = rewrite1(root, FOLD_CONSTS)
    self.assertIs(r.op, Ops.CONST)
    self.assertAlmostEqual(r.arg[0], 12.0)

  def test_max_consts(self):
    root = maxop(const(3.0), const(7.0))
    r    = rewrite1(root, FOLD_CONSTS)
    self.assertIs(r.op, Ops.CONST)
    self.assertAlmostEqual(r.arg[0], 7.0)

  def test_does_not_fold_non_const(self):
    a    = load((4,))
    root = add(a, const(2.0))
    r    = rewrite1(root, FOLD_CONSTS)
    self.assertIs(r, root) # unchanged

  def test_output_dtype_preserved(self):
    root = add(const(1.0, F32), const(2.0, F32))
    r    = rewrite1(root, FOLD_CONSTS)
    self.assertIs(r.dtype, F32)


class TestElimIdentity(unittest.TestCase):
  def test_add_zero_right(self):
    a = load((4,))
    self.assertIs(rewrite1(add(a, const(0.0)), ELIM_IDENTITY), a)

  def test_add_zero_left(self):
    """commutative=True must catch CONST on the left"""
    a = load((4,))
    self.assertIs(rewrite1(add(const(0.0), a), ELIM_IDENTITY), a)

  def test_mul_one_right(self):
    a = load((4,))
    self.assertIs(rewrite1(mul(a, const(1.0)), ELIM_IDENTITY), a)

  def test_mul_one_left(self):
    a = load((4,))
    self.assertIs(rewrite1(mul(const(1.0), a), ELIM_IDENTITY), a)

  def test_add_nonzero_does_not_fire(self):
    a    = load((4,))
    root = add(a, const(2.0))
    self.assertIs(rewrite1(root, ELIM_IDENTITY), root)

  def test_mul_non_one_does_not_fire(self):
    a    = load((4,))
    root = mul(a, const(2.0))
    self.assertIs(rewrite1(root, ELIM_IDENTITY), root)

  def test_returns_input_not_copy(self):
    """Must return the exact UOp object, not a new one."""
    a   = load((4,))
    r   = rewrite1(add(a, const(0.0)), ELIM_IDENTITY)
    self.assertIs(r, a)


class TestElimAbsorbing(unittest.TestCase):
  def test_mul_zero_right(self):
    a = load((4,))
    r = rewrite1(mul(a, const(0.0)), ELIM_ABSORBING)
    self.assertIs(r.op, Ops.CONST)
    self.assertEqual(r.arg[0], 0)

  def test_mul_zero_left(self):
    a = load((4,))
    r = rewrite1(mul(const(0.0), a), ELIM_ABSORBING)
    self.assertIs(r.op, Ops.CONST)
    self.assertEqual(r.arg[0], 0)

  def test_output_has_correct_dtype(self):
    """Absorbing element must have the same dtype as the op output."""
    a = load((4,), dtype=F32)
    r = rewrite1(mul(a, const(0.0, F32)), ELIM_ABSORBING)
    self.assertIs(r.dtype, F32)

  def test_mul_nonzero_does_not_fire(self):
    a    = load((4,))
    root = mul(a, const(2.0))
    self.assertIs(rewrite1(root, ELIM_ABSORBING), root)

  def test_and_false_absorbing(self):
    a = load((4,), dtype=dtypes.bool)
    c = UOp(Ops.CONST, dtypes.bool, (), (False, CPU))
    r = rewrite1(andop(a, c, dtype=dtypes.bool), ELIM_ABSORBING)
    self.assertIs(r.op, Ops.CONST)


class TestElimDoubleCast(unittest.TestCase):
  def test_double_cast_eliminated(self):
    a    = load((4,), dtype=I32)
    c1   = cast(a, F16)
    c2   = cast(c1, F32)
    r    = rewrite1(c2, ELIM_DOUBLE_CAST)
    self.assertIs(r.op, Ops.CAST)
    self.assertIs(r.dtype, F32)
    # inner src must skip c1 and go straight to a
    self.assertIs(r.src[0], a)

  def test_single_cast_unchanged(self):
    a    = load((4,), dtype=I32)
    c    = cast(a, F32)
    self.assertIs(rewrite1(c, ELIM_DOUBLE_CAST), c)

  def test_triple_cast_fully_collapsed(self):
    """Triple cast should collapse to one in two fixpoint passes."""
    a    = load((4,), dtype=I32)
    c1   = cast(a, F16)
    c2   = cast(c1, F32)
    c3   = cast(c2, F16)
    r    = rewrite_graph(c3, [ELIM_DOUBLE_CAST])
    self.assertIs(r.op, Ops.CAST)
    self.assertIs(r.dtype, F16)
    self.assertIs(r.src[0], a)

  def test_output_dtype_is_outer_cast_dtype(self):
    a  = load((4,), dtype=I32)
    r  = rewrite1(cast(cast(a, F16), F32), ELIM_DOUBLE_CAST)
    self.assertIs(r.dtype, F32)


class TestStrengthReducePow2(unittest.TestCase):
  def test_pow2_becomes_mul(self):
    a    = load((4,))
    root = powop(a, const(2.0))
    r    = rewrite1(root, STRENGTH_REDUCE_POW2)
    self.assertIs(r.op, Ops.MUL)
    # both srcs must be the same UOp
    self.assertIs(r.src[0], a)
    self.assertIs(r.src[1], a)

  def test_pow3_does_not_fire(self):
    a    = load((4,))
    root = powop(a, const(3.0))
    self.assertIs(rewrite1(root, STRENGTH_REDUCE_POW2), root)

  def test_pow1_does_not_fire(self):
    a    = load((4,))
    root = powop(a, const(1.0))
    self.assertIs(rewrite1(root, STRENGTH_REDUCE_POW2), root)

  def test_output_dtype_preserved(self):
    a    = load((4,), dtype=F32)
    root = powop(a, const(2.0, F32))
    r    = rewrite1(root, STRENGTH_REDUCE_POW2)
    self.assertIs(r.dtype, F32)


class TestElimDoubleNeg(unittest.TestCase):
  def test_double_neg_eliminated(self):
    a    = load((4,))
    root = neg(neg(a)) # MUL(MUL(a, -1), -1)
    r    = rewrite1(root, ELIM_DOUBLE_NEG)
    self.assertIs(r, a)

  def test_single_neg_unchanged(self):
    a    = load((4,))
    root = neg(a)
    self.assertIs(rewrite1(root, ELIM_DOUBLE_NEG), root)

  def test_neg_non_minus1_does_not_fire(self):
    a    = load((4,))
    root = mul(mul(a, const(2.0)), const(-1.0))
    self.assertIs(rewrite1(root, ELIM_DOUBLE_NEG), root)

  def test_commutative_const_on_left(self):
    """MUL(CONST(-1), MUL(CONST(-1), x)) must also be caught."""
    a    = load((4,))
    inner = mul(const(-1.0), a)   # CONST on left
    root  = mul(const(-1.0), inner)
    r     = rewrite1(root, ELIM_DOUBLE_NEG)
    self.assertIs(r, a)

  def test_integer_neg_eliminated(self):
    """neg with int -1 (not float -1.0) must also fire."""
    a    = load((4,), dtype=I32)
    root = neg(neg(a))
    r    = rewrite1(root, ELIM_DOUBLE_NEG)
    self.assertIs(r, a)


class TestFoldRecipConst(unittest.TestCase):
  def test_recip_const_folds(self):
    root = recip(const(2.0))
    r    = rewrite1(root, FOLD_RECIP_CONST)
    self.assertIs(r.op, Ops.CONST)
    self.assertAlmostEqual(r.arg[0], 0.5)

  def test_recip_const4_folds(self):
    root = recip(const(4.0))
    r    = rewrite1(root, FOLD_RECIP_CONST)
    self.assertAlmostEqual(r.arg[0], 0.25)

  def test_recip_zero_does_not_fold(self):
    """RECIP(0) must not fold — division by zero."""
    root = recip(const(0.0))
    self.assertIs(rewrite1(root, FOLD_RECIP_CONST), root)

  def test_recip_non_const_does_not_fold(self):
    a    = load((4,))
    root = recip(a)
    self.assertIs(rewrite1(root, FOLD_RECIP_CONST), root)

  def test_output_dtype_preserved(self):
    root = recip(const(2.0, F32))
    r    = rewrite1(root, FOLD_RECIP_CONST)
    self.assertIs(r.dtype, F32)


class TestFuseMulacc(unittest.TestCase):
  def test_add_mul_fuses(self):
    a, b, c = load((4,)), load((4,)), load((4,))
    root    = add(mul(a, b), c)
    r       = rewrite1(root, FUSE_MULACC)
    self.assertIs(r.op, Ops.MULACC)
    self.assertIs(r.src[0], a)
    self.assertIs(r.src[1], b)
    self.assertIs(r.src[2], c)

  def test_commutative_mul_on_right(self):
    """ADD(c, MUL(a,b)) must also fuse."""
    a, b, c = load((4,)), load((4,)), load((4,))
    root    = add(c, mul(a, b))
    r       = rewrite1(root, FUSE_MULACC)
    self.assertIs(r.op, Ops.MULACC)

  def test_integer_does_not_fuse(self):
    """MULACC only valid for floats — integers must not fuse."""
    a = load((4,), dtype=I32)
    b = load((4,), dtype=I32)
    c = load((4,), dtype=I32)
    root = add(mul(a, b, dtype=I32), c, dtype=I32)
    self.assertIs(rewrite1(root, FUSE_MULACC), root)

  def test_output_dtype_is_mul_dtype(self):
    a, b, c = load((4,)), load((4,)), load((4,))
    r       = rewrite1(add(mul(a, b), c), FUSE_MULACC)
    self.assertIs(r.dtype, F32)

  def test_plain_add_does_not_fuse(self):
    a, b = load((4,)), load((4,))
    root = add(a, b)
    self.assertIs(rewrite1(root, FUSE_MULACC), root)



# **** DEFAULT_RULES; integration tests for rule interactions ****
class TestDefaultRules(unittest.TestCase):
  def _opt(self, root: UOp) -> UOp:
    return rewrite_graph(root, DEFAULT_RULES)

  def test_recip_const_then_fold(self):
    """RECIP(CONST(2)) * CONST(4) → CONST(0.5) * CONST(4) → CONST(2.0)"""
    root = mul(recip(const(2.0)), const(4.0))
    r    = self._opt(root)
    self.assertIs(r.op, Ops.CONST)
    self.assertAlmostEqual(r.arg[0], 2.0)

  def test_double_cast_then_identity(self):
    """CAST(CAST(x, F16), F32) eliminated, leaving just x with cast."""
    a = load((4,), dtype=I32)
    r = self._opt(cast(cast(a, F16), F32))
    self.assertIs(r.op, Ops.CAST)
    self.assertIs(r.src[0], a)

  def test_pow2_then_mulacc(self):
    """x**2 + c → MUL(x,x) + c → MULACC(x,x,c)"""
    a = load((4,))
    c = load((4,))
    root = add(powop(a, const(2.0)), c)
    r    = self._opt(root)
    self.assertIs(r.op, Ops.MULACC)
    self.assertIs(r.src[0], a)
    self.assertIs(r.src[1], a)
    self.assertIs(r.src[2], c)

  def test_double_neg_eliminated(self):
    a = load((4,))
    r = self._opt(neg(neg(a)))
    self.assertIs(r, a)

  def test_add_zero_eliminated(self):
    a = load((4,))
    r = self._opt(add(a, const(0.0)))
    self.assertIs(r, a)

  def test_mul_one_eliminated(self):
    a = load((4,))
    r = self._opt(mul(a, const(1.0)))
    self.assertIs(r, a)

  def test_mul_zero_absorbed(self):
    a = load((4,))
    r = self._opt(mul(a, const(0.0)))
    self.assertIs(r.op, Ops.CONST)
    self.assertEqual(r.arg[0], 0)

  def test_all_const_expression_folds(self):
    """CONST(2) + CONST(3) * CONST(4) → CONST(2) + CONST(12) → CONST(14)"""
    root = add(const(2.0), mul(const(3.0), const(4.0)))
    r    = self._opt(root)
    self.assertIs(r.op, Ops.CONST)
    self.assertAlmostEqual(r.arg[0], 14.0)

  def test_no_change_returns_same_object(self):
    """A graph with no optimizable patterns must return the same root."""
    a, b = load((4,)), load((4,))
    root = add(a, b)
    self.assertIs(self._opt(root), root)

  def test_mse_pattern(self):
    """
    (pred - target)^2 where sub = add(pred, neg(target)) and neg = mul(target, -1)
    After pow2 reduction: MUL(add(pred, neg(target)), add(pred, neg(target)))
    The double-neg in -(-x) should fire if present.
    """
    pred   = load((4,))
    target = load((4,))
    diff   = add(pred, neg(target))  # pred - target via decomposition
    sq     = powop(diff, const(2.0))
    r      = self._opt(sq)
    # pow2 must have been strength-reduced to MUL
    self.assertIs(r.op, Ops.MUL)
    self.assertIs(r.src[0], r.src[1])  # both srcs are the same diff node

  def test_layer_norm_div_folds(self):
    """
    In layernorm: x / 8.0 = mul(x, RECIP(CONST(8.0)))
    RECIP(CONST(8.0)) must fold to CONST(0.125)
    then mul(x, CONST(0.125)) stays as a fast multiply
    """
    x    = load((4,))
    n    = const(8.0)
    root = mul(x, recip(n))   # div(x, 8.0) decomposed
    r    = self._opt(root)
    self.assertIs(r.op, Ops.MUL)
    # right src must now be CONST(0.125)
    rhs = r.src[1] if r.src[1].op is Ops.CONST else r.src[0]
    self.assertIs(rhs.op, Ops.CONST)
    self.assertAlmostEqual(rhs.arg[0], 0.125)


if __name__ == "__main__":
  unittest.main(verbosity=2)
