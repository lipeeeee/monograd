from __future__ import annotations
from typing import Any, Callable, ClassVar, Generic, TypeVar
import numpy as np
import os

T = TypeVar("T")

def flat_mv(mv:memoryview) -> memoryview: return mv if len(mv) == 0 else mv.cast("B", shape=(mv.nbytes,))
def argfix(*x):
  if x and x[0].__class__ in (tuple, list):
    if len(x) != 1: raise ValueError(f"bad arg {x}")
    return tuple(x[0])
  return x

# **** Context/Env vars ****
def getenv(key:str, default:Any=0): return type(default)(os.getenv(key, default))
class ContextVar(Generic[T]):
  _cache: ClassVar[dict[str, ContextVar]] = {}
  value: T
  key: str
  def __init__(self, key: str, default_value: T):
    if key in ContextVar._cache: raise RuntimeError(f"attempt to recreate ContextVar {key}")
    ContextVar._cache[key] = self
    self.value, self.key = getenv(key, default_value), key
  def __bool__(self): return bool(self.value)
  def __eq__(self, x): return self.value == x
  def __ge__(self, x): return self.value >= x
  def __gt__(self, x): return self.value > x
  def __lt__(self, x): return self.value < x
  def tolist(self, obj=None):
    assert isinstance(self.value, str)
    return [getattr(obj, x) if obj else x for x in self.value.split(',') if x]

DEBUG = ContextVar("DEBUG", 0)

# **** generic toposort ****
# usage:  toposort(uop, lambda u: u.src)
#         toposort(tensor, lambda t: t.src)
def toposort(root:T, get_children:Callable[[T], tuple[T, ...]]) -> list[T]:
  visited: set[int] = set()
  order: list[T] = []
  def dfs(node:T):
    if id(node) in visited: return
    visited.add(id(node))
    for child in get_children(node): dfs(child)
    order.append(node)
  dfs(root)
  return order

#### 
DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("true", "1", "t")
def dbg(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_width) % stride == 0
    
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    # Creating vectors for rows and columns
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    
    # Broadcast to create indices
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    
    if isinstance(p, tuple) or isinstance(p, list):
        p = p[0]
        
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    
    # Using np.add.at for atomic addition (handling overlapping patches correctly)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped) # pyright: ignore
    
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]
