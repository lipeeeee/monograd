# NOTE: This should handle pre-processing of stuff
# like environment variables & pyopencl processing

try:
  import pyopencl
except ImportError:
  # pyopencl was not installed with monograd
  # should implement ctype import but that should NOT be the normal way to do it
  pass