# next
- opmixin


# important
- environmentalize/globalize data/ folder for downloding and caching (is it a good name?)
- review latest ops
- remove device class from utils
- DEBUG=int flag like in tinygrad
- UOp
- do adam & adamw


# important longterm
- refactor to support gpu ops (we will use pyopencl)
  - If user doesnt have it installed, we extract it via ctypes(not very supported )
- UPAT to lower number of gpu instructions, see tinygrad
- LazyBuffer for tensor data
- Kernel fusion(god pls help)


# unimportant
- mnist.py should be cleaned