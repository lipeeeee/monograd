from enum import IntEnum, auto

class Device(IntEnum):
  CPU = auto()
  GPU = auto() # This is not real, pure "gpu" support is fake, we support libs
  OPENCL = auto()

DeviceLike = str | Device
def to_device(device: DeviceLike): return device if isinstance(device, Device) else getattr(Device, device.upper())
