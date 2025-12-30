import os

DEBUG_MODE = os.environ.get("DEBUG", "").lower() in ("true", "1", "t")
def dbg(*args, **kwargs):
    if DEBUG_MODE:
        print(*args, **kwargs)
