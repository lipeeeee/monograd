from typing import Callable
import traceback

class Tester():
    count_fn_passed:int
    count_fn_total:int

    def __init__(self):
        self.count_fn_passed = 0
        self.count_fn_total = 0

    def test_all(self):
        raise NotImplementedError

    def test_fn(self, fn:Callable):
        assert isinstance(fn, Callable), f"Invalid test fn({fn})"
        error_msg:str|None = None
        err_loc:object|None = None
        self.count_fn_total += 1

        try:
            fn()
            self.count_fn_passed += 1
        except AssertionError as ae:
            error_msg = str(ae)
            err_loc = traceback.extract_tb(ae.__traceback__)[-1]
        except Exception as e:
            error_msg = f"Unhandled Exception: {str(e)}"

        print(f"[{fn.__qualname__}] {'✔' if not error_msg else f'❌ {error_msg}'}")
        if err_loc: print(f"\tline {err_loc.lineno}: {err_loc.line}")

    def show_metrics(self):
        div = 11
        print(f"{'-'*div} METRICS {type(self).__name__} {'-'*div}")
        print(f"count_fn_total={self.count_fn_total} ;;; count_fn_passed={self.count_fn_passed}")
