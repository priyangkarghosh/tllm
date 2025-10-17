import time
from contextlib import contextmanager

@contextmanager
def Timer():
    start = time.perf_counter()
    class _Timer:
        @property
        def elapsed(self):
            return time.perf_counter() - start
    t = _Timer()
    yield t
