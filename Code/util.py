from collections.abc import Iterable
from datetime import datetime


def isonow(): 
    return datetime.now().replace(microsecond=0).isoformat().replace(":", "-")

def get_only(it: Iterable):
    ret_val = next(it)
    assert_stopiter(it)
    return ret_val

def assert_stopiter(it):
    try:
        extra = next(it)
        raise AssertionError(f"No StopIteration encountered, got {extra}")
    except StopIteration:
        pass
