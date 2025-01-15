import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory, sharedctypes
import numpy as np
from typing import List, Tuple, Dict, Callable
from functools import reduce
from operator import mul

def numpy_to_shared_array(buf: np.ndarray) -> Tuple[sharedctypes.SynchronizedArray, Tuple[int, ...]]:
    if not buf.data.c_contiguous or buf.dtype != np.float32:
        raise RuntimeError("expecting c_contiguous float ndarray")
    array_type = ctypes.POINTER(ctypes.c_float * buf.size)
    data = buf.ctypes.data_as(array_type).contents
    return sharedctypes.Array('f', data, lock=False), buf.shape

def alloc_shared_array(shape: Tuple[int, ...]) -> Tuple[sharedctypes.SynchronizedArray, Tuple[int, ...]]:
    return sharedctypes.Array('f', reduce(mul, shape), lock=False), shape

def shared_array_to_numpy(buf: sharedctypes.SynchronizedArray, shape: Tuple[int, ...]) -> np.ndarray:
    return np.ctypeslib.as_array(buf).reshape(shape)

keep_alive = {}
def init(data: Dict[str, List[Tuple[sharedctypes.SynchronizedArray, Tuple[int, ...]]]], func: Callable):
    out = {}
    for k, (buf, shape) in data.items():
        out[k] = shared_array_to_numpy(buf, shape)
        keep_alive[k] = buf
    func(**out)

def test_setter(a: np.ndarray):
    global _cached
    _cached = a

def test_worker(i):
    global _cached
    _cached[1, i] += 1

if __name__ == "__main__":
    _cached: np.ndarray = None
    buf = np.array([[1,2,3], [3,2,1]], dtype="float32")
    bufs = {"a": numpy_to_shared_array(buf)}
    init(bufs, test_setter)
    import concurrent
    import concurrent.futures
    pool = concurrent.futures.ProcessPoolExecutor(1, initializer=init, initargs=(bufs, test_setter))
    concurrent.futures.wait([pool.submit(test_worker, i) for i in range(3)])
    print(_cached)
