import KunQuant
from KunQuant.Driver import KunCompilerConfig
from KunQuant.Op import Input, Output, Builder, Rank
from KunQuant.Stage import Function
from KunQuant.jit import cfake
from KunQuant.runner import KunRunner as kr
import geppy as gep
from typing import Dict, Union, List
import Raven.Ops
import pandas as pd
import Raven.Eval.loader
import Raven.Eval.kunquant_impl
from Raven.Result.corr import get_corr_with_existing_alphas
from dataclasses import dataclass
import numpy as np

def evaluate(individual, pset, inputs, idx):
    func = gep.compile_(individual, pset)
    ast: Union[Raven.Ops.Op, int] = func(**Raven.Ops.input_nodes)
    if isinstance(ast, int):
        return None
    ast, _ = ast.legalize(inputs, Raven.Eval.kunquant_impl)
    if isinstance(ast, Raven.Ops.Ops.Constant) or isinstance(ast, Raven.Ops.Ops.Input):
        return None
    return Output(ast.compute_recursive(inputs, Raven.Eval.kunquant_impl), "out" + str(idx))

_executor = None
def _get_executor():
    global _executor
    if _executor:
        return _executor
    _executor = kr.createMultiThreadExecutor(8)
    return _executor
drop_head=30

@dataclass
class SharedMemory:
    ret_data: np.ndarray
    in_data: Dict[str, np.ndarray]
_cached: SharedMemory = None
def test_setter(*args, **kwargs):
    global _cached
    _cached = SharedMemory(args, kwargs)

def evaluate_exist_corr(c):
  if c > 0.9:
    return 0
  if c > 0.8:
    return 1
  if c > 0.7:
    return 2
  if c > 0.6:
    return 3
  if c > 0.5:
    return 4
  return 5
def evaluate_exist_corr_lv1(c):
  if c > 0.8:
    return 1
  if c > 0.6:
    return 3
  return 5

_returns: np.ndarray = None
def evaluate_batch(indvs, pset, data):
    global _returns
    builder = Builder()
    outs = []
    with builder:
        inputs = {"open": Input("open"), "close": Input("close"), "high": Input("high"), "low": Input("low"),
            "volume": Input("volume"), "amount": Input("amount"),}
        for idx, ind in enumerate(indvs):
            out = evaluate(ind, pset, inputs, idx)
            outs.append(out)
    func = [("test", Function(builder.ops), KunCompilerConfig(input_layout="TS", output_layout="TS", split_source=15, partition_factor=6))]
    lib = cfake.compileit(func, "test", cfake.CppCompilerConfig(opt_level=2, fast_linker_threads=4))
    # outbuf = {}
    # for idx, o in enumerate(outs):
    #     outbuf[o.attrs["name"]] = _cached.ret_data[idx]
    time = data["open"].shape[0]
    out = kr.runGraph(_get_executor(), lib.getModule("test"), data, 0, time)
    if _returns is None:
      _returns = pd.DataFrame(data["returns"]).rank(axis=1, pct=True).astype("float32").to_numpy()
    valid_in = []
    valid_ic = []
    ind_buf = []
    for idx, o in enumerate(outs):
        if o is not None:
            inbuf = out[o.attrs["name"]]
            outbuf = np.empty((time,), dtype="float32")
            valid_in.append(inbuf)
            valid_ic.append(outbuf)
            ind_buf.append((idx, outbuf))
        else:
            outs[idx] = (0,0,0,0,0)
    kr.corrWith(_get_executor(), valid_in, _returns, valid_ic, rank_inputs = True)
    corrs = get_corr_with_existing_alphas(_get_executor(), data, valid_in)
    for (idx, ic), exist_ic in zip(ind_buf, corrs):
        ic = ic[drop_head:]
        ic = np.nan_to_num(ic, nan=0)
        mean_ic = abs(np.mean(ic))
        #print(exist_ic)
        outs[idx] = (int(mean_ic/0.02), evaluate_exist_corr_lv1(exist_ic), int(mean_ic/0.005), evaluate_exist_corr(exist_ic), mean_ic,)
    return outs
        