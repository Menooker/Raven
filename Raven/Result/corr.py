from KunQuant.runner import KunRunner as kr
from KunQuant.Op import Builder, Input, Output, Rank
from KunQuant.jit import cfake
from KunQuant.Stage import Function
from KunQuant.Driver import KunCompilerConfig
import numpy as np
import Raven.Result.result
from typing import Dict, List
import warnings

warnings.filterwarnings('ignore', "Mean of empty slice")
def _build():
    builder = Builder()
    with builder:
        low=Input("low")
        high=Input("high")
        close=Input("close")
        open=Input("open")
        amount=Input("amount")
        volume=Input("volume")
        names = []
        for idx, v in enumerate(Raven.Result.result.get_alphas(low, high, close, open, amount, volume)):
          name = f"alpha{idx}"
          Output(Rank(v), name)
          names.append(name)
    f = Function(builder.ops)
    print("Total names: ", len(names))

    jitm = cfake.compileit([("test", f, KunCompilerConfig(split_source=20, input_layout="TS", output_layout="TS"))], "test", cfake.CppCompilerConfig())
    return jitm, names

_alpha_data: Dict[str, np.ndarray] = None
_start_window: Dict[str, int] = None
def calc_existing_alphas(executor, np_data: Dict[str, np.ndarray]):
  global _alpha_data, _start_window
  if _alpha_data is not None:
    return _alpha_data, _start_window
  jitm, names = _build()
  mod = jitm.getModule("test")
  time = np_data["open"].shape[0]
  _start_window = mod.getOutputUnreliableCount()
  _alpha_data = kr.runGraph(executor, mod, np_data, 0, time)
  return _alpha_data, _start_window

def get_corr(executor, np_data: List[np.ndarray], corrwith: np.ndarray) -> np.ndarray:
  valid_ic = []
  time = corrwith.shape[0]
  outbuf_shared = np.empty((len(np_data), time), dtype="float32")
  for idx, inbuf in enumerate(np_data):
      outbuf = outbuf_shared[idx]
      valid_ic.append(outbuf)
  kr.corrWith(executor, np_data, corrwith, valid_ic, rank_inputs = True)
  ic = np.nanmean(outbuf_shared, axis=1)
  return ic

def get_corr_with_existing_alphas(executor, ochl: Dict[str, np.ndarray], newalpha_data: List[np.ndarray], printall = False) -> List[float]:
  oldalpha_data, _ = calc_existing_alphas(executor, ochl)
  ret = [0] * len(newalpha_data)
  for name, data in oldalpha_data.items():
    if printall:
      print("=======\nOLD", name)
    for idx, corr in enumerate(get_corr(executor, newalpha_data, data)):
      if printall:
        print(idx, abs(corr))
      ret[idx] = max(ret[idx], abs(corr))
  return ret