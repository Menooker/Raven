from KunQuant.Driver import KunCompilerConfig
import Raven.Eval.loader as loader
from datetime import datetime
import KunQuant.ops as KunOps
import KunQuant.Op as KunOp
from KunQuant.jit import cfake
from KunQuant.predefined import Alpha101, Alpha158
from KunQuant.runner import KunRunner as kr
from KunQuant.Op import Builder, Input, Output, Rank
from KunQuant.Stage import Function
from KunQuant.predefined.Alpha158 import AllData as AllData158
from KunQuant.predefined.Alpha101 import AllData as AllData101
import numpy as np
import pandas as pd

def build():
    builder = Builder()
    with builder:
        pack_158 = AllData158(low=Input("low"), high=Input("high"), close=Input(
            "close"), open=Input("open"), amount=Input("amount"), volume=Input("volume"))
        alpha158, names = pack_158.build({
            'kbar': {},  # whether to use some hard-code kbar features
            "price": {
                "windows": [0],
                "feature": [("OPEN", pack_158.open), ("HIGH", pack_158.high), ("LOW", pack_158.low), ("VWAP", pack_158.vwap)],
            },
            # 'volume': { # whether to use raw volume features
            #     'windows': [0, 1, 2, 3, 4], # use volume at n days ago
            # },
            'rolling': {  # whether to use rolling operator based features
                'windows': [5, 10, 20, 30, 60],  # rolling windows size
                # if include is None we will use default operators
                # 'exclude': ['RANK'], # rolling operator not to use
            }
        })
        for v, k in zip(alpha158, names):
            Output(Rank(v), k)
        all_data = AllData101(low=pack_158.low,high=pack_158.high,close=pack_158.close,open=pack_158.open, amount=pack_158.amount, volume=pack_158.volume)
        for f in Alpha101.all_alpha:
            out = f(all_data)
            Output(Rank(out), f.__name__)
            names.append(f.__name__)
        newalpha = KunOps.Mul(
                KunOps.Add(
                    Rank(
                        KunOps.Add(
                            Rank(KunOps.WindowedStddev(pack_158.volume, 33)),
                            Rank(KunOps.Div(pack_158.open, pack_158.low))
                            )),
                    Rank(KunOps.Div(pack_158.close, pack_158.low))
                    ),
                KunOps.TsRank(pack_158.close, 3))
        Output(Rank(newalpha), "newalpha")
        names.append("newalpha")
        newalpha2 = KunOps.Div(
            KunOps.Add(
                KunOps.WindowedMax(pack_158.close/ 9, 3),
                KunOps.Select(KunOps.WindowedMax(pack_158.high, 7) > pack_158.close, pack_158.low, pack_158.close)
                ),
            pack_158.close)
        Output(Rank(newalpha2), "newalpha2")
        names.append("newalpha2")


        newalpha3 = KunOps.Div(KunOps.Div(KunOp.Scale(KunOps.Min(pack_158.open,pack_158.close)),KunOps.ExpMovingAvg(KunOps.Div(pack_158.open,pack_158.close), 3)),pack_158.low)
        Output(Rank(newalpha3), "newalpha3")
        names.append("newalpha3")

        newalpha4 = KunOps.Div(KunOps.BackRef(KunOps.ExpMovingAvg(pack_158.low, 29), 1), KunOps.Sub(KunOps.Select(pack_158.high> pack_158.close, pack_158.close, pack_158.open), pack_158.low))
        Output(Rank(newalpha4), "newalpha4")
        names.append("newalpha4")

        simple_return = KunOps.Div(KunOps.BackRef(pack_158.close, 1), pack_158.close)
        Output(Rank(simple_return), "simple_return")
        names.append("simple_return")
    f = Function(builder.ops)
    print("Total names: ", len(names))

    jitm = cfake.compileit([("test", f, KunCompilerConfig(split_source=20, input_layout="TS", output_layout="TS"))], "test", cfake.CppCompilerConfig(), tempdir="/tmp/icir", keep_files=True)
    return jitm, names
jitm, names = build()
mod = jitm.getModule("test")

data = loader.loaddata("/mnt/d/Menooker/quant_data/12y_5m/out.npz", "/mnt/d/Menooker/quant_data/12y_5m/dates.pkl", datetime(2020, 1, 2).date(), datetime(2023, 1, 3).date())
np_data = {}
for k, v in data.items():
    np_data[k] = np.ascontiguousarray(v.to_numpy())
del data
executor = kr.createMultiThreadExecutor(4)
time = np_data["open"].shape[0]
start_window = mod.getOutputUnreliableCount()
out = kr.runGraph(executor, mod, np_data, 0, time)
returns = pd.DataFrame(np_data["returns"]).rank(axis=1, pct=True).astype("float32").to_numpy()
valid_in = []
valid_ic = []
for idx, name in enumerate(names):
    inbuf = out[name]
    outbuf = np.empty((time,), dtype="float32")
    valid_in.append(inbuf)
    valid_ic.append(outbuf)
kr.corrWith(executor,"TS", valid_in, returns, valid_ic)
for idx, (name,ic) in enumerate(zip(names, valid_ic)):
    ic = ic[start_window[name]:]
    ic = np.nan_to_num(ic, nan=0)
    #vic, vir = loader.get_ic_ir(pd.DataFrame(out[name]), pd.DataFrame(returns), start_window[name])
    print(name, np.mean(ic), np.mean(ic)/np.std(ic))#, vic, vir)