import enum
import numpy as np
import pandas as pd
import pickle
from typing import Dict
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", message="An input array is constant; the correlation coefficient is not defined.")

cols = ["open", "high", "low", "close", "volume", "amount"]
def loaddata(path: str, date: str, start_dt: datetime, end_dt: datetime) -> Dict[str, pd.DataFrame]:
    with open(date, "rb") as f:
        dates = pickle.load(f)
        start_idx = -1
        end_idx = -1
        for idx, d in enumerate(dates):
            if d == start_dt:
                start_idx = idx
            if d == end_dt:
                end_idx = idx
        if start_idx == -1 or end_idx == -1:
            raise RuntimeError("Cannot find date")
        print("idx", start_idx, end_idx, end_idx - start_idx)
    mat: np.ndarray = np.load(path)["data"]
    # F, S, T, s
    mat_1d: np.ndarray = mat[-1]
    F, S, T, s = mat_1d.shape
    data = mat_1d.transpose((0,1,3,2)).reshape((F, S*s, T))
    ret = dict()
    for idx, col in enumerate(cols):
        ret[col] = pd.DataFrame(data[idx].transpose()[start_idx:end_idx+1])
    ret["returns"] = ret["close"].pct_change().shift(-1)
    return ret

def alpha001(data: Dict[str, pd.DataFrame]):
    inner = data["close"].copy()
    inner[data["returns"] < 0] =  data["returns"].rolling(20).std()
    return ((inner*inner).rolling(5).apply(np.argmax) + 1).rank(axis=1, pct=True)

def get_return_corr(factor: pd.DataFrame, returns: pd.DataFrame):
    return factor.corrwith(returns, axis=1, method="spearman")

def get_ic_ir(factor: pd.DataFrame, returns: pd.DataFrame, drophead: int):
    corr = get_return_corr(factor, returns).iloc[drophead:]
    corr.fillna(0, inplace=True)
    mean = corr.mean()
    std = corr.std()
    if std < 0.0001:
        std = 10000000
    return mean, mean / std

# print(data["close"])
# print(data["returns"])
# a001= alpha001(data)
# print(a001)
# print(get_ic_ir(a001, data["returns"], 30))
    
