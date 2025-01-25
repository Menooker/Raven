from KunQuant.ops import *
from KunQuant.Op import *
import KunQuant.ops

Rank

TsStddev = WindowedStddev
TsSum = WindowedSum
TsMean = WindowedAvg


def TsCorrelation(x, y, window):
    return WindowedCorrelation(x,window,y)

def TsCovariance(x, y, window):
    return WindowedCovariance(x,window,y)

Delay = BackRef

def Delta(df, period):
    return df - BackRef(df, period)

TsMin = WindowedMin
TsMax = WindowedMax
TsArgMax
TsArgMin
DecayLinear
Scale


def is_number(v):
    return isinstance(v, int) or isinstance(v, float)

def min_like(number_min, op_min):
    def inner(a,b):
        if is_number(a) and is_number(b):
            return number_min(a,b)
        if is_number(a):
            a = ConstantOp(a)
        if is_number(b):
            b = ConstantOp(b)
        return op_min(a, b)
    return inner

Min = min_like(min, KunQuant.ops.Min)
Max = min_like(max, KunQuant.ops.Max)

def SelectIfGreater(lhs, rhs, vlhs, vrhs):
    return Select(lhs > rhs, vlhs, vrhs)