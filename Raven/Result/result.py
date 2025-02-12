from KunQuant.ops import *
from KunQuant.Op import *

def get_alphas(low,
        high,
        close,
        open,
        amount,
        volume):
  return [
    Mul(
                Add(
                    Rank(
                        Add(
                            Rank(WindowedStddev(volume, 33)),
                            Rank(Div(open, low))
                            )),
                    Rank(Div(close, low))
                    ),
                TsRank(close, 3)),
    Div(
            Add(
                WindowedMax(close/ 9, 3),
                Select(WindowedMax(high, 7) > close, low, close)
                ),
            close),
    Div(Div(Scale(Min(open,close)),ExpMovingAvg(Div(open,close), 3)),low),
    Div(BackRef(ExpMovingAvg(low, 29), 1), Sub(Select(high> close, close, open), low) + 0.0001),
    #Div(Sub(Select(high> close, close, open), low), BackRef(ExpMovingAvg(low, 29), 32)),
    # Div(Add(WindowedMax(low, 3), low* 4), close* 19),
    Mul(volume, Sub(close, DecayLinear(low, 3))),
    Mul(Sub(TsRank(close, 2), WindowedCorrelation(high, 20, low)), Add(Rank(volume), Rank(open))),
    Div(Sub(open, Select(close> open, low, open)), BackRef(low, 32)),
    #Scale(Div(Select(high> close, low, close)/ 26, close)),
    Div(DecayLinear(low, 3), Select(Rank(WindowedCovariance(amount, 33, open))> Rank(open), close, open)),
    Min(ExpMovingAvg(Div(high, open), 16), TsRank(TsRank(close, 30), 4)),
    Add(Rank(Add(Rank(Div(close,low)),Rank(WindowedCovariance(high,10,volume)))),Rank(WindowedLinearRegressionSlope(volume,8))),
    # Div(Add(WindowedMax(low, 3), low* 4), close* 19),
    # Scale(Div(Select(high> close, low, close)/ 26, close)),
    Sub(Rank(amount), Rank(Add(Rank(Div(open, close)), Rank(BackRef(amount, 34))))),
    Max(Rank(Div(Mul(close,volume),Scale(amount))),Rank(TsArgMax(low,2))),
    Max(Rank(Mul(WindowedLinearRegressionRSqaure(amount,12),Select(close>open,Rank(volume),Rank(close)))),Rank(Div(close,low))),
    Select(Rank(WindowedCovariance(amount,4,low))>Rank(Div(low,close)),Rank(Div(close,low)),Rank(WindowedCovariance(volume,15, high))),
    ExpMovingAvg(Min(Rank(Div(Div(amount,volume),close)),Rank(Div(open,high))),2),
  ]