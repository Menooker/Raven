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
    Div(BackRef(ExpMovingAvg(low, 29), 1), Sub(Select(high> close, close, open), low)),
    #Div(Sub(Select(high> close, close, open), low), BackRef(ExpMovingAvg(low, 29), 32)),
    Div(Add(WindowedMax(low, 3), low* 4), close* 19),
    #Scale(Div(Select(high> close, low, close)/ 26, close)),
  ]