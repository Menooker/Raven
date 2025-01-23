from typing import Any, List, Dict, Tuple, Union
from abc import ABC, abstractclassmethod, abstractmethod, abstractstaticmethod

def _create_instance(curcls, cls, *args, **kwargs):
    if not hasattr(cls, "_instance"):
        cls._instance = super(curcls, cls).__new__(cls, *args, **kwargs)
    return cls._instance

def nearly_eq(a: float, b: float):
    return abs(a-b) < 0.0005

class Dimension(ABC):

    def __pow__(self, ex: float) -> 'Dimension':
        if isinstance(self, ComplexDimension):
            return ComplexDimension()
        if self == PureNumber():
            return self
        cb = self.get_combine()
        newcb = cb.copy()
        for k, v in newcb.items():
            newcb[k] = v * ex
        return MixedDimension.create(newcb)

    def __mul__(self, v: 'Dimension') -> 'Dimension':
        if isinstance(self, ComplexDimension) or isinstance(v, ComplexDimension):
            return ComplexDimension()
        if self == PureNumber():
            return v
        if v == PureNumber():
            return self
        cb = self.get_combine()
        cb2 = v.get_combine()
        newcb = cb.copy()
        for k, v in cb2.items():
            newcb[k] = newcb.get(k, 0) + v
        return MixedDimension.create(newcb)
    
    def get_combine(self) -> Dict['Dimension', float]:
        return {self: 1}

class ComplexDimension(Dimension):
    def __new__(cls):
        return _create_instance(ComplexDimension, cls)
    def get_combine(self) -> Dict['Dimension', float]:
        raise RuntimeError("ComplexDimension: get_combine")

class MixedDimension(Dimension):
    @staticmethod
    def create(types: Dict[Dimension, float]) -> Dimension:
        if len(types) == 1 and nearly_eq(next(iter(types.values())), 1):
            return next(iter(types.keys()))
        else:
            return MixedDimension(types)

    def __init__(self, types: Dict[Dimension, float]) -> None:
        super().__init__()
        self.types = types
    def __eq__(self, o: object) -> bool:
        if not isinstance(o, ComplexDimension):
            return False
        return self.types == o.types
    def get_combine(self) -> Dict['Dimension', float]:
        return self.types

class PureNumber(Dimension):
    def __new__(cls):
        return _create_instance(PureNumber, cls)

class Money(Dimension):
    def __new__(cls):
        return _create_instance(Money, cls)

class Amount(Dimension):
    def __new__(cls):
        return _create_instance(Amount, cls)

class Op(ABC):
    @abstractstaticmethod
    def num_args() -> int:
        pass
    @abstractstaticmethod
    def has_const_arg() -> bool:
        return False

    def __init__(self, operands: List[Union['Op', float, int]]) -> None:
        super().__init__()
        self.operands: List['Op'] = [op if isinstance(op, Op) else Constant(op) for op in operands]

    def legalize_operands(self, operands: List['Op'], inputs, namespace) -> Tuple[List['Op'], List[Dimension], 'Constant', bool]:
        '''
        return newoperands, dims, first_const, all_const
        '''
        all_const = True
        newoperands = []
        dims = []
        first_const = None
        for op in operands:
            newop, dim = op.legalize(inputs, namespace)
            if not isinstance(newop, Constant):
                all_const = False
            elif first_const is None:
                first_const = newop
            newoperands.append(newop)
            dims.append(dim)
        return newoperands, dims, first_const, all_const
    
    def compute_recursive(self, inputs, namespace) -> Any:
        args = []
        for op in self.operands:
            args.append(op.compute_recursive(inputs, namespace))
        # args = [op.compute_recurive(inputs, namespace) for op in self.operands]
        return self.compute(inputs, namespace, args)

    def compute(self, inputs, namespace, args) -> Any:
        return getattr(namespace, self.__class__.__name__)(*args)

    @abstractmethod
    def legalize(self, inputs, namespace) -> Tuple['Op', Dimension]:
        pass

class ZeroArgsTrait:
    @staticmethod
    def num_args() -> int:
        return 0
class OneArgTrait:
    @staticmethod
    def num_args() -> int:
        return 1
class TwoArgsTrait:
    @staticmethod
    def num_args() -> int:
        return 2
class ThreeArgsTrait:
    @staticmethod
    def num_args() -> int:
        return 3

class NoConstArgTrait:
    @staticmethod
    def has_const_arg() -> bool:
        return False

class HasConstArgTrait:
    @staticmethod
    def has_const_arg() -> bool:
        return True

class Input(NoConstArgTrait, ZeroArgsTrait, Op):
    def __init__(self, name: str, dim: Dimension) -> None:
        super().__init__([])
        self.name = name
        self.dim = dim
    def compute(self, inputs, namespace, args) -> Any:
        return inputs[self.name]
    def legalize(self, inputs, namespace) -> Tuple['Op', Dimension]:
        return self, self.dim

vopen = Input("open", Money())
vclose = Input("close", Money())
vhigh = Input("high", Money())
vlow = Input("low", Money())
volume = Input("volume", Amount())
amount = Input("amount", Money())

class Constant(NoConstArgTrait, ZeroArgsTrait, Op):
    def __init__(self, v) -> None:
        super().__init__([])
        self.val = v
    def compute(self, inputs, namespace, args) -> Any:
        return self.val
    def legalize(self, inputs, namespace) -> Tuple['Op', Dimension]:
        return self, PureNumber()

def _get_first_non_const_dim(newoperands, dims) -> Dimension:
        for op, dim in zip(newoperands, dims):
            if not isinstance(op, Constant):
                return dim
        return None

def _match_dims(ops: List['Op'], dims: List[Dimension], target: Dimension, allow_const_convert: bool) -> Dimension:
    need_match = False
    for op, dim in zip(ops, dims):
        if allow_const_convert and isinstance(op, Constant):
            continue
        if dim != target:
            need_match = True
            break
    if not need_match:
        return target
    for idx, (op, dim) in enumerate(zip(ops, dims)):
        if isinstance(op, Constant):
            continue
        ops[idx] = Rank(op)
    return PureNumber()


class BinaryOp(NoConstArgTrait, TwoArgsTrait, Op):
    def __init__(self, v1: Op, v2: Op) -> None:
        super().__init__([v1, v2])

def _legalize_or_compute(op: Op, inputs, namespace):
    newoperands, dims, _, all_const = op.legalize_operands(op.operands, inputs, namespace)
    if all_const:
        return True, (Constant(op.compute(inputs, namespace, [v.val for v in newoperands])), PureNumber())
    return False, (newoperands, dims)

class AutoPassdownAndFoldTrait(ABC):
    @abstractmethod
    def on_fold_fail(*args) -> Tuple['Op', Dimension]:
        pass

    def legalize(self, inputs, namespace) -> Tuple['Op', Dimension]:
        all_const, ret =  _legalize_or_compute(self, inputs, namespace)
        if all_const:
            return ret
        self.operands = ret[0]
        return self.on_fold_fail(ret[1])

class WindowedOp(HasConstArgTrait, Op):
    def __init__(self, *args) -> None:
        self.window = args[-1]
        Op.__init__(self, args[:-1])
    def compute(self, inputs, namespace, args) -> Any:
        return getattr(namespace, self.__class__.__name__)(*(args+[self.window]))

class AutoPassdownForbidConstantTrait(ABC):
    @abstractmethod
    def on_non_const(self, dims: List[Dimension]) -> Tuple['Op', Dimension]:
        pass
    def legalize(self, inputs, namespace) -> Tuple['Op', Dimension]:
        non_const_operands = self.operands[0:self.num_args()]
        newoperands, dims, first_const, all_const = self.legalize_operands(non_const_operands, inputs, namespace)
        if first_const is not None:
            return first_const, PureNumber()
        self.operands = newoperands + self.operands[self.num_args():]
        return self.on_non_const(dims)


class AddLike(AutoPassdownAndFoldTrait, BinaryOp):
    def __init__(self, v1: Op, v2: Op) -> None:
        super(BinaryOp, self).__init__([v1, v2])
    def on_fold_fail(self, dims: List[Dimension]) -> Tuple['Op', Dimension]:
        target = _get_first_non_const_dim(self.operands, dims)
        return self, _match_dims(self.operands, dims, target, True)


class RankLike(NoConstArgTrait, OneArgTrait, Op):
    def __init__(self, v: Op) -> None:
        super().__init__([v])
    def legalize(self, inputs, namespace) -> Tuple['Op', Dimension]:
        newoperands, _, _, all_const = self.legalize_operands(self.operands, inputs, namespace)
        if all_const:
            return newoperands[0], PureNumber()
        self.operands = newoperands
        return self, PureNumber()
class Rank(RankLike):
    pass
class Scale(RankLike):
    pass


class Add(AddLike):
    def compute(self, inputs, namespace, args) -> Any:
        return args[0] + args[1]

class Sub(AddLike):
    def compute(self, inputs, namespace, args) -> Any:
        return args[0] - args[1]

class Min(AddLike):
    pass

class Max(AddLike):
    pass

class Mul(AutoPassdownAndFoldTrait, BinaryOp):
    def compute(self, inputs, namespace, args) -> Any:
        return args[0] * args[1]
    def on_fold_fail(self, dims: List[Dimension]) -> Tuple['Op', Dimension]:
        return self, dims[0] * dims[1]

class Div(AutoPassdownAndFoldTrait, BinaryOp):
    def compute(self, inputs, namespace, args) -> Any:
        if isinstance(args[1], int) and args[1] == 0:
            return args[0]
        return args[0] / args[1]
    def on_fold_fail(self, dims: List[Dimension]) -> Tuple['Op', Dimension]:
        if isinstance(self.operands[1], Constant) and self.operands[1].val == 0:
            return self.operands[0], dims[0] 
        return self, dims[0] * (dims[1] ** -1)




class TsRankLike(OneArgTrait, AutoPassdownForbidConstantTrait, WindowedOp):
    def on_non_const(self, dims: List[Dimension]) -> Tuple['Op', Dimension]:
        return self, PureNumber()
class TsRank(TsRankLike):
    pass
class TsArgMax(TsRankLike):
    pass
class TsArgMin(TsRankLike):
    pass

class TsMeanLike(OneArgTrait, AutoPassdownForbidConstantTrait, WindowedOp):
    def on_non_const(self, dims: List[Dimension]) -> Tuple['Op', Dimension]:
        return self, dims[0]

class TsMin(TsMeanLike):
    pass
class TsMax(TsMeanLike):
    pass
class Delay(TsMeanLike):
    pass
class Delta(TsMeanLike):
    pass
class DecayLinear(TsMeanLike):
    pass

class TsStddev(TsMeanLike):
    pass

class TsSum(TsMeanLike):
    pass   

class TsMean(TsMeanLike):
    pass

class TsCorrelation(TwoArgsTrait, AutoPassdownForbidConstantTrait, WindowedOp):
    def on_non_const(self, dims: List[Dimension]) -> Tuple['Op', Dimension]:
        return self, PureNumber()

class TsCovariance(TwoArgsTrait, AutoPassdownForbidConstantTrait, WindowedOp):
    def on_non_const(self, dims: List[Dimension]) -> Tuple['Op', Dimension]:
        return self, dims[0]



input_nodes = {
    "open": vopen,
    "close": vclose,
    "high": vhigh,
    "low": vlow,
    "volume": volume,
    "amount": amount
}
all_ops = [Rank, Add, Sub, Mul, Div, TsSum, TsStddev, TsMean, TsCorrelation, TsCovariance, TsRank, TsMin, TsMax, Delay, Delta,
    TsArgMax, TsArgMin, DecayLinear, Scale, Min, Max]
opname_2_class = dict([(clzz.__name__, clzz) for clzz in all_ops])


# TsMean(Div(volume, amount), 19).compute_recursive(None, None)