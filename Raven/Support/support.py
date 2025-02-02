
from geppy.core.entity import GeneDc, KExpression
from geppy.core.symbol import Function, RNCTerminal, Terminal
from geppy.algorithms.basic import _validate_basic_toolbox, _apply_modification, _apply_crossover
import deap
import Raven.Ops
import pickle
import random

class PartialFunction(Function):
    def __init__(self, name, arity, val):
        super().__init__(name, arity)
        self.val = val
        args = ', '.join('{{{}}}'.format(index)
                         for index in range(arity + len(val)))  # like '{0}, {1}, {2}'
        self._seq = name + '(' + args + ')'  # e.g., add, --> 'add({0}, {1})'
    def format(self, *args):
        assert len(args) == self.arity, "Function {} requires {} arguments while {} are provided.".format(
            self.name, self.arity, len(*args))
        return self._seq.format(*(args + tuple(self.val)))

class GeneDc2(GeneDc):
    @property
    def kexpression(self):
        """
        Get the K-expression of type :class:`KExpression` represented by this gene. The involved RNC terminal will be
        replaced by a constant terminal with its value retrived from the :meth:`rnc_array` according to the GEP-RNC
        algorithm.
        """
        n_rnc = 0

        def convert_RNC(p):
            nonlocal n_rnc
            clazz = Raven.Ops.opname_2_class.get(p.name, None)
            if isinstance(p, Function) and clazz is not None and clazz.has_const_arg():
                index = self.dc[n_rnc  % len(self.dc)]
                value = self.rnc_array[index]
                n_rnc += 1
                t = PartialFunction(p.name, p.arity, [value])
                return t
            if isinstance(p, RNCTerminal):
                index = self.dc[n_rnc  % len(self.dc)]
                value = self.rnc_array[index]
                n_rnc += 1
                t = Terminal(str(value), value)
                return t
            return p

        # level-order
        expr = KExpression([convert_RNC(self[0])])
        i = 0
        j = 1
        while i < len(expr):
            for _ in range(expr[i].arity):
                expr.append(convert_RNC(self[j]))
                j += 1
            i += 1
        return expr

    def __repr__(self):
        return super().__repr__() + ', rnc_array=[' + ', '.join(str(num) for num in self.rnc_array) + ']'

def gep_simple(logbook, population, toolbox, start_gen, n_generations=100, n_elites=1,
               stats=None, hall_of_fame=None, verbose=__debug__):
    _validate_basic_toolbox(toolbox)

    for gen in range(start_gen, n_generations+1):
        # evaluate: only evaluate the invalid ones, i.e., no need to reevaluate the unchanged ones
        invalid_individuals = list(dict([(str(ind),ind) for ind in population if not ind.fitness.valid]).values())
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individuals)
        for ind, fit in zip(invalid_individuals, fitnesses):
            ind.fitness.values = fit
        no_rep = list(dict([(ind.fitness.values[0] ,ind) for ind in population if ind.fitness.valid]).values())
        # record statistics and log
        if hall_of_fame is not None:
            hall_of_fame.update(no_rep)
        record = stats.compile(no_rep) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_individuals), **record)
        if verbose:
            print(logbook.stream)

        if gen == n_generations:
            break

        # selection with elitism
        elites = deap.tools.selTournament(no_rep, k=n_elites-5, tournsize=5) + deap.tools.selBest(no_rep, k=5)
        offspring = toolbox.select(no_rep, len(population) - n_elites)

        # replication
        offspring = [toolbox.clone(ind) for ind in offspring]

        # mutation
        for op in toolbox.pbs:
            if op.startswith('mut'):
                offspring = _apply_modification(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # crossover
        for op in toolbox.pbs:
            if op.startswith('cx'):
                offspring = _apply_crossover(offspring, getattr(toolbox, op), toolbox.pbs[op])

        # replace the current population with the offsprings
        population = elites + offspring
        if (gen + 1) % 50 == 0:
            # Fill the dictionary using the dict(key=value[, ...]) constructor
            cp = dict(population=population, generation=gen, halloffame=hall_of_fame,
                      logbook=logbook, rndstate=random.getstate())

            with open("./tmp/checkpoint_name.pkl", "wb") as cp_file:
                pickle.dump(cp, cp_file)

    return population, logbook