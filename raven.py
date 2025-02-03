import geppy as gep
from deap import creator, base, tools
import numpy as np
import pandas as pd
import random
import operator
import Raven.Ops
import Raven.Eval.loader
import Raven.Eval.pandas
from datetime import datetime
from typing import Dict, Union
from dataclasses import dataclass
from Raven.Support import GeneDc2, PartialFunction, gep_simple
from concurrent.futures import ProcessPoolExecutor
import Raven.Multiprocess
import Raven.Eval.kunquant_eval
import pickle
import deap

@dataclass
class SharedMemory:
    data: Dict[str, pd.DataFrame]
_cached: SharedMemory = None
def test_setter(**kwargs):
    global _cached
    data = {}
    for k, v in kwargs.items():
        data[k] = pd.DataFrame(v)
    _cached = SharedMemory(data)

drop_head = 30
def evaluate(individual):
    data = _cached.data
    """Evalute the fitness of an individual: MAE (mean absolute error)"""
    func = toolbox.compile(individual)
    ast: Union[Raven.Ops.Op, int] = func(**Raven.Ops.input_nodes)
    if isinstance(ast, int):
        return 0,
    ast, _ = ast.legalize()
    if isinstance(ast, Raven.Ops.Ops.Constant) or isinstance(ast, Raven.Ops.Ops.Input):
        return 0,
    ic, ir = Raven.Eval.loader.get_ic_ir(ast.compute_recursive(data, Raven.Eval.pandas), data["returns"], drop_head)
    return abs(ic),

def get_prim_set():
    pset = gep.PrimitiveSet('Main', input_names=['open','close', 'high', 'low', 'volume', 'amount'])
    for op in Raven.Ops.all_ops:
        pset.add_function(op, op.num_args())
    # pset.add_function(protected_div, 2)
    pset.add_rnc_terminal()
    return pset


s = 19198101
random.seed(s)
np.random.seed(s)
pset = get_prim_set()



creator.create("FitnessMax", base.Fitness, weights=(1,))  # to minimize the objective (fitness)
creator.create("Individual", gep.Chromosome, fitness=creator.FitnessMax)

h = 6 # head length
n_genes = 1   # number of genes in a chromosome
r = 4   # length of the RNC array

toolbox = gep.Toolbox()
toolbox.register('rnc_gen', random.randint, a=1, b=40)   # each RNC is random integer within [-5, 5]
toolbox.register('gene_gen', GeneDc2, pset=pset, head_length=h, rnc_gen=toolbox.rnc_gen, rnc_array_length=r)
toolbox.register('individual', creator.Individual, gene_gen=toolbox.gene_gen, n_genes=n_genes, linker=None)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# compile utility: which translates an individual into an executable function (Lambda)
toolbox.register('compile', gep.compile_, pset=pset)


toolbox.register('evaluate', evaluate)

toolbox.register('select', tools.selTournament, tournsize=3)
# 1. general operators
toolbox.register('mut_uniform', gep.mutate_uniform, pset=pset, ind_pb=0.05, pb=1)
toolbox.register('mut_invert', gep.invert, pb=0.1)
toolbox.register('mut_is_transpose', gep.is_transpose, pb=0.1)
toolbox.register('mut_ris_transpose', gep.ris_transpose, pb=0.1)
toolbox.register('mut_gene_transpose', gep.gene_transpose, pb=0.1)
toolbox.register('cx_1p', gep.crossover_one_point, pb=0.3)
toolbox.register('cx_2p', gep.crossover_two_point, pb=0.2)
toolbox.register('cx_gene', gep.crossover_gene, pb=0.1)
# 2. Dc-specific operators
toolbox.register('mut_dc', gep.mutate_uniform_dc, ind_pb=0.05, pb=1)
toolbox.register('mut_invert_dc', gep.invert_dc, pb=0.1)
toolbox.register('mut_transpose_dc', gep.transpose_dc, pb=0.1)
# for some uniform mutations, we can also assign the ind_pb a string to indicate our expected number of point mutations in an individual
toolbox.register('mut_rnc_array_dc', gep.mutate_rnc_array_dc, rnc_gen=toolbox.rnc_gen, ind_pb='0.5p')
toolbox.pbs['mut_rnc_array_dc'] = 1  # we can also give the probability via the pbs property

def load_chkpt(n_pop, fn = None):
    if fn:
        # A file name has been given, then load the data from the file
        with open(fn, "rb") as cp_file:
            cp = pickle.load(cp_file)
        population = cp["population"]
        start_gen = cp["generation"]
        halloffame = cp["halloffame"]
        logbook = cp["logbook"]
        random.setstate(cp["rndstate"])
        return population, start_gen, halloffame, logbook
    else:
        pop = toolbox.population(n=n_pop)
        hof = tools.HallOfFame(30, similar=is_pop_similar)   # only record the best three individuals ever found in all generations
        logbook = deap.tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        return pop, 0, hof, logbook

def is_pop_similar(x, y):
    return str(x) == str(y) or x.fitness == y.fitness
if __name__ == "__main__":
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    # stats.register("std", np.std)
    # stats.register("min", np.min)
    stats.register("max", np.max)

    # size of population and number of generations
    n_pop = 350
    n_gen = 8000
    n_elites = 5

    pop, start_gen, hof, logbook = load_chkpt(n_pop)#, "./tmp/checkpoint_name.pkl")


    #data = Raven.Eval.loader.loaddata("D:\\Menooker\\quant_data\\12y_5m\\out.npz", "D:\\Menooker\\quant_data\\12y_5m\\dates.pkl", datetime(2020, 1, 2).date(), datetime(2023, 1, 3).date())
    data = Raven.Eval.loader.loaddata("/mnt/d/Menooker/quant_data/12y_5m/out.npz", "/mnt/d/Menooker/quant_data/12y_5m/dates.pkl", datetime(2020, 1, 2).date(), datetime(2023, 1, 3).date())
    
    use_pandas = False
    if use_pandas:
        np_data = {}
        for k, v in data.items():
            np_data[k] = Raven.Multiprocess.numpy_to_shared_array(np.ascontiguousarray(v.to_numpy()))
        del data
        Raven.Multiprocess.init(np_data, test_setter)
        pool = ProcessPoolExecutor(6, initializer=Raven.Multiprocess.init, initargs=(np_data, test_setter))
        toolbox.register("map", pool.map)
        # start evolution
        pop, log = gep_simple(logbook, pop, toolbox, start_gen, n_generations=n_gen, n_elites=n_elites,
                                stats=stats, hall_of_fame=hof, verbose=True)
        pool.shutdown()
    else:
        np_data = {}
        for k, v in data.items():
            np_data[k] = np.ascontiguousarray(v.to_numpy())
        del data
        def mapper(dummy, indvs):
            return Raven.Eval.kunquant_eval.evaluate_batch(indvs, pset, np_data)
        toolbox.register("map", mapper)
        # start evolution
        pop, log = gep_simple(logbook, pop, toolbox, start_gen, n_generations=n_gen, n_elites=n_elites,
                                stats=stats, hall_of_fame=hof, verbose=True)
    for h in hof:
        print(h)