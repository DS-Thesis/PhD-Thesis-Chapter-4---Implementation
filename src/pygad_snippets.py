import pygad
import config
import numpy as np


def on_mutation(ga_instance, offspring_mutation):
    #print("on_mutation()-----------------------------------------------")
    pass


def init(fitness_function, nb_genes, pbar):
    keep_parents = 1
    domain = dict()  # [low; high[
    domain["low"] = config.DOMAIN[0]
    domain["high"] = config.DOMAIN[1]
    domain["step"] = 1 / 10**config.PRECISION
    initial_population = np.full((config.POPSIZE, nb_genes), (config.DOMAIN[0] + config.DOMAIN[1]) / 2)#np.zeros((config.POPSIZE, nb_genes))
    ga_instance = pygad.GA(num_generations=config.STEPS,
                    num_parents_mating=config.PARENT_MATING,
                    #mutation_percent_genes=0.1,
                    fitness_func=fitness_function,
                    sol_per_pop=config.POPSIZE,
                    num_genes=nb_genes,
                    #init_range_low=0.0,
                    #init_range_high=0*1.0,
                    parent_selection_type=config.PARENT_SELECTION,
                    keep_parents=keep_parents,
                    crossover_type=config.CROSSOVER_TYPE,
                    mutation_type=config.MUTATION_TYPE,
                    # gene_space=[i / 10 for i in range(11)])  # [0.0; 1.0]
                    #gene_space=[0.1],
                    initial_population=initial_population,
                    gene_space=domain,
                    on_mutation=on_mutation,
                    on_generation=lambda x: pbar.update(1))
    return ga_instance