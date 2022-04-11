import copy
import math
import numpy
import random
from numpy.random import random_integers


class GeneticAlgorithm:
    def __init__(self, population):
        self.population = population
        self.fitnesses = []

    """
     Genetic Algorithm
    """

    def next_generation(self, env, prop_elite, prob_mutation, prop_offsprings):
        size = len(self.population)
        next_generation = []
        num_elite = round(size * prop_elite)
        num_mutated = round(size * prob_mutation)
        num_offsprings = round(size * prop_offsprings)
        num_not_changed = size - (num_elite + num_mutated + num_offsprings)

        # pick best n individuals (elites)
        if self.fitnesses:
            for i in range(num_elite):
                elite_index = numpy.argmax(self.fitnesses)
                next_generation.append(self.population[elite_index])
                self.fitnesses[elite_index] = -1
            num_not_changed -= num_elite

        # mutation
        to_mutate = random.sample(self.population, num_mutated)
        for i in range(num_mutated):
            while True:
                mutated = []
                mutated.extend(mutation(env, to_mutate[i]))
                if mutated not in next_generation:
                    next_generation.append(mutated)
                    break

        # individuals that are survived to the next generation
        set_pop = [x for x in self.population if x not in next_generation]
        if num_not_changed > 0:
            survived = random.sample(set_pop, num_not_changed)
            next_generation.extend(survived)

        # crossover
        num_parents = num_offsprings
        if num_offsprings / 2 != 0:
            num_parents += 1
        crossover(self.population, math.ceil(num_offsprings/2), next_generation)
        if len(next_generation) != len(self.population):
            del next_generation[-1]

        self.population = [sorted(i) for i in next_generation]
        self.fitnesses = []


"""
 Creating initial sub-goal set for genetic algorithm 
"""
def initial_population(env, population_size, num_sub_goal):
    prohibited = copy.deepcopy(env.obstacles)
    prohibited.extend([(1, 1), env.goal_position])
    population = []
    for i in range(population_size):
        population.append([])
        for j in range(num_sub_goal):
            while True:
                x = random_integers(env.width - 2)
                y = random_integers(env.height - 2)
                if (x, y) not in prohibited and (x, y) not in population[i]:
                    population[i].append((x, y))
                    break
    return GeneticAlgorithm([sorted(i) for i in population])


def mutation(env, individual):
    prohibited = copy.deepcopy(env.obstacles)
    prohibited.extend([(1, 1), env.goal_position] + individual)
    mutated = []
    mutated.extend(individual)
    while True:
        x = random_integers(env.width - 2)
        y = random_integers(env.height - 2)
        if (x, y) not in prohibited and (x, y) not in mutated:
            mutated[-1] = (x, y)
            break
    return mutated


def crossover(population, num_pair, next_generation):
    pool = copy.deepcopy(population)
    num_subgoals = len(pool[0])
    for i in range(num_pair):
        while True:
            p1, p2 = random.sample(pool, 2)
            crossover_point = round(num_subgoals / 2)
            o1 = p1[:crossover_point] + p2[crossover_point:]
            o2 = p2[:crossover_point] + p1[crossover_point:]
            if set(o1) != set(o2) and o1 not in next_generation and o2 not in next_generation and len(set(o1)) == num_subgoals and len(set(o2)) == num_subgoals:
                next_generation.extend([o1,o2])
                del p1
                del p2
                break


"""
For Diversity Control
"""
def dist_ind(A, B):
    return min([min([math.dist(a, b) for b in B]) for a in A])


def dist_inp(X, P, limit):
    ran_p = copy.deepcopy(P)
    if limit <= len(P):
        ran_p = random.sample(P, limit)
    return numpy.mean([dist_ind(X, e) for e in ran_p])
