import copy
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
                del self.fitnesses[elite_index]
            num_not_changed -= num_elite

        # mutation
        to_mutate = random.sample(self.population, num_mutated)  # randomly mutating
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
        parents = random.sample(self.population, num_parents)
        for i in range(num_parents // 2):
            offsprings = crossover(parents[i], parents[i + num_parents // 2])
            while offsprings[0] in next_generation or offsprings[1] in next_generation:
                set_parents = [x for x in self.population if x not in parents]
                alternative_parents = random.sample(set_parents, 2)
                offsprings = crossover(alternative_parents[0], alternative_parents[1])
            next_generation.extend(offsprings)

        if len(next_generation) != len(self.population):
            del next_generation[len(next_generation) - 1]

        self.population = next_generation
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
                x = random_integers(env.height - 2)
                y = random_integers(env.width - 2)
                if (x, y) not in prohibited and (x, y) not in population[i]:
                    population[i].append((x, y))
                    break
    return GeneticAlgorithm(population)


def mutation(env, individual):
    prohibited = copy.deepcopy(env.obstacles)
    prohibited.extend([(1, 1), env.goal_position] + individual)
    mutated = []
    mutated.extend(individual)
    while True:
        x = random_integers(env.height - 2)
        y = random_integers(env.width - 2)
        if (x, y) not in prohibited:
            i = random_integers(len(individual) - 1)
            mutated[i] = (x, y)
            break
    return mutated


def crossover(parent1, parent2):
    num_subgoals = len(parent1)
    crossover_point = round(num_subgoals / 2)
    offsprings = []

    p1_1 = parent1[:crossover_point]
    p1_2 = parent1[crossover_point:]
    p2_1 = parent2[:crossover_point]
    p2_2 = parent2[crossover_point:]

    offsprings.append(p1_1 + p2_2)
    offsprings.append(p2_1 + p1_2)

    return offsprings
