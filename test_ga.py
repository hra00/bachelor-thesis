import copy
import sys
import time

import numpy as np
import rooms
import agent as a
import matplotlib.pyplot as plot
import genitic_algorithm as gena
import os
import csv


"""
test - parameters 
"""
layouts = [[9, 9, 4], [11, 11, 4], [13, 13, 4], [17, 17, 4], [25, 13, 8]]
w, h, r = layouts[4]
file_path = f'./evaluation/{w}_{h}_{r}/'
test_csv_name = f'{w}_{h}_{r}_test.csv'

def episode_with_subgoals(env, agent, subgoal, diversity_control, dist, g, i, it):
    state = env.reset()
    env.set_subgoals(subgoal)
    discounted_return = 0
    discounted_return_without_subgoals = 0
    discount_factor = 0.99
    done = False
    time_step = 0
    fitness = 0
    while not done:
        # 1. Select action according to policy
        action = agent.policy(state)
        # 2. Execute selected action
        next_state, reward, done, _ = env.step(action)
        # 3. Integrate new experience into agent
        agent.update(state, action, reward, next_state, done)
        state = next_state
        discounted_return += reward * (discount_factor ** time_step)
        time_step += 1
    if done and env.agent_position == env.goal_position:
        if diversity_control:
            fitness = 1 - (time_step / max_time_step) + dist[i] / max(dist)
        else:
            fitness = 1 - (time_step / max_time_step)
        discounted_return_without_subgoals = reward * (discount_factor ** time_step)
        save_video_g_i_it(env, g, i, it)
    return np.array([1 if env.agent_position == env.goal_position else 0, time_step, discounted_return, fitness, discounted_return_without_subgoals], dtype=float)


def subgoal_evolution(env, agent, n_generation, population_size, n_iteration, diversity_control, termination_criteria):
    ga = gena.initial_population(env, population_size, n_subgoals)
    bests = []

    # test_results csv
    test_results = open(file_path + test_csv_name, "w")
    wr = csv.writer(test_results)
    wr.writerow(["generation", "individual", "iteration","subgoal","done", "time_step", "discounted_return", "fitness", "discounted_return_without_subgoals"])

    for gen in range(n_generation):

        # check if termination_criteria is satisfied
        if termination_criteria and len(bests) >= termination_criteria:
            if is_not_changed(bests, termination_criteria):
                return

        # for each j in generation, calculate distance between j and population (if diversity_control = true)
        dist = [gena.dist_inp(individual, ga.population, 10) for individual in ga.population] if diversity_control else []
        # calculate fitness of each individual in the population
        for ind in range(population_size):
            subgoal = ga.population[ind]
            ind_fitness = []
            ind_result_total = np.array([0,0,0,0,0],dtype=float)
            ind_s = ""
            for i in range(n_subgoals):
                if i > 0:
                    ind_s += " | "
                ind_s += str(subgoal[i][0]) + " " + str(subgoal[i][1])

            for it in range(n_iteration):
                ind_result = episode_with_subgoals(env, agent, subgoal, diversity_control, dist, gen, ind, it)
                #test_results.write(f'{gen},{ind},{it},' + ",".join([str(i) for i in ind_result]))
                wr.writerow([gen, ind, it, ind_s] + ind_result.tolist())
                ind_fitness.append(ind_result[3])
                ind_result_total += np.array(ind_result)
            ga.fitnesses.append(np.mean(ind_fitness))
            wr.writerow([gen, ind, n_iteration, ind_s] + (ind_result_total/n_iteration).tolist())
            print(gen, ind, subgoal, 'finished')
        ga.next_generation(env, prop_elite, prob_mutation, prop_offsprings)
        bests.append(ga.population[0])

"""
functions
"""
def is_not_changed(bests, termination_criteria):
    last_n_sgs = bests[-termination_criteria:]
    cps = [0 if last_n_sgs[0] == last_n_sgs[i] else 1 for i in range(1, termination_criteria)]
    return True if sum(cps) == 0 else False

def save_video_g_i_it(env, g, i, it):
    env.movie_filename = file_path + f'G_{g} | I_{i} | iteration {it}.mp4'
    env.save_video()

"""
env - parameters
"""
max_time_step = 100
params = {}
env = rooms.load_env(f'layouts/rooms_{w}_{h}_{r}.txt', "rooms.mp4", time_limit=max_time_step)
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["horizon"] = 10
params["simulations"] = 100
params["env"] = env

agent = a.MonteCarloTreeSearchPlanner(params)

"""
ga - parameters
"""
population_size = 10
n_subgoals = 4

prop_elite = 0.01
prob_mutation = 0.4
prop_offsprings = 0.5

n_generation = 10
n_iteration = 1

diversity_control = False
termination_criteria = 0

"""
test
"""
if __name__ == '__main__':
    start = time.time()
    subgoal_evolution(env, agent, n_generation, population_size, n_iteration, diversity_control, termination_criteria)
    print("time :", time.time() - start)