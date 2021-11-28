import copy
import random
import numpy as np
import rooms
import agent as a
import matplotlib.pyplot as plot
import genitic_algorithm as gena

"""
episode without subgoals
"""
def episode(env, agent, nr_episode=0):
    state = env.reset()
    discounted_return = 0
    discount_factor = 0.99
    done = False
    time_step = 0
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
    print(nr_episode, ":", discounted_return)
    return discounted_return


"""
episode with subgoals
"""
def episode_with_ga(env, agent, subgoal, g, i, it, diversity_control, dist):
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
            fitness = 1 - (time_step / 100) + dist[i]/max(dist)
        else:
            fitness = 1 - (time_step / 100)
        discounted_return_without_subgoals = reward * (discount_factor ** time_step)
        save_video_g_i_it(env, g, i, it)
    return [discounted_return, fitness, discounted_return_without_subgoals]


def subgoal_evolution(env, agent, ga, n_generation, n_iteration, diversity_control):
    discounted_returns_xsg_g = []
    discounted_returns_g = []
    fitnesses_g = []
    for i in range(n_generation):  # for each generation
        discounted_returns_xsg_i = []  # discounted returns without subgoal in each generation
        discounted_returns_i = []
        # for each j in generation, calculate distance between j and population (if diversity_control = true)
        dist = [gena.dist_inp(individual, ga.population, 10) for individual in ga.population] if diversity_control else []
        for j in range(len(ga.population)):  # for each individual in population
            # iteration
            returns = [episode_with_ga(env, agent, ga.population[j], i, j, it, diversity_control, dist) for it in range(n_iteration)]
            print("g :", i, " j : ",ga.population[j], returns)
            # mean returns for individual j
            mean_dc_j, mean_fitness_j, mean_dc_xsg_j = np.mean(returns, axis=0)
            discounted_returns_i.append(mean_dc_j)
            discounted_returns_xsg_i.append(mean_dc_xsg_j)
            ga.fitnesses.append(mean_fitness_j)
        fitnesses_g = copy.deepcopy(ga.fitnesses)
        ga.next_generation(env, prop_elite, prob_mutation, prop_offsprings)
        discounted_returns_xsg_g.append(max(discounted_returns_xsg_i))
    return discounted_returns_xsg_g, discounted_returns_g, fitnesses_g


def save_video_g_i_it(env, g, i, it):
    env.movie_filename = "G" + str(g) + " | I" + str(i) + "_" + str(it) + ".mp4"
    env.save_video()


"""
parameters for agent & environment
"""
params = {}
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["horizon"] = 10
params["simulations"] = 100
params["env"] = env

agent = a.MonteCarloTreeSearchPlanner(params)


"""
parameters for genetic algorithms
"""
population_size = 10
n_subgoals = 2
init_p = gena.initial_population(env, population_size, n_subgoals)
prop_elite = 0.1
prob_mutation = 0.4
prop_offsprings = 0.5
n_generation = 10
n_iteration = 1
diversity_control = True


"""
extract results
"""
results = subgoal_evolution(env, agent, init_p, n_generation, n_iteration, diversity_control)
print(results)
x = range(n_generation)
y = results[0]

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("generation")
plot.ylabel("discounted return")
plot.show()
