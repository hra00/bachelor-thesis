import random

import numpy

import rooms
import agent as a
import matplotlib.pyplot as plot
import genitic_algorithm as ga

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

def episode_with_ga(env, agent):
    state = env.reset()
    discounted_return = 0
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
    if done and discounted_return > 0:
        fitness = 1
    return [discounted_return, fitness]

def subgoal_evolution(env, agent, ga, nr_generation, num_iteration):
    for i in range(nr_generation):
        fitnesses = []
        print("generation :", i)
        for j in range(len(ga.population)):
            fitness = []
            discounted_return = []
            for k in range(num_iteration):
                env.set_subgoals(ga.population[j])
                result = episode_with_ga(env, agent)
                fitness.append(result[1])
                discounted_return.append(result[0])
            print("individual ", j, " : ", ga.population[j])
            print("   - fitness :", fitness, " mean :", numpy.mean(fitness))
            print("   - discounted_return :", discounted_return, " mean :", numpy.mean(discounted_return))
            ga.fitnesses.append(numpy.mean(fitnesses))
        ga.next_generation(env, prop_elite, prob_mutation, prop_offsprings)
    return fitness


params = {}
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["horizon"] = 10
params["simulations"] = 100
params["env"] = env
agent = a.MonteCarloTreeSearchPlanner(params)
nr_episodes = 1

ga = ga.initial_population(env, 10, 4)
prop_elite = 0.1
prob_mutation = 0.4
prop_offsprings = 0.5
nr_generation = 10
num_iteration = 10

subgoal_evolution(env, agent, ga, nr_generation, num_iteration)

"""
returns = [episode(env, agent, i) for i in range(nr_episodes)]

x = range(nr_episodes)
y = returns

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("discounted return")
plot.show()

env.save_video()
"""