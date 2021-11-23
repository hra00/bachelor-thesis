import copy
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


def episode_with_ga(env, agent, subgoal, g, i, it):
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
        fitness = 1 - (time_step / 100)
        discounted_return_without_subgoals = reward * (discount_factor ** time_step)
        save_video_g_i_it(env, g, i, it)
    return [discounted_return, fitness, discounted_return_without_subgoals]


def subgoal_evolution(env, agent, ga, nr_generation, num_iteration):
    discounted_returns_xsg_g = []
    discounted_returns_g = []
    fitnesses_g = []
    for i in range(nr_generation):
        discounted_returns_xsg_i = []  # discounted returns without subgoal in each generation
        discounted_returns_i = []
        for j in range(len(ga.population)):
            sum_dc_j, sum_fitness_j, sum_dc_xsg_j = 0, 0, 0
            returns = [episode_with_ga(env, agent, ga.population[j], i, j, it) for it in range(num_iteration)]
            print(ga.population[j], returns)
            for discounted_return_it, fitness_it, dc_xsg_it in returns:
                sum_dc_j += discounted_return_it
                sum_fitness_j += fitness_it
                sum_dc_xsg_j += dc_xsg_it
            ga.fitnesses.append(sum_fitness_j / num_iteration)
            discounted_returns_i.append(sum_dc_j / num_iteration)
            discounted_returns_xsg_i.append(sum_dc_xsg_j / num_iteration)
        fitnesses_g = copy.deepcopy(ga.fitnesses)
        ga.next_generation(env, prop_elite, prob_mutation, prop_offsprings)
        discounted_returns_xsg_g.append(max(discounted_returns_xsg_i))
    return discounted_returns_xsg_g, discounted_returns_g, fitnesses_g


def save_video_g_i_it(env, g, i, it):
    env.movie_filename = "G" + str(g) + " | I" + str(i) + "_" + str(it) + ".mp4"
    env.save_video()


params = {}
env = rooms.load_env("layouts/rooms_11_11_4.txt", "rooms.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["horizon"] = 10
params["simulations"] = 100
params["env"] = env

agent = a.MonteCarloTreeSearchPlanner(params)

population_size = 20
num_subgoals = 2
ga = ga.initial_population(env, population_size, num_subgoals)
prop_elite = 0.1
prob_mutation = 0.4
prop_offsprings = 0.5
nr_generation = 20
num_iteration = 5

results = subgoal_evolution(env, agent, ga, nr_generation, num_iteration)
print(results)
x = range(nr_generation)
y = results[0]

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("generation")
plot.ylabel("discounted return")
plot.show()
