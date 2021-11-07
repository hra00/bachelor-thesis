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


def episode_with_ga(env, agent, subgoal):
    state = env.reset()
    env.set_subgoals(subgoal)
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
    if done and discounted_return > 1:
        fitness = 1 - (time_step/100)
    return [discounted_return, fitness]


def subgoal_evolution(env, agent, ga, nr_generation, num_iteration):
    for i in range(nr_generation):
        discounted_returns = []
        for j in range(len(ga.population)):
            discounted_return, fitness = 0, 0
            returns = [episode_with_ga(env, agent, ga.population[j]) for _ in range(num_iteration)]
            print(returns)
            for discounted_return_i, fitness_i in returns:
                discounted_return += discounted_return_i
                fitness += fitness_i
            ga.fitnesses.append(fitness/num_iteration)
            discounted_returns.append(discounted_return/num_iteration)
            save_result(env,i,j,num_iteration,returns)
        ga.next_generation(env, prop_elite, prob_mutation, prop_offsprings)
    return discounted_returns

def save_result(env, nr_generation ,nr_iteration, num_iteration, returns):
    env.movie_filename = "Genaration_" + str(nr_generation) + " | " + str(nr_iteration) + " of " + str(num_iteration) + ".mp4"
    plot_name = "Genaration_" + str(nr_generation) + " | " + str(nr_iteration) + " of " + str(num_iteration) + ".png"
    x = range(num_iteration)
    y = returns
    plot.plot(x, y)
    plot.title("Progress")
    plot.xlabel("episode")
    plot.ylabel("discounted return & fitness")
    plot.savefig(plot_name)
    env.save_video()

params = {}
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["horizon"] = 10
params["simulations"] = 100
params["env"] = env

agent = a.MonteCarloTreeSearchPlanner(params)

population_size = 20
num_subgoals = 4
ga = ga.initial_population(env, population_size, num_subgoals)
prop_elite = 0.1
prob_mutation = 0.4
prop_offsprings = 0.5
nr_generation = 20
num_iteration = 1

print(subgoal_evolution(env,agent,ga,nr_generation,num_iteration))

