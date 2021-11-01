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


params = {}
env = rooms.load_env("layouts/rooms_9_9_4.txt", "rooms.mp4")
params["nr_actions"] = env.action_space.n
params["gamma"] = 0.99
params["horizon"] = 10
params["simulations"] = 100
params["env"] = env

agent = a.MonteCarloTreeSearchPlanner(params)
nr_episodes = 10
returns = [episode(env, agent, i) for i in range(nr_episodes)]

x = range(nr_episodes)
y = returns

plot.plot(x, y)
plot.title("Progress")
plot.xlabel("episode")
plot.ylabel("discounted return")
plot.show()

env.save_video()