# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy
import time
# Environment
env = gym.make("Taxi-v3")
# Training parameters for Q learning
alpha = 0.9 # Learning rate
gamma = 0.9 # Future reward discount factor
num_of_episodes = 1000
num_of_steps = 500 # per each episode

# Q tables for rewards
#Q_reward = -100000*numpy.ones((500, 6))
Q_reward = numpy.zeros((500, 6))
# Training w/ random sampling of actions
# YOU WRITE YOUR CODE HERE
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.0001
for num in range(num_of_episodes):
    state = env.reset()
    done = False
    while not done:
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = numpy.argmax(Q_reward[state])
        else:
            action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        old_value = Q_reward[state, action]
        new_state_max = numpy.max(Q_reward[new_state])
        Q_reward[state, action] = (1-alpha)*old_value + alpha*(reward + gamma*new_state_max)
        state = new_state
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * numpy.exp(-exploration_decay_rate*num)
tot_reward = 0
tot_action = 0
reward_list = []
action_list = []
Total_re = 0
Total_ac = 0
for i in range(10):
    state = env.reset()
    tot_reward = 0
    for t in range(50):
        action = numpy.argmax(Q_reward[state, :])
        state, reward, done, info = env.step(action)
        print('reward', reward)
        tot_reward += reward
        print('tot_reward', tot_reward)
        env.render()
        #time.sleep(1)
        if done:
            print("Total reward %d" % tot_reward)
            reward_list.append(tot_reward)
            action_list.append(t+1)
            break

for i in range(len(reward_list)):
    Total_re = Total_re + reward_list[i]
    Total_ac = Total_ac + action_list[i]

print("Average reward", Total_re/10)
print("Average action", Total_ac/10)
