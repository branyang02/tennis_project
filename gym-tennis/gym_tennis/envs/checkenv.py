
from stable_baselines3.common.env_checker import check_env
from tennis_env import TennisEnv
from simulator import *

env = TennisEnv()

#check_env(env)

# episodes = 50
# for episode in range(episodes):
#     done = False
#     obs = env.reset()
#     while True:
#         random_action = env.action_space.sample()
#         print("action", random_action)
#         obs, reward, done, info = env.step(random_action)
#         print("reward", reward)


while True:
    random_action = env.action_space.sample()
    obs, reward = step(random_action)
    print("observation\n", obs)
    print("reward\n", reward)