import os, subprocess, time, signal
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

from simulator import Speed, PositionStates
from simulator import lower_bound_list, upper_bound_list, dof_states

import torch
import random


#generate random action
def GenerateAction():
  for i in range(len(dof_states)):
    random_action = random.uniform(lower_bound_list[i], upper_bound_list[i])
    dof_states[i] = torch.tensor([random_action, 0], device="cuda:0")
    return dof_states

"""
class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    ...
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human'):
    ...
  def close(self):
    ...
"""