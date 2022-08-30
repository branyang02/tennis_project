import gym
from gym import spaces
import numpy as np
import math
import random
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch

import torch

from simulator import *

gym.logger.set_level(40)

class TennisEnv(gym.Env):
    
    def __init__(self):
        super(TennisEnv, self).__init__()

        self.action_space = spaces.Box(
            low=lower_bound_list, high=upper_bound_list, dtype=np.float64
        )
        # observation shape = (17, 7)
        # 17 rigid bodies
        # 7 observations: 3 for cartesian positions and 4 for quaternion
        self.observation_space = spaces.Box(
            low=-np.inf, hihg=np.inf, shape=(119,), dtype=np.float64
        )