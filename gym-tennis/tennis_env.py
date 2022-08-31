from cgitb import reset
import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque

from simulator import *

class TennisEnv(gym.Env):    

    def __init__(self):
        super(TennisEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(
            low=lower_bound_list, high=upper_bound_list, dtype=np.float64
        )
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(17,7), dtype=np.float64
        )
        

    def step(self, action):
        # step the physics
        gym1.simulate(sim)
        gym1.fetch_results(sim, True)

        # refresh
        gym1.refresh_actor_root_state_tensor(sim)
        gym1.refresh_rigid_body_state_tensor(sim)
        gym1.refresh_dof_state_tensor(sim)

        # compute velocity of the racket head
        # 13 is the index of the racket; rigid body 7:10 are the index for xyz velocities respecitvely.
        xyz_velocity = rb_states[13][7:10]
        self.speed = math.sqrt(xyz_velocity[0].item()**2 + xyz_velocity[1].item()**2 + xyz_velocity[2].item()**2)

        # position = (3 position, 4 orientation)
        self.position = rb_states[:, 0:7]

        # perform action
        # action is a list of 17, corresponds to 17 rigid bodies in the actor.
        for i in range(len(dof_states)):
            dof_states[i] = torch.tensor([action[i], 0], device="cuda:0")
            dof_states[i].detach().cpu().numpy()

        # applies all the values in the tensor (reverse kinematics)
        gym1.set_dof_state_tensor(sim, dof_states_update)

        # update the viewer
        gym1.step_graphics(sim)
        gym1.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym1.sync_frame_time(sim)

        observation = self.position
        self.reward = self.speed
        
        self.done = False

        # ADD POSITION REWARD FUNCTION

        if self.done:
            self.reward = -10
        info = {}


        return observation, self.reward, self.done, info

    def reset(self):
        reset_action = np.zeros(23)
        observation, __, __, __ = self.step(reset_action)
        observation = observation.detach().cpu().numpy()
        return observation