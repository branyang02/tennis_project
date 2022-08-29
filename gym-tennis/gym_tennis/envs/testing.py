

import gym
from gym import spaces
import numpy as np

from simulator import *


class TennisEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(TennisEnv, self).__init__()
        
        # action space
        self.action_space = spaces.Box(
        low=lower_bound_list, high=upper_bound_list, dtype=np.float16)

        x = np.zeros((17))
        # observation space
        self.observation_space = spaces.Box(
            np.zeros((17)), np.full_like(x, 7, dtype=np.int), dtype=np.int
        )

    def step():
        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # refresh
        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)
        gym.refresh_dof_state_tensor(sim)

        # compute velocity of the "racket head"
        # 13 is the index of the racket; rigid body 7:10 are the index for xyz velocities respecitvely
        xyz_velocity = rb_states[13][7:10]
        speed = math.sqrt(xyz_velocity[0].item()**2 + xyz_velocity[1].item()**2 + xyz_velocity[2].item()**2)
        #print(speed)

        # position = (3 position, 4 orientation), 17 rigid bodies
        position = rb_states[:, 0:7]
        #print(position)

        # perform random action
        for i in range(len(dof_states)):
            random_action = random.uniform(lower_bound_list[i], upper_bound_list[i])
            dof_states[i] = torch.tensor([random_action, 0], device="cuda:0")

        # applies all the values in the tensor (reverse kinematics)
        gym.set_dof_state_tensor(sim, _dof_states)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        return position, speed

    def reset(self):
        self.positions = np.zeros((17,7))

    def render(self, mode='human'):
        pass
    def close(self):
        pass