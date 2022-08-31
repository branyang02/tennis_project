import math
import random
import numpy as np
from isaacgym import gymapi, gymutil
from isaacgym import gymtorch

import torch

# simulation setup
gym1 = gymapi.acquire_gym()

# Creating a Simulation
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = 0
sim_params.physx.use_gpu = True
# activate use_gpu_pipeline to enable Tensor API
sim_params.use_gpu_pipeline = True

# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# uses PhysX backend
sim = gym1.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams() 
gym1.add_ground(sim, plane_params)

# creating a viewer
viewer = gym1.create_viewer(sim, gymapi.CameraProperties())

# load the humanoid with tennis racket mjcf file
asset_root = "/home/brandon/isaacgym/assets"
asset_file = "mjcf/nv_humanoid.xml"
# asset options
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True
# load asset function
asset = gym1.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 1
num_per_row = 1
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(4.2, 2.0, 4)
cam_target = gymapi.Vec3(0, 0, 0)
gym1.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cashe some common handles for later use
envs = []
actor_handles = []

# print("Creating %d environment" % num_envs)

for i in range(num_envs):

    env = gym1.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym1.create_actor(env, asset, pose, "actor", i, 1)
    actor_handles.append(actor_handle)

num_bodies = gym1.get_actor_rigid_body_count(env, actor_handle)
num_joints = gym1.get_actor_joint_count(env, actor_handle)
num_dofs = gym1.get_actor_dof_count(env, actor_handle)
print("number of bodies: ", num_bodies)
print("number of joints: ", num_joints)
print("number of dofs: ", num_dofs)


gym1.prepare_sim(sim)

# Actor Root State Tensor
_root_tensor = gym1.acquire_actor_root_state_tensor(sim)
# wrap in PyTorch Tensor
root_tensor = gymtorch.wrap_tensor(_root_tensor)
# size = (num_actors, the number of elements)
root_positions = root_tensor[:, 0:3]
root_orientations = root_tensor[:, 3:7]
root_linvels = root_tensor[:, 7:10]
root_angvels = root_tensor[:, 10:13]


torch.set_printoptions(profile="full")

# size of rb_states: (num_rigid_bodies, 13), (17, 13) in this case
# num_rigid_bodies = num_bodies * num_actors
# 13 is: 3 position + 4 orientation + 3 linear velociy + 3 angular velocity
_rb_states = gym1.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)


# dof_states = position, velocity
dof_states_update = gym1.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(dof_states_update)
# print(dof_states.size())


props = gym1.get_actor_dof_properties(env, actor_handle)
lower_bound_list = props["lower"]
upper_bound_list = props["upper"]
# print(lower_bound_list)
# print(upper_bound_list)