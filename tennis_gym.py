from logging import root
import math
import random
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

speed_scale = 1.0

gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0

sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 0
sim_params.physx.num_threads = 0
sim_params.physx.use_gpu = True
sim_params.use_gpu_pipeline = True

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())

asset_root = "/home/brandon/isaacgym/assets"
asset_file = "mjcf/nv_humanoid.xml"

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True


print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 36
num_per_row = 6
spacing = 2.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(17.2, 2.0, 16)
cam_target = gymapi.Vec3(5, -2.5, 13)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


# force sensor



envs = []
actor_handles = []

for i in range(num_envs):

    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.32, 0.0)
    pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "actor", i, 1)
    gym.enable_actor_dof_force_sensors(env, actor_handle)
    actor_handles.append(actor_handle)

    num_bodies = gym.get_actor_rigid_body_count(env, actor_handle)
    num_joints = gym.get_actor_joint_count(env, actor_handle)
    num_dofs = gym.get_actor_dof_count(env, actor_handle)


    props = gym.get_actor_dof_properties(env, actor_handle)
    lower_limits = props['lower']
    upper_limits = props['upper']
    ranges = upper_limits - lower_limits

    

    props["driveMode"].fill(gymapi.DOF_MODE_POS)
    props["stiffness"].fill(1000)
    props["damping"].fill(200.0)
    props["effort"].fill(100.0)
    gym.set_actor_dof_properties(env, actor_handle, props)

    pos_targets = lower_limits + ranges * np.random.random(num_dofs).astype('f')

    #print(pos_targets)
    #gym.set_actor_dof_position_targets(env, actor_handle, pos_targets)


    targets = np.zeros(num_dofs).astype('f')
    # targets = np.array([0.7854, -0.0168, -0.3007, -0.2887, -0.0651, 0.0248, -0.8278, -0.4744,
    # 0.0806, -0.0953, -0.2267, 0.0101, 0.0349, 0.4654, -0.0419, -1.0767, -0.0170, -0.3491, -1.5708, 1.5708,
    # 1.2217, -1.5708, -1.5708]).astype('f')
    gym.set_actor_dof_position_targets(env, actor_handle, targets)

gym.prepare_sim(sim)

_root_tensor = gym.acquire_actor_root_state_tensor(sim)

# root_tensor = 13 floats: 3 for position, 4 for quaternion, 3 for velociy, 3 for angular velocity
root_tensor = gymtorch.wrap_tensor(_root_tensor)
root_positions = root_tensor[:, 0:3]
root_orientations = root_tensor[:, 3:7]
root_linvels = root_tensor[:, 7:10]
root_angvels = root_tensor[:, 10:13]
num_actors = 36


# dof_states = position, velocity
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)



step = 0
position_list = [0.7854, -0.0168, -0.3007, -0.2887, -0.0651, 0.0248, -0.8278, -0.4744,
    0.0806, -0.0953, -0.2267, 0.0101, 0.0349, 0.4654, -0.0419, -1.0767, -0.0170, -0.3491, -1.5708, 1.5708,
    1.2217, -1.5708, -1.5708]

props = gym.get_actor_dof_properties(env, actor_handle)
lower_bound_list = props["lower"]
upper_bound_list = props["upper"]

_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

while not gym.query_viewer_has_closed(viewer):

    step += 1

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    count = 0
    #for i in range(23):
    for i in range(len(dof_states)):
        # apply to all the environments
        if i % len(lower_bound_list) == 0 and i != 0:
            count = 0
        random_action = random.uniform(lower_bound_list[count], upper_bound_list[count])
        dof_states[i] = torch.tensor([random_action, 0], device="cuda:0")
        count += 1

    gym.set_dof_state_tensor(sim, _dof_states)

    gym.refresh_rigid_body_state_tensor(sim)
    # 3 position, orientation, linear velocity, angular velocity
    print(rb_states.size())

    # if step % 100 == 0:
    #     offsets = torch.tensor([0,1,0], device="cuda:0")
    #     root_positions += offsets

    gym.set_actor_root_state_tensor(sim,_root_tensor)
   

    

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destory_sim(sim)