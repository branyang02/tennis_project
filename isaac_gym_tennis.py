import random
from isaacgym import gymapi
import numpy as np

# initialize gym
gym = gymapi.acquire_gym()

# Devices
compute_device_id = 0
graphics_device_id = 0

# Creating a Simulation
sim_params = gymapi.SimParams()

# set common parameters
# sim_params.dt = 1 / 60
# sim_params.substeps = 2
# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

# set PhysX-specific parameters
sim_params.physx.use_gpu = True
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.num_threads = 0
# sim_params.physx.contact_offset = 0.01
# sim_params.physx.rest_offset = 0.0


sim = gym.create_sim(compute_device_id, graphics_device_id, gymapi.SIM_PHYSX, sim_params)

# configure the ground plane
plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
# plane_params.distance = 0
# plane_params.static_friction = 1
# plane_params.dynamic_friction = 1
# plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# load asset
asset_root = "/home/brandon/isaacgym/assets"
asset_file = "mjcf/nv_humanoid.xml"
#asset_file = "urdf/franka_description/robots/franka_panda.urdf"

asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = True
# asset_options.armature = 0.01
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 64
envs_per_row = 8
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
envs = []
actor_handles = []

# create and populate the environments
for i in range(num_envs):
    env = gym.create_env(sim, env_lower, env_upper, envs_per_row)
    envs.append(env)

    #height = random.uniform(1.0, 2.5)
    height = 1.2

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, height, 0.0)
    #pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

    actor_handle = gym.create_actor(env, asset, pose, "Humanoid", i, 1)
    actor_handles.append(actor_handle)

    # num_bodies = gym.get_actor_rigid_body_count(env, actor_handle)
    # num_joints = gym.get_actor_joint_count(env, actor_handle)
    num_dofs = gym.get_actor_dof_count(env, actor_handle)
    # print("num_bodies", num_bodies)
    # print("num_joints", num_joints)
    # print("num_dofs", num_dofs)

    # configure the joints for effort control mode (once)
    # props = gym.get_actor_dof_properties(env, actor_handle)
    # props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    # props["stiffness"].fill(0.0)
    # props["damping"].fill(0.0)
    # gym.set_actor_dof_properties(env, actor_handle, props)

    # apply efforts (every frame)
    # efforts = np.full(num_dofs, 100.0).astype(np.float32)
    # gym.apply_actor_dof_efforts(env, actor_handle, efforts)





# Running the Simulation
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)




