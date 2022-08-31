# Humanoid Tennis Serve Reinforcement Learning
Uses Reinforcement Learning to demonstrate the best tennis serve strategy


<table>
  <tr>
     <td align="center"><em>Before Training</em></td>
     <td align="center"><em>After Training</em></td>
  </tr>
  <tr>
    <td><img src="./images/pre_train.png" width="300"></td>
    <td><img src="./images/post_train.png" width="300"></td>
  </tr>
</table>

## Description

Goals of the project: 
1. Create a tennis racket with appropiate joints in different degrees of freedom using MuJoCo physics engine.
2. Train the agent to perform a tennis serve using Reinforcement Learning algorithms. The main reward funciton is the speed of the "racket head".

Future improvements:
1. Perform Imitation Learning from expert demonstration using Adversarial Motion Priors(AMP) (https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/rl_examples.md) 
2. Train the agent to perform a tennis serve with an added tennis ball in the environment using [Residual Reinforcement Learning](https://arxiv.org/abs/1812.03201) based on the policy learned from the AMP algorithm.
3. Train the agent to toss the ball with stability as the main reward funciton and height as the secondary reward.
4. Hit the ball at the desired height with the speed of the ball after the serve as the main reward function.
5. Create a virtual tennis court environment in MuJoCo physics engine, and train the Humanoid agent to perform a fast tennis serve while keeping the ball within the service area after perfoming a serve. 


## Getting Started

### Dependencies

* [IsaacGym](https://developer.nvidia.com/isaac-gym)
* Optional: [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
* Cuda >= 11.4
* PyTorch >= 1.12.0
* [StableBaseline3](https://stable-baselines3.readthedocs.io/en/master/)

### Installing

* See [IsaacGym](https://developer.nvidia.com/isaac-gym) for more installation and dependencies info.

### Executing program
1. Set up the anaconda environment.
```
conda activate rlgpu
export LD_LIBRARY_PATH=/home/xyz/anaconda3/envs/rlgpu/lib
```
2. Run the trainning file.
```
cd gym-tennis
python learning.py
```


### Troubleshooting
See included "index.html" for more troubleshooting tips.

### Tennis Racket Creation
The following code is added to "nv_humanoid.xml" to create the tennis racket.
```
 <joint name="right_wrist" axis="-1 -1 -1" range="-90 90" class="big_joint"/>
            <joint name="right_wrist_2" axis="0 -1 1" range="-30 90" class="small_joint" pos=".18 -.18 -.18"/>
            <geom name="right_hand" fromto="0 0 0 .17 .17 .17" size=".031"/>
            <site name="right_hand" class="touch" type="sphere" size=".041"/>
            <body name="racket" pos=".16 .20 .16">
              <geom name="racket" type="cylinder" fromto="0 0 0 .027 -.054 .027" size="0.15"/>
              <site name="racket" class="touch" type="cylinder" size="0.041"/> 
```
Two joints are created to simulate wrist movement.
