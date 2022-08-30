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
* 
* ex. Windows 10

### Installing

* How/where to download your program
* Any modifications needed to be made to files/folders

### Executing program

* How to run the program
* Step-by-step bullets
```
code blocks for commands
```

## Help

Any advise for common problems or issues.
```
command to run if program contains helper info
```

## Authors

Contributors names and contact info

ex. Dominique Pizzie  
ex. [@DomPizzie](https://twitter.com/dompizzie)

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [NAME HERE] License - see the LICENSE.md file for details

## Acknowledgments

Inspiration, code snippets, etc.
* [awesome-readme](https://github.com/matiassingers/awesome-readme)
* [PurpleBooth](https://gist.github.com/PurpleBooth/109311bb0361f32d87a2)
* [dbader](https://github.com/dbader/readme-template)
* [zenorocha](https://gist.github.com/zenorocha/4526327)
* [fvcproductions](https://gist.github.com/fvcproductions/1bfc2d4aecb01a834b46)
