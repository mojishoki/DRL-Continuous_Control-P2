# DRL-Continuous_Control-P2-Report
Project 2 Udacity's Deep RL nanodegree Report

##### &nbsp;

## Contents

1. Goal, State & Action Space.
2. DDPG
3. PPO
4. D4PG
5. Possible Future Improvements and Directions

##### &nbsp;

### 1. Goal, State & Action Space

In this project, the goal is to get 20 different robotic arms within Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment to maintain contact with the green spheres for as long as possible.

Most of the times, a reward of +0.04 is provided for each timestep that the agent's hand is in the goal location. In order to solve the environment, the agent must achieve a score of +30 averaged across all 20 agents for 100 consecutive episodes.

![Trained Agent][image1]

#### Summary of Environment (by Unity's [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher)
- Set-up: Double-jointed arm which can move to target locations.
- Goal: Each agent must move its hand to the goal location, and keep it there.
- Agents: The environment contains 20 agents linked to a single Brain.
- Agent Reward Function (independent):
  - +0.04 for each timestep agent's hand is in goal location.
- Brains: One Brain with the following observation/action space.
  - Vector Observation space: 33 variables corresponding to position, rotation, velocity, and angular velocities of the two arm Rigidbodies.
  - Vector Action space: (Continuous) Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.
  - Visual Observations: None.
- Reset Parameters: Two, corresponding to goal size, and goal movement speed.
- Benchmark Mean Reward: 30

##### &nbsp;

### 2. DDPG



<img src="assets/DQNetwork_training_plot.PNG" width="50%" align="top-left" alt="" title="DQNetwork Graph" />