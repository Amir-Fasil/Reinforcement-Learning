# FrozenLake Q-Learning Agent

This project implements a Q-learning agent to solve the FrozenLake-v1 environment from OpenAI Gymnasium using reinforcement learning. The agent is trained to navigate an 8x8 frozen lake without falling into holes while maximizing rewards.

## Features

Implements Q-learning with an epsilon-greedy exploration strategy.

Supports training and testing modes.

Saves and loads a Q-table using pickle.

Plots training performance as a PNG file (frozen_lake8x8.png).

## Requirements

Ensure you have the following dependencies installed:
```
pip install gymnasium numpy matplotlib pickle5
```

## How to Run

### Training the Agent


To train the agent and save the learned Q-table:
```
python FrozenLakeUsingQLearning.py
```

This will create a frozen_lake8x8.pkl file storing the trained Q-table.

### Testing the Agent


After training, you can test the agent with:

you just need to change the boolean value for is_training to false, the value for render to true, and set number of episodes to 1 to execute the test one time

This will load the saved Q-table and render the environment.

### Training Details

Learning rate (): 0.9 (decays over time)

Discount factor (): 0.9

Exploration rate (): Starts at 1 and decays over time

Actions: Left (0), Down (1), Right (2), Up (3)
### Performance Visualization

The script saves a plot of rewards over time as frozen_lake8x8.pngwhen training, allowing you to see how learning progresses.



# MineRL Model

This project implements PPO, a reinforcement algorithm for navigating through minecraft enviroment to find a location.

requirments
```
pip install minerl gym matplotlib numpy
```
### Problem I faced

I have tried to install gym and minerl in the normal way for the MineRl task on my window, but both of which were not being installed properlly, so I have found a way to install both from a github, which is just a fix for the problem I had with the first method. the consequnce for this is that i don't have access to the enviroment i was currentlly working on. the enviroments it has only are the one from Diamond and Basalt Competition.

I have also implemented the MineRL for different enviroment for the one that is found on Basalt Competition. the task is turning the camera 360 degree and the images on the colabs are to show the camera is turning.
