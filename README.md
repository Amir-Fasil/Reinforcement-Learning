# FrozenLake Q-Learning Agent

This project implements a Q-learning agent to solve the FrozenLake-v1 environment from OpenAI Gymnasium using reinforcement learning. The agent is trained to navigate an 8x8 frozen lake without falling into holes while maximizing rewards.

## Features

### Implements Q-learning with an epsilon-greedy exploration strategy.

#### Supports training and testing modes.

### Saves and loads a Q-table using pickle.

### Plots training performance as a PNG file (frozen_lake8x8.png).

Requirements

Ensure you have the following dependencies installed:

pip install gymnasium numpy matplotlib pickle5

How to Run

Training the Agent

To train the agent and save the learned Q-table:

python frozen_lake_qlearning.py

This will create a frozen_lake8x8.pkl file storing the trained Q-table.

Testing the Agent

After training, you can test the agent with:

python frozen_lake_qlearning.py --test

This will load the saved Q-table and render the environment.

Training Details

Learning rate (): 0.9 (decays over time)

Discount factor (): 0.9

Exploration rate (): Starts at 1 and decays over time

Actions: Left (0), Down (1), Right (2), Up (3)

Performance Visualization

The script saves a plot of rewards over time as frozen_lake8x8.png, allowing you to see how learning progresses.

Demo

If you'd like to add a visual demo, convert a gameplay video into a GIF and embed it in this README:

![FrozenLake Demo](demo.gif)

Author

Created by [Your Name]


