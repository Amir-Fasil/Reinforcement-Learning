import gymnasium as gym
import minerl
from stable_baselines3 import PPO
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create the MineRL environment
env = gym.make('MineRLNavigateSparse-v0')

# Step 2: Initialize the PPO model (you can change the model here)
model = PPO('MlpPolicy', env, verbose=1)

# Step 3: Train the model (let's train for 10,000 timesteps)
model.learn(total_timesteps=10000)

# Step 4: Save the trained model
model.save("ppo_minerl_navigate_sparse")

# Step 5: Evaluate the trained model (run it for a few episodes)
total_rewards = []
for episode in range(5):  # Running 5 test episodes
    obs, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
    
    total_rewards.append(total_reward)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

# Step 6: Plot the rewards across episodes
plt.plot(total_rewards)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Reward per Episode')
plt.show()

# Step 7: Close the environment
env.close()
