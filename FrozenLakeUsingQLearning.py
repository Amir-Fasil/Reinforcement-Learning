import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training = True, render = False):
    
    env = gym.make("FrozenLake-v1", map_name = "8x8", is_slippery = False, render_mode = "human" if render else None)
    
    numOfState = env.observation_space.n # number of states in this case 64
    lenOfactionSpace = env.action_space.n # number of action in action space 
    if (is_training):
        Q_table = np.zeros((numOfState,lenOfactionSpace)) # initialization of 64x4 array 
    else:
        f = open("frozen_lake8x8.pkl", "rb")
        Q_table = pickle.load(f)
        f.close()
    
    learning_rate_a = 0.9
    discount_factor_g = 0.9
    
    epsilon = 1 # when epsilon is one we take 100% random action
    epsilon_decay_rate = 0.0001 # epsilon decay rate
    rng = np.random.default_rng() # random number generator
    
    rewards_per_episode = np.zeros(episodes)
    
    for i in range(episodes):
        
        state = env.reset()[0] # states since it is 8x8 we have 64 states from 0 to 63
        terminated = False # when we fall in hole or reached goal
        truncated = False # True when the number of action we took is > 200
        
        while(not terminated and not truncated):
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample() # 0-left, 1-down, 2-right, 3-up
            else:
                action = np.argmax(Q_table[state,:])
                
            new_state, reward, terminated, truncated,_ = env.step(action)
            
            if (is_training):  # this is updating our Q table using Bellman equation
                Q_table[state,action] = Q_table[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(Q_table[new_state,:]) - Q_table[state, action]
                )
            state = new_state
            
        epsilon = max(epsilon - epsilon_decay_rate, 0)
        
        if (epsilon == 0):
            learning_rate_a = 0.0001
            
        if reward == 1:
            rewards_per_episode[i] = 1
        
    env.close()
    
    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.plot(sum_rewards)
    plt.savefig("frozen_lake8x8.png")
    if (is_training):
        f = open("frozen_lake8x8.pkl", "wb")
        pickle.dump(Q_table,f)
        f.close()
    
if __name__ == "__main__":
    run(1, is_training = False, render=True)
    
#I have trained the agent
#Now it is time to test
    
    