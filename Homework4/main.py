

import gym
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


# hyper parameters
learning_rate = 0.1         # How fast to learn, alpha for
n_episodes = 1000           # number of epsiodes
start_epsilon = 0.1         # dictates exploration against exploration
gamma = 0.99
max_steps = 100

class CliffAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        discount_factor: float,
    ):
        

        # Q-table: maps (state, action)
        #rows = states, columns of rows = actions and associated q value
        #numpy and defaultdict create structure all initialized to zero
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        #learning parameters
        self.lr = learning_rate #alpha
        self.discount_factor = discount_factor  # gamma, the effect of future rewards
        self.epsilon = initial_epsilon  # exploration parameter
    
    

        #action space - Discrete (4) moving [u,r,d,l]
        #observation space - 48 - players current position

    def get_action(self, obs: tuple[int, int, bool]) -> int:
           
        # decide to explore or exploit 
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # exploit best action in q_table
        else:
            return int(np.argmax(self.q_values[obs]))
                



    def sarsa_learn(
            self,
            obs: tuple[int, int, bool],
            action: int,
            reward: float,
            terminated: bool,
            next_obs: tuple[int, int, bool],
            next_action: int,                
    ):
        

        #sarsa bellman-equation
        value = self.q_values[obs][action] + self.lr * (reward + self.discount_factor * self.q_values[next_obs][next_action] - self.q_values[obs][action])
        
        self.q_values[obs][action] = value



def training_sars(agent,episodes,steps):

    episode_returns = []
    env = agent.env

    for _ in range(episodes):

        #start a new training session
        obs, info = env.reset()
        total_reward = 0

        #choose an initial action
        action = agent.get_action(obs)


        #carry out training
        for _ in range(steps):

           
            #take the action and see what happens
            next_obs,reward, terminated,truncated,info = env.step(action)

            #choose nex action
            next_action = agent.get_action(next_obs)


            #learn from experience
            agent.sarsa_learn(obs,action,reward,terminated,next_obs,next_action)


            total_reward += reward
           
            obs = next_obs
            action = next_action

            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        

    return episode_returns
  


# Initialise the environment and agent
#env = gym.make('CliffWalking-v0')

#env = gym.make('CliffWalking-v0', render_mode='human')
#human for visual window, rgb_array for image arrays



# runs and plotting
all_returns_sarsa = []


for i in range(10):

    # Initialise the environment and agent
    env = gym.make('CliffWalking-v1')

    agent = CliffAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    discount_factor=gamma
    )


    all_returns_sarsa.append(training_sars(agent,n_episodes,max_steps))
    env.close()

avg_sarsa = np.mean(all_returns_sarsa,axis = 0)


plt.figure(figsize=(10, 6))
plt.plot(avg_sarsa, color ='b', label='SARSA')
plt.xlabel('Episodes')
plt.ylabel('Average Return')
plt.title('Q-Learning vs SARSA on CliffWalking Environment')
plt.legend()

plt.grid(True, alpha=0.2)
plt.show()







