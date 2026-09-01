

import gym
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt


# hyper parameters
learning_rate = 0.5        # alpha
n_episodes = 200           # number of epsiodes
epsilon = 0.1         
gamma = 0.99
max_steps = 100
lamda_vals = [0,0.3,0.5]   #run 200 episodes on each lamda val,
n_runs = 100               #for 100 runs


class CliffAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        discount_factor: float,
        eligibility_decay: float,
    ):
        

        # Q-table: maps (state, action)
        #rows = states, columns of rows = actions and associated q value
        #numpy and defaultdict create structure all initialized to zero
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.e_trace = defaultdict(lambda: np.zeros(env.action_space.n))

        #learning parameters
        self.lr = learning_rate #alpha
        self.discount_factor = discount_factor  # gamma, the effect of future rewards
        self.epsilon = initial_epsilon  # exploration parameter
        self.eligibility = eligibility_decay

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


        # td error
        if terminated:
            td_target = reward
        else:
            td_target = reward + self.discount_factor * self.q_values[next_obs][next_action]
        
        td_error = td_target - self.q_values[obs][action]

        # trace update
        self.e_trace[obs][action] += 1

        # update to all state-action pairs
        for state in self.q_values:
            for a in range(self.env.action_space.n):
                self.q_values[state][a] = self.q_values[state][a] + self.lr * td_error * self.e_trace[state][a]
                self.e_trace[state][a] *= self.discount_factor * self.eligibility 



def training_sars(agent,episodes,steps):

    episode_returns = []
    env = agent.env

    for _ in range(episodes):

        agent.e_trace = defaultdict(lambda: np.zeros(env.action_space.n))
        obs, info = env.reset()
        total_reward = 0

        action = agent.get_action(obs)

        #carry out training
        for _ in range(steps):

            next_obs,reward, terminated,truncated,info = env.step(action)
            next_action = agent.get_action(next_obs)


            agent.sarsa_learn(obs,action,reward,terminated,next_obs,next_action)

            total_reward += reward
           
            obs = next_obs
            action = next_action

            if terminated or truncated:
                break

        episode_returns.append(total_reward)
        

    return episode_returns
  



# runs and plotting
returns_per_trace = {}


for lambda_val in lamda_vals:

    all_returns_sarsa = []

    for i in range(n_runs):

        # Initialise the environment and agent
        env = gym.make('CliffWalking-v1')

        agent = CliffAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=epsilon,
        discount_factor=gamma,
        eligibility_decay=lambda_val
        )


        all_returns_sarsa.append(training_sars(agent,n_episodes,max_steps))
        env.close()

    avg_sarsa = np.mean(all_returns_sarsa,axis = 0)
    returns_per_trace[lambda_val] = avg_sarsa




plt.figure(figsize=(10, 6))
for lambda_val, avg_returns in returns_per_trace.items():
    plt.plot(avg_returns, label=f"lambda = {lambda_val}")

plt.xlabel('Episodes')
plt.ylabel('Average Return')
plt.title('SARSA on CliffWalking Environment')
plt.legend()

plt.grid(True, alpha=0.2)
plt.show()







