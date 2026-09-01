

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



    def get_action(self, obs) -> int:
           
        # decide to explore or exploit 
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # exploit best action in q_table
        else:
            return int(np.argmax(self.q_values[obs]))
                



    def sarsa_learn(
            self,
            obs: int,
            action: int,
            reward: float,
            done: bool,
            next_obs: int,
            next_action: int,                
    ):


        # td error
        if done:
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


def make_value_grid(agent):
    n_states = agent.env.observation_space.n
    grid = np.array([np.max(agent.q_values[s]) for s in range(n_states)])
    return grid



def training_sars(agent,episodes,steps):

    episode_returns = []
    episode_heatmaps = []
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
        episode_heatmaps.append(make_value_grid(agent))
        

    return episode_returns, episode_heatmaps




# runs and plotting

#heatmaps 
run_heatmaps = {}
for lambda_val in lamda_vals:

    # Initialise the environment and agent
    env = gym.make('CliffWalking-v0')

    agent = CliffAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=epsilon,
    discount_factor=gamma,
    eligibility_decay=lambda_val
    )

    _, cliffmaps = training_sars(agent,n_episodes,max_steps)

    run_heatmaps[lambda_val] = cliffmaps
    env.close()



for i in range(n_episodes):
    fig, axes = plt.subplots(1, len(lamda_vals), figsize=(15,6))
    for ax, lambda_val in zip(axes, lamda_vals):
        im = ax.imshow(run_heatmaps[lambda_val][i], cmap = 'magma', vmin = -100, vmax = 0)
        ax.set_title(f"lambda = {lambda_val}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Episode {i + 1}")
    fig.colorbar(im,ax = axes, fraction = 0.02)
    imname = f"episode_{i+1}.png"
    fig.savefig(imname, dpi =150)
    plt.close(fig)



# 100 run average and variance

returns_per_trace = {}


for lambda_val in lamda_vals:

    all_returns_sarsa = []

    for i in range(n_runs):

        # Initialise the environment and agent
        env = gym.make('CliffWalking-v0')

        agent = CliffAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=epsilon,
        discount_factor=gamma,
        eligibility_decay=lambda_val
        )

        training_sars_returns,_ = training_sars(agent,n_episodes,max_steps)


        all_returns_sarsa.append(training_sars_returns)
        env.close()


    avg_sarsa = np.mean(all_returns_sarsa,axis = 0)
    std_sarsa = np.std(all_returns_sarsa,axis = 0)
    returns_per_trace[lambda_val] = (avg_sarsa, std_sarsa)




plt.figure(figsize=(10, 6))
for lambda_val, (avg_returns,var_returns) in returns_per_trace.items():
    episodes = np.arange(len(avg_returns))
    plt.plot(episodes, avg_returns, label=f"lambda = {lambda_val}")
    plt.errorbar(episodes,avg_returns, yerr=var_returns, fmt = 'o',capsize=5,capthick=2)

plt.xlabel('Episodes')
plt.ylabel('Average Return')
plt.title('Average SARSA Returns on CliffWalking Environment')
plt.legend()

plt.grid(True, alpha=0.2)
plt.show()







