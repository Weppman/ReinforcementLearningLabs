from gym import spaces
import numpy as np
import torch
import torch.nn.functional as F


from dqn.model import DQN
from dqn.replay_buffer import ReplayBuffer

device = "cuda"


class DQNAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """

        # Initialise agent's networks, optimiser and replay buffer

        self.policy_network = DQN(observation_space, action_space).to(device)
        self.target_network = DQN(observation_space, action_space).to(device)

        self.optimiser = torch.optim.Adam(self.policy_network.parameters(), lr =lr)

        self.replay_buffer = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.batch_size = batch_size
        self.gamma = gamma      


        self.update_target_network() 
        self.target_network.eval() 
        



    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.from_numpy(states).to(device)
        actions = torch.from_numpy(actions).long().to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).to(device)
        dones = torch.from_numpy(dones).float().to(device)


        q_vals = self.policy_network(states)
        q_sa = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                best_next = self.policy_network(next_states).argmax(dim = 1, keepdim = True)
                next_q = self.target_network(next_states).gather(1, best_next).squeeze(1)
            else:

                next_q = self.target_network(next_states).max(dim=1)[0]

            targets = rewards + self.gamma * (1 - dones) * next_q

        loss = F.smooth_l1_loss(q_sa, targets)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        return loss.item()





    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        # update target_network parameters with policy_network parameters
        self.target_network.load_state_dict(self.policy_network.state_dict())



    def act(self, state: np.ndarray):
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # Select action greedily from the Q-network given the state
        state = torch.from_numpy(np.array(state)).unsqueeze(0).to(device)
        with torch.no_grad():
            q_vals = self.policy_network(state)

        return q_vals.argmax(dim=1).item()

