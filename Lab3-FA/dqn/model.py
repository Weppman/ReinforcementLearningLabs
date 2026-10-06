from gym import spaces
import torch.nn as nn


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space: spaces.Box, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        assert (
            type(observation_space) == spaces.Box
        ), "observation_space must be of type Box"
        assert (
            len(observation_space.shape) == 3
        ), "observation space must have the form channels x width x height"
        assert (
            type(action_space) == spaces.Discrete
        ), "action_space must be of type Discrete"



        # implement DQN Network
        self.conv = nn.Sequential(

            nn.Conv2d(observation_space.shape[0], 32, kernel_size = 8, stride = 4),
            nn.ReLU(),
            nn.Conv2d(32,64,kernel_size = 4, stride = 2),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size = 3, stride = 1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(

            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n)
        )
    

    

    def forward(self, x):
        # implement forward pass

        #normalize pixels intensity between [0 - 1]
        return self.fc(self.conv(x / 255.0))
    

        
