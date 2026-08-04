import math
import random
import numpy as np
from Arm import Arm

def run_eGreedy(epsilon=0.1, runs=100, iterations=1000):

    rewards = np.zeros((runs, iterations))

    for run in range(runs):

        arms = [Arm() for _ in range(10)]
        Q = [0.0] * 10
        N = [0] * 10

        for i in range(iterations):
            if random.random() > epsilon:
                indexPicked = Q.index(max(Q))
            else:
                indexPicked = random.randint(0, 9)

            valueGenerated = arms[indexPicked].pull()
            N[indexPicked] += 1
            Q[indexPicked] += (1 / N[indexPicked]) * (valueGenerated - Q[indexPicked])

            rewards[run, i] = valueGenerated

    return np.mean(rewards, axis=0)

def run_greedyInitialization(initial_Q=5.0, runs=100, iterations=1000):
    rewards = np.zeros((runs, iterations))

    for run in range(runs):
        arms = [Arm() for _ in range(10)]
        Q = [float(initial_Q)] * 10  # Start all estimates optimistically high
        N = [0] * 10

        for i in range(iterations):
            indexPicked = Q.index(max(Q))

            valueGenerated = arms[indexPicked].pull()
            N[indexPicked] += 1
            Q[indexPicked] += (1 / N[indexPicked]) * (valueGenerated - Q[indexPicked])

            rewards[run, i] = valueGenerated

    return np.mean(rewards, axis=0)

def run_UCB(c=2.0, runs=100, iterations=1000):
    rewards = np.zeros((runs, iterations))

    for run in range(runs):
        arms = [Arm() for _ in range(10)]
        Q = [0.0] * 10
        N = [0] * 10

        for arm_idx in range(10):
            reward = arms[arm_idx].pull()
            N[arm_idx] = 1
            Q[arm_idx] = reward
            rewards[run, arm_idx] = reward 

        for i in range(10, iterations):

            t = i + 1 
            
            ucb_values = [Q[x] + c * math.sqrt(math.log(t) / N[x]) for x in range(10)]
            
            indexPicked = ucb_values.index(max(ucb_values))

            valueGenerated = arms[indexPicked].pull()
            N[indexPicked] += 1
            Q[indexPicked] += (1 / N[indexPicked]) * (valueGenerated - Q[indexPicked])

            rewards[run, i] = valueGenerated

    return np.mean(rewards, axis=0)

