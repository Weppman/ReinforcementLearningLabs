###
# Group Members
# Joshua Weppelman:2591952
# Michael Anokye-Boateng:2382971
# Suhail Jadwat:2430921
# Name:Student Number
###

import numpy as np
from environments.gridworld import GridworldEnv
import timeit
import time
import matplotlib.pyplot as plt
import math
import gym

def policy_evaluation(env, policy, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:

        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        policy: [S, A] shaped matrix representing the policy.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        Vector of length env.observation_space.n representing the value function.
    """
    final_Vec = np.zeros(env.observation_space.n)

    valueFunction = 1

    while valueFunction >= theta:
        valueFunction = 0

        for s in range(env.observation_space.n):
            v_old = final_Vec[s]
            v_new = 0

            for a in range(env.action_space.n):
                action_prob = policy[s][a]

                for (prob, next_state, reward, done) in (env.P[s][a]):
                    if done:
                        v_new += action_prob * prob * reward
                    else:
                        v_new += action_prob * prob * (reward + discount_factor * final_Vec[next_state])

            final_Vec[s] = v_new
            valueFunction = max(valueFunction, abs(v_old - final_Vec[s]))


    return final_Vec


def policy_iteration(env, policy_evaluation_fn=policy_evaluation, discount_factor=1.0):
    """
    Iteratively evaluates and improves a policy until an optimal policy is found.

    Args:
        env: The OpenAI environment.
        policy_evaluation_fn: Policy Evaluation function that takes 3 arguments:
            env, policy, discount_factor.
        discount_factor: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V):
            """
            Helper function to calculate the value for all action in a given state.
    
            Args:
                state: The state to consider (int)
                V: The value to use as an estimator, Vector of length env.observation_space.n
    
            Returns:
                A vector of length env.action_space.n containing the expected value of each action.
            """
            final_vec = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                v_new = 0.0
                for (prob, next_state, reward, done) in (env.P[state][a]):
                    if done:
                        v_new += prob * reward
                    else:
                        v_new += prob * (reward + discount_factor * V[next_state])
                final_vec[a] = v_new
            return final_vec
            


    policy = np.ones((env.observation_space.n,env.action_space.n)) / env.action_space.n

    while True:
        V = policy_evaluation_fn(env, policy, discount_factor)

        policy_stable = True

        for s in range(env.observation_space.n):
            chosen_action = np.argmax(policy[s])

            action_values = one_step_lookahead(s, V)

            best_action = np.argmax(action_values)

            if chosen_action != best_action:
                policy_stable = False

            tmp = [0]*env.action_space.n 
            tmp[best_action] = 1
            policy[s] = tmp

        if policy_stable:
            return(policy,V)



def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment.
            env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.observation_space.n is a number of states in the environment.
            env.action_space.n is a number of actions in the environment.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V):
            """
            Helper function to calculate the value for all action in a given state.
    
            Args:
                state: The state to consider (int)
                V: The value to use as an estimator, Vector of length env.observation_space.n
    
            Returns:
                A vector of length env.action_space.n containing the expected value of each action.
            """
            final_vec = np.zeros(env.action_space.n)
            for a in range(env.action_space.n):
                v_new = 0.0
                for (prob, next_state, reward, done) in (env.P[state][a]):
                    if done:
                        v_new += prob * reward
                    else:
                        v_new += prob * (reward + discount_factor * V[next_state])
                final_vec[a] = v_new
            return final_vec

    V = np.zeros(env.observation_space.n)

    while True:
        delta = 0

        for s in range(env.observation_space.n):
            action_values = one_step_lookahead(s,V)

            best_action_value = max(action_values)

            delta = max(delta, abs(best_action_value - V[s]))

            V[s] = best_action_value

        if delta < theta:
            break


    policy  = np.zeros((env.observation_space.n, env.action_space.n))
    for s in range(env.observation_space.n):
        for a in range(env.action_space.n):
            action_values = one_step_lookahead(s, V)
            best_action = np.argmax(action_values)

            policy[s][best_action] = 1

    return(policy, V)

    
    

def printPath(env, actions):
    ACTION_MAP = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}

    grid_visual = np.full(env.observation_space.n, 'o', dtype=object)

    state = env._GridworldEnv__current_state

    for action in actions:

        next_state, reward, done, _ = env.step(action)
        
        grid_visual[state] = ACTION_MAP[action]
        
        state = next_state
        #trajectory_matrix = grid_visual.reshape(env.shape)
        # for row in trajectory_matrix:
        #     print(" ".join(row))
        # Uncomment Above to get a step by step guide for this
        print()

    grid_visual[state] = 'X'

    trajectory_matrix = grid_visual.reshape(env.shape)

    for row in trajectory_matrix:
        print(" ".join(row))


def main():
    # Create Gridworld environment with size of 5 by 5, with the goal at state 24. Reward for getting to goal state is 0, and each step reward is -1
    env = GridworldEnv(shape=[5, 5], terminal_states=[
                       24], terminal_reward=0, step_reward=-1)

    print("*" * 5 + " Policy evaluation " + "*" * 5)
    print("")

    random_actions =[]

    state = env.reset()
    print("")
    env.render()
    print("")

    randomPolicy = np.ones((env.observation_space.n,env.action_space.n)) / env.action_space.n

    for x in range(5):
        random_actions.append(np.random.choice(env.action_space.n, p=randomPolicy[state]))

    printPath(env, random_actions)

    ACTION_MAP = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}

    print("*" * 5 + " Policy iteration " + "*" * 5)
    print("")

    #Range of discounts
    discountRate = np.logspace(-0.2,0,num=30)

    policyRecordings = open("policy.txt","w")
    valueRecordings = open("value.txt","w")

    for rate in discountRate:
        # Call policy_iteration
        start = time.time()
        policy, v = policy_iteration(env,discount_factor=rate)
        end = time.time()
        policyRecordings.write(str(end-start) + " " + str(rate) + "\n")

    # Print out best action for each state in grid shape
    best_actions = np.array([ACTION_MAP[np.argmax(p)] for p in policy]).reshape(env.shape)
    print("Grid Policy (Best Actions):")
    for row in best_actions:
        print(" ".join(row))
    print("")

    # Print state value for each state, as grid shape
    print("Grid State Values:")
    print(v.reshape(env.shape))
    print("")

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                        -7., -6., -5., -4., -3.,
                        -6., -5., -4., -3., -2.,
                        -5., -4., -3., -2., -1.,
                        -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)


    print("*" * 5 + " Value iteration " + "*" * 5)
    print("")

    for rate in discountRate:
        # Call value_iteration
        start = time.time()
        policy, v = value_iteration(env,discount_factor=rate)
        end = time.time()
        valueRecordings.write(str(end-start) + " " + str(rate) + "\n")

    # Print out best action for each state in grid shape
    best_actions = np.array([ACTION_MAP[np.argmax(p)] for p in policy]).reshape(env.shape)
    print("Grid Policy (Best Actions):")
    for row in best_actions:
        print(" ".join(row))
    print("")

    # Print state value for each state, as grid shape
    print("Grid State Values:")
    print(v.reshape(env.shape))
    print("")

    # Test: Make sure the value function is what we expected
    expected_v = np.array([-8., -7., -6., -5., -4.,
                        -7., -6., -5., -4., -3.,
                        -6., -5., -4., -3., -2.,
                        -5., -4., -3., -2., -1.,
                        -4., -3., -2., -1., 0.])
    np.testing.assert_array_almost_equal(v, expected_v, decimal=1)

if __name__ == "__main__":
    main()
