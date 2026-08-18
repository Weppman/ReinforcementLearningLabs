from GridWorld import gridWorld, actionLeft, actionDown, actionRight, actionUp
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import numpy as np


def environment_step(current_pos,action):

    pos = [current_pos[0], current_pos[1]]

    if action == "Up":
        pos, grid, success = actionUp(pos, gridWorld)
    elif action == "Down":
        pos, grid, success = actionDown(pos, gridWorld)
    elif action == "Left":
        pos, grid, success = actionLeft(pos, gridWorld)
    elif action == "Right":
        pos, grid, success = actionRight(pos, gridWorld)

    
    next_pos = pos
    reward = -1

    #top left [0,0] is goal state
    terminated = (next_pos == [0,0])


    return (next_pos, reward, terminated)


# in-place policy iteration
def policy_eval_in_place(gamma, theta):
    cols, rows = 4,4
    V =[[0 for i in range(cols)] for j in range(rows)]

    actions = ["Up","Down","Left","Right"]
    goal_pos =[0,0]
    prob = 0.25

    done = False

    iterations = 0
    while not done:
        delta = 0

        for r in range(rows):
            for c in range(cols):

                if ([r,c] == goal_pos):
                    continue


                val = 0
                for action in actions:
                    next_pos,reward,terminated = environment_step([r,c],action)
                    nr, nc = next_pos
                    val += prob * (reward + gamma * V[nr][nc])


                temp_v = V[r][c]
                V[r][c] = val
                delta = max(delta,abs(temp_v - V[r][c]))

        iterations += 1

        if delta < theta:
            done = True

    return V, iterations


        


# two array policy iteration
def policy_eval_two_array(gamma, theta):
    cols,rows = 4,4
    V = [[0 for i in range(cols)] for j in range(rows)]

    actions = ["Up","Down","Left","Right"]
    goal_pos =[0,0]
    prob = 0.25

    done = False

    iterations = 0
    while not done:
        new_V = deepcopy(V)
        delta = 0

        for r in range(rows):
            for c in range(cols):

                if [r,c] == goal_pos:
                    continue


                val = 0
                for action in actions:
                    next_pos, reward, terminated = environment_step([r,c],action)
                    nr, nc = next_pos
                    val += prob * (reward + gamma * V[nr][nc])

                new_V[r][c] = val

                if (abs(new_V[r][c] - V[r][c]) > delta):
                    delta = abs(new_V[r][c] - V[r][c])


        V = new_V
        iterations += 1

        if delta < theta:
            done = True

    return V, iterations


#heat map

y = 1
V, iterations = policy_eval_two_array(gamma = y, theta = 0.01)
print(f"For gamma = {y}, it took {iterations} iterations until convergence")


v_data = np.array(V)
plt.imshow(v_data,cmap = 'magma')

for r in range(4):
    for c in range(4):
        plt.annotate(f"{v_data[r][c]:.2f}",
                     xy = (c,r), 
                     ha = 'center',
                     va = 'center',
                     color = 'green'
                     )


plt.colorbar()

plt.title(r"Value Function for $\gamma = 1$")
plt.xlabel("Cols")
plt.ylabel("Rows")
plt.xticks(range(4))
plt.yticks(range(4))

plt.savefig("value_heatmap.png",dpi = 150)
plt.show()






