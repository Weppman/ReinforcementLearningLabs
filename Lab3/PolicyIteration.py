from GridWorld import gridWorld, actionLeft, actionDown, actionRight, actionUp


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








