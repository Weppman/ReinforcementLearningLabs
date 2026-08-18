from GridWorld import gridWorld, actionLeft, actionDown, actionRight, actionUp


def environment_step(current_pos,action):

    moves = {
        "Up": (-1,0),
        "Down": (1,0),
        "Left": (0,-1),
        "Right": (0,1)
    }

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

    # dir_row, dir_col = moves[action]
    # r,c = current_pos[0], current_pos[1]

    # next_row, next_col = r + dir_row, c + dir_col

    # if((next_row < 0 or next_row > 4) or
    #     (next_col < 0 or next_col > 4)):

    #      next_row, next_col = r,c


    # next_pos = [next_row, next_col]
    # reward = -1

    #top left [0,0] is goal state
    terminated = (next_pos == [0,0])


    return (next_pos, reward, terminated)




