import math
import random

def printWorld(grid):
    for x in range(7):
        print(grid[x])

def actionDown(currentPos, grid):
    if (currentPos[0] + 1 < 7):
        if (grid[currentPos[0] + 1][currentPos[1]] >= 0):
            currentPos[0] = currentPos[0] + 1
            return currentPos, grid, True
        else:
            return currentPos, grid, False
    else:
        return currentPos, grid, False

def actionUp(currentPos, grid):
    if (currentPos[0] - 1 >= 0):
        if (grid[currentPos[0] - 1][currentPos[1]] >= 0):
                currentPos[0] = currentPos[0] - 1
                return currentPos, grid, True
        else:
            return currentPos, grid, False
    else:
        return currentPos, grid, False

def actionRight(currentPos, grid):
    if (currentPos[1] + 1 < 7):
        if (grid[currentPos[0]][currentPos[1] + 1] >= 0):
            currentPos[1] = currentPos[1] + 1
            return currentPos, grid, True
        else:
            return currentPos, grid, False
    else:
        return currentPos, grid, False

def actionLeft(currentPos, grid):
    if (currentPos[1] - 1 >= 0):
        if (grid[currentPos[0]][currentPos[1] - 1] >= 0):
            currentPos[1] = currentPos[1] - 1
            return currentPos, grid, True
        else:
            return currentPos, grid, False
    else:
        return currentPos, grid, False

def executeActionSequence(currentPos, grid, moveList):
    """
    Executes a list of string moves on the grid using existing action functions
    """
    totalDistance = 0
    totalReward = 0
    

    pos = [currentPos[0], currentPos[1]]

    for move in moveList:
        start_r, start_c = pos[0], pos[1]
        success = False

        if move == "Up":
            pos, grid, success = actionUp(pos, grid)
        elif move == "Down":
            pos, grid, success = actionDown(pos, grid)
        elif move == "Left":
            pos, grid, success = actionLeft(pos, grid)
        elif move == "Right":
            pos, grid, success = actionRight(pos, grid)

        if success:
            dist_moved = abs(pos[0] - start_r) + abs(pos[1] - start_c)
            totalDistance += dist_moved

        if grid[pos[0]][pos[1]] == 20:
            totalReward += 20
            print(f"Executed: {move:<5} -> New Pos: {pos} | Goal Reached!")
            break
        else:
            totalReward -= 1
            print(f"Executed: {move:<5} -> New Pos: {pos} | Success: {success}")

    return totalDistance, totalReward, pos


def getOptimalPath(grid, startPos, goalPos):
    """
    Finds the shortest sequence of actions from startPos to goalPos using BFS
    """

    queue = [(startPos, [])]
    visited = {tuple(startPos)}

    moves = [
        ("Up", -1, 0),
        ("Down", 1, 0),
        ("Left", 0, -1),
        ("Right", 0, 1)
    ]

    while queue:
        (r, c), path = queue.pop(0)

        if [r, c] == goalPos:
            return path

        for moveName, dr, dc in moves:
            nr, nc = r + dr, c + dc

            if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]):
                if grid[nr][nc] >= 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(([nr, nc], path + [moveName]))

    return []

cols = 7
rows = 7
gridWorld = [[0 for i in range(cols)] for j in range(rows)]
agent = 42
currentPos = [6, 0]
gridWorld[0][6] = 20

for x in range(6):
    gridWorld[2][x] = -1

direction = "word"
state = False

optimalPath = getOptimalPath(gridWorld, currentPos, [0,6] )
executeActionSequence(currentPos,gridWorld,optimalPath)


for x in range(50):
    move = random.randint(1,4)
    match move:
        case 1:
            direction = "Left"
            currentPos, gridWorld, state = actionLeft(currentPos, gridWorld)
            if not state:
                direction = "None"
        case 2:
            direction = "Right"
            currentPos, gridWorld, state = actionRight(currentPos, gridWorld)
            if not state:
                direction = "None"
        case 3:
            direction = "Up"
            currentPos, gridWorld, state = actionUp(currentPos, gridWorld)
            if not state:
                direction = "None"
        case 4:
            direction = "Down"
            currentPos, gridWorld, state = actionDown(currentPos, gridWorld)
            if not state:
                direction = "None"
    print(direction)
    


