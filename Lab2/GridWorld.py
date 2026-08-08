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
    


