import random
import math

class Arm:


    def __init__(self):
        self.true_mean = random.gauss(0, math.sqrt(3))
 
    def pull(self):
        return random.gauss(self.true_mean, 1.0)
