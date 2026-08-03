class Arm:

    def __init__(self):
        import math
        import random
        randomVal = random.uniform(-6,6)
        self.mean = (1/math.sqrt(2* math.pi * 3)*math.exp(-1*((randomVal - 0)**2)/(2*3)))
 
    def pull(self):
        import math
        import random
        randomVal = random.uniform(self.mean - 4,self.mean + 4)
        gaussReward =(1/math.sqrt(2* math.pi * 1)*math.exp(-1*((randomVal - self.mean)**2)/(2*1)))
        return round(gaussReward,4)
