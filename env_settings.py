import numpy as np
class envSettings():
    def __init__(self):
        self.actions = np.arange(-1, 1.02, 0.05)
        for i in range(len(self.actions)):
            self.actions[i] = float("%.2f" % self.actions[i])
        self.action_gap = 0.05
        self.human_time = 1.1
        self.t = 0.01
        pass