import numpy as np
from PID import PID
from env import env
class env_aim_lowspeed(env):
    def __init__(self,ego_speed,aim_speed,aim_d):
        super().__init__()
        self.init_variable(ego_v=ego_speed,aim_v=aim_speed,aim_x=aim_d)
        self.init_constent()
        pass
    def aim_change(self):
        self.aim_a=0
        pass
    def is_done(self):
        if abs(self.gap_v)<=0.05 and abs(self.gap_a)<=0.05:
            return True
        pass
