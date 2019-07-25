import numpy as np
from PID import PID
from env import env
class env_aim_decelerate(env):
    def __init__(self,ego_speed,aim_speed,decelerateion,aim_x):
        super().__init__()
        self.aim_deceleration=decelerateion
        self.init_variable(ego_v=ego_speed/3.6,aim_v=aim_speed/3.6,aim_x=aim_x)
        self.init_constent()
        pass
    def aim_change(self):
        switch=True
        if switch:
            if self.steady_judge():
                switch=False
        if switch:
            pass
        else:
            self.aim_a=self.aim_deceleration
        if self.aim_v <= 0:
            self.aim_a = 0
            self.aim_v = 0
        pass
    def steady_judge(self):
        state=False
        if abs(self.gap_v)<=0.05 and abs(self.gap_a)<=0.05:
            state=True
            print("开始减速")
        return state
        pass
    def is_done(self):
        if abs(self.gap_v)<=0.001 and abs(self.gap_a)<=0.001:
            print(self.gap_v,self.gap_a)
            return True
        pass