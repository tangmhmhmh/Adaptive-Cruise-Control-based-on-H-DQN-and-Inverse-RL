import numpy as np
from PID import PID
from env import env
from env_settings import envSettings
class env_aim_stop(env):

    def __init__(self,ego_speed,aim_d):

        super().__init__()
        self.s=envSettings()
        self.init_variable(ego_v=ego_speed, aim_x=aim_d)
        self.init_constent()
        pass
    def aim_change(self):
        self.aim_v=0
        self.aim_v=0
        pass
    def is_done(self):
        if  self.ego_v<=0 and abs(self.gap_a)<=0.05:
            return True
        pass