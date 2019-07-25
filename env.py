'''
设计指标：
1. 完成环境计算，包括自车和前车位置，速度等
2. 设计奖励函数
3. 参数化的选择前车运动信息
4. 加入结果输出，文件打包以及上传网盘并发邮件提醒功能
'''
import csv
from env_settings import envSettings
import numpy as np
from PID import PID
class env():
    def __init__(self):
        self.s=envSettings()
        self.init_variable()
        self.init_constent()
        pass
        pass
    def init_constent(self):
        self.action_gap = self.s.action_gap
        self.actions = self.s.actions
        self.gap_x = self.aim_x - self.ego_x
        self.gap_v = self.ego_v - self.aim_v
        self.gap_v_old = 0
        self.gap_a = self.ego_a - self.aim_a
        self.gap_a_0=self.ego_a - self.aim_a
        self.TTC = 0
        self.get_TTC()
        self.human_time = self.s.human_time
        self.advise = 0
        self.done = False
        self.save_title = []
        self.save_data = []
        self.steps = 0
        self.reward = 0
        self.rewards = 0
        self.loss = 0
        self.choosed_action = 0
        self.t = self.s.t
        self.pid = PID(1.5, 0.5, self.t)
        self.pid.setSampleTime(0.01)
        self.savedatas=[
            {"序号":self.steps},{"单步奖励":self.reward},{"累积奖励":self.rewards},
                   {'损失值':self.loss},{'自车位置':self.ego_x},{'自车速度':self.ego_v},
                   {'自车加速度':self.ego_a},{'前车位置':self.aim_x},{'前车速度':self.aim_v},
                   {'前车加速度':self.aim_a},{'两车距离':self.gap_x},{'两车速度差':self.gap_v},
                   {'危险系数':0},{'时效性系数':0},{'舒适度系数':0},
                   {'建议加速度':self.advise},{"安全系数奖励":0},{"时效系数奖励":0},{"舒适系数奖励":0},
                    {"所选动作":self.choosed_action},{"TTC":self.TTC}
                   ]
        self.get_title()
        self.state = np.array([self.gap_x, self.gap_v])
        pass
    def init_variable(self,ego_x=0,ego_v=120,ego_a=0,ego_v_set=20,aim_x=150,aim_v=80,aim_a=0):
        self.ego_x = ego_x
        self.ego_v = ego_v/3.6
        self.ego_a = ego_a
        self.ego_v_set = ego_v_set
        self.aim_x = aim_x
        self.aim_v = aim_v/3.6
        self.aim_a = aim_a
        pass
    def calculate(self):
        self.ego_x+=self.ego_v*self.t+0.5*self.ego_a*self.t*self.t
        self.aim_x += self.aim_v * self.t + 0.5 * self.aim_a * self.t * self.t
        self.ego_v += self.ego_a * self.t
        self.aim_v+=self.aim_a*self.t
        self.gap_x = self.aim_x - self.ego_x
        self.gap_v = self.ego_v - self.aim_v
        self.gap_a = self.ego_a - self.aim_a
        self.get_TTC()

        pass
    def calcu(self,action):
        #-----------------------------------------------------------------
        #PID的控制目标为两车相对速度
        #-----------------------------------------------------------------
        self.pid.SetPoint=self.actions[action]
        self.pid.update(self.gap_v)
        self.ego_a=self.pid.output
        #time.sleep(0.05)
        #time.sleep(0.05)
        pass
    def reset(self):
        '''
        :return:当前状态
        '''
        self.init_variable()
        self.init_constent()
        return self.state
        pass
    def step(self,action):
        #self.ego_a=self.actions[action]
        self.choosed_action=self.actions[action]
        self.aim_change()
        self.calcu(action)
        self.calculate()
        self.reward=self.get_reward()
        self.advise=self.get_advise()
        self.done=self.is_done()
        self.update_savedata()
        self.rewards+=self.reward
        return np.array([self.aim_x-self.ego_x,self.ego_v-self.aim_v]),self.reward,self.done,self.advise
        pass
    def update_savedata(self):
        self.savedatas=[
            {"序号":self.steps},{"单步奖励":self.reward},{"累积奖励":self.rewards},
                   {'损失值':self.loss},{'自车位置':self.ego_x},{'自车速度':self.ego_v},
                   {'自车加速度':self.ego_a},{'前车位置':self.aim_x},{'前车速度':self.aim_v},
                   {'前车加速度':self.aim_a},{'两车距离':self.gap_x},{'两车速度差':self.gap_v},
                   {'危险系数':0},{'时效性系数':0},{'舒适度系数':0},
                   {'建议加速度':self.advise},{"安全系数奖励":0},{"时效系数奖励":0},{"舒适系数奖励":0},
                    {"所选动作":self.choosed_action},{"TTC":self.TTC}
                   ]
        self.save_data=[]
        for i in range(len(self.savedatas)):
            try:
                self.save_data.append("%.2f" % float(list(self.savedatas[i].values())[0]))
            except:
                self.save_data.append(list(self.savedatas[i].values())[0])
        pass
    def get_title(self):
        self.save_title=[]
        for i in range(len(self.savedatas)):
            self.save_title.append(list(self.savedatas[i].keys())[0])
        pass
    def is_done(self):
        dmin=abs(self.ego_v*self.ego_v/(2*self.actions[0]))
        if self.TTC < self.human_time or self.gap_x >= dmin:
            done=True
            #print(self.gap_x, dmin)
        else:
            done=False
        done=False
        return done
        pass
    def get_TTC(self):
        if self.gap_v<=0:
            self.TTC=100000
        else:
            self.TTC=self.gap_x/self.gap_v
        pass
    def get_advise(self):
        return 0
        pass
    def get_reward(self):
        """
        :return:
        """
        reward=-abs(150-self.gap_x)

        '''
        if self.gap_x == 0:
            reward = 5
        elif abs(self.gap_x) > 0 and abs(self.gap_x) <= 0.05:
            reward = 5 * (0.05 - abs(self.gap_x)) / 0.05
        else:
            reward = -abs(self.gap_x)
            
            
        
        if self.gap_v==0:
            reward=5
        elif abs(self.gap_v)>0 and abs(self.gap_v)<=0.05:
            reward=5*(0.05-abs(self.gap_v))/0.05
        else:
            reward=-abs(self.gap_v)
        
        gap_a_change=(self.gap_a-self.gap_a_0)/self.t
        reward-=5*(abs(gap_a_change)-0.05)/0.05
        self.gap_a_0=self.gap_a
       
        if self.ego_v>0 and self.ego_v<=5:
            reward-=5*((abs(self.ego_a)-4)/4)
        elif self.ego_v>5 and self.ego_v<=20:
            max_a=5-(3/15)*(self.ego_v-5)
            reward-=5*((abs(self.ego_a)-max_a))/max_a
        else:
            reward-=5*((abs(self.ego_a)-2)/2)
        '''
        return reward
        pass
    def aim_change(self):
        if self.steps>0 and self.steps<=10000:
            temp = self.steps % 5000
            if temp >= 0 and temp < 1000:
                self.aim_a = 0
            elif temp >= 1000 and temp < 2000:
                self.aim_a = 2
            elif temp >= 2000 and temp < 3000:
                self.aim_a = 0
            elif temp >= 3000 and temp < 4000:
                self.aim_a = -2
            else:
                self.aim_a = 0
        else:
            if self.steps>10000 and self.steps<=20000:
                if self.steps>10000 and self.steps<=12000:
                    self.aim_a=0
                elif self.steps>12000 and self.steps <=14000:
                    self.aim_a=-0.22
                elif self.steps>14000 and self.steps<=16000:
                    self.aim_a=0
                elif self.steps >16000 and self.steps<=18000:
                    self.aim_a=-0.85
                else:
                    self.aim_a=0
            else:
                self.aim_a=0
        '''
        temp=self.steps%1000
        if temp>=0 and temp<200:
            self.aim_a=0
        elif temp>=200 and temp<400:
            self.aim_a=0.5
        elif temp>=400 and temp<600:
            self.aim_a=0
        elif temp>=600 and temp<800:
            self.aim_a=-0.5
        else:
            self.aim_a=0
        '''
        pass
if __name__=="__main__":
    tt=env()
    print(tt.actions)