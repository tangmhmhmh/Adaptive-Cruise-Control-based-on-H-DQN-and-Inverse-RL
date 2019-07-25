from DQN_For_Test import DeepQNetwork
from env import env
from add_save_data import SaveData
from env_aim_stop import env_aim_stop
from env_aim_lowspeed import env_aim_lowspeed
from env_aim_decelerate import env_aim_decelerate
import time
from path_tools import *
import time
localtime = time.localtime(time.time())
mytyme=str(localtime.tm_year)+"-"+str(localtime.tm_mon)+"-"+str(localtime.tm_mday)+"---"+\
       str(localtime.tm_hour)+"-"+str(localtime.tm_min)+"-"+str(localtime.tm_sec)+"/"

ENVS=[env(),env_aim_stop(120,150),env_aim_lowspeed(ego_speed=120,aim_speed=30,aim_d=150),
      env_aim_decelerate(ego_speed=120,aim_speed=70,decelerateion=-3,aim_x=150)]
'''
for envs in range(len(ENVS)):
    print(envs)
    Env=ENVS[envs]
'''
max_steps=20000
Env=env()
root="./Result/"
child0='2019-6-15---15-15-23/'
child1=['Datas/',"Src/"]
child2=["DQN/",'Double-DQN/','Dueling-DQN/',"Double-Dueling-DQN/"]
child3=["data/","map/","memory/"]
kind=0
envs=1
support=Support(root=root,child0=child0,child1=child1,child2=child2,child3=child3)
DQNS_ALL=[{"DQN":DeepQNetwork(double_q=False,dueling_q=False,env=str(envs),memory_name=root+child0+child1[0]+child2[0]+child3[1]+"56th"),"kind":0},
          {"DQN":DeepQNetwork(double_q=True,dueling_q=False,env=str(envs),memory_name=root+child0+child1[0]+child2[1]+child3[1]+"2th"),"kind":1},
          {"DQN":DeepQNetwork(double_q=False,dueling_q=True,env=str(envs),memory_name=root+child0+child1[0]+child2[2]+child3[1]+"2th"),"kind":2},
          {"DQN":DeepQNetwork(double_q=True,dueling_q=True,env=str(envs),memory_name=root+child0+child1[0]+child2[3]+child3[1]+"2th"),"kind":3},
          ]
for DQNS in DQNS_ALL:
    print(kind)
    kind=DQNS["kind"]
    DQN=DQNS["DQN"]
    steps = 0
    for i in range(100):
        start=time.time()
        support.create_csv(Env.save_title,kind=kind,i=i+1)
        s = Env.reset()
        while not Env.done:
            action=DQN.choose_actions(s)
            s_,r,done,advise=Env.step(action)
            steps+=1
            Env.steps+=1
            if Env.steps!=0:
                support.save_data2csv(Env.save_data,kind=kind,i=i+1)
            if Env.steps>=max_steps:
                break
        support.save_data2csv_end(kind=kind,i=i+1)
        support.save_fig(kind=kind, i=i + 1)
        DQN.store_results(support.root+support.child1[0]+support.child2[kind]+support.child3[1]+str(i+1)+"th/"+"models",i+1)
        DQN.store_graph(support.root+support.child1[0]+support.child2[kind]+support.child3[2]+str(i+1)+"th/")
        print(str(kind+1)+"--"+str(i+1)+"th 结束,用时： "+str(time.time()-start)+"累计损失： "+
              str(np.array(support.data[3]["data"]).sum())+"  累计奖励： "+str(support.data[2]["data"][-1]))
support.saveandupload()