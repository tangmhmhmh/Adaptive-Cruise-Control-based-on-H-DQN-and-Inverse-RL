import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
class plot_fig():
    def __init__(self,data,output,title):
        self.data=data
        self.steps=data[0]
        self.reward=data[1]
        self.rewards=data[2]
        self.loss=data[3]
        self.ego_x=data[4]
        self.ego_v=data[5]
        self.ego_a=data[6]
        self.aim_x=data[7]
        self.aim_v=data[8]
        self.aim_a=data[9]
        self.dis=data[10]
        self.gap_v=data[11]
        self.actions=data[19]
        self.save_fig(output,title=title)
        pass
    def save_fig(self,filename,title):
        plt.figure(figsize=(19.20, 10.80), dpi=100)
        plt.rcParams['savefig.dpi'] = 500
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.subplots_adjust(wspace=0.2, hspace=0.4)  # 调整子图间距
        rewards=plt.subplot(421)
        self.plot_line(rewards,self.rewards,"r",self.rewards["title"],"步数",self.rewards["title"])
        loss=plt.subplot(422)
        self.plot_line(loss, self.loss, "r", self.loss["title"], "步数", self.loss["title"])
        dis=plt.subplot(423)
        self.plot_line(dis, self.dis, "r", self.dis["title"], "步数", self.dis["title"]+" m")
        gap_v=plt.subplot(424)
        #self.plot_line(gap_v, self.gap_v, "g", self.gap_v["title"], "步数", self.gap_v["title"]+" m/s")
        self.plot_scatter(gap_v, self.gap_v,plt.cm.Blues , self.gap_v["title"], "步数", self.gap_v["title"]+" m/s",[abs(self.gap_v["data"][i]) for i in range(len(self.gap_v["data"]))])
        gap_v.plot(self.steps["data"],[0.05 for i in range(len(self.steps["data"]))],"--r")
        gap_v.plot(self.steps["data"],[-0.05 for i in range(len(self.steps["data"]))],"--r")
        velocities=plt.subplot(425)
        self.plot_lines(velocities, [self.ego_v,self.aim_v], ["r","g"], "两车速度", "步数", "速度 m/s")
        actions=plt.subplot(426)
        self.plot_bar(actions,self.actions,kinds='r',title="分布",xlabel="所选动作",ylabel="动作(期望速度差) m/s")
        accelerations=plt.subplot(427)
        #self.plot_lines(accelerations, [self.ego_a,self.aim_a], "r", "两车加速度", "步数", "加速度 m/s2")
        accelerations=self.plot_line_scatter(accelerations,self.ego_v,self.ego_a,self.steps,'r',"加速度-速度关系","速度 m/s","加速度绝对值 m/s2",plt)
        expect_gap_v=plt.subplot(428)
        self.plot_scatter(expect_gap_v, self.actions, plt.cm.Blues, self.actions["title"], "步数", self.actions["title"]+"m/s",self.reward["data"])
        plt.suptitle(title,fontsize=24)
        plt.savefig(filename)
        #plt.show()
        pass
    def plot_line(self,plt,data,kind,title,xlabel,ylabel):
        plt.set_title(title)
        plt.set_xlabel(xlabel)
        plt.set_ylabel(ylabel)
        plt.plot(data["data"], kind,linewidth=1)
        return plt
        pass
    def plot_line_scatter(self,plt,ego_v,ego_a,steps,kind,title,xlabel,ylabel,mother):
        plt.set_title(title)
        plt.set_xlabel(xlabel)
        plt.set_ylabel(ylabel)
        max_v=max(ego_v["data"])
        min_v=min(ego_v["data"])
        if min_v>0 and min_v<=5:
            if max_v>0 and max_v<=5:
                x=[0,max_v]
                y=[4,4]
            elif max_v>5 and max_v<=20:
                x=[0,5,max_v]
                y=[4,4,(4-2/15*(max_v-5))]
            else:
                x=[0,5,20,max_v]
                y=[4,4,2,2]
        elif min_v>5 and min_v<=20:
            if max_v>5 and max_v<=20:
                x=[min_v,max_v]
                y=[(4-2/15*(min_v-5)),(4-2/15*(max_v-5))]
            else:
                x=[min_v,20,max_v]
                y=[(4-2/15*(min_v-5)),2,2]
        else:
            x=[min_v,max_v]
            y=[2,2]
        plt.plot(x,y,kind)
        plt.plot(x,[-y[k] for k in range(len(y))],kind)
        #for i in range(len(ego_a["data"])):
        #    ego_a["data"][i]=abs(ego_a["data"][i])
        #plt.scatter(ego_v["data"],ego_a["data"],s=2,cmap=mother.cm.get_cmap('RdYlBu'),c=steps["data"],edgecolor="none")
        #plt.scatter(ego_v["data"],ego_a["data"],s=2,cmap=mother.cm.Blues,c=steps["data"],edgecolor="none",alpha=0.5)
        plt.scatter(ego_v["data"],ego_a["data"],s=2,edgecolor="none")

        return plt
        pass
    def plot_lines(self,plt,datas,kinds,title,xlabel,ylabel):
        plt.set_title(title)
        plt.set_xlabel(xlabel)
        plt.set_ylabel(ylabel)
        lines=[]
        lengens=[]
        for i in range(len(datas)):
            lines.append(plt.plot(datas[i]["data"],linewidth=1))
            lengens.append(datas[i]["title"])
        plt.legend([i[0] for i in lines], lengens, loc='best')
        return plt
        pass
    def plot_bar(self,plt,data,kinds,title,xlabel,ylabel):
        plt.set_title(title)
        plt.set_xlabel(xlabel)
        plt.set_ylabel(ylabel)
        num = []
        #plt.tick_params(labelsize=6)
        a=np.arange(-1,1.02,0.05)
        for i in range(len(a)):
            a[i] = float("%.2f" % a[i])
            num.append(data["data"].count(a[i]))
            #a[i] = str(a[i])
        print(a)
        print(num)
        plt.bar(a, num, width=0.01, align='center')
        for x, b in zip(a, num):
            plt.text(x, b + 0.1, b, ha='center', va='bottom',fontsize=6)

        return plt
        pass
    def plot_scatter(self,plt,data,kind,title,xlabel,ylabel,steps):
        plt.set_title(title)
        plt.set_xlabel(xlabel)
        plt.set_ylabel(ylabel)
        #plt.scatter(range(len(data["data"])),data["data"],s=2,c=steps,cmap=kind)
        plt.scatter(range(len(data["data"])),data["data"],s=2,edgecolor="none")
        return plt
        pass
