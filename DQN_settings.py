from env import env
class Settings():
    def __init__(self):
        self.env=env()
        self.n_actions=len(self.env.actions)
        self.n_feature=len(self.env.state)
        self.learning_rate=0.01
        self.gamma=0.8
        self.e_greedy=0
        self.e_greedy_increment=0.0001
        self.memory_size=100000
        self.batch_size=10000
        self.replace_target_iter=500
        self.output_graph=True
        self.double_q=True
        self.dueling=True
        pass
