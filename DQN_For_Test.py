import tensorflow as tf
import numpy as np
from DQN_settings import *
s=Settings()
class DeepQNetwork():
    def __init__(self,n_actions=s.n_actions,n_feature=s.n_feature,
                 learning_rate=s.learning_rate,gamma=s.gamma,
                 e_greedy=s.e_greedy,e_greedy_increment=s.e_greedy_increment,
                 memory_size=s.memory_size,batch_size=s.batch_size,
                 replace_target_iter=s.replace_target_iter,
                 output_graph=s.output_graph,double_q=s.double_q,
                 dueling_q=s.dueling,env="",memory_name=""
                 ):
        self.n_actions=n_actions
        self.n_feature=n_feature
        self.lr=learning_rate
        self.gamma=gamma
        self.e_greedy=e_greedy
        self.e_greedy_increment=e_greedy_increment
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.replace_target_iter=replace_target_iter
        self.output_graph=output_graph
        self.double_q=double_q
        self.dueling_q=dueling_q
        self.memory=np.zeros((self.memory_size,self.n_feature*2+2))
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.cost_his=[]
        self.cost=0
        self.replace_time=1
        self.base=env+"variable"
        if self.double_q:
            self.base+="_double_q"
        if self.dueling_q:
            self.base+="_dueling_q"
        self._build_net()
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)
        self.restore_results(memory_name)
        pass
    def _build_net(self):
        layers=[100,70,50,70,100]
        self.s=tf.placeholder(tf.float32,[None,self.n_feature],name="s")
        self.q_target=tf.placeholder(tf.float32,[None,self.n_actions],name="q_target")
        w_initializer = tf.random_normal_initializer(0.0, 0.3)
        b_initializer = tf.constant_initializer(0.1)
        with tf.variable_scope(self.base+"dqn_enva_net"):
            c_name=["dqn_e_param",tf.GraphKeys.GLOBAL_VARIABLES]
            l1=self.build_nets(self.s,self.n_feature,100,self.base+"dqn_l1",c_name,w_initializer,b_initializer,tf.nn.relu)
            l2=self.build_nets(l1,100,70,self.base+"dqn_l2",c_name,w_initializer,b_initializer,tf.nn.relu)
            l3=self.build_nets(l2,70,50,self.base+"dqn_l3",c_name,w_initializer,b_initializer,tf.nn.relu)
            l4=self.build_nets(l3,50,70,self.base+"dqn_l4",c_name,w_initializer,b_initializer,tf.nn.relu)
            l5=self.build_nets(l4,70,100,self.base+"dqn_l5",c_name,w_initializer,b_initializer,tf.nn.relu)
            if self.dueling_q:
                with tf.variable_scope(self.base+'dueling_Value'):
                    w3 = tf.get_variable(self.base+'dueling_w3', [100, 1], initializer=w_initializer, collections=c_name)
                    b3 = tf.get_variable(self.base+'dueling_b3', [1, 1], initializer=b_initializer, collections=c_name)
                    self.V = tf.matmul(l5, w3) + b3

                with tf.variable_scope(self.base+'dueling_Advantage'):
                    w3 = tf.get_variable(self.base+'dueling_w3', [100, self.n_actions], initializer=w_initializer, collections=c_name)
                    b3 = tf.get_variable(self.base+'dueling_b3', [1, self.n_actions], initializer=b_initializer, collections=c_name)
                    self.A = tf.matmul(l5, w3) + b3

                with tf.variable_scope(self.base+'dueling_Q'):
                    self.q_eval = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                self.q_eval=self.build_nets(l5,100,self.n_actions,self.base+"dqn_l6",c_name,w_initializer,b_initializer,None)
        with tf.variable_scope(self.base+"dqn_loss"):
            self.loss=tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        with tf.variable_scope(self.base+"dqn_train"):
            self._train_op=tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
        self.s_ = tf.placeholder(tf.float32, [None, self.n_feature], name="s_")
        with tf.variable_scope(self.base+"dqn_target_net"):
            c_name=["dqn_t_param",tf.GraphKeys.GLOBAL_VARIABLES]
            l1=self.build_nets(self.s,self.n_feature,100,"dqn_l1",c_name,w_initializer,b_initializer,tf.nn.relu)
            l2=self.build_nets(l1,100,70,self.base+"dqn_l2",c_name,w_initializer,b_initializer,tf.nn.relu)
            l3=self.build_nets(l2,70,50,self.base+"dqn_l3",c_name,w_initializer,b_initializer,tf.nn.relu)
            l4=self.build_nets(l3,50,70,self.base+"dqn_l4",c_name,w_initializer,b_initializer,tf.nn.relu)
            l5=self.build_nets(l4,70,100,self.base+"dqn_l5",c_name,w_initializer,b_initializer,tf.nn.relu)
            if self.dueling_q:
                with tf.variable_scope(self.base+'dueling_Value'):
                    w3 = tf.get_variable(self.base+'dueling_w3', [100, 1], initializer=w_initializer, collections=c_name)
                    b3 = tf.get_variable(self.base+'dueling_b3', [1, 1], initializer=b_initializer, collections=c_name)
                    self.V = tf.matmul(l5, w3) + b3

                with tf.variable_scope(self.base+'dueling_Advantage'):
                    w3 = tf.get_variable(self.base+'dueling_w3', [100, self.n_actions], initializer=w_initializer, collections=c_name)
                    b3 = tf.get_variable(self.base+'dueling_b3', [1, self.n_actions], initializer=b_initializer, collections=c_name)
                    self.A = tf.matmul(l5, w3) + b3

                with tf.variable_scope(self.base+'dueling_Q'):
                    self.q_next = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                self.q_next=self.build_nets(l5,100,self.n_actions,self.base+"dqn_l6",c_name,w_initializer,b_initializer,None)
        pass
    def build_nets(self,input,input_num,output_num,name,c_name,w_initializer,b_initializer,activation):
        with tf.variable_scope(name):
            w = tf.get_variable(self.base+"dqn_w", [input_num,output_num], initializer=w_initializer, collections=c_name)
            tf.summary.histogram(self.base+"dqn_w",w)
            b = tf.get_variable(self.base+"dqn_b", [1, output_num], initializer=b_initializer, collections=c_name)
            if activation is None:
                output = tf.matmul(input, w) + b
            else:
                output=activation(tf.matmul(input, w) + b)
        return output
        pass
    def choose_actions(self,observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.e_greedy:
            action_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(action_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
        pass
    def restore_results(self,filename):
        self.saver.restore(self.sess,filename)
        pass
if __name__=="__main__":
    pp=DeepQNetwork()