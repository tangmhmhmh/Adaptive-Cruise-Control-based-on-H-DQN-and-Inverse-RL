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
                 dueling_q=s.dueling,env=""
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
        t_param=tf.get_collection("dqn_t_param")
        e_param=tf.get_collection("dqn_e_param")
        self.replace_target_op=[tf.assign(t,e) for t,e in  zip(t_param,e_param)]
        self.sess=tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver(max_to_keep=10)

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
    def store_transition(self,s,a,r,s_):
        self.memory[self.memory_counter%self.memory_size,:]=np.hstack((s,[a,r],s_))
        self.memory_counter+=1
        pass
    def learn(self):
        if self.learn_step_counter%self.replace_target_iter==0:
            self.sess.run(self.replace_target_op)
            print(str(self.replace_time)+"th target-evaluation  替换成功")
            self.replace_time+=1
        if self.memory_counter>self.memory_size:
            sample_index=np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index=np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory=self.memory[sample_index,:]


        q_next,q_eval4next=self.sess.run([self.q_next,self.q_eval],
                                    feed_dict=
                                    {
                                        self.s_: batch_memory[:, -self.n_feature:],  # fixed params
                                        self.s: batch_memory[:, :self.n_feature],  # newest params
                                     }
                                    )
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_feature]})

        q_target=q_eval.copy()

        batch_index=np.arange(self.batch_size,dtype=np.int32)
        eval_act_index=batch_memory[:,self.n_feature].astype(int)
        reward=batch_memory[:,self.n_feature+1]
        if self.double_q:
            max_act4next=np.argmax(q_eval4next,axis=1)
            seected_q_next=q_next[batch_index,max_act4next]
        else:
            seected_q_next=np.argmax(q_next,axis=1)

        q_target[batch_index,eval_act_index]=reward+self.gamma*seected_q_next
        _,self.cost=self.sess.run([self._train_op,self.loss],
                                  feed_dict={self.s:batch_memory[:,:self.n_feature],
                                             self.q_target:q_target
                                             }
                                  )
        self.cost_his.append(self.cost)
        if self.e_greedy<0.9:
            self.e_greedy+=self.e_greedy_increment
        self.learn_step_counter+=1
        pass
    def store_results(self,fileame,i):
        self.saver.save(self.sess,fileame, global_step=i)
        pass
    def store_graph(self,filename):
        tf.summary.FileWriter(filename,self.sess.graph)
        pass
if __name__=="__main__":
    pp=DeepQNetwork()