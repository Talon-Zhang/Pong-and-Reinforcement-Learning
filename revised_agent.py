import utils
import numpy as np
import random

class Agent:
    
    def __init__(self, actions, two_sided = False): 
        self.two_sided = two_sided
        self._actions = actions
        self._train = True
        self._x_bins = utils.X_BINS
        self._y_bins = utils.Y_BINS
        self._v_x = utils.V_X
        self._v_y = utils.V_Y
        self._paddle_locations = utils.PADDLE_LOCATIONS
        self._num_actions = utils.NUM_ACTIONS
        # Create the Q Table to work with
        self.Q = utils.create_q_table()

        self.Nsa = {} # a dictionary storing the number of appreance of a (state, action) pair
        self.tot_bounce = 0 # the total number of bounce so far
        self.pre_state = None#(6,6,1,-1,8) # the initial state described in the assignment
        self.pre_action=None#0
        self.pre_reward=None#0

    def act(self, state, bounces, done, won):
        print(state)

        #if the game ends
        if done:
            self.pre_state = None
            self.pre_action = None
            self.pre_reward = None
            self.tot_bounce = 0
            return 0

        def get_Nsa(some_state, some_action):
            if (some_state, some_action) not in self.Nsa:
                return 0
            else:
                return self.Nsa[(some_state, some_action)]
        def increment_Nsa(some_state, some_action):
            if (some_state, some_action) not in self.Nsa:
                self.Nsa[(some_state, some_action)]=1
            else:
                self.Nsa[(some_state, some_action)]+=1


        # divide the space into 12x12 grid
        bins=np.linspace(0,1,12)

        # determine velocity in x and y
        vx=1
        vy=1
        if state[2]<0:
            vx=-1
        if abs(state[3])<0.015:
            vy=0
        elif state[3]<0:
            vy=-1

        # convert paddle location to discrete value
        paddle_height=0.2
        if state[-1]==1 - paddle_height:
            discrete_paddle=11
        else:
            discrete_paddle=int(np.floor(12 * state[-1] / (1 - paddle_height)))

        # a new state would be (xbin, ybin, vx, vy, discrete paddle)
        new_state=(int(np.digitize(state[0],bins,right=True)), int(np.digitize(state[1],bins,right=True)), vx, vy, discrete_paddle)

        # determin the reward r
        r=0
        if bounces>self.tot_bounce:
            self.tot_bounce+=1
            r+=1
        if won:
            r+=1

        # training mode
        if self._train:
            
            if self.pre_state!=None:
                # increment Nsa
                increment_Nsa(self.pre_state, self.pre_action)

                num_acti=[]
                for acti in [0,1,-1]:
                    num_acti.append(get_Nsa(self.pre_state,acti))


                if min(num_acti)<50:
                    # the action with least Nsa
                    action=[0,1,-1][np.argmin(np.array(num_acti))]
                else:
                    # if Nsa for the pre_state and all three actions are >=50, choose the best from Q table
                    max_Q=-999
                    action=-999
                    for acti in [0,1,-1]:
                        Q_value = self.Q[self.pre_state[0],self.pre_state[1],self.pre_state[2],self.pre_state[3],self.pre_state[4],acti]
                        if Q_value>max_Q:
                            max_Q=Q_value
                            action=acti


                alpha=50/(get_Nsa(self.pre_state,action)+50)
                # discount factor beta
                gamma=0.7
                
                increase = alpha*(self.pre_reward + gamma*
                self.Q[new_state[0],new_state[1],new_state[2],new_state[3],new_state[4],action] - 
                self.Q[self.pre_state[0],self.pre_state[1],self.pre_state[2],self.pre_state[3],self.pre_state[4],self.pre_action])

                # update Q(s,a)
                self.Q[self.pre_state[0],self.pre_state[1],self.pre_state[2],self.pre_state[3],self.pre_state[4],self.pre_action]+=increase
                #print(increase)
            # the first run. No pre_state info available
            else:
                action=0

                   

        # test section
        if not self._train:
            # use the Q table to determin best action
            max_Q=-999
            action=-999
            for acti in [0,1,-1]:
                Q_value = self.Q[new_state[0],new_state[1],new_state[2],new_state[3],new_state[4],acti]
                if Q_value>max_Q:
                    max_Q=Q_value
                    action=acti
            
        self.pre_state=new_state
        self.pre_reward=r
        self.pre_action=action

        
        return self.pre_action

    


    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    def save_model(self,model_path):
        # At the end of training save the trained model
        utils.save(model_path,self.Q)

    def load_model(self,model_path):
        # Load the trained model for evaluation
        self.Q = utils.load(model_path)



