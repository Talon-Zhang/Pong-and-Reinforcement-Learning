import utils
import numpy as np
import random
import math


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
        self.pre_bounce = 0 # the total number of bounce so far
        self.pre_state = None # (6,6,1,-1,8) # the initial state described in the assignment
        self.pre_action = None # 0

    def reward(self, bounces, done, won):
        if won:
            return 40
        elif self.pre_bounce < bounces:
            return math.floor(bounces / 5) + 1
        elif done:
            if bounces < 5:
                return -18
            elif bounces < 10:
                return -14
            elif bounces < 15:
                return -10
            else:
                return -18
        else:
            return 0

    def split(self, value, index):
        if index == 0 or index == 1:
            if value >= 1:
                return 11
            return math.floor(value * 11.0)
        elif index == 2:
            if value >= 0.3:
                return 1
            else:
                return -1
        elif index == 3:
            if value > 0.015:
                return 1
            elif value < -0.015:
                return -1
            else:
                return 0
        elif index == 4:
            if value >= 0.8:
                return 11
            return math.floor(12 * (value / 0.8))

    def act(self, state, bounces, done, won):
        current = self.Q[self.split(state[0], 0)][self.split(state[1], 1)][self.split(state[2], 2)] \
                    [self.split(state[3], 3)][self.split(state[4], 4)]
        if random.randint(0, 99) < 1:
            ret = random.choice([-1, 0, 1])
        else:
            action_max = [i-1 for i, x in enumerate(current) if x == max(current)]
            if len(action_max) == 1:
                ret = action_max[0]
            else:
                ret = random.choice(action_max)
        if self.pre_state is not None and self._train:
            reward = self.reward(bounces, done, won)
            pre_state = self.Q[self.split(self.pre_state[0], 0)][self.split(self.pre_state[1], 1)]\
                        [self.split(self.pre_state[2], 2)][self.split(self.pre_state[3], 3)]\
                        [self.split(self.pre_state[4], 4)]
            pre_state[self.pre_action+1] = pre_state[self.pre_action+1] + \
                                           0.1*(reward-pre_state[self.pre_action+1] + 0.9*current[ret+1])
        self.pre_state = state
        self.pre_bounce = bounces
        self.pre_action = ret
        return ret

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



