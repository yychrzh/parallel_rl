#encoding=utf-8
from osim.env import RunEnv
import numpy as np
from wrapper_env.observation_processor import generate_observation as go


class WrapperEnv():
    def __init__(self, game='l2r', visualize=False, max_obstacles=10, skip_count=1):
        self.env = RunEnv(visualize=visualize, max_obstacles=max_obstacles)
        self.step_count = 0
        self.old_observation = None
        self.skip_count = 1  # skip_count  # 4
        self.last_x = 0
        self.current_x = 0
        self.observation_space_shape = (76,)
        self.action_space = self.env.action_space
        self.difficulty = 2

    def obg(self, plain_obs):
        # observation generator
        # derivatives of observations extracted here.
        processed_observation, self.old_observation = go(plain_obs, self.old_observation, step=self.step_count)
        return np.array(processed_observation)

    def process_action(self, action):
        processed_action = [(v+1.0)/2 for v in action]
        return processed_action

    def step(self, action):
        action = [float(action[i]) for i in range(len(action))]
        action = self.process_action(action)

        import math
        for num in action:
            if math.isnan(num):
                print('NaN met', action)
                raise RuntimeError('this is bullshit')

        sr = 0
        sp = 0
        o, oo = [], []
        d, i = 0, 0
        self.last_x = self.current_x
        for j in range(self.skip_count):
            self.step_count += 1
            oo, r, d, i = self.env.step(action)
            self.current_x = oo[1]
            headx = oo[22]
            px = oo[1]
            py = oo[2]
            kneer = oo[7]
            kneel = oo[10]
            lean = min(0.3, max(0, px - headx - 0.15)) * 0.05
            joint = sum([max(0, k - 0.1) for k in [kneer, kneel]]) * 0.03   # * 0.03
            penalty = lean + joint

            o = self.obg(oo)
            sr += r
            sp += penalty

            if d is True:
               break
        res = [o, sr, d, sp]
        # res = [o, sr, d, i]
        return res

    def reset(self, difficulty=2):
        self.difficulty = difficulty
        self.step_count = 0
        self.old_observation = None
        oo = self.env.reset(difficulty=difficulty)
        self.last_x = oo[1]
        self.current_x = oo[1]
        o = self.obg(oo)
        return o

    def seed(self, s):
        self.env.seed(s)