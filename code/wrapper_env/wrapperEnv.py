import gym
import numpy as np


GAME = 'Pendulum-v0'  # 'Pendulum-v0'


class WrapperEnv():
    def __init__(self, game='Pendulum-v0', visualize=False):
        self.env = gym.make(game).unwrapped
        self.observation_space_shape = self.env.observation_space.shape
        self.action_space = self.env.action_space
        self.difficulty = 0
        self.visualize = visualize

    def step(self, action):
        action = self.process_action(action)
        o, sr, d, i = self.env.step(action)
        oo = np.array(o)
        res = [oo, (sr+8)/8, d, i]
        # res = [oo, sr, d, i]
        return res

    def process_action(self, action):
        # the output of the actor is tanh
        action = [float(action[i]) for i in range(len(action))]
        for i in range(len(action)):
            action_gain = self.env.action_space.high[i] - self.env.action_space.low[i]
            action[i] = (action[i] + 1) / 2 * action_gain + self.env.action_space.low[i]
        action = np.array(action)
        return action

    def reset(self, difficulty=0):
        self.difficulty = difficulty
        o = self.env.reset()
        return o

    def render(self):
        self.env.render()

    def seed(self, s):
        self.env.seed(s)
        pass