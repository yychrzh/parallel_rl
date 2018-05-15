# rollouts to sample and fit with ddpg:

import time
from termcolor import *
import psutil
import numpy as np


# print the time , reward and step info after a episode end
def episode_print(ep_num, total_steps, relative_time, ep_value):
    info = psutil.virtual_memory()
    print(" ")
    print(colored("*" * 150, "blue"))
    print(colored("*", "blue"), colored("ep_num:{:4d}|".format(ep_num), "green"),
          " step:{:3d}| time:{:6.3f}sec| reward:{:7.3f}| total_steps:{:7d}| total_time:{:6.2f}min|"
          " average_steps:{:3d}|".format(ep_value["episode_steps"], ep_value["episode_time"],
                                         ep_value["episode_reward"], total_steps, relative_time,
                                         ep_value["average_steps"]), " memory: ", info.percent)
    print(colored("*", "blue"), "actor_loss: ", ep_value["actor_loss"], "critic_loss: ", ep_value["critic_loss"])
    print(colored("*", "blue"), "fit_time: %7.5f" % ep_value["fit_time"], "step_time: %7.5f" % ep_value["step_time"],
          "actor_time: %7.5f" % ep_value["actor_train_time"], "critic_time: %7.5f" % ep_value["critic_train_time"],
          "sample_time: %7.5f" % ep_value["sample_time"], "target_update_time: % 7.5f" % ep_value["target_update_time"])
    print(colored("*" * 150, "blue"))


# a single thread rollout without farm, the env should be given
class fast_rollouts():
    def __init__(self, agent, env):
        self.env = env
        self.agent = agent
        self.relative_time = 0
        self.ep_num, self.total_steps = 0, 0
        self.history_reward = []
        self.ep_value = {}
        self.average_steps = self.agent.args.max_pathlength
        self.start_time = time.time()

    def fit(self):
        vars = [0]*7
        for i in range(self.average_steps):
            var = self.agent.fit()
            for i, v in enumerate(var):
                vars[i] += v / self.average_steps
        self.ep_value['actor_loss'], self.ep_value['critic_loss'], self.ep_value['fit_time'] = vars[0], vars[1], vars[2]
        self.ep_value['sample_time'], self.ep_value['actor_train_time'] = vars[3], vars[4]
        self.ep_value['critic_train_time'], self.ep_value['target_update_time'] = vars[5], vars[6]

    def rollout(self):
        while self.total_steps < self.agent.args.n_steps and self.ep_num < self.agent.args.n_episodes:
            noise_level = 2 * max(0., ((self.agent.args.n_steps/5 - self.total_steps) / (self.agent.args.n_steps/5)))
            ep_step, ep_reward, ep_time, step_time, ep_memory = self.agent.run_an_episode(noise_level, self.env)
            for t in ep_memory:
                self.agent.feed_one(t)
            self.ep_value['episode_steps'], self.ep_value['episode_reward'] = ep_step, ep_reward
            self.ep_value['episode_time'], self.ep_value['step_time'] = ep_time, step_time
            self.ep_num += 1
            self.total_steps += self.ep_value["episode_steps"]
            self.average_steps = int(self.total_steps / self.ep_num)
            self.ep_value["average_steps"] = self.average_steps
            self.history_reward.append(self.ep_value["episode_reward"])
            self.relative_time = (time.time() - self.start_time) / 60.0
            episode_print(self.ep_num, self.total_steps, self.relative_time, self.ep_value)
            if (self.ep_num + 1) % 100 == 0:
                self.agent.saveModel(0)
        self.agent.saveModel(0)
        return self.history_reward


import threading as th
from remote_env.farmer import farmer as farmer_class
import tensorflow as tf

graph = tf.get_default_graph()


# parallel remote_env, need farm
class parallel_rollouts():
    def __init__(self, agent, env=None):
        self.lock = th.Lock()
        self.agent = agent
        # one and only
        self.para_list = self.get_parameter_list()
        self.farmer = farmer_class(self.para_list)
        self.ep_num = 0
        self.total_steps = 0
        self.history_reward = []
        self.ep_value = {}
        self.value_init()
        self.relative_time = 0
        self.average_steps = self.agent.args.max_pathlength
        self.start_time = time.time()

    def get_parameter_list(self):
        para_list = {"model": self.agent.args.model, "task": self.agent.args.task, 
                     "policy_layer_norm": self.agent.args.policy_layer_norm, 
                     "value_layer_norm": self.agent.args.value_layer_norm, 
                     "policy_act_fn": self.agent.args.policy_act_fn,
                     "value_act_fn": self.agent.args.value_act_fn,
                     "max_pathlength": self.agent.args.max_pathlength,
                     "farmer_port": self.agent.args.farmer_port,
                     "farm_list_base": self.agent.args.farm_list_base,
                     "farmer_debug_print": self.agent.args.farmer_debug_print,
                     "farm_debug_print": self.agent.args.farm_debug_print
        }
        return para_list

    def refarm(self):    # most time no use
        del self.farmer
        self.farmer = farmer_class()

    def value_init(self):
        self.ep_value['actor_loss'], self.ep_value['critic_loss'], self.ep_value['fit_time'] = 0, 0, 0
        self.ep_value['sample_time'], self.ep_value['actor_train_time'] = 0, 0
        self.ep_value['critic_train_time'], self.ep_value['target_update_time'] = 0, 0

    def fit(self):
        vars = [0]*7
        for i in range(self.average_steps):
            var = self.agent.fit()
            for i, v in enumerate(var):
                vars[i] += v / self.average_steps
        self.ep_value['actor_loss'], self.ep_value['critic_loss'], self.ep_value['fit_time'] = vars[0], vars[1], vars[2]
        self.ep_value['sample_time'], self.ep_value['actor_train_time'] = vars[3], vars[4]
        self.ep_value['critic_train_time'], self.ep_value['target_update_time'] = vars[5], vars[6]

    def rollout_an_episode(self, noise_level, env):
        # self.ep_value = self.agent.run_an_episode(noise_level, env)  # this function is thread safe
        global graph
        with graph.as_default():
            ep_step, ep_reward, ep_time, step_time, ep_memory = self.agent.run_an_episode(noise_level, env)
            self.fit()
        self.lock.acquire()
        for t in ep_memory:
            self.agent.feed_one(t)
        self.ep_value['episode_steps'], self.ep_value['episode_reward'] = ep_step, ep_reward
        self.ep_value['episode_time'], self.ep_value['step_time'] = ep_time, step_time
        self.ep_num += 1
        self.total_steps += ep_step  # self.ep_value["episode_steps"]
        self.average_steps = int(self.total_steps / self.ep_num)
        self.ep_value["average_steps"] = self.average_steps
        self.history_reward.append(ep_reward)  # (self.ep_value["episode_reward"])
        self.relative_time = (time.time() - self.start_time) / 60.0
        episode_print(self.ep_num, self.total_steps, self.relative_time, self.ep_value)
        self.lock.release()
        env.rel()

    def rollout_if_available(self, noise_level):
        while True:
            remote_env = self.farmer.acq_env()  # call for a remote_env
            if remote_env is False:  # no free environment
                # time.sleep(0.1)
                pass
            else:
                t = th.Thread(target=self.rollout_an_episode, args=(noise_level, remote_env), daemon=True)
                t.start()
                break

    def rollout(self):
        for i in range(self.agent.args.n_episodes):
            noise_level = 2 * max(0., ((self.agent.args.n_steps/5 - self.total_steps) / (self.agent.args.n_steps/5)))
            nl = noise_level if np.random.uniform() > 0.05 else 0
            self.rollout_if_available(nl)
            if (i + 1) % 100 == 0:
                self.agent.saveModel(0)
            if self.total_steps > self.agent.args.n_steps:
                break

        self.agent.saveModel(0)
        return self.history_reward
