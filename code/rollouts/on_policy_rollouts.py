import time
from termcolor import *
import psutil
from model.logger import Logger


# print the time , reward and step info after a episode end
def episode_print(iteration, total_steps, relative_time, ep_value):
    info = psutil.virtual_memory()
    print(" ")
    print(colored("*" * 150, "blue"))
    print(colored("*", "blue"), colored("iteration:{:4d}|".format(iteration), "green"),
          " ep_num:{:3d}| step:{:3d}| time:{:6.3f}sec| reward:{:7.3f}| total_steps:{:7d}| total_time:{:6.2f}min|"
          " average_steps:{:3d}|".format(ep_value["episode_nums"], ep_value["episode_steps"], ep_value["episode_time"],
                                         ep_value["episode_reward"], total_steps, relative_time,
                                         ep_value["average_steps"]), " memory: ", info.percent)
    print(colored("*", "blue"), "surro_loss: ", ep_value["surrogate_loss"], "value_loss: ", ep_value["value_loss"],
                                "learning_rate: ", ep_value["learning_rate"])
    print(colored("*", "blue"), "iter_time: %7.5f" % ep_value["iter_time"], "step_time: %7.5f" % ep_value["step_time"],
          "train_time: %7.5f" % ep_value["train_time"], "update_old_time: %7.5f" % ep_value["update_time"],
          "kl_after: %8.6f" % ep_value["kl_after"], "entropy_after: % 8.6f" % ep_value["entropy_after"])
    print(colored("*" * 150, "blue"))


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
        self.average_len_of_episode = self.agent.args.max_pathlength
        self.num_rollouts = int(self.agent.args.timesteps_per_batch / self.average_len_of_episode)
        self.rollout_count = 0
        self.rollout_paths = []
        self.iteration = 0
        self.log_scalar_name_list = ['reward', 'kl_div', 'entropy', 'surrogate_loss', 'value_loss']
        self.log_scalar_type_list = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        self.logger = Logger(self.agent.session, self.agent.args.log_path + 'train',
                             self.log_scalar_name_list, self.log_scalar_type_list)
        self.write_log = self.logger.create_scalar_log_method()
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
        # self.ep_value['actor_loss'], self.ep_value['critic_loss'], self.ep_value['fit_time'] = 0, 0, 0
        # self.ep_value['sample_time'], self.ep_value['actor_train_time'] = 0, 0
        # self.ep_value['critic_train_time'], self.ep_value['target_update_time'] = 0, 0
        pass

    def fit(self):
        # calculate real leaning rate with a linear schedule
        final_learning_rate = 5e-5
        policy_delta_rate = (self.agent.args.policy_learning_rate - final_learning_rate) / self.agent.args.n_steps
        value_delta_rate = (self.agent.args.value_learning_rate - final_learning_rate) / self.agent.args.n_steps
        policy_learning_rate = self.agent.args.policy_learning_rate - policy_delta_rate * self.total_steps
        value_learning_rate = self.agent.args.policy_learning_rate - value_delta_rate * self.total_steps

        self.ep_value["learning_rate"] = policy_learning_rate

        # vars: update_old_policy_time, train_time, fit_time, surrogate_loss, kl_after, entropy_after, value_loss
        vars = self.agent.fit(self.rollout_paths, [policy_learning_rate, value_learning_rate])

        # update print info:
        self.ep_value["episode_nums"] = self.num_rollouts
        self.ep_value["episode_steps"] = sum([len(path["rewards"]) for path in self.rollout_paths])
        self.ep_value["episode_time"] = sum([path["ep_time"] for path in self.rollout_paths]) / self.num_rollouts
        self.ep_value["episode_reward"] = sum([path["rewards"].sum() for
                                               path in self.rollout_paths]) / self.num_rollouts
        self.ep_value["average_steps"] = int(self.average_len_of_episode)
        self.ep_value["step_time"] = sum([path["step_time"] for path in self.rollout_paths]) / self.num_rollouts
        self.ep_value['update_time'], self.ep_value['train_time'], self.ep_value['iter_time'] = vars[0], vars[1], vars[2]
        self.ep_value['surrogate_loss'], self.ep_value['kl_after'] = vars[3], vars[4]
        self.ep_value['entropy_after'], self.ep_value['value_loss'] = vars[5], vars[6]
        self.relative_time = (time.time() - self.start_time) / 60

        # write_log with tensorboard: ['reward', 'kl_div', 'entropy', 'surrogate_loss', 'value_loss']
        self.write_log([self.ep_value['episode_reward'], self.ep_value['kl_after'], self.ep_value['entropy_after'],
                        self.ep_value['surrogate_loss'], self.ep_value['value_loss']], self.iteration)

        # print iteration information:
        episode_print(self.iteration, self.total_steps, self.relative_time, self.ep_value)
        self.history_reward.append(self.ep_value["episode_reward"])

    def rollout_an_episode(self, env):
        global graph
        with graph.as_default():
            path = self.agent.run_an_episode(env)
        self.lock.acquire()
        self.ep_num += 1
        self.rollout_count += 1
        self.rollout_paths.append(path)
        self.total_steps += len(path["actions"])
        self.lock.release()
        env.rel()

    def rollout_if_available(self):
        while True:
            remote_env = self.farmer.acq_env()  # call for a remote_env
            if remote_env is False:  # no free environment
                # time.sleep(0.1)
                pass
            else:
                t = th.Thread(target=self.rollout_an_episode, args=(remote_env, ), daemon=True)
                t.start()
                break

    def rollout(self):
        while self.total_steps < self.agent.args.n_steps:
            if self.num_rollouts == 0:
                raise RuntimeError('wrong, div 0!!!')
            self.iteration += 1
            for i in range(self.num_rollouts):
                self.rollout_if_available()

            while self.rollout_count != self.num_rollouts:
                pass

            if (self.iteration + 1) % 10 == 0:
                self.agent.saveModel(0)

            self.average_len_of_episode = sum([len(path["rewards"]) for path in self.rollout_paths]) / self.num_rollouts
            if self.average_len_of_episode == 0:
                raise RuntimeError('wrong, div 0!!!')
            self.fit()
            self.rollout_count = 0
            self.rollout_paths = []
            self.num_rollouts = int(self.agent.args.timesteps_per_batch / self.average_len_of_episode)

        self.agent.saveModel(0)
        return self.history_reward