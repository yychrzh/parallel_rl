import time
from termcolor import *
import psutil
import threading as th
from model.logger import Logger
from model.utils import *
from remote_env.farmer import farmer as farmer_class


graph = tf.get_default_graph()


# print the time , reward and step info after a episode end
def on_policy_iteration_print(iteration, total_steps, relative_time, ep_value):
    info = psutil.virtual_memory()
    print(" ")
    print(colored("*" * 150, "blue"))
    print(colored("*", "blue"), colored("iteration:{:4d}|".format(iteration), "green"),
          " ep_nums:{:3d}| step:{:3d}| time:{:6.3f}sec| reward:{:7.3f}| total_steps:{:7d}| total_time:{:6.2f}min|"
          " average_steps:{:3d}|".format(ep_value["episode_nums"], ep_value["episode_steps"], ep_value["episode_time"],
                                         ep_value["episode_reward"], total_steps, relative_time,
                                         ep_value["average_steps"]), " memory: ", info.percent)
    print(colored("*", "blue"), "surro_loss: ", ep_value["surrogate_loss"], "value_loss: ", ep_value["value_loss"],
                                "learning_rate: ", ep_value["learning_rate"])
    print(colored("*", "blue"), "iter_time: %7.5f" % ep_value["iter_time"], "step_time: %7.5f" % ep_value["step_time"],
          "train_time: %7.5f" % ep_value["train_time"], "update_old_time: %7.5f" % ep_value["update_time"],
          "kl_after: %8.6f" % ep_value["kl_after"], "entropy_after: % 8.6f" % ep_value["entropy_after"])
    print(colored("*" * 150, "blue"))


# print the time , reward and step info after a episode end
def off_policy_episode_print(ep_num, total_steps, relative_time, ep_value):
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


# linear decay learning rate
def learning_rate_schedule(para_list, current_steps):
    # calculate real leaning rate with a linear schedule
    final_learning_rate = 5e-5
    policy_delta_rate = (para_list["policy_learning_rate"] - final_learning_rate) / para_list["n_steps"]
    value_delta_rate = (para_list["value_learning_rate"] - final_learning_rate) / para_list["n_steps"]
    policy_learning_rate = para_list["policy_learning_rate"] - policy_delta_rate * current_steps
    value_learning_rate = para_list["policy_learning_rate"] - value_delta_rate * current_steps
    return [policy_learning_rate, value_learning_rate]


# linear decay noise_level
def noise_level_schedule(para_list, current_steps):
    noise_level = 2 * max(0., ((para_list["n_steps"] / 5 - current_steps) / (para_list["n_steps"] / 5)))
    nl = noise_level if np.random.uniform() > 0.05 else 0
    return nl


# rollout with giving agent in giving env, the function itself should be thread safe
def rollout_an_episode(agent, env, th_lock, noise_level=0):
    if agent.para_list["collect_every_step"] is False:
        # send current actor parameters to the remote env
        th_lock.acquire()
        policy_params_flat = floatify(agent.get_from_flat())
        if agent.para_list["model"] == 'ddpg':
            policy_params_flat = parameter_space_noise(noise_level, policy_params_flat)
        # else:
        #    old_policy_params_flat = floatify(agent.get_from_flat_old())
        #    policy_params_flat = parameter_space_noise(noise_level, policy_params_flat, old_policy_params_flat)
        th_lock.release()

        for num, i in enumerate(policy_params_flat):
            if math.isnan(num):
                print('NaN met in %d th element of parameters' % i)
                raise RuntimeError('this is bullshit')
        env.set(policy_params_flat)
        [obs, rewards, dones, actions, ep_time, step_time] = env.reset_and_rollout()
    else:
        steps, steps_time, ep_time = 0, 0, 0
        start_time = time.time()
        obs, actions, rewards, dones = [], [], [], []

        o = env.reset()
        while True:
            o_before_a = o
            before_step_time = time.time()
            a = agent.act(o_before_a)
            [o, r, d, _] = env.step(a)
            obs.append(o_before_a)
            actions.append(a)
            rewards.append(r)
            dones.append(d)
            step_time = time.time() - before_step_time
            steps_time += step_time
            steps += 1
            if d or steps > (agent.para_list["max_pathlength"] - 1):
                ep_time = time.time() - start_time
                step_time = steps_time / steps
                obs.append(o)
                break

    path = {"obs": np.array(obs),
            "rewards": np.array(rewards),
            "dones": np.array(dones),
            "actions": np.array(actions),
            "ep_time": ep_time,
            "step_time": step_time
            }
    return path


# parallel remote_env, need farm
class on_policy_parallel_rollouts():
    def __init__(self, agent, env=None):
        self.lock = th.Lock()
        self.agent = agent
        # one and only
        self.farmer = farmer_class(self.agent.para_list)
        self.ep_num = 0
        self.total_steps = 0
        self.history_reward = []
        self.ep_value = {}
        self.relative_time = 0
        self.average_len_of_episode = self.agent.para_list["max_pathlength"]
        self.num_rollouts = int(self.agent.para_list["timesteps_per_batch"] / self.average_len_of_episode)
        self.rollout_count = 0
        self.rollout_paths = []
        self.iteration = 0
        self.log_scalar_name_list = ['mean_reward', 'kl_div', 'entropy', 'surrogate_loss', 'value_loss']
        self.log_scalar_type_list = [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
        self.logger = Logger(self.agent.session, self.agent.para_list["log_path"] + 'train',
                             self.log_scalar_name_list, self.log_scalar_type_list)
        self.write_log = self.logger.create_scalar_log_method()
        self.start_time = time.time()

    def refarm(self):    # most time no use
        del self.farmer
        self.farmer = farmer_class()

    def fit(self):
        # calculate real leaning rate with a linear schedule
        learning_rate = learning_rate_schedule(self.agent.para_list, self.total_steps)

        # vars: update_old_policy_time, train_time, fit_time, surrogate_loss, kl_after, entropy_after, value_loss
        vars = self.agent.fit(self.rollout_paths, learning_rate)

        # update print info:
        self.ep_value["learning_rate"] = (learning_rate[0] + learning_rate[1]) / 2
        self.ep_value["episode_nums"] = self.num_rollouts
        self.ep_value["episode_steps"] = sum([len(path["rewards"]) for path in self.rollout_paths])
        self.ep_value["episode_time"] = sum([path["ep_time"] for path in self.rollout_paths]) / self.num_rollouts
        self.ep_value["episode_reward"] = sum([path["rewards"].sum() for
                                               path in self.rollout_paths]) / self.num_rollouts
        self.ep_value["average_steps"] = int(self.average_len_of_episode)
        self.ep_value["step_time"] = sum([path["step_time"] for path in self.rollout_paths]) / self.num_rollouts
        # copy fit info to the ep_value
        self.ep_value['update_time'], self.ep_value['train_time'], self.ep_value['iter_time'] = vars[0], vars[1], vars[2]
        self.ep_value['surrogate_loss'], self.ep_value['kl_after'] = vars[3], vars[4]
        self.ep_value['entropy_after'], self.ep_value['value_loss'] = vars[5], vars[6]
        self.relative_time = (time.time() - self.start_time) / 60

        # write_log with tensorboard: ['reward', 'kl_div', 'entropy', 'surrogate_loss', 'value_loss']
        self.write_log([self.ep_value['episode_reward'], self.ep_value['kl_after'], self.ep_value['entropy_after'],
                        self.ep_value['surrogate_loss'], self.ep_value['value_loss']], self.iteration)

        # print iteration information:
        on_policy_iteration_print(self.iteration, self.total_steps, self.relative_time, self.ep_value)
        self.history_reward.append(self.ep_value["episode_reward"])

    def rollout_an_episode(self, env):
        global graph
        with graph.as_default():
            path = rollout_an_episode(self.agent, env, self.lock)
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
        while self.total_steps < self.agent.para_list["n_steps"]:
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
            self.num_rollouts = int(self.agent.para_list["timesteps_per_batch"] / self.average_len_of_episode)

        self.agent.saveModel(0)
        return self.history_reward


# parallel remote_env, need farm
class off_policy_parallel_rollouts():
    def __init__(self, agent, env=None):
        self.lock = th.Lock()
        self.agent = agent
        # one and only
        self.farmer = farmer_class(self.agent.para_list)
        self.ep_num = 0
        self.total_steps = 0
        self.history_reward = []
        self.ep_value = {}
        self.value_init()
        self.relative_time = 0
        self.average_steps = self.agent.para_list["max_pathlength"]
        self.log_scalar_name_list = ['mean_reward', 'actor_loss', 'critic_loss']
        self.log_scalar_type_list = [tf.float32, tf.float32, tf.float32]
        self.logger = Logger(self.agent.session, self.agent.para_list["log_path"] + 'train',
                             self.log_scalar_name_list, self.log_scalar_type_list)
        self.write_log = self.logger.create_scalar_log_method()
        self.start_time = time.time()

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

    def process_path(self, path):
        ep_step = len(path["rewards"])
        ep_time = path["ep_time"]
        step_time = path["step_time"]
        ep_reward = sum(path["rewards"])
        ep_memory = []
        for i in range(ep_step):
            t = [path["obs"][i], path["actions"][i], path["rewards"][i], path["dones"][i], path["obs"][i+1]]
            ep_memory.append(t)
        return ep_step, ep_reward, ep_time, step_time, ep_memory

    def rollout_an_episode(self, noise_level, env):  # this function is tread safe
        global graph
        with graph.as_default():
            path = rollout_an_episode(self.agent, env, self.lock, noise_level)
            self.fit()

        ep_step, ep_reward, ep_time, step_time, ep_memory = self.process_path(path)
        for t in ep_memory:
            self.agent.feed_one(t)

        self.lock.acquire()
        self.ep_num += 1
        self.total_steps += ep_step  # self.ep_value["episode_steps"]
        self.average_steps = int(self.total_steps / self.ep_num)
        self.ep_value["average_steps"] = self.average_steps
        self.ep_value['episode_steps'], self.ep_value['episode_reward'] = ep_step, ep_reward
        self.ep_value['episode_time'], self.ep_value['step_time'] = ep_time, step_time
        self.history_reward.append(ep_reward)  # (self.ep_value["episode_reward"])
        self.relative_time = (time.time() - self.start_time) / 60.0
        off_policy_episode_print(self.ep_num, self.total_steps, self.relative_time, self.ep_value)
        self.write_log([ep_reward, self.ep_value["actor_loss"], self.ep_value["critic_loss"]], self.ep_num)
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
        for i in range(self.agent.para_list["n_episodes"]):
            nl = noise_level_schedule(self.agent.para_list, self.total_steps)
            self.rollout_if_available(nl)
            if (i + 1) % 100 == 0:
                self.agent.saveModel(0)
            if self.total_steps > self.agent.para_list["n_steps"]:
                break

        self.agent.saveModel(0)
        return self.history_reward