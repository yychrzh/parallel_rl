from wrapperEnv import WrapperEnv
import multiprocessing
from utils import *
import time
from random import randint

HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 64


class Actor(multiprocessing.Process):
    def __init__(self, args, task_q, result_q, actor_id, monitor):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.args = args
        self.monitor = monitor
        self.actor_id = actor_id

    # parameters are shared between policy and value network
    def create_shared_network(self, name, state, trainable=True):
        with tf.variable_scope(name):
            if self.args.policy_act_fn == 'selu':
                activate_func = tf.nn.selu
            else:
                activate_func = tf.nn.relu

            if self.args.policy_layer_norm:
                # two hidden layer
                l1 = tf.contrib.layers.layer_norm(
                    tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable), name='layer-1')
                l2 = tf.contrib.layers.layer_norm(
                    tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable), name='layer-2')
            else:
                # two hidden layer
                l1 = tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable, name='layer-1')
                l2 = tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable, name='layer-2')
            mu = tf.layers.dense(l2, self.action_size, tf.nn.tanh, trainable=trainable, name='policy-mu')
            sigma = tf.layers.dense(l2, self.action_size, tf.nn.softplus, trainable=trainable, name='policy-sigma')
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma, name='policy-dist')
            value = tf.layers.dense(l2, 1, trainable=trainable, name='value')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params, value

    def create_policy_network(self, name, state, trainable=True):
        with tf.variable_scope(name):
            if self.args.policy_act_fn == 'selu':
                activate_func = tf.nn.selu
            else:
                activate_func = tf.nn.relu

            if self.args.policy_layer_norm:
                # two hidden layer
                l1 = tf.contrib.layers.layer_norm(
                    tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable), name='policy-l1')
                l2 = tf.contrib.layers.layer_norm(
                    tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable), name='policy-l2')
            else:
                # two hidden layer
                l1 = tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable, name='policy-l1')
                l2 = tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable, name='policy-l2')
            mu = tf.layers.dense(l2, self.action_size, tf.nn.tanh, trainable=trainable, name='policy-mu')
            sigma = tf.layers.dense(l2, self.action_size, tf.nn.softplus, trainable=trainable, name='policy-sigma')
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma, name='policy_dist')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def make_model(self):
        # tensorflow variables (same as in model.py)
        self.observation_size = self.env.observation_space_shape[0]
        self.action_size = np.prod(self.env.action_space.shape)
        self.hidden_size_1 = HIDDEN_SIZE_1
        self.hidden_size_2 = HIDDEN_SIZE_2

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size])

        # built policy network
        # sample_policy
        pi, self.pi_params, value = self.create_shared_network('policy-a', self.obs, trainable=False)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )

        self.session = tf.Session(config=config)

        def act(obs):
            s = np.array(obs)
            s = s[np.newaxis, :]
            a = self.session.run(self.sample_op, {self.obs: s})[0]
            return a

        self.act = act

        # set policy for sample actor
        self.set_policy = SetPolicyWeights(self.session, self.pi_params)
        self.session.run(tf.global_variables_initializer())

    def run(self):
        print(self.actor_id, "actor start")
        self.env = WrapperEnv(visualize=False)
        self.env.seed(randint(0, 899999) + self.actor_id * 10)

        self.make_model()

        last_ob = self.env.reset()
        last_step = 0
        last_reward = 0
        last_real_reward = 0
        while True:
            # get a task, or wait until it gets one
            next_task = self.task_q.get(block=True)
            if next_task == 1:
                # the task is an actor request to collect experience
                path, last_ob, last_step, last_reward, last_real_reward = self.episode_rollout(last_ob, last_step,
                                                                                               last_reward,
                                                                                               last_real_reward)
                self.task_q.task_done()
                self.result_q.put(path)
            elif next_task == 2:
                print("kill message")
                self.task_q.task_done()
                break
            else:
                # the task is to set parameters of the actor policy
                self.set_policy(next_task)
                # super hacky method to make sure when we fill the queue with set parameter tasks,
                # an actor doesn't finish updating before the other actors can accept their own tasks.
                time.sleep(0.1)
                self.task_q.task_done()
        return

    def episode_rollout(self, last_ob, last_step, last_reward, last_real_reward):
        obs, actions, rewards, dones, real_rewards, action_dists_mu, action_dists_logstd = [], [], [], [], [], [], []
        episode_rewards = []
        ob = np.array(last_ob)
        o = []
        sample_step = 0
        for i in range(self.args.sample_length):
            obs.append(ob)
            action = self.act(ob)
            actions.append(action)
            [o, r, d, p] = self.env.step(action)
            ob = np.array(o)
            rewards.append(r)  # (r-p)
            real_rewards.append(r)
            dones.append(d)
            last_reward += r
            last_real_reward += r
            last_step += 1
            sample_step += 1
            if d or last_step > self.args.max_pathlength - 1:
                episode_rewards.append([last_reward, last_real_reward, last_step, sample_step])
                last_step = 0
                last_reward = 0
                last_real_reward = 0
                o = self.env.reset()
                ob = np.array(o)

        obs.append(ob)
        path = {"obs": np.concatenate(np.expand_dims(obs, 0)),
                "rewards": np.array(rewards),
                "real_rewards": np.array(real_rewards),
                "dones": np.array(dones),
                "actions": np.array(actions),
                "episode_rewards": np.array(episode_rewards)
                }
        return path, o, last_step, last_reward, last_real_reward


class ParallelRollout():
    def __init__(self, args):
        self.args = args

        self.tasks = multiprocessing.JoinableQueue()
        self.results = multiprocessing.Queue()

        self.actors = []
        self.actors.append(Actor(self.args, self.tasks, self.results, 9999, args.monitor))

        for i in range(self.args.num_threads - 1):
            self.actors.append(Actor(self.args, self.tasks, self.results, 37 * (i + 3), False))

        print("start all actor")
        for a in self.actors:
            a.start()

            # we will start by running 20,000 / 1000 = 20 episodes for the first ieration

            # self.average_timesteps_in_episode = 200

    def rollout(self):
        # keep 20,000 timesteps per update
        num_rollouts = self.args.num_threads  # self.args.timesteps_per_batch // self.average_timesteps_in_episode
        # print("the next iteration will call %d episodes"%(num_rollouts))

        for i in range(num_rollouts):
            self.tasks.put(1)

        self.tasks.join()

        paths = []
        while num_rollouts:
            num_rollouts -= 1
            paths.append(self.results.get())

        # self.average_timesteps_in_episode = int(sum([len(path["rewards"]) for path in paths]) / len(paths))
        return paths

    def set_policy_weights(self, parameters):
        for i in range(self.args.num_threads):
            self.tasks.put(parameters)
        self.tasks.join()

    def end(self):
        for i in range(self.args.num_threads):
            self.tasks.put(2)