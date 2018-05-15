import tensorflow as tf
from termcolor import *
import threading as th
import time
import numpy as np
from model.rpm import rpm
from model.noise import one_fsq_noise


HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 64
DEBUG_PRINT = False


# deep deterministic policy gradient


class DDPG(object):
    def __init__(self, args, observation_space_shape, action_space):
        self.observation_space_shape = observation_space_shape
        self.action_space = action_space
        self.args = args
        self.rpm = rpm(500000)
        self.training = True
        self.noise_source = one_fsq_noise()
        self.lock = th.Lock()     # to make the agent thread safe with distributed training
        self.make_model()

    # deterministic actor network
    def create_actor_network(self, name, state, trainable=True):
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
            actor = tf.layers.dense(l2, self.action_size, tf.nn.tanh, trainable=trainable, name='actor_output')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return actor, params

    # deterministic actor network
    def create_critic_network(self, name, state, action, trainable=True):
        with tf.variable_scope(name):
            if self.args.value_act_fn == 'selu':
                activate_func = tf.nn.selu
            else:
                activate_func = tf.nn.relu

            if self.args.value_layer_norm:
                la = tf.contrib.layers.layer_norm(
                    tf.layers.dense(action, self.hidden_size_1/2, activate_func, trainable=trainable), name='layer-a')
                # two hidden layer
                l1 = tf.contrib.layers.layer_norm(
                    tf.layers.dense(tf.concat([state, la], 1), self.hidden_size_1, activate_func, trainable=trainable), name='layer-1')
                l2 = tf.contrib.layers.layer_norm(
                    tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable), name='layer-2')
            else:
                la = tf.layers.dense(action, self.hidden_size_1/2, activate_func, trainable=trainable, name='layer-a')
                # two hidden layer
                l1 = tf.layers.dense(tf.concat([state, la], 1), self.hidden_size_1, activate_func, trainable=trainable, name='layer-1')
                l2 = tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable, name='layer-2')
            critic = tf.layers.dense(l2, 1, trainable=trainable, name='critic_output')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return critic, params

    # create tensorboard logs: need to be revised
    def create_log_method(self):
        # create logs method:
        r = tf.placeholder(tf.float32)
        r_r = tf.placeholder(tf.float32)
        kl = tf.placeholder(tf.float32)
        ent = tf.placeholder(tf.float32)
        s_loss = tf.placeholder(tf.float32)
        v_loss = tf.placeholder(tf.float32)
        tf.summary.scalar('reward', r)
        tf.summary.scalar('real_reward', r_r)
        tf.summary.scalar('kl_div', kl)
        tf.summary.scalar('entropy', ent)
        tf.summary.scalar('surrogate_loss', s_loss)
        tf.summary.scalar('value_loss', v_loss)
        # summaries merged
        merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.args.log_path + 'train', self.session.graph)

        def save_summary(summary_data, step):
            [reward, real_reward, kl_div, entropy, surrogate_loss, value_loss] = summary_data
            if reward != 0:
                feed_dict = {r: reward, r_r: real_reward, kl: kl_div, ent: entropy, s_loss: surrogate_loss,
                             v_loss: value_loss}
                summary = self.session.run(merged, feed_dict=feed_dict)
                self.train_writer.add_summary(summary, step)
        return save_summary

    def create_actor_train_method(self):
        self.q_gradient_input = tf.placeholder(tf.float32, [None, self.action_size], 'q_gradient')
        self.actor_parameters_gradients = tf.gradients(self.actor, self.actor_params, -self.q_gradient_input)
        self.actor_train_op = tf.train.AdamOptimizer(self.actor_lr).apply_gradients(zip(self.actor_parameters_gradients, self.actor_params))

    def create_critic_train_method(self):
        # define training optimizer: y = r + discount_factor * q_target_value
        self.y_target = tf.placeholder(tf.float32, [None, 1], 'y_target')
        self.c_loss = tf.reduce_mean(tf.square(self.y_target - self.critic))  # + weight_decay
        self.critic_train_op = tf.train.AdamOptimizer(self.critic_lr).minimize(self.c_loss, var_list=self.critic_params)
        self.action_gradients = tf.gradients(self.critic, self.action)

    def create_soft_update_method(self):
        self.get_actor_flat = GetFlat(self.session, self.actor_params)
        self.get_critic_flat = GetFlat(self.session, self.critic_params)
        self.set_target_actor_flat = SetFromFlat(self.session, self.target_actor_params)
        self.set_target_critic_flat = SetFromFlat(self.session, self.target_critic_params)

    # tau: soft update rat: the Recommended value in the paper is 1e-3
    def update_target_network(self, tau):
        start_time = time.time()
        self.lock.acquire()
        # update target actor network from actor network:
        actor_params_flat = self.get_actor_flat()
        self.set_target_actor_flat(actor_params_flat, tau)
        # update target critic network from critic network:
        critic_params_flat = self.get_critic_flat()
        self.set_target_critic_flat(critic_params_flat, tau)
        self.lock.release()
        cost_time = time.time() - start_time
        return cost_time

    def make_model(self):
        self.soft_update_rate = 1e-3
        self.nbatch_train = self.args.policy_opt_batch
        self.observation_size = self.observation_space_shape[0]
        self.action_size = np.prod(self.action_space.shape)
        self.hidden_size_1 = HIDDEN_SIZE_1
        self.hidden_size_2 = HIDDEN_SIZE_2

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size], "observation")
        self.action = tf.placeholder(tf.float32, [None, self.action_size], "action")
        self.actor_lr = tf.Variable(self.args.policy_learning_rate, trainable=False)
        self.critic_lr = tf.Variable(self.args.value_learning_rate, trainable=False)
        self.actor, self.actor_params = self.create_actor_network('exu_actor', self.obs, trainable=True)
        self.target_actor, self.target_actor_params = self.create_actor_network('target_actor', self.obs, trainable=False)
        self.critic, self.critic_params = self.create_critic_network('exu_critic', self.obs, self.action, trainable=True)
        self.target_critic, self.target_critic_params = self.create_critic_network('target_critic', self.obs, self.action, trainable=False)

        self.create_actor_train_method()
        self.create_critic_train_method()
        self.create_soft_update_method()

        self.session.run(tf.global_variables_initializer())

        def act(obs):
            obs = np.array([obs])
            self.lock.acquire()
            a = self.session.run(self.actor, {self.obs: obs})[0]
            self.lock.release()
            return a

        def noise_act(obs, noise_level):
            obs = np.array([obs])
            self.lock.acquire()
            a = self.session.run(self.actor, {self.obs: obs})[0]
            if noise_level <= 0:
                a_noise = np.zeros(self.action_size)
            else:
                a_noise = self.noise_source.one((self.action_size,), noise_level)
            self.lock.release()
            return np.array(a) + np.array(a_noise)

        def act_batch(obs_batch):
            self.lock.acquire()
            a = self.session.run(self.actor, {self.obs: obs_batch})
            self.lock.release()
            return a

        def target_act_batch(obs_batch):
            self.lock.acquire()
            a = self.session.run(self.target_actor, {self.obs: obs_batch})
            self.lock.release()
            return a

        # predict the value of one state
        def value(obs, action):
            o, a = np.array([obs]), np.array([action])
            self.lock.acquire()
            q_value = self.session.run(self.critic, {self.obs: o, self.action: a})[0][0]
            self.lock.release()
            return q_value

        def value_batch(obs_batch, action_batch):
            self.lock.acquire()
            q_value_batch = self.session.run(self.critic, {self.obs: obs_batch, self.action: action_batch})
            self.lock.release()
            return q_value_batch

        def target_value_batch(obs_batch, action_batch):
            self.lock.acquire()
            q_value_batch = self.session.run(self.target_critic, {self.obs: obs_batch, self.action: action_batch})
            self.lock.release()
            return q_value_batch

        # predict action and value
        self.act = act
        self.noise_act = noise_act
        self.act_batch = act_batch
        self.target_act_batch = target_act_batch
        self.value = value
        self.value_batch = value_batch
        self.target_value_batch = target_value_batch

        # weights saver
        self.saver = tf.train.Saver()
        # tensorboard logs
        self.save_summary = self.create_log_method()

        # copy params from network to target network
        self.update_target_network(1.0)

    def calculate_y_target(self, batch_samples):
        [s1, _, r1, isdone, s2] = batch_samples
        batch_size = len(s1)
        target_a2 = self.target_act_batch(s2)
        target_q2 = self.target_value_batch(s2, target_a2)
        y_target = []
        for i in range(batch_size):
            if isdone[i]:
                y_target.append(r1[i])
            else:
                y_target.append(r1[i] + self.args.gamma * target_q2[i])
        y_target = np.resize(y_target, [batch_size, 1])
        return y_target

    def update_actor_network(self, batch_samples):
        start_time = time.time()
        [s1, _, _, _, _] = batch_samples
        a_predict = self.act_batch(s1)  # predict actions for s1
        self.lock.acquire()
        q_gradient_batch = self.session.run(self.action_gradients,
                                            feed_dict={self.obs: s1, self.action: a_predict})[0]
        self.session.run(self.actor_train_op, feed_dict={self.obs: s1, self.q_gradient_input: q_gradient_batch})
        self.lock.release()
        q_predict = self.value_batch(s1, a_predict)
        actor_loss = -np.concatenate(q_predict).mean()
        # the under code is bug with too much calculate time consumed!
        # self.session.run(tf.reduce_mean(-self.critic), feed_dict={self.obs: s1, self.action: a_predict})
        cost_time = time.time() - start_time
        return actor_loss, cost_time

    def update_critic_network(self, batch_samples):
        start_time = time.time()
        [s1, a1, _, _, _] = batch_samples
        y_target = self.calculate_y_target(batch_samples)
        self.lock.acquire()
        [critic_loss, _] = self.session.run([self.c_loss, self.critic_train_op],
                                            feed_dict={self.obs: s1, self.action: a1, self.y_target: y_target})
        self.lock.release()
        cost_time = time.time() - start_time
        return critic_loss, cost_time

    def feed_one(self, tup):
        self.rpm.add(tup)  # the add method itself is thread safe

    # sample a batch samples from rpm with batch_size
    def sample_one(self, batch_size):
        start_time = time.time()
        samples = self.rpm.sample_batch(batch_size)
        cost_time = time.time() - start_time
        return samples, cost_time

    def fit(self):
        batch_size = self.args.policy_opt_batch
        opt_times = self.args.policy_opt_epochs
        actor_loss, critic_loss = 0, 0
        mean_s_time, mean_a_time, mean_c_time, mean_u_time = 0, 0, 0, 0
        start_time = time.time()
        if self.training and self.rpm.size() > batch_size * 32:
            for i in range(opt_times):
                samples, s_cost_time = self.sample_one(batch_size)
                a_loss, a_cost_time = self.update_actor_network(samples)
                c_loss, c_cost_time = self.update_critic_network(samples)
                u_cost_time = self.update_target_network(self.soft_update_rate)

                actor_loss += a_loss / opt_times
                critic_loss += c_loss / opt_times
                mean_s_time += s_cost_time / opt_times
                mean_a_time += a_cost_time / opt_times
                mean_c_time += c_cost_time / opt_times
                mean_u_time += u_cost_time / opt_times
        fit_cost_time = time.time() - start_time
        return actor_loss, critic_loss, fit_cost_time, mean_s_time, mean_a_time, mean_c_time, mean_u_time

    def run_an_episode(self, noise_level, env):
        steps, steps_time, ep_reward, ep_time = 0, 0, 0, 0
        # [a_loss, c_loss, fit_time, s_time, a_time, c_time, u_time]
        vars = [0, 0, 0, 0, 0, 0, 0]
        episode_memory = []
        start_time = time.time()

        # send current actor parameters to the remote env
        self.lock.acquire()
        actor_params_flat = floatify(self.get_actor_flat())
        self.lock.release()
        env.set([actor_params_flat, noise_level])

        o = env.reset()
        while True:
            o_before_a = o
            # action = floatify(self.noise_act(o_before_a, noise_level))
            before_step_time = time.time()
            nl = [noise_level]
            [o, r, d, a] = env.step(nl)
            step_time = time.time() - before_step_time
            steps_time += step_time
            # self.feed_one([o_before_a, action, r, d, o])
            episode_memory.append([o_before_a, a, r, d, o])
            var = np.zeros(7)  # self.fit()
            ep_reward += r
            steps += 1
            for i, v in enumerate(var):
                vars[i] += v
            if d or steps > (self.args.max_pathlength - 1):
                ep_time = time.time() - start_time
                for i in range(len(vars)):
                    vars[i] /= steps
                break

        # episode_value = {'actor_loss': vars[0], 'critic_loss': vars[1], 'fit_time': vars[2], 'sample_time': vars[3],
        #                  'actor_train_time': vars[4], 'critic_train_time': vars[5], 'target_update_time': vars[6],
        #                  'episode_steps': steps, 'episode_reward': ep_reward, 'episode_time': ep_time,
        #                  'step_time': steps_time / steps}
        # return episode_value
        return steps, ep_reward, ep_time, steps_time / steps, episode_memory

    def saveModel(self, step):
        if self.saver is not None:
            self.lock.acquire()
            self.saver.save(self.session, self.args.weights_path + "model", global_step=step)
            self.lock.release()

    def loadModel(self, save_path, step):
        if self.saver is not None:
            self.lock.acquire()
            self.saver.restore(self.session, save_path + "model-" + str(step))
            self.lock.release()
            print("Model loaded successfully")


# when use the remote_env, the np.array data will make the communication failed
def floatify(np):
    return [float(np[i]) for i in range(len(np))]


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


def numel(x):
    return np.prod(var_shape(x))


class SetFromFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        # shapes = map(var_shape, var_list)  # this is error for python 3.x
        shapes = [self.session.run(tf.shape(v)) for v in var_list]
        self.size = sum(np.prod(shape) for shape in shapes)
        self.tau = tf.Variable(1e-3)
        self.theta = tf.placeholder(tf.float32, [self.size])
        self.own_theta = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)
        self.new_theta = (self.theta - self.own_theta) * self.tau + self.own_theta

        start = 0
        assigns = []
        i = 0
        for (shape, v) in zip(shapes, var_list):
            i += 1
            size = np.prod(shape)
            assigns.append(tf.assign(v, tf.reshape(self.new_theta[start:start + size], shape)))
            start += size
        self.op = tf.group(*assigns)

    def __call__(self, theta, tau=1.0):
        self.session.run(self.op, feed_dict={self.theta: theta, self.tau: tau})


class GetFlat(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)

    def __call__(self):
        return self.op.eval(session=self.session)