import time
import threading as th
from model.policy import shared_stochastic_policy_and_value, stochastic_gaussian_policy
from model.value_function import value_function
from model.utils import *


class PPO(object):
    def __init__(self, para_list, observation_space_shape, action_space):
        self.observation_space_shape = observation_space_shape
        self.action_space = action_space
        self.para_list = para_list
        self.lock = th.Lock()  # to make the agent thread safe with distributed training
        self.make_model()

    def make_model(self):
        self.epsilon = 0.2
        self.observation_size = self.observation_space_shape[0]
        self.action_size = np.prod(self.action_space.shape)

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size], "observation")
        self.action = tf.placeholder(tf.float32, [None, self.action_size], "action")
        self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")
        self.returns = tf.placeholder(tf.float32, [None, 1], name="returns")
        self.lr = tf.Variable(self.para_list["policy_learning_rate"], trainable=False)

        # built policy network
        # new policy, will update
        if self.para_list["shared_network"]:
            self.net = shared_stochastic_policy_and_value(
                'policy_and_value', self.session, self.obs, self.action_size, self.para_list, trainable=True)
            self.old_net = shared_stochastic_policy_and_value(
                'old_policy_and_value', self.session, self.obs, self.action_size, self.para_list, trainable=False)
        else:
            self.net = stochastic_gaussian_policy(
                'policy', self.session, self.obs, self.action_size, self.para_list, trainable=True)
            # old policy, don't update
            self.old_net = stochastic_gaussian_policy(
                'old_policy', self.session, self.obs, self.action_size, self.para_list, trainable=False)
            self.vf = value_function('value_function', self.session, self.observation_size, self.para_list,
                                     trainable=True)

        with tf.variable_scope('entropy_pen'):
            self.entropy = tf.reduce_mean(self.net.policy.entropy())
        with tf.variable_scope('kl_divergence'):
            kl_div = tf.distributions.kl_divergence(self.old_net.policy, self.net.policy)
            self.kl = tf.reduce_mean(kl_div)
        with tf.variable_scope('surrogate_loss'):
            self.ratio = self.net.policy.prob(self.action) / (self.old_net.policy.prob(self.action) + 1e-8)
            self.surr = self.ratio * self.advantage    # surrogate loss
            self.clip_value = tf.clip_by_value(self.ratio, 1. - self.epsilon, 1. + self.epsilon)*self.advantage
            self.min_value = tf.minimum(self.surr, self.clip_value)
            self.aloss = -tf.reduce_mean(self.min_value)
            self.aloss -= self.para_list["entropy_coefficient"] * self.entropy
        with tf.variable_scope('train_op'):
            if self.para_list["shared_network"]:
                self.vloss = tf.reduce_mean(tf.square(self.returns - self.net.value))
                self.total_loss = self.aloss + self.para_list["value_coefficient"] * self.vloss
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, var_list=self.net.params)
                self.get_value = self.net.get_value
            else:
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.aloss, var_list=self.net.params)
                self.get_value = self.vf.get_value

        # predict action and value
        self.act = self.net.act

        # methods of getting or setting policy parameters
        self.get_from_flat = self.net.get_params_flat
        self.set_from_flat = self.net.set_params_flat
        self.get_from_flat_old = self.old_net.get_params_flat
        self.set_from_flat_old = self.old_net.set_params_flat

        self.session.run(tf.global_variables_initializer())

        # weights saver
        self.saver = tf.train.Saver()

    def update_old_policy(self):
        start_time = time.time()
        new_theta = self.get_from_flat()
        self.set_from_flat_old(new_theta)
        return time.time() - start_time

    def train_with_loss(self, obs_n, action_n, advant_n, return_n, learning_rate):
        start_time = time.time()
        epochs = self.para_list["policy_opt_epochs"]
        if self.para_list["policy_batch_fit"] is not True:
            feed_dict = {self.obs: obs_n, self.action: action_n, self.advantage: advant_n,
                         self.returns: return_n, self.lr: learning_rate[0]}
            # training with surrogate loss
            for i in range(epochs):
                self.session.run(self.train_op, feed_dict=feed_dict)
        else:
            # fit with epochs:
            nbatch = len(obs_n)
            nbatch_train = self.para_list["policy_opt_batch"]
            inds = np.arange(nbatch)
            for _ in range(epochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    feed_dict = {self.obs: obs_n[mbinds], self.action: action_n[mbinds],
                                 self.advantage: advant_n[mbinds], self.returns: return_n[mbinds],
                                 self.lr: learning_rate[0]}
                    self.session.run(self.train_op, feed_dict=feed_dict)

        feed_dict = {self.obs: obs_n, self.action: action_n, self.returns: return_n, self.advantage: advant_n}

        if self.para_list["shared_network"]:
            value_loss = self.session.run(self.vloss, feed_dict)
        else:
            # train value function based on rollout paths
            value_fit_time, value_loss = self.vf.fit([obs_n, return_n], learning_rate[1])

        [surrogate_loss, kl_after, entropy_after] = self.session.run([self.aloss, self.kl, self.entropy], feed_dict)
        train_time = time.time() - start_time
        return surrogate_loss, kl_after, entropy_after, value_loss, train_time

    def fit(self, paths, learning_rate=(1e-3, 1e-3)):
        start_time = time.time()

        # copy the parameters of new policy to old policy
        update_old_policy_time = self.update_old_policy()

        # get fit data
        obs_n, action_n, return_n, advant_n = path_process(
            paths, self.para_list["gamma"], self.para_list["lam"], self.get_value)

        # optimize policy and value network
        surrogate_loss, kl_after, entropy_after, value_loss, train_time = self.train_with_loss(
            obs_n, action_n, advant_n, return_n, learning_rate)

        # just for debug
        if self.para_list["debug_print"]:
            # data shape:
            print("")
            [r0, surr0, clip0, min0] = self.session.run(
                [self.ratio, self.surr, self.clip_value, self.min_value],
                feed_dict={self.obs: obs_n, self.action: action_n, self.advantage: advant_n})
            print("surro 0: ", surr0[0])
            print("clip  0: ", clip0[0])
            print("min   0: ", min0[0])
            print("ratio 0: ", r0[0])
            print("adv_n 0: ", advant_n[0])
            print("adv_n max: ", advant_n.max(), ", adv_n min: ", advant_n.min())

        fit_time = time.time() - start_time
        return update_old_policy_time, train_time, fit_time, surrogate_loss, kl_after, entropy_after, value_loss

    def loadModel(self, save_path, step):
        if self.saver is not None:
            self.saver.restore(self.session, save_path + "model-" + str(step))
            print("Model loaded successfully")

    def saveModel(self, step):
        if self.saver is not None:
            self.saver.save(self.session, self.para_list["weights_path"] + "model", global_step=step)