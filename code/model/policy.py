from model.noise import one_fsq_noise
from model.utils import *
import threading as th


class deterministic_policy(object):
    def __init__(self, name, session, state, action_size, para_list=None, trainable=True):
        self.name = name
        self.para_list = para_list
        self.trainable = trainable
        self.session = session
        self.state = state
        self.action_size = action_size
        self.lock = th.Lock()
        self.noise_source = one_fsq_noise()
        self.make_model()

    def make_model(self):
        self.policy, self.params = create_deterministic_network(self.name, self.state, self.action_size,
                                                                       self.para_list["policy_network_shape"],
                                                                       self.para_list["policy_act_fn"],
                                                                       self.para_list["policy_layer_norm"],
                                                                       output_act=tf.nn.tanh,
                                                                       trainable=self.trainable)
        self.learning_rate = tf.Variable(1e-3, trainable=False)

        def act(obs):
            self.lock.acquire()
            s = np.array([obs])
            a = self.session.run(self.policy, {self.state: s})[0]
            self.lock.release()
            return a

        def act_batch(obs_batch):
            self.lock.acquire()
            a = self.session.run(self.policy, {self.state: obs_batch})
            self.lock.release()
            return a

        def noise_act(obs, noise_level):
            self.lock.acquire()
            obs = np.array([obs])
            a = self.session.run(self.policy, {self.state: obs})[0]
            if noise_level <= 0:
                a_noise = np.zeros(self.action_size)
            else:
                a_noise = self.noise_source.one((self.action_size,), noise_level)
            self.lock.release()
            return np.array(a) + np.array(a_noise)

        self.act = act
        self.act_batch = act_batch
        self.noise_act = noise_act
        self.set_params_flat = SetFromFlat(self.session, self.params)
        self.get_params_flat = GetFlat(self.session, self.params)
        self.session.run(tf.global_variables_initializer())


class stochastic_gaussian_policy(object):
    def __init__(self, name, session, state, action_size, para_list=None, trainable=True):
        self.name = name
        self.para_list = para_list
        self.trainable = trainable
        self.session = session
        self.state = state
        self.action_size = action_size
        self.lock = th.Lock()
        self.make_model()

    def make_model(self):
        self.policy, self.params = create_norm_dist_network(self.name, self.state, self.action_size,
                                                                   self.para_list["policy_network_shape"],
                                                                   self.para_list["policy_act_fn"],
                                                                   self.para_list["policy_layer_norm"],
                                                                   output_act=tf.nn.tanh,
                                                                   trainable=self.trainable)
        self.learning_rate = tf.Variable(1e-3, trainable=False)
        self.sample_op = tf.squeeze(self.policy.sample(1), axis=0)  # operation of choosing action

        def act(obs):
            self.lock.acquire()
            s = np.array([obs])
            a = self.session.run(self.sample_op, {self.state: s})[0]
            self.lock.release()
            return a

        self.act = act
        self.set_params_flat = SetFromFlat(self.session, self.params)
        self.get_params_flat = GetFlat(self.session, self.params)
        self.session.run(tf.global_variables_initializer())


class shared_stochastic_policy_and_value(object):
    def __init__(self, name, session, state, action_size, para_list=None, trainable=True):
        self.name = name
        self.para_list = para_list
        self.trainable = trainable
        self.session = session
        self.state = state
        self.action_size = action_size
        self.lock = th.Lock()
        self.make_model()

    def create_neural_network(self):
        with tf.variable_scope(self.name):
            layer = fully_connected_network(self.state, self.para_list["policy_network_shape"],
                                            self.para_list["policy_act_fn"],
                                            self.para_list["policy_layer_norm"],
                                            trainable=self.trainable)
            mu = tf.layers.dense(layer, self.action_size, tf.nn.tanh, trainable=self.trainable, name='pi-mu')
            sigma = tf.layers.dense(layer, self.action_size, tf.nn.softplus, trainable=self.trainable, name='pi-sigma')
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma, name='policy-dist')
            value = tf.layers.dense(layer, 1, trainable=self.trainable, name='value')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        return norm_dist, params, value

    def make_model(self):
        self.policy, self.params, self.value = self.create_neural_network()
        self.learning_rate = tf.Variable(1e-3, trainable=False)
        self.sample_op = tf.squeeze(self.policy.sample(1), axis=0)  # operation of choosing action

        def act(obs):
            self.lock.acquire()
            s = np.array([obs])
            a = self.session.run(self.sample_op, {self.state: s})[0]
            self.lock.release()
            return a

        # predict the value of one state
        def get_value(state):
            self.lock.acquire()
            s = np.array(state)
            if s.ndim < 2:
                s = s[np.newaxis, :]
            self.lock.release()
            return self.session.run(self.value, {self.state: s})[0][0]

        self.act = act
        self.get_value = get_value
        self.set_params_flat = SetFromFlat(self.session, self.params)
        self.get_params_flat = GetFlat(self.session, self.params)
        self.session.run(tf.global_variables_initializer())