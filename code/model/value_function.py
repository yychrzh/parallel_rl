import time
from model.utils import *
import threading as th


class q_value_function(object):
    def __init__(self, name, session, observation_size, action_size, para_list=None, trainable=True):
        self.name = name
        self.observation_size = observation_size
        self.action_size = action_size
        self.para_list = para_list
        self.trainable = trainable
        self.session = session
        self.lock = th.Lock()
        self.make_model()

    def create_nerual_network(self):
        with tf.variable_scope(self.name):
            la = fully_connected_network(self.action, [self.para_list["value_network_shape"][0]/2],
                                         self.para_list["value_act_fn"], self.para_list["value_layer_norm"],
                                         trainable=self.trainable)
            mid_layer = fully_connected_network(tf.concat([self.obs, la], 1), self.para_list["value_network_shape"],
                                                self.para_list["value_act_fn"], self.para_list["value_layer_norm"],
                                                trainable=self.trainable)
            value = tf.layers.dense(mid_layer, 1, trainable=self.trainable, name='critic_output')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        return value, params

    def make_model(self):
        self.obs = tf.placeholder(tf.float32, shape=[None, self.observation_size], name="obs")
        self.action = tf.placeholder(tf.float32, shape=[None, self.action_size], name="action")
        self.y_target = tf.placeholder(tf.float32, shape=[None, 1], name="y_target")
        self.value, self.params = self.create_nerual_network()
        self.learning_rate = tf.Variable(1e-3, trainable=False, name='lr')
        self.value_loss = tf.reduce_mean(tf.square(self.y_target - self.value))
        self.action_gradients = tf.gradients(self.value, self.action)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.value_loss, var_list=self.params)
        self.session.run(tf.global_variables_initializer())

    def fit(self, fit_data, learning_rate=1e-3):  # paths: state and returns
        start_time = time.time()
        value_loss = 0
        [observations, actions, y_target] = fit_data
        for _ in range(self.para_list["value_opt_epochs"]):
            feed_dict = {self.obs: observations, self.action: actions,
                         self.y_target: y_target, self.learning_rate: learning_rate}
            res = self.session.run([self.value_loss, self.train_op], feed_dict=feed_dict)
            value_loss += res[0] / self.para_list["value_opt_epochs"]
        value_fit_time = time.time() - start_time
        return value_loss, value_fit_time

    # predict the value of one state
    def get_value(self, state):
        self.lock.acquire()
        s = np.array(state)
        if s.ndim < 2:
            s = s[np.newaxis, :]
            self.lock.release()
        return self.session.run(self.value, {self.obs: s})[0][0]

    # predict the value of a batch of state
    def value_batch(self, obs_batch, action_batch):
        self.lock.acquire()
        q_value_batch = self.session.run(self.value, {self.obs: obs_batch, self.action: action_batch})
        self.lock.release()
        return q_value_batch

    def value_action_gradients(self, obs_bacth, action_batch):
        self.lock.acquire()
        q_gradient_batch = self.session.run(self.action_gradients,
                                            feed_dict={self.obs: obs_bacth, self.action: action_batch})[0]
        self.lock.release()
        return q_gradient_batch


class value_function(object):
    def __init__(self, name, session, observation_size, para_list=None, trainable=True):
        self.name = name
        self.para_list = para_list
        self.trainable = trainable
        self.session = session
        self.observation_size = observation_size
        self.make_model()

    def make_model(self):
        self.state = tf.placeholder(tf.float32, shape=[None, self.observation_size], name="state")
        self.returns = tf.placeholder(tf.float32, shape=[None, 1], name="returns")
        self.value, self.value_params = create_deterministic_network(self.name, self.state, 1,
                                                                     self.para_list["value_network_shape"],
                                                                     self.para_list["value_act_fn"],
                                                                     self.para_list["value_layer_norm"],
                                                                     trainable=self.trainable)
        self.learning_rate = tf.Variable(1e-3, trainable=False, name='lr')
        self.value_loss = tf.reduce_mean(tf.square(self.returns - self.value))
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.value_loss)
        self.session.run(tf.global_variables_initializer())

    def fit(self, fit_data, learning_rate=1e-3):  # paths: state and returns
        start_time = time.time()
        value_loss = 0
        [observations, returns] = fit_data
        if self.para_list["value_batch_fit"] is False:
            for _ in range(self.para_list["value_opt_epochs"]):
                feed_dict = {self.state: observations, self.returns: returns, self.learning_rate: learning_rate}
                res = self.session.run([self.value_loss, self.train_op], feed_dict=feed_dict)
                value_loss += res[0] / self.para_list["value_opt_epochs"]
        else:
            # fit with epochs and minibatchs:
            nbatch = len(returns)
            inds = np.arange(nbatch)
            for _ in range(self.para_list["value_opt_epochs"]):
                np.random.shuffle(inds)
                for start in range(0, nbatch, self.para_list["value_opt_batch"]):
                    end = start + self.para_list["value_opt_batch"]
                    mbinds = inds[start:end]
                    feed_dict = {self.state: observations[mbinds], self.returns: returns[mbinds],
                                 self.learning_rate: learning_rate}
                    res = self.session.run([self.value_loss, self.train_op], feed_dict=feed_dict)
                    value_loss += res[0] / self.para_list["value_opt_epochs"]
        value_fit_time = time.time() - start_time
        return value_fit_time, value_loss

    # predict the value of one state
    def get_value(self, state):
        s = np.array(state)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.session.run(self.value, {self.state: s})[0][0]


class VF(object):
    coeffs = None

    def __init__(self, session, observation_size, args):
        self.session = session
        self.args = args
        self.make_model(observation_size)

    def create_value_network(self, name, state, trainable=True):
        if self.args.policy_act_fn == 'selu':
            activate_func = tf.nn.selu
        else:
            activate_func = tf.nn.relu

        with tf.variable_scope(name):
            if self.args.value_layer_norm:
                l1 = tf.contrib.layers.layer_norm(tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable))
                l2 = tf.contrib.layers.layer_norm(tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable))
            else:
                l1 = tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable)
                l2 = tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable)
            value = tf.layers.dense(l2, 1, trainable=trainable)
        return value

    def make_model(self, shape):
        self.hidden_size_1 = HIDDEN_SIZE_1
        self.hidden_size_2 = HIDDEN_SIZE_2
        print("observation size: ", shape)
        self.state = tf.placeholder(tf.float32, shape=[None, shape], name="state")
        self.returns = tf.placeholder(tf.float32, shape=[None, 1], name="returns")
        self.lr = tf.Variable(1e-3, trainable=False)
        self.value = self.create_value_network('VF', self.state)
        self.v_loss = tf.reduce_mean(tf.square(self.returns - self.value))
        self.train = tf.train.AdamOptimizer(self.lr).minimize(self.v_loss)
        self.session.run(tf.global_variables_initializer())

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        # act = path["action_dists"].astype('float32')
        # l = len(path["rewards"])
        # al = np.arange(l).reshape(-1, 1) / 10.0
        # ret = np.concatenate([o, act, al, np.ones((l, 1))], axis=1)
        ret = o
        return ret

    def fit(self, paths):
        # featmat = np.concatenate([self._features(path) for path in paths])
        observation = np.concatenate([path["obs"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        returns = np.vstack(returns)
        res = []
        if self.args.value_batch_fit is not True:
            for _ in range(self.args.value_opt_epochs):
                res = self.session.run([self.v_loss, self.train], {self.state: observation, self.returns: returns})
        else:
            # fit with epochs and minibatchs:
            nbatch = len(returns)
            inds = np.arange(nbatch)
            for _ in range(self.args.value_opt_epochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, self.args.value_opt_batch):
                    end = start + self.args.value_opt_batch
                    mbinds = inds[start:end]
                    feed_dict = {self.state: observation[mbinds], self.returns: returns[mbinds], self.lr: self.args.value_learning_rate}
                    res = self.session.run([self.v_loss, self.train], feed_dict=feed_dict)
        return res[0]

    def predict(self, path):
        if self.value is None:
            return np.zeros(len(path["rewards"]))
        else:
            ret = self.session.run(self.value, {self.state: self._features(path)})
            return np.reshape(ret, (ret.shape[0], ))

    # predict the value of one state
    def get_value(self, state):
        s = np.array(state)
        if s.ndim < 2:
            s = s[np.newaxis, :]
        return self.session.run(self.value, {self.state: s})[0][0]


class LinearVF(object):
    coeffs = None

    def _features(self, path):
        o = path["obs"].astype('float32')
        o = o.reshape(o.shape[0], -1)
        l = len(path["rewards"])
        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o**2, al, al**2, np.ones((l, 1))], axis=1)

    def fit(self, paths):
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        n_col = featmat.shape[1]
        lamb = 2.0
        self.coeffs = np.linalg.lstsq(featmat.T.dot(featmat) + lamb * np.identity(n_col), featmat.T.dot(returns))[0]

    def predict(self, path):
        return np.zeros(len(path["rewards"])) if self.coeffs is None else self._features(
            path).dot(self.coeffs)