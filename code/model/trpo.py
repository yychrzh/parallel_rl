from model.value_function import *
import math
import time
from model.policy import stochastic_gaussian_policy
from model.value_function import value_function
from model.utils import *
import threading as th


class TRPO(object):
    def __init__(self, para_list, observation_space_shape, action_space):
        self.observation_space_shape = observation_space_shape
        self.action_space = action_space
        self.para_list = para_list
        self.lock = th.Lock()
        self.make_model()

    def make_model(self):
        self.observation_size = self.observation_space_shape[0]
        self.action_size = np.prod(self.action_space.shape)

        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        self.session = tf.Session(config=config)

        self.obs = tf.placeholder(tf.float32, [None, self.observation_size], name='obs')
        self.action = tf.placeholder(tf.float32, [None, self.action_size], name='action')
        self.advantage = tf.placeholder(tf.float32, [None, 1], name='advantage')
        self.learning_rate = tf.Variable(self.para_list["policy_learning_rate"], trainable=False)

        # built policy network
        # new policy, will update
        self.pi = stochastic_gaussian_policy(
            'policy', self.session, self.obs, self.action_size, self.para_list, trainable=True)
        # old policy, don't update
        self.old_pi = stochastic_gaussian_policy(
            'old_policy', self.session, self.obs, self.action_size, self.para_list, trainable=False)

        with tf.variable_scope('entropy_pen'):
            self.entropy = tf.reduce_mean(self.pi.policy.entropy())

        with tf.variable_scope('kl_divergence'):
            kl_div = tf.distributions.kl_divergence(self.old_pi.policy, self.pi.policy)
            self.kl = tf.reduce_mean(kl_div)

        with tf.variable_scope('surrogate_loss'):
            self.ratio = self.pi.policy.prob(self.action) / (self.old_pi.policy.prob(self.action) + 1e-8)
            self.surr = self.ratio * self.advantage
            self.surrogate_loss = -tf.reduce_mean(self.surr)
            self.policy_loss = self.surrogate_loss - self.para_list["entropy_coefficient"] * self.entropy

        self.create_policy_optimization_method()

        # methods of getting or setting policy parameters
        self.get_from_flat = self.pi.get_params_flat
        self.set_from_flat = self.pi.set_params_flat
        self.get_from_flat_old = self.old_pi.get_params_flat
        self.set_from_flat_old = self.old_pi.set_params_flat
        self.session.run(tf.global_variables_initializer())

        self.vf = value_function('value_function', self.session, self.observation_size, self.para_list, trainable=True)
        self.act = self.pi.act

        self.saver = tf.train.Saver()

    def update_old_policy(self):
        start_time = time.time()
        new_theta = self.get_from_flat()
        self.set_from_flat_old(new_theta)
        return time.time() - start_time

    def create_policy_optimization_method(self):
        self.policy_gradients = flatgrad(self.policy_loss, self.pi.params)
        self.kl_gradients = tf.gradients(self.kl, self.pi.params)
        self.flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        # flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
        # shapes = [var.get_shape().as_list() for var in self.pi.policy_params]
        shapes = [self.session.run(tf.shape(v)) for v in self.pi.params]
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            tangents.append(tf.reshape(self.flat_tangent[start:start + size], shape))
            start += size

        # gradient of KL w/ itself * tangent
        self.gvp = [tf.reduce_sum(g * t) for (g, t) in zip(self.kl_gradients, tangents)]
        # 2nd gradient of KL w/ itself * tangent
        self.fvp = flatgrad(self.gvp, self.pi.params)

    def optimize_policy(self, fit_data):
        start_time = time.time()
        [obs_n, action_n, advant_n] = fit_data
        feed_dict = {self.obs: obs_n, self.action: action_n, self.advantage: advant_n}
        # get parameters before optimization
        thprev = self.get_from_flat()

        # computes fisher vector product: F * [self.pg]
        def fisher_vector_product(p):
            feed_dict[self.flat_tangent] = p
            return self.session.run(self.fvp, feed_dict) + p * self.para_list["cg_damping"]

        pg = self.session.run(self.policy_gradients, feed_dict)

        # solve Ax = g, where A is Fisher information metrix and g is gradient of parameters
        # stepdir = A_inverse * g = x
        stepdir = conjugate_gradient(fisher_vector_product, -pg)

        # let stepdir =  change in theta / direction that theta changes in
        # KL divergence approximated by 0.5 x stepdir_transpose * [Fisher Information Matrix] * stepdir
        # where the [Fisher Information Matrix] acts like a metric
        # ([Fisher Information Matrix] * stepdir) is computed using the function,
        # and then stepdir * [above] is computed manually.
        shs = 0.5 * stepdir.dot(fisher_vector_product(stepdir))

        lm = np.sqrt(shs / self.para_list["max_kl"])

        fullstep = stepdir / lm
        negative_g_dot_steppdir = -pg.dot(stepdir)

        # finds best parameter by starting with a big step and working backwards
        # theta = linesearch(loss, thprev, fullstep, negative_g_dot_steppdir / lm)    # this line didn't work
        # i guess we just take a fullstep no matter what
        theta = thprev + fullstep  # theta: new weights, is equal to old weights + improvements
        # set new parameters
        self.set_from_flat(theta)

        [surrogate_loss, kl_after, entropy_after] = self.session.run(
            [self.surrogate_loss, self.kl, self.entropy], feed_dict=feed_dict)
        policy_fit_time = time.time() - start_time
        return surrogate_loss, kl_after, entropy_after, policy_fit_time

    def fit(self, paths, learning_rate=(1e-3, 1e-3)):
        start_time = time.time()

        # copy the parameters of new policy to old policy
        update_old_policy_time = self.update_old_policy()

        # get fit data
        obs_n, action_n, return_n, advant_n = path_process(
            paths, self.para_list["gamma"], self.para_list["lam"], self.vf.get_value)

        # optimize policy network on rollout paths
        surrogate_loss, kl_after, entropy_after, policy_fit_time = self.optimize_policy([obs_n, action_n, advant_n])

        # train value function based on rollout paths
        value_fit_time, value_loss = self.vf.fit([obs_n, return_n], learning_rate[1])

        # just for debug
        if self.para_list["debug_print"]:
            print("")
            r0 = self.session.run(self.ratio, feed_dict={self.obs: obs_n, self.action: action_n})
            surr0 = self.session.run(self.surr,
                                     feed_dict={self.obs: obs_n, self.action: action_n, self.advantage: advant_n})
            print("surro 0: ", surr0[0])
            print("ratio 0: ", r0[0])
            print("adv_n 0: ", advant_n[0])
            print("adv_n max: ", advant_n.max(), ", adv_n min: ", advant_n.min())
            print("value_loss: ", value_loss)

        train_time = policy_fit_time + value_fit_time
        fit_time = time.time() - start_time
        return update_old_policy_time, train_time, fit_time, surrogate_loss, kl_after, entropy_after, value_loss

    def loadModel(self, save_path, step):
        if self.saver is not None:
            self.saver.restore(self.session, save_path + "model-" + str(step))
            print("Model loaded successfully")

    def saveModel(self, step):
        if self.saver is not None:
            self.saver.save(self.session, self.para_list["weights_path"] + "model", global_step=step)