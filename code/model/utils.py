import tensorflow as tf
import numpy as np
import math
import scipy.signal


# KL divergence with itself, holding first argument fixed
def gauss_selfKL_firstfixed(mu, logstd):
    mu1, logstd1 = map(tf.stop_gradient, [mu, logstd])
    mu2, logstd2 = mu, logstd

    return gauss_KL(mu1, logstd1, mu2, logstd2)


# probability to take action x, given paramaterized guassian distribution
def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2*logstd)
    gp = -tf.square(x - mu)/(2*var) - .5*tf.log(tf.constant(2*np.pi)) - logstd
    return  tf.reduce_sum(gp, [1])


# KL divergence between two paramaterized guassian distributions
def gauss_KL(mu1, logstd1, mu2, logstd2):
    var1 = tf.exp(2*logstd1)
    var2 = tf.exp(2*logstd2)

    kl = tf.reduce_sum(logstd2 - logstd1 + (var1 + tf.square(mu1 - mu2))/(2*var2) - 0.5)
    return kl


# Shannon entropy for a paramaterized guassian distributions
def gauss_ent(mu, logstd):
    h = tf.reduce_sum(logstd + tf.constant(0.5*np.log(2*np.pi*np.e), tf.float32))
    return h


def discount(x, gamma):
    assert x.ndim >= 1
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def cat_sample(prob_nk):
    assert prob_nk.ndim == 2
    # prob_nk: batchsize x actions
    N = prob_nk.shape[0]
    csprob_nk = np.cumsum(prob_nk, axis=1)
    out = np.zeros(N, dtype='i')
    for (n, csprob_k, r) in zip(range(N), csprob_nk, np.random.rand(N)):
        for (k, csprob) in enumerate(csprob_k):
            if csprob > r:
                out[n] = k
                break
    return out


def slice_2d(x, inds0, inds1):
    inds0 = tf.cast(inds0, tf.int64)
    inds1 = tf.cast(inds1, tf.int64)
    shape = tf.cast(tf.shape(x), tf.int64)
    ncols = shape[1]
    x_flat = tf.reshape(x, [-1])
    return tf.gather(x_flat, inds0 * ncols + inds1)


def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out


class Filter:
    def __init__(self, filter_mean=True):
        self.m1 = 0
        self.v = 0
        self.n = 0.
        self.filter_mean = filter_mean

    def __call__(self, o):
        self.m1 = self.m1 * (self.n / (self.n + 1)) + o * 1/(1 + self.n)
        self.v = self.v * (self.n / (self.n + 1)) + (o - self.m1)**2 * 1/(1 + self.n)
        self.std = (self.v + 1e-6)**.5  # std
        self.n += 1
        if self.filter_mean:
            o1 = (o - self.m1)/self.std
        else:
            o1 = o/self.std
        o1 = (o1 > 10) * 10 + (o1 < -10) * (-10) + (o1 < 10) * (o1 > -10) * o1
        return o1


filter = Filter()
filter_std = Filter()


def numel(x):
    return np.prod(var_shape(x))


def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    # return tf.concat(0, [tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)])
    return tf.concat([tf.reshape(grad, [numel(v)]) for (v, grad) in zip(var_list, grads)], 0)


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    # in numpy
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)
    for i in range(cg_iters):
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v * p
        r -= v * z
        newrdotr = r.dot(r)
        mu = newrdotr / rdotr
        p = r + mu * p
        rdotr = newrdotr
        if rdotr < residual_tol:
            break
    return x


def linesearch(f, x, fullstep, expected_improve_rate):
    accept_ratio = .1
    max_backtracks = 10
    fval = f(x)
    for (_n_backtracks, stepfrac) in enumerate(.5**np.arange(max_backtracks)):
        xnew = x + stepfrac * fullstep
        newfval = f(xnew)
        actual_improve = fval - newfval
        expected_improve = expected_improve_rate * stepfrac
        ratio = actual_improve / expected_improve
        if ratio > accept_ratio and actual_improve > 0:
            return xnew
    return x


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


class GetPolicyWeights(object):
    def __init__(self, session, var_list):
        self.session = session
        self.op = [var for var in var_list if 'policy' in var.name]

    def __call__(self):
        return self.session.run(self.op)


class SetPolicyWeights(object):
    def __init__(self, session, var_list):
        self.session = session
        self.policy_vars = [var for var in var_list if 'policy' in var.name]
        self.placeholders = {}
        self.assigns = []
        for var in self.policy_vars:
            self.placeholders[var.name] = tf.placeholder(tf.float32, var.get_shape())
            self.assigns.append(tf.assign(var, self.placeholders[var.name]))

    def __call__(self, weights):
        feed_dict = {}
        count = 0
        for var in self.policy_vars:
            feed_dict[self.placeholders[var.name]] = weights[count]
            count += 1
        self.session.run(self.assigns, feed_dict)


def xavier_initializer(self, shape):
    dim_sum = np.sum(shape)
    if len(shape) == 1:
        dim_sum += 1
    bound = np.sqrt(6.0 / dim_sum)
    return tf.random_uniform(shape, minval=-bound, maxval=bound)


def fully_connected(input_layer, input_size, output_size, weight_init, bias_init, scope):
    with tf.variable_scope(scope):
        w = tf.get_variable("w", [input_size, output_size], initializer=weight_init)
        # w = tf.Variable(xavier_initializer([input_size, output_size]), name="w")
        b = tf.get_variable("b", [output_size], initializer=bias_init)
    return tf.matmul(input_layer, w) + b


# use tf.layer.dense relu or selu activate function
def fully_connected_network(input_layer, network_shape, act_func='relu', layer_norm=False, trainable=True):
    network_depth = len(network_shape)
    if act_func == 'selu':
        activate_func = tf.nn.selu
    else:
        activate_func = tf.nn.relu

    if layer_norm:
        layer = input_layer
        for i in range(network_depth):
            layer = tf.contrib.layers.layer_norm(
                tf.layers.dense(
                    layer, network_shape[i], activate_func, trainable=trainable))
    else:
        layer = input_layer
        for i in range(network_depth):
            layer = tf.layers.dense(
                layer, network_shape[i], activate_func, trainable=trainable)
    return layer


def create_norm_dist_network(
                            name,
                            input_layer,
                            output_size,
                            network_shape,
                            act_func='relu',
                            layer_norm=False,
                            output_act=tf.nn.tanh,
                            trainable=True
):
    with tf.variable_scope(name):
        layer = fully_connected_network(input_layer, network_shape, act_func, layer_norm, trainable)
        mu = tf.layers.dense(layer, output_size, activation=output_act, trainable=trainable, name='mu')
        sigma = tf.layers.dense(layer, output_size, tf.nn.softplus, trainable=trainable, name='sigma')
        norm_dist = tf.distributions.Normal(loc=mu, scale=sigma, name='norm_dist')
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return norm_dist, params


def create_deterministic_network(
                             name,
                             input_layer,
                             output_size,
                             network_shape,
                             act_func='relu',
                             layer_norm=False,
                             output_act=None,
                             trainable=True
):
    with tf.variable_scope(name):
        layer = fully_connected_network(input_layer, network_shape, act_func, layer_norm, trainable)
        output = tf.layers.dense(layer, output_size, activation=output_act, trainable=trainable, name='output')
    params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
    return output, params


# calculate the T time-steps advantage function A1, A2, A3, ...., AT
# generalized advantage estimate: return advantage && returns
# path: dictionary
def calculate_advantage(path, gamma, lam, value_predict):
    bs = path["obs"]
    ba = path["actions"]
    br = path["rewards"]
    bd = path["dones"]

    T = len(ba)

    last_adv = 0
    advantage = [None] * T
    discounted_return = [None] * T
    for t in reversed(range(T)):  # step T-1, T-2 ... 1
        delta = br[t] + (0 if bd[t] else gamma * value_predict(bs[t + 1])) - value_predict(bs[t])
        advantage[t] = delta + gamma * lam * (1 - bd[t]) * last_adv
        last_adv = advantage[t]

        if t == T - 1:
            discounted_return[t] = br[t] + (0 if bd[t] else gamma * value_predict(bs[t + 1]))
        else:
            discounted_return[t] = br[t] + gamma * (1 - bd[t]) * discounted_return[t + 1]
    return advantage, discounted_return


def calculate_y_target(batch_samples, target_act, target_q_value, gamma):
    [_, _, r1, isdone, s2] = batch_samples
    batch_size = len(r1)
    target_a2 = target_act.act_batch(s2)
    target_q2 = target_q_value.value_batch(s2, target_a2)
    y_target = []
    for i in range(batch_size):
        if isdone[i]:
            y_target.append(r1[i])
        else:
            y_target.append(r1[i] + gamma * target_q2[i])
    y_target = np.resize(y_target, [batch_size, 1])
    return y_target


# get obs_n, action_n, return_n, advant_n from the paths dict
def path_process(paths, gamma, lam, value_predict):
    # is it possible to replace A(s,a) with Q(s,a)?
    for path in paths:
        path["advantage"], path["returns"] = calculate_advantage(
            path, gamma, lam, value_predict)
        # remove the last observation of the batch
        path["obs"] = path["obs"][0:-1]

    # puts all the experiences in a matrix: total_timesteps * options
    obs_n = np.concatenate([path["obs"] for path in paths])
    action_n = np.concatenate([path["actions"] for path in paths])
    return_n = np.concatenate([path["returns"] for path in paths])
    obs_n, action_n, return_n = np.vstack(obs_n), np.vstack(action_n), np.vstack(return_n)

    eps = 1e-8
    # standardize to mean 0 stddev 1
    advant_n = np.concatenate([path["advantage"] for path in paths])
    advant_n -= advant_n.mean()
    advant_n /= (advant_n.std() + eps)
    advant_n = np.vstack(advant_n)
    return obs_n, action_n, return_n, advant_n


# when use the remote_env, the np.array data will make the communication failed
def floatify(np):
    return [float(np[i]) for i in range(len(np))]


# update parameters' value from args:
def parameters_update(para_list, args):
    para_list = {} if para_list is None else para_list
    for key, _ in para_list.items():
        value = getattr(args, key)
        para_list[key] = value


# add parameter space noise
def parameter_space_noise(noise_level, theta, old_theta=None):
    from model.noise import one_fsq_noise
    parameter_noise = one_fsq_noise()
    if noise_level <= 0:
        return theta
    theta_len = len(theta)
    para_noise = parameter_noise.one((theta_len,), noise_level)
    for i in range(theta_len):
        if old_theta is not None:
            theta[i] += math.fabs(theta[i] - old_theta[i]) * para_noise[i]
        else:
            theta[i] += math.fabs(theta[i]) * para_noise[i]
    return theta