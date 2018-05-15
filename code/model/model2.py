from model.value_function import *
import multiprocessing
from termcolor import *
import time

HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 64
DEBUG_PRINT = False


class PPO1(multiprocessing.Process):
    def __init__(self, args, observation_space_shape, action_space, task_q, result_q):
        multiprocessing.Process.__init__(self)
        self.task_q = task_q
        self.result_q = result_q
        self.observation_space_shape = observation_space_shape
        self.action_space = action_space
        self.args = args

    def run(self):
        self.make_model()
        # print("load learning weights")
        # self.loadModel("save_data/2018-04-09/data-01/weights/", 0)
        iteration = 0
        while True:
            paths = self.task_q.get()
            if paths is None:
                # kill the learner
                self.task_q.task_done()
                break
            elif paths == 1:
                # just get params, no learn
                self.task_q.task_done()
                self.result_q.put(self.get_policy())
            elif paths[0] == 2:
                # adjusting the max KL.
                self.args.max_kl = paths[1]
                self.task_q.task_done()
            else:
                iteration += 1
                mean_reward = self.fit(paths, iteration)
                self.task_q.task_done()
                self.result_q.put((self.get_policy(), mean_reward))
                self.saveModel(0)
                # print("Model saved")
                # if iteration > 100:
                #    self.train_writer.close()
                #    print('train end, close the log file')
        return

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
                      tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable), name='policy-1')
                l2 = tf.contrib.layers.layer_norm(
                      tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable), name='policy-2')
            else:
                # two hidden layer
                l1 = tf.layers.dense(state, self.hidden_size_1, activate_func, trainable=trainable, name='policy-1')
                l2 = tf.layers.dense(l1, self.hidden_size_2, activate_func, trainable=trainable, name='policy-2')
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
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma, name='policy-dist')
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

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

    def make_model(self):
        self.epsilon = 0.2
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
        self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")
        self.returns = tf.placeholder(tf.float32, [None, 1], name="returns")
        self.lr = tf.Variable(self.args.policy_learning_rate, trainable=False)

        # built policy network
        # new policy, will update
        pi, self.pi_params, self.value = self.create_shared_network('policy', self.obs, trainable=True)
        # old policy, don't update
        old_pi, self.old_pi_params, old_value = self.create_shared_network('old_policy', self.obs, trainable=False)

        with tf.variable_scope('entropy_pen'):
            self.entropy = tf.reduce_mean(pi.entropy())
        with tf.variable_scope('kl_divergence'):
            kl_div = tf.distributions.kl_divergence(old_pi, pi)
            self.kl = tf.reduce_mean(kl_div)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(self.pi_params, self.old_pi_params)]
        with tf.variable_scope('surrogate_loss'):
            # self.ratio = pi.prob(self.action) / (old_pi.prob(self.action) + 1e-5)
            self.ratio = tf.exp(pi.log_prob(self.action) - old_pi.log_prob(self.action))
            self.surr = self.ratio * self.advantage    # surrogate loss
            self.aloss = -tf.reduce_mean(tf.minimum(
                          self.surr, tf.clip_by_value(self.ratio, 1. - self.epsilon, 1. + self.epsilon)*self.advantage))
            self.aloss -= self.args.entropy_coefficient * self.entropy
            # self.atrain_op = tf.train.AdamOptimizer(self.lr).minimize(self.aloss, var_list=self.pi_params)
        with tf.variable_scope('value_loss'):
            self.vloss = tf.reduce_mean(tf.square(self.returns - self.value))
        with tf.variable_scope('train_op'):
            self.total_loss = self.aloss + self.args.value_coefficient * self.vloss
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.total_loss, var_list=self.pi_params)

        self.session.run(tf.global_variables_initializer())

        def act(obs):
            s = np.array(obs)
            s = s[np.newaxis, :]
            a = self.session.run(self.sample_op, {self.obs: s})[0]
            return a

        # predict the value of one state
        def get_value(state):
            s = np.array(state)
            if s.ndim < 2:
                s = s[np.newaxis, :]
            return self.session.run(self.value, {self.obs: s})[0][0]

        # predict action and value
        self.act = act
        self.get_value = get_value

        # get policy for sample actor
        self.get_policy = GetPolicyWeights(self.session, self.pi_params)
        # weights saver
        self.saver = tf.train.Saver()
        # tensorboard logs
        self.save_summary = self.create_log_method()

    # calculate the T time-steps advantage function A1, A2, A3, ...., AT
    def calculate_advantage(self, path):
        bs = path["obs"]
        ba = path["actions"]
        br = path["rewards"]
        bd = path["dones"]

        # print(bs.shape, ba.shape, br.shape, bd.shape)

        T = len(ba)
        gamma = self.args.gamma
        lam = self.args.lam

        last_adv = 0
        advantage = [None] * T
        discounted_return = [None] * T
        for t in reversed(range(T)):  # step T-1, T-2 ... 1
            delta = br[t] + (0 if bd[t] else gamma * self.get_value(bs[t + 1])) - self.get_value(bs[t])
            advantage[t] = delta + gamma * lam * (1 - bd[t]) * last_adv
            last_adv = advantage[t]

            if t == T - 1:
                discounted_return[t] = br[t] + (0 if bd[t] else gamma * self.get_value(bs[t + 1]))
            else:
                discounted_return[t] = br[t] + gamma * (1 - bd[t]) * discounted_return[t + 1]
        return advantage, discounted_return

    def fit(self, paths, iteration):
        start_time = time.time()
        # is it possible to replace A(s,a) with Q(s,a)? i think is no...
        for path in paths:
            path["advantage"], path["returns"] = self.calculate_advantage(path)
            # remove the last observation of the batch
            path["obs"] = path["obs"][0:-1]

        # puts all the experiences in a matrix: total_timesteps x options
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

        if DEBUG_PRINT:
            # data shape:
            print("")
            r0 = self.session.run(self.ratio, feed_dict={self.obs: obs_n, self.action: action_n})
            surr0 = self.session.run(self.surr, feed_dict={self.obs: obs_n, self.action: action_n, self.advantage: advant_n})
            print("ratio shape: ", r0.shape)
            print("advan shape: ", advant_n.shape)
            print("surro shape: ", surr0.shape)
            print("surro 0: ", surr0[0])
            print("ratio 0: ", r0[0])
            print("adv_n 0: ", advant_n[0])

        # update the old and new policy:
        self.session.run(self.update_oldpi_op)

        if self.args.policy_batch_fit is not True:
            feed_dict = {self.obs: obs_n, self.action: action_n, self.advantage: advant_n,
                         self.returns: return_n, self.lr: self.args.policy_learning_rate}
            # training with surrogate loss
            for i in range(self.args.policy_opt_epochs):
                self.session.run(self.train_op, feed_dict=feed_dict)
        else:
            # fit with epochs:
            nbatch = len(obs_n)
            inds = np.arange(nbatch)
            for _ in range(self.args.policy_opt_epochs):
                np.random.shuffle(inds)
                for start in range(0, nbatch, self.nbatch_train):
                    end = start + self.nbatch_train
                    mbinds = inds[start:end]
                    feed_dict = {self.obs: obs_n[mbinds], self.action: action_n[mbinds],
                                 self.advantage: advant_n[mbinds], self.returns: return_n[mbinds],
                                 self.lr: self.args.policy_learning_rate}
                    self.session.run(self.train_op, feed_dict=feed_dict)

        feed_dict = {self.obs: obs_n, self.action: action_n, self.returns: return_n, self.advantage: advant_n}
        [surrogate_loss, kl_after, entropy_after, value_loss] = self.session.run([self.aloss, self.kl, self.entropy, self.vloss], feed_dict)

        fit_time = time.time() - start_time

        episoderewards = np.array([path["rewards"].sum() for path in paths])
        episode_real_rewards = np.array([path["real_rewards"].sum() for path in paths])
        time_steps = sum([len(path["rewards"]) for path in paths])

        print(" ")
        print(colored("*"*90, "blue"))
        print(colored("iter:{:3d}|".format(iteration), 'green'),
              " reward:{:7.4f}| r_reward:{:7.4f}| ent:{:7.4f}| steps:{:5d}| kl:{:8.6f}".format(
              episoderewards.mean(), episode_real_rewards.mean(), entropy_after, time_steps, kl_after))
        print("s_loss:{:10.8f}| v_loss:{:10.7f}| fit_time:{:.3f} sec|".format(surrogate_loss, value_loss, fit_time))

        def extract_reward(paths):
            reward_list = []
            real_reward_list = []
            for path in paths:
                for episode_reward in path["episode_rewards"]:
                    reward_list.append(episode_reward[0])
                    real_reward_list.append(episode_reward[1])
            reward_lens = len(reward_list)
            if reward_lens:
                r, r_r = np.array(reward_list), np.array(real_reward_list)
                print(colored("%2d episode end| mean reward: %7.4f| mean real reward: %7.4f" % (
                reward_lens, r.mean(), r_r.mean()), "red"))
                return r.mean(), r_r.mean()
            else:
                return 0, 0

        mean_reward, mean_real_reward = extract_reward(paths)

        # if not close the train_writer, some data will lost
        self.save_summary([mean_reward, mean_real_reward, kl_after, entropy_after, surrogate_loss, value_loss], iteration)

        # return stats["Average sum of rewards per episode"]
        return mean_real_reward

    def loadModel(self, save_path, step):
        if self.saver is not None:
            self.saver.restore(self.session, save_path + "model-" + str(step))
            print("Model loaded successfully")

    def saveModel(self, step):
        if self.saver is not None:
            self.saver.save(self.session, self.args.weights_path + "model", global_step=step)

